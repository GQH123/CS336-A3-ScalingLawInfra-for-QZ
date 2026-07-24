from __future__ import annotations

import re
import time
from dataclasses import dataclass
from http.cookies import CookieError, SimpleCookie
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from scaling_backend.providers.qz_distributed.crypto import encrypt_password


BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
)


class QzApiError(Exception):
    def __init__(
        self,
        public_message: str,
        *,
        code: int | None = None,
        staff_message: str = "",
    ):
        super().__init__(public_message)
        self.code = code
        self.staff_message = staff_message or public_message


class QzAuthError(QzApiError):
    pass


class QzTransientError(QzApiError):
    pass


@dataclass(frozen=True)
class QzClientConfig:
    base_url: str = "https://qz.sii.edu.cn"
    username: str = ""
    password: str = ""
    cookie: str = ""
    request_timeout_seconds: int = 60
    login_timeout_seconds: int = 30
    login_max_tries: int = 3
    proxy: str = ""


class QzClient:
    def __init__(
        self,
        config: QzClientConfig,
        session_factory: Callable[[], requests.Session] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self._session_factory = session_factory or requests.Session
        self._sleep = sleep
        self._cookie = config.cookie

    @property
    def cookie(self) -> str:
        return self._cookie

    def login_with_cas(self) -> str:
        last_error: QzApiError | None = None
        for attempt in range(self.config.login_max_tries):
            try:
                self._cookie = self._login_with_cas_once()
                return self._cookie
            except QzTransientError as exc:
                last_error = exc
                if attempt < self.config.login_max_tries - 1:
                    self._sleep(min(2.0, 0.5 * (2**attempt)))
        if last_error:
            raise last_error
        raise QzAuthError("QZ login failed", staff_message="CAS login failed")

    def _login_with_cas_once(self) -> str:
        if not self.config.username or not self.config.password:
            raise QzAuthError(
                "QZ credentials are not configured",
                staff_message="missing QZ username or password",
            )

        session = self._session_factory()
        session.headers.update(
            {
                "User-Agent": BROWSER_UA,
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            }
        )
        if self.config.proxy:
            session.trust_env = False
            proxy_url = self.config.proxy.replace("socks5h://", "socks5://")
            session.proxies = {"http": proxy_url, "https": proxy_url}

        try:
            response = session.get(
                self.base_url,
                timeout=self.config.login_timeout_seconds,
                allow_redirects=True,
            )
        except Exception as exc:
            raise QzTransientError(
                "QZ login transport failed", staff_message=str(exc)
            ) from exc
        if response.status_code >= 500:
            raise QzTransientError(
                "QZ login service unavailable",
                code=response.status_code,
                staff_message=response.text,
            )

        current_url = response.url
        if urlparse(current_url).netloc == urlparse(self.base_url).netloc:
            existing = self._cookie_string_from_session(session)
            if self._has_session_cookie(existing):
                return existing

        if "keycloak" in current_url:
            response = self._follow_keycloak_to_cas(session, response)
            current_url = response.url

        if "cas.sii.edu.cn" not in current_url:
            raise QzAuthError(
                "QZ login did not reach CAS",
                staff_message=f"unexpected login URL: {current_url}",
            )

        response = self._submit_cas_login(session, current_url, response.text)
        if "cas.sii.edu.cn" in response.url and "login" in response.url:
            if "用户名或密码错误" in response.text or "账号或密码错误" in response.text:
                raise QzAuthError(
                    "用户名或密码错误", staff_message="CAS rejected username/password"
                )
            if "验证码" in response.text:
                raise QzAuthError(
                    "QZ login requires captcha",
                    staff_message="CAS login page requested captcha",
                )
            raise QzAuthError("QZ login failed", staff_message=response.text[:500])

        if urlparse(response.url).netloc != urlparse(self.base_url).netloc:
            try:
                session.get(
                    self.base_url,
                    timeout=self.config.login_timeout_seconds,
                    allow_redirects=True,
                )
            except Exception as exc:
                raise QzTransientError(
                    "QZ session finalization failed", staff_message=str(exc)
                ) from exc

        cookie = self._cookie_string_from_session(session)
        if not self._has_session_cookie(cookie):
            raise QzAuthError(
                "QZ login did not return a session cookie",
                staff_message="missing session-like qz cookie after CAS login",
            )
        return cookie

    def _follow_keycloak_to_cas(self, session: requests.Session, response):
        match = re.search(
            r'"loginUrl":\s*"([^"]*broker/cas/login[^"]*)"', response.text
        )
        if not match:
            raise QzAuthError(
                "QZ login page did not expose CAS link",
                staff_message="Keycloak page missing broker/cas/login URL",
            )
        cas_url = match.group(1).replace("\\/", "/")
        if not cas_url.startswith("http"):
            parsed = urlparse(response.url)
            cas_url = f"{parsed.scheme}://{parsed.netloc}{cas_url}"
        try:
            return session.get(
                cas_url,
                timeout=self.config.login_timeout_seconds,
                allow_redirects=True,
            )
        except Exception as exc:
            raise QzTransientError(
                "QZ CAS redirect failed", staff_message=str(exc)
            ) from exc

    def _submit_cas_login(self, session: requests.Session, cas_url: str, html: str):
        login_data = {
            "username": self.config.username,
            "password": encrypt_password(self.config.password),
            "_eventId": "submit",
            "submit": "登 录",
            "loginType": "1",
            "encrypted": "true",
        }
        lt = re.search(r'name="lt"\s+value="([^"]+)"', html)
        execution = re.search(r'name="execution"\s+value="([^"]+)"', html)
        if lt:
            login_data["lt"] = lt.group(1)
        if execution:
            login_data["execution"] = execution.group(1)
        try:
            return session.post(
                cas_url,
                data=login_data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "https://cas.sii.edu.cn",
                    "Referer": cas_url,
                },
                timeout=self.config.login_timeout_seconds,
                allow_redirects=True,
            )
        except Exception as exc:
            raise QzTransientError(
                "QZ CAS form submission failed", staff_message=str(exc)
            ) from exc

    @staticmethod
    def _cookie_string_from_session(session: requests.Session) -> str:
        cookies = {}
        for cookie in session.cookies:
            if "qz.sii.edu.cn" in cookie.domain:
                cookies[cookie.name] = cookie.value
        return "; ".join(f"{key}={value}" for key, value in cookies.items())

    @staticmethod
    def _has_session_cookie(cookie_string: str) -> bool:
        return any(
            "session" in part.split("=", 1)[0].strip().lower()
            for part in cookie_string.split(";")
            if part.strip()
        )

    def create_train_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        workspace_id = str(payload.get("workspace_id", ""))
        return self._post_api_v1(
            "/api/v1/train_job/create",
            payload,
            referer=f"{self.base_url}/jobs/distributedTraining?spaceId={workspace_id}",
        )

    def probe_cookie_auth(self, *, workspace_id: str) -> dict[str, Any]:
        if not self._cookie:
            raise QzAuthError(
                "QZ cookie is not configured",
                staff_message="missing QZ_COOKIE for cookie-auth probe",
            )
        return self._post_api_v1(
            "/api/v1/train_job/list",
            {
                "page_num": 1,
                "page_size": 1,
                "workspace_id": workspace_id,
            },
            referer=f"{self.base_url}/jobs/distributedTraining?spaceId={workspace_id}",
            reauth_on_401=False,
        )

    def get_train_job_detail(self, job_id: str) -> dict[str, Any]:
        return self._post_api_v1(
            "/api/v1/train_job/detail",
            {"job_id": job_id},
            referer=f"{self.base_url}/jobs/distributedTrainingDetail/{job_id}",
        )

    def stop_train_job(self, job_id: str) -> bool:
        self._post_api_v1(
            "/api/v1/train_job/stop",
            {"job_id": job_id},
            referer=f"{self.base_url}/jobs/distributedTrainingDetail/{job_id}",
        )
        return True

    def get_train_job_logs(
        self,
        job_id: str,
        *,
        page_size: int = 200,
        pod_names: list[str] | None = None,
        sort: str = "ascend",
    ) -> dict[str, Any]:
        if pod_names is None:
            pod_names = [f"{job_id}-worker-0"]
        body = {
            "page_size": page_size,
            "filter": {"podNames": pod_names},
            "sorter": [
                {"field": "time", "sort": sort},
                {"field": "log-id.keyword", "sort": sort},
            ],
        }
        result = self._post_api_v2("/api/v2/train/GetJobLog", body)
        if isinstance(result.get("Result"), dict):
            return result["Result"]
        return result

    def _post_api_v1(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        referer: str,
        reauth_on_401: bool = True,
    ) -> dict[str, Any]:
        return self._post_json_with_reauth(
            path,
            payload,
            referer=referer,
            v2=False,
            reauth_on_401=reauth_on_401,
        )

    def _post_api_v2(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json_with_reauth(
            path, payload, referer=f"{self.base_url}/", v2=True
        )

    def _post_json_with_reauth(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        referer: str,
        v2: bool,
        reauth_on_401: bool = True,
        _retried: bool = False,
    ) -> dict[str, Any]:
        if not self._cookie:
            if not self.config.username or not self.config.password:
                raise QzAuthError(
                    "QZ cookie is not configured",
                    staff_message=(
                        "missing QZ_COOKIE; configure QZ_COOKIE or CAS credentials"
                    ),
                )
            self.login_with_cas()
        try:
            response = self._session_factory().post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._browser_json_headers(referer=referer, v2=v2),
                timeout=self.config.request_timeout_seconds,
            )
        except Exception as exc:
            raise QzTransientError(
                "QZ API transport failed", staff_message=str(exc)
            ) from exc

        self._refresh_cookie_from_response(response)

        if response.status_code == 401:
            if _retried or not reauth_on_401:
                raise QzAuthError(
                    "QZ session expired",
                    code=401,
                    staff_message=(
                        "QZ returned 401; refresh QZ_COOKIE or configure CAS "
                        "credentials for automatic re-authentication"
                    ),
                )
            if not self.config.username or not self.config.password:
                raise QzAuthError(
                    "QZ session expired",
                    code=401,
                    staff_message=(
                        "QZ returned 401; refresh QZ_COOKIE or configure CAS "
                        "credentials for automatic re-authentication"
                    ),
                )
            self.login_with_cas()
            return self._post_json_with_reauth(
                path,
                payload,
                referer=referer,
                v2=v2,
                reauth_on_401=reauth_on_401,
                _retried=True,
            )
        if response.status_code >= 500:
            raise QzTransientError(
                "QZ API service unavailable",
                code=response.status_code,
                staff_message=response.text,
            )
        if response.status_code != 200:
            raise QzApiError(
                "QZ API request failed",
                code=response.status_code,
                staff_message=response.text,
            )

        try:
            result = response.json()
        except Exception as exc:
            raise QzApiError(
                "QZ API response was not valid JSON", staff_message=str(exc)
            ) from exc

        code = result.get("code")
        if code not in (None, 0, "0"):
            raise QzApiError(
                "QZ API request failed",
                code=code,
                staff_message=str(result.get("message", result)),
            )
        return result.get("data", result)

    def _browser_json_headers(self, *, referer: str, v2: bool) -> dict[str, str]:
        headers = {
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json",
            "cookie": self._cookie,
            "origin": self.base_url,
            "referer": referer,
            "user-agent": BROWSER_UA,
        }
        if v2:
            headers["x-inspire-client-source"] = "scaling-backend/qz-distributed"
        return headers

    def _refresh_cookie_from_response(self, response: Any) -> None:
        updates = self._cookie_updates_from_response(response)
        if not updates:
            return

        cookies = self._cookie_pairs_from_cookie_header(self._cookie)
        for name, value in updates:
            if value is None:
                cookies.pop(name, None)
            else:
                cookies[name] = value
        self._cookie = "; ".join(f"{name}={value}" for name, value in cookies.items())

    @classmethod
    def _cookie_updates_from_response(
        cls, response: Any
    ) -> list[tuple[str, str | None]]:
        updates: list[tuple[str, str | None]] = []
        for header_value in cls._set_cookie_header_values(response):
            for set_cookie in cls._split_set_cookie_header(header_value):
                parsed = SimpleCookie()
                try:
                    parsed.load(set_cookie)
                except CookieError:
                    continue
                for name, morsel in parsed.items():
                    if not name:
                        continue
                    max_age = str(morsel["max-age"]).strip()
                    expires = str(morsel["expires"]).strip()
                    is_delete = max_age == "0" or (
                        morsel.value == "" and bool(expires)
                    )
                    updates.append((name, None if is_delete else morsel.value))
        return updates

    @staticmethod
    def _set_cookie_header_values(response: Any) -> list[str]:
        values: list[str] = []

        raw_headers = getattr(getattr(response, "raw", None), "headers", None)
        for method_name in ("getlist", "get_all"):
            getter = getattr(raw_headers, method_name, None)
            if not getter:
                continue
            for value in getter("Set-Cookie") or []:
                text = str(value).strip()
                if text and text not in values:
                    values.append(text)

        headers = getattr(response, "headers", {}) or {}
        header_items = headers.items() if hasattr(headers, "items") else []
        for key, value in header_items:
            if str(key).lower() != "set-cookie":
                continue
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
        return values

    @staticmethod
    def _split_set_cookie_header(header_value: str) -> list[str]:
        parts: list[str] = []
        start = 0
        for index, char in enumerate(header_value):
            if char != ",":
                continue
            remainder = header_value[index + 1 :].lstrip()
            first_attr = remainder.split(";", 1)[0]
            if re.match(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+=", first_attr):
                parts.append(header_value[start:index].strip())
                start = index + 1
        parts.append(header_value[start:].strip())
        return [part for part in parts if part]

    @staticmethod
    def _cookie_pairs_from_cookie_header(cookie_header: str) -> dict[str, str]:
        cookies: dict[str, str] = {}
        parsed = SimpleCookie()
        try:
            parsed.load(cookie_header)
        except CookieError:
            parsed = SimpleCookie()
        for name, morsel in parsed.items():
            if name:
                cookies[name] = morsel.value
        if cookies:
            return cookies

        for part in cookie_header.split(";"):
            item = part.strip()
            if "=" not in item:
                continue
            name, value = item.split("=", 1)
            name = name.strip()
            if name:
                cookies[name] = value.strip()
        return cookies
