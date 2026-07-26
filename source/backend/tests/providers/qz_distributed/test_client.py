import pytest

from scaling_backend.providers.qz_distributed.client import (
    QzApiError,
    QzAuthError,
    QzClient,
    QzClientConfig,
    QzTransientError,
)


class FakeResponse:
    def __init__(
        self,
        status_code=200,
        payload=None,
        text="",
        url="https://qz.sii.edu.cn",
        headers=None,
    ):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.url = url
        self.headers = {"content-type": "application/json"}
        if headers:
            self.headers.update(headers)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeCookie:
    def __init__(self, name, value, domain="qz.sii.edu.cn"):
        self.name = name
        self.value = value
        self.domain = domain


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.cookies = []
        self.get_responses = []
        self.post_responses = []
        self.posts = []
        self.gets = []
        self.trust_env = True
        self.proxies = {}

    def get(self, url, timeout=None, allow_redirects=None):
        self.gets.append(
            {"url": url, "timeout": timeout, "allow_redirects": allow_redirects}
        )
        response = self.get_responses.pop(0)
        if callable(response):
            return response()
        return response

    def post(self, url, json=None, data=None, headers=None, timeout=None, allow_redirects=None):
        call = {
            "url": url,
            "json": json,
            "data": data,
            "headers": headers or {},
            "timeout": timeout,
            "allow_redirects": allow_redirects,
        }
        self.posts.append(call)
        response = self.post_responses.pop(0)
        if callable(response):
            return response(call)
        return response


def _client(
    session=None,
    cookie="session=old",
    password="a" * 256,
    cookie_file_path="",
) -> QzClient:
    return QzClient(
        QzClientConfig(
            base_url="https://qz.sii.edu.cn",
            username="253108120093",
            password=password,
            cookie=cookie,
            cookie_file_path=cookie_file_path,
        ),
        session_factory=lambda: session or FakeSession(),
    )


def test_login_with_cas_posts_encrypted_form_and_returns_qz_session_cookie():
    session = FakeSession()
    session.cookies = [FakeCookie("inspire-session", "abc")]
    session.get_responses = [
        FakeResponse(
            url="https://keycloak.sii.edu.cn/realms/qz",
            text='"loginUrl": "/realms/qz/broker/cas/login"',
        ),
        FakeResponse(
            url="https://cas.sii.edu.cn/login",
            text='<input name="lt" value="LT-1"><input name="execution" value="EX-1">',
        ),
    ]
    session.post_responses = [FakeResponse(url="https://qz.sii.edu.cn")]

    cookie = _client(session=session, cookie="").login_with_cas()

    assert cookie == "inspire-session=abc"
    post = session.posts[0]
    assert post["url"] == "https://cas.sii.edu.cn/login"
    assert post["data"]["username"] == "253108120093"
    assert post["data"]["encrypted"] == "true"
    assert post["data"]["password"] == "a" * 256
    assert post["data"]["lt"] == "LT-1"
    assert post["data"]["execution"] == "EX-1"


def test_login_with_cas_raises_auth_error_for_bad_password_page():
    session = FakeSession()
    session.get_responses = [
        FakeResponse(
            url="https://keycloak.sii.edu.cn/realms/qz",
            text='"loginUrl": "/realms/qz/broker/cas/login"',
        ),
        FakeResponse(url="https://cas.sii.edu.cn/login", text=""),
    ]
    session.post_responses = [
        FakeResponse(url="https://cas.sii.edu.cn/login", text="用户名或密码错误")
    ]

    with pytest.raises(QzAuthError, match="用户名或密码错误"):
        _client(session=session, cookie="").login_with_cas()


def test_post_api_v1_reauthenticates_once_on_401_and_retries_request():
    session = FakeSession()
    session.cookies = [FakeCookie("inspire-session", "fresh")]
    session.get_responses = [
        FakeResponse(
            url="https://keycloak.sii.edu.cn/realms/qz",
            text='"loginUrl": "/realms/qz/broker/cas/login"',
        ),
        FakeResponse(url="https://cas.sii.edu.cn/login", text=""),
    ]
    session.post_responses = [
        FakeResponse(status_code=401, payload={"code": 401, "message": "expired"}),
        FakeResponse(url="https://qz.sii.edu.cn"),
        FakeResponse(payload={"code": 0, "data": {"job_id": "job-1"}}),
    ]
    client = _client(session=session, cookie="session=old")

    result = client.create_train_job({"workspace_id": "ws-1", "name": "job"})

    assert result == {"job_id": "job-1"}
    assert session.posts[0]["headers"]["cookie"] == "session=old"
    assert session.posts[2]["headers"]["cookie"] == "inspire-session=fresh"


def test_create_train_job_uses_distributed_training_endpoint_and_browser_headers():
    session = FakeSession()
    session.post_responses = [
        FakeResponse(payload={"code": 0, "data": {"job_id": "job-1"}})
    ]
    payload = {"workspace_id": "ws-1", "name": "job"}

    result = _client(session=session).create_train_job(payload)

    assert result == {"job_id": "job-1"}
    post = session.posts[0]
    assert post["url"] == "https://qz.sii.edu.cn/api/v1/train_job/create"
    assert post["json"] == payload
    assert post["headers"]["cookie"] == "session=old"
    assert post["headers"]["referer"].endswith("/jobs/distributedTraining?spaceId=ws-1")


def test_api_refreshes_cookie_from_set_cookie_header_before_next_request():
    session = FakeSession()
    session.post_responses = [
        FakeResponse(
            payload={"code": 0, "data": {"job_id": "job-1"}},
            headers={
                "Set-Cookie": (
                    "inspire-session=fresh; Path=/; HttpOnly, "
                    "throttle-token=ok; Max-Age=30; Path=/"
                )
            },
        ),
        FakeResponse(payload={"code": 0, "data": {"status": "RUNNING"}}),
    ]
    client = _client(session=session, cookie="inspire-session=old; stable=keep")

    assert client.create_train_job({"workspace_id": "ws-1"}) == {"job_id": "job-1"}
    assert client.cookie == "inspire-session=fresh; stable=keep; throttle-token=ok"
    assert client.get_train_job_detail("job-1") == {"status": "RUNNING"}
    assert session.posts[1]["headers"]["cookie"] == (
        "inspire-session=fresh; stable=keep; throttle-token=ok"
    )


def test_client_loads_cookie_from_file_when_config_cookie_is_absent(tmp_path):
    cookie_file = tmp_path / "qz.cookie"
    cookie_file.write_text("inspire-session=file-fresh\n", encoding="utf-8")

    client = _client(cookie="", cookie_file_path=str(cookie_file))

    assert client.cookie == "inspire-session=file-fresh"


def test_config_cookie_takes_precedence_over_cookie_file(tmp_path):
    cookie_file = tmp_path / "qz.cookie"
    cookie_file.write_text("inspire-session=file-stale\n", encoding="utf-8")

    client = _client(
        cookie="inspire-session=env-fresh",
        cookie_file_path=str(cookie_file),
    )

    assert client.cookie == "inspire-session=env-fresh"


def test_api_refresh_persists_cookie_file_after_set_cookie(tmp_path):
    cookie_file = tmp_path / "qz.cookie"
    session = FakeSession()
    session.post_responses = [
        FakeResponse(
            payload={"code": 0, "data": {"job_id": "job-1"}},
            headers={"Set-Cookie": "inspire-session=fresh; Path=/; HttpOnly"},
        )
    ]
    client = _client(
        session=session,
        cookie="inspire-session=old; stable=keep",
        cookie_file_path=str(cookie_file),
    )

    assert client.create_train_job({"workspace_id": "ws-1"}) == {"job_id": "job-1"}

    assert cookie_file.read_text(encoding="utf-8") == (
        "inspire-session=fresh; stable=keep\n"
    )
    assert oct(cookie_file.stat().st_mode & 0o777) == "0o600"


def test_login_with_cas_persists_cookie_file(tmp_path):
    cookie_file = tmp_path / "qz.cookie"
    session = FakeSession()
    session.cookies = [FakeCookie("inspire-session", "fresh")]
    session.get_responses = [
        FakeResponse(
            url="https://keycloak.sii.edu.cn/realms/qz",
            text='"loginUrl": "/realms/qz/broker/cas/login"',
        ),
        FakeResponse(url="https://cas.sii.edu.cn/login", text=""),
    ]
    session.post_responses = [FakeResponse(url="https://qz.sii.edu.cn")]

    cookie = _client(
        session=session,
        cookie="",
        cookie_file_path=str(cookie_file),
    ).login_with_cas()

    assert cookie == "inspire-session=fresh"
    assert cookie_file.read_text(encoding="utf-8") == "inspire-session=fresh\n"


def test_probe_cookie_auth_uses_read_only_train_job_list_endpoint():
    session = FakeSession()
    session.post_responses = [
        FakeResponse(payload={"code": 0, "data": {"list": [], "total": 0}})
    ]
    client = _client(session=session, cookie="inspire-session=manual")

    result = client.probe_cookie_auth(workspace_id="ws-1")

    assert result == {"list": [], "total": 0}
    assert session.gets == []
    assert len(session.posts) == 1
    post = session.posts[0]
    assert post["url"] == "https://qz.sii.edu.cn/api/v1/train_job/list"
    assert post["json"] == {
        "page_num": 1,
        "page_size": 1,
        "workspace_id": "ws-1",
    }
    assert post["headers"]["cookie"] == "inspire-session=manual"
    assert post["headers"]["referer"].endswith(
        "/jobs/distributedTraining?spaceId=ws-1"
    )


def test_api_accepts_string_zero_success_code():
    session = FakeSession()
    session.post_responses = [
        FakeResponse(payload={"code": "0", "data": {"job_id": "job-1"}})
    ]

    result = _client(session=session).create_train_job({"workspace_id": "ws-1"})

    assert result == {"job_id": "job-1"}


def test_get_detail_stop_and_logs_call_expected_platform_surfaces():
    session = FakeSession()
    session.post_responses = [
        FakeResponse(payload={"code": 0, "data": {"status": "RUNNING"}}),
        FakeResponse(payload={"code": 0, "data": {}}),
        FakeResponse(payload={"Result": {"logs": [{"message": "hello"}], "total": 1}}),
    ]
    client = _client(session=session)

    assert client.get_train_job_detail("job-1") == {"status": "RUNNING"}
    assert client.stop_train_job("job-1") is True
    assert client.get_train_job_logs("job-1") == {
        "logs": [{"message": "hello"}],
        "total": 1,
    }

    assert session.posts[0]["url"] == "https://qz.sii.edu.cn/api/v1/train_job/detail"
    assert session.posts[1]["url"] == "https://qz.sii.edu.cn/api/v1/train_job/stop"
    assert session.posts[2]["url"] == "https://qz.sii.edu.cn/api/v2/train/GetJobLog"
    assert session.posts[2]["json"]["filter"]["podNames"] == ["job-1-worker-0"]


def test_api_error_redacts_provider_details_from_public_message():
    session = FakeSession()
    session.post_responses = [
        FakeResponse(payload={"code": 1001, "message": "quota_id secret failure"})
    ]

    with pytest.raises(QzApiError) as exc_info:
        _client(session=session).create_train_job({"workspace_id": "ws-1"})

    assert str(exc_info.value) == "QZ API request failed"
    assert exc_info.value.staff_message == "quota_id secret failure"
    assert exc_info.value.code == 1001


def test_transient_transport_error_is_typed_for_retry_policy():
    session = FakeSession()

    def raise_timeout(_call):
        raise TimeoutError("network timeout")

    session.post_responses = [raise_timeout]

    with pytest.raises(QzTransientError):
        _client(session=session).create_train_job({"workspace_id": "ws-1"})
