from __future__ import annotations

from fastapi import FastAPI

from scaling_backend.runtime import RuntimeConfigError, build_app_from_env


def _build_import_safe_app() -> FastAPI:
    try:
        return build_app_from_env()
    except RuntimeConfigError as exc:
        app = FastAPI(title="Scaling Laws Assignment Backend")

        @app.get("/healthz")
        def healthz():
            return {
                "ok": False,
                "error": "runtime_not_configured",
                "message": str(exc),
            }

        return app


app = _build_import_safe_app()
