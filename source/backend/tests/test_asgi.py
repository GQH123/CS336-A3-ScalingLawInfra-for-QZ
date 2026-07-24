import importlib


def test_asgi_entrypoint_exposes_deferred_config_error_app(monkeypatch):
    monkeypatch.delenv("SCALING_STUDENT_KEYS_CSV", raising=False)
    monkeypatch.delenv("SCALING_INTERNAL_CALLBACK_TOKEN", raising=False)
    monkeypatch.delenv("SCALING_ADMIN_API_TOKEN", raising=False)

    import scaling_backend.asgi

    module = importlib.reload(scaling_backend.asgi)
    app = module.app

    assert app.title == "Scaling Laws Assignment Backend"
