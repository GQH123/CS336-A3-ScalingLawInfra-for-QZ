import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = ROOT / "docs" / "examples" / "scaling_laws_student_workflow.ipynb"
ZH_NOTEBOOK_PATH = (
    ROOT / "docs" / "examples" / "scaling_laws_student_workflow.zh-CN.ipynb"
)
NOTEBOOK_PATHS = [NOTEBOOK_PATH, ZH_NOTEBOOK_PATH]
QUICKSTART_PATHS = [
    ROOT / "docs" / "scaling-laws-api-quickstart.md",
    ROOT / "docs" / "scaling-laws-api-quickstart.zh-CN.md",
]
LATEX_QUICKSTART_PATHS = [
    ROOT / "docs" / "latex" / "scaling-laws-api-quickstart.tex",
    ROOT / "docs" / "latex" / "scaling-laws-api-quickstart.zh-CN.tex",
]


def _notebook_text() -> str:
    return NOTEBOOK_PATH.read_text(encoding="utf-8")


def _all_notebook_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in NOTEBOOK_PATHS)


def _all_copyable_quickstart_text() -> str:
    paths = [*NOTEBOOK_PATHS, *QUICKSTART_PATHS, *LATEX_QUICKSTART_PATHS]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_student_workflow_notebook_is_valid_and_release_safe():
    stale_patterns = [
        "X-API-Key",
        "architecture_config",
        "optimizer_config",
        "status_type",
        "total_train_tokens",
        "max_runtime_seconds",
        '"detail"',
        "ScalingApiClient",
        "small_example_config",
        "scaling_backend.student_client",
        "student_client.py",
        "source/backend",
        "PYTHONPATH",
    ]

    for path in NOTEBOOK_PATHS:
        notebook = json.loads(path.read_text(encoding="utf-8"))

        assert notebook["nbformat"] >= 4
        assert notebook["metadata"]["kernelspec"]["language"] == "python"
        assert notebook["metadata"]["language_info"]["name"] == "python"

        text = path.read_text(encoding="utf-8")
        for pattern in stale_patterns:
            assert pattern not in text

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                compile("".join(cell["source"]), str(path), "exec")


def test_student_workflow_notebook_covers_submit_poll_analyze_and_final_submission():
    required_snippets = [
        "import requests",
        "api_base_url",
        "api_key",
        "requests.request(",
        "def api_request(",
        "def make_small_config(",
        "num_attention_heads",
        "def wait_for_experiment(",
        "SAMPLE_BUDGET_RESPONSE",
        "SAMPLE_SUBMIT_SUCCESS_RESPONSE",
        "SAMPLE_DUPLICATE_CONFIG_RESPONSE",
        "SAMPLE_INVALID_CONFIG_RESPONSE",
        "SAMPLE_COMPLETED_EXPERIMENT",
        "SAMPLE_FAILED_EXPERIMENT",
        'api_request("GET", "/budget")',
        "api_request(",
        '"POST",',
        '"/submit"',
        '"requested_runtime_seconds": 300',
        'api_request("GET", "/experiments")',
        '"/final_submission"',
        "final_validation_loss",
        "validation_losses",
        "predicted_final_loss_lower",
        "predicted_final_loss_upper",
    ]
    for path in NOTEBOOK_PATHS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for snippet in required_snippets:
            assert snippet in code


def test_copyable_student_api_helpers_print_error_json_without_http_tracebacks():
    text = _all_copyable_quickstart_text()

    assert "raise_for_status" not in text
    assert "raw_response" in text
    assert "status_code < 200 or response.status_code >= 300" in text
    assert "SystemExit(1)" in text


def test_student_workflow_notebook_explains_validation_loss_history():
    text = _all_notebook_text()

    assert "validation-loss history" in text
    assert "not a prediction interval" in text
    assert "有序验证损失历史" in text
    assert "不是预测区间" in text
    assert "not `[lower, upper]`" in text
    assert "final_validation_loss` is the scalar result" in text
