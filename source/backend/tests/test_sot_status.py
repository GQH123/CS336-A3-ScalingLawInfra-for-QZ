import json

from scaling_backend import sot_status


def _by_id(report):
    return {milestone["id"]: milestone for milestone in report["milestones"]}


def test_sot_status_reports_all_implementation_milestones():
    report = sot_status.run_sot_status()
    milestones = _by_id(report)

    assert report["ok"] is True
    assert report["summary"] == {"pass": 7, "warn": 0, "fail": 0}
    assert list(milestones) == ["M0", "M1", "M2", "M2.5", "M3", "M4", "M5"]

    for milestone in report["milestones"]:
        assert milestone["status"] == "pass"
        assert milestone["evidence"]
        assert not milestone["missing"]

    assert "docs/scaling-assignment-sot.md" in milestones["M0"]["evidence"]
    assert "source/backend/tests/test_smoke_submit_to_result.py" in milestones["M1"]["evidence"]
    assert "source/data/tests/test_data_pipeline.py" in milestones["M2.5"]["evidence"]
    assert "source/backend/scaling_backend/readiness.py" in milestones["M5"]["evidence"]


def test_sot_status_fails_with_actionable_missing_evidence(tmp_path):
    report = sot_status.run_sot_status(root=tmp_path)
    first = report["milestones"][0]

    assert report["ok"] is False
    assert report["summary"]["fail"] == len(report["milestones"])
    assert first["status"] == "fail"
    assert "docs/scaling-assignment-sot.md" in first["missing"]
    assert "Missing milestone evidence" in first["message"]


def test_sot_status_cli_writes_json_report(tmp_path, capsys):
    output_path = tmp_path / "sot-status.json"

    exit_code = sot_status.main(["--output-json", str(output_path)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert printed == written
    assert printed["ok"] is True
    assert printed["summary"]["fail"] == 0
