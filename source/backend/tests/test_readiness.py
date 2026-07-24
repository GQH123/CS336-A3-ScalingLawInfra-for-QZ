import json

from scaling_backend import readiness


def _env(tmp_path):
    roster = tmp_path / "student_keys.csv"
    roster.write_text("student_id,api_key\nstudent-1,key-1\n", encoding="utf-8")
    return {
        "SCALING_PROVIDER": "fake",
        "SCALING_STUDENT_KEYS_CSV": str(roster),
        "SCALING_INTERNAL_CALLBACK_TOKEN": "internal-token",
        "SCALING_ADMIN_API_TOKEN": "admin-token",
        "SCALING_CALLBACK_URL": "https://backend.example/internal/provider-events",
        "SCALING_MANIFEST_BASE_URI": "memory://manifests",
        "SCALING_LOCAL_MANIFEST_DIR": str(tmp_path / "manifests"),
        "SCALING_STATE_SNAPSHOT_PATH": str(tmp_path / "state" / "snapshot.json"),
        "SCALING_TOTAL_BUDGET_SECONDS": "43200",
        "SCALING_MAX_ACTIVE_EXPERIMENTS_PER_STUDENT": "1",
        "SCALING_MAX_ACTIVE_EXPERIMENTS_GLOBAL": "8",
    }


def test_run_readiness_gate_combines_preflight_release_audit_and_rehearsal(tmp_path):
    report = readiness.run_readiness_gate(
        env=_env(tmp_path),
        create_dirs=True,
        rehearsal_output_dir=tmp_path / "rehearsal",
    )

    assert report["ok"] is True
    assert report["summary"] == {"pass": 4, "warn": 1, "fail": 0}
    assert report["checks"]["preflight"]["status"] == "pass"
    assert report["checks"]["sot_status"]["status"] == "pass"
    assert report["checks"]["release_audit"]["status"] == "pass"
    assert report["checks"]["rehearsal"]["status"] == "pass"
    assert report["preflight"]["ok"] is True
    assert report["sot_status"]["ok"] is True
    assert report["sot_status"]["summary"]["fail"] == 0
    assert report["release_audit"]["ok"] is True
    assert report["rehearsal"]["acceptance_checks"]["final_batch_launched"] is True
    assert (tmp_path / "rehearsal" / "exports" / "grading.csv").exists()


def test_run_readiness_gate_fails_when_rehearsal_acceptance_check_fails(tmp_path):
    def bad_rehearsal(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "rehearsal-report.json"
        path.write_text(
            json.dumps(
                {
                    "acceptance_checks": {
                        "seeded_students": True,
                        "final_batch_launched": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    report = readiness.run_readiness_gate(
        env=_env(tmp_path),
        create_dirs=True,
        rehearsal_output_dir=tmp_path / "rehearsal",
        run_rehearsal=bad_rehearsal,
    )

    assert report["ok"] is False
    assert report["summary"]["fail"] == 1
    assert report["checks"]["rehearsal"]["status"] == "fail"
    assert "final_batch_launched" in report["checks"]["rehearsal"]["message"]


def test_readiness_cli_writes_json_report(tmp_path, capsys):
    output_path = tmp_path / "readiness.json"

    exit_code = readiness.main(
        [
            "--create-dirs",
            "--rehearsal-output-dir",
            str(tmp_path / "rehearsal"),
            "--output-json",
            str(output_path),
        ],
        env=_env(tmp_path),
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == printed
    assert printed["ok"] is True
    assert printed["checks"]["rehearsal"]["status"] == "pass"
