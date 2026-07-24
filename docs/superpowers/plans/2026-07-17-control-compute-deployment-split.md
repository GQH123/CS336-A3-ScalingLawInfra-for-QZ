# Control/Compute Deployment Split Implementation Plan

> **Execution note:** Implemented by the primary agent without subagents. Steps use checkbox (`- [x]`) syntax for completed tracking.

**Goal:** Split deployment contracts so the API/control-node backend submits and orchestrates jobs while the compute-node worker image owns Stanford-style training runtime and mounted data access.

**Architecture:** `SCALING_API_CONFIG` becomes the primary control-node config source. It maps API, manifest-default, and QZ submission settings into existing runtime environment keys. Compute-node worker settings move into separate deployment docs/config samples; worker/data validation remains in the worker runtime and data package.

**Tech Stack:** Python config loader, FastAPI runtime factory, pytest, JSON deployment samples, Markdown operations docs.

---

### Task 1: API Config Tests

**Files:**
- Modify: `source/backend/tests/test_course_config.py`
- Modify: `source/backend/tests/test_preflight.py`

- [x] Write tests for `load_api_config_env(...)` using a top-level `api`, `worker_manifest_defaults`, and `qz` schema.
- [x] Assert `SCALING_API_CONFIG` is preferred over `SCALING_COURSE_CONFIG`.
- [x] Assert worker-visible tokenized index URIs are passed through as opaque strings and are not derived from API-local mounted paths.
- [x] Run: `PYTHONPATH=source/backend pytest -q source/backend/tests/test_course_config.py source/backend/tests/test_preflight.py::test_preflight_loads_standard_api_config_file`
- [x] Expected red state: imports or assertions fail because only the combined course config exists.

### Task 2: API Config Loader

**Files:**
- Modify: `source/backend/scaling_backend/course_config.py`
- Modify: `source/backend/scaling_backend/runtime.py`
- Modify: `source/backend/scaling_backend/preflight.py`

- [x] Add `load_api_config_env(...)`.
- [x] Keep `load_course_config_env(...)` as a compatibility wrapper.
- [x] Make `resolve_runtime_env(...)` prefer `SCALING_API_CONFIG`, falling back to `SCALING_COURSE_CONFIG`.
- [x] Map `worker_manifest_defaults.tokenized_*_index_uri` into existing `SCALING_TOKENIZED_*_INDEX_URI` runtime keys without local path checks.
- [x] Run focused config/preflight tests and make them pass.

### Task 3: Split Deployment Artifacts

**Files:**
- Delete: `deploy/course-runtime.sample.json`
- Create: `deploy/control-node/api-runtime.sample.json`
- Create: `deploy/compute-node/worker-runtime.sample.json`
- Create: `deploy/compute-node/data-contract.md`
- Modify: `deploy/README.md`
- Modify: `source/backend/scaling_backend/release_audit.py`
- Modify: `source/backend/tests/test_release_audit.py`

- [x] Add role-specific sample configs and docs.
- [x] Update release audit to require both role-specific deployment packages.
- [x] Run release audit tests and make them pass.

### Task 4: Documentation Alignment

**Files:**
- Modify: `source/backend/README.md`
- Modify: `docs/instructor-operations-guide.md`
- Modify: `docs/provider-adapter-guide.md`
- Modify: `docs/scaling-assignment-sot.md`

- [x] Replace “single course runtime config” language with control-node API config plus compute-node worker/data config.
- [x] State that the API treats worker data URIs as opaque manifest references.
- [x] State that compute nodes own mounted data validation and training backend execution.
- [x] Scan for stale `course-runtime` and `SCALING_COURSE_CONFIG` deployment language.

### Task 5: Verification

**Files:**
- Modify only files from earlier tasks.

- [x] Run: `PYTHONPATH=source/backend pytest -q source/backend/tests`
- [x] Run: `PYTHONPATH=source/data python -m unittest discover -s source/data/tests -v`
- [x] Run: `PYTHONPATH=source/backend python -m scaling_backend.sot_status`
- [x] Run: `PYTHONPATH=source/backend python -m scaling_backend.release_audit`
- [x] Run: `PYTHONPATH=source/backend python -m scaling_backend.readiness`
- [x] Record expected readiness limitation if local deployment env/secrets are absent.
