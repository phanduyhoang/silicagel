# Session Summary (Detailed Handoff)

This document summarizes all major work completed in this chat, including project reorganization, app isolation, telemetry additions, log analysis findings, and code-level fixes made to address long-run latency drift.

---

## 1) Initial User Goal

You asked to:

1. Organize everything related to `backend/web_line_motion_infer_dual_yolo_history.py` into a new folder.
2. Keep old files (do not delete existing project data initially).
3. Split/clean the large codebase into a more maintainable structure.
4. Include relevant models and stored images.
5. Later, enforce **no legacy bridge**: new folder must be self-contained and runnable independently.
6. Improve defect history browsing performance.
7. Add detailed GPU/runtime logging to debug processing time increase over long runs.

---

## 2) Workspace Created

Created isolated workspace:

- `backend/line_motion_dual_yolo_workspace`

Key current structure:

- `app/`
  - `main.py`
  - `server.py`
  - `modbus_tcp_send.py`
  - `__init__.py`
- `assets/models/`
  - `yolo/best_finetune1902_v3_beads4x.pt`
  - `classifier/best_classifier_20260219_123823.pt`
- `data/history/`
  - `beads_defect_frames_live/`
  - `beads_defect_frames_live_annotated/`
  - `beads_package_crops_live/`
  - `history_index.sqlite` (created/used by app)
- `data/logs/`
  - `gpu_runtime_metrics_*.csv`
  - `runtime_events_*.jsonl`
  - `hardware_info.json`

Old app files were not used by runtime after isolation updates.

---

## 3) Isolation Work (No Legacy Dependency)

### What was changed for isolation

1. `app/main.py` now imports local server app only:
   - `from .server import app`
2. `app/server.py` is standalone and no longer imports from old backend app module.
3. Modbus helper made local:
   - added `app/modbus_tcp_send.py`
   - switched pulse call to local import (`from . import modbus_tcp_send`)
4. Default runtime paths were moved to this workspace:
   - model paths -> `../assets/models/...`
   - output/history image paths -> `../data/history/...`
5. `legacy/` folder was removed from this workspace when you requested clean-only setup.

### Isolation verification done

Search checks were run to ensure no references to:
- `backend.web_line_motion_infer_dual_yolo_history`
- `bead_detection_interface`

inside the workspace app/docs after cleanup.

---

## 4) History Performance Upgrade

You asked to make defect history serving fast/correct, potentially DB-backed.

### Implemented approach

Implemented local SQLite index for annotated defect images:

- DB file: `data/history/history_index.sqlite`
- Table: `defect_history`
- Indexed fields include:
  - filename
  - ts_ms
  - day
  - minute_of_day
  - shift_key
  - shift_label
  - mtime_ns

### Runtime behavior

1. Annotated directory is scanned incrementally.
2. New/changed files are upserted by mtime.
3. Deleted files are removed from DB index.
4. `/history-data` now queries DB for pagination/filters instead of repeatedly scanning all image files.

### Added endpoint

- `POST /history/reindex`
  - force reindex
  - returns indexed item count and DB path

---

## 5) Telemetry/Diagnostics Added

You requested detailed long-run diagnostics for 5-12 hour monitoring.

### Main telemetry files

1. `gpu_runtime_metrics_*.csv`
2. `runtime_events_*.jsonl`
3. `hardware_info.json`

### APIs added

- `POST /metrics/start?hours=<H>&interval_s=<S>`
- `GET /metrics/status`
- telemetry paths also included in `/status`

### Logged metric categories

#### Existing + expanded runtime metrics
- state, result id, session counters
- capture/infer/total timings
- camera staleness ages
- result age
- python active thread count
- GC counters

#### PyTorch CUDA memory
- allocated/reserved
- max allocated/max reserved

#### System/process
- system CPU%, RAM%
- process CPU%
- RSS/VMS
- threads
- handles (Windows if available)

#### GPU (nvidia-smi)
- util %
- mem util %, mem used/total
- temp
- power / power limit
- SM/mem clocks
- fan %
- pstate
- PCIe gen/width
- throttle reason fields (if supported)

#### Hardware/software snapshot
- OS/platform/python
- torch/cuda/cudnn details
- GPU info from torch and nvidia-smi
- host memory/cpu counts if psutil available

---

## 6) First Major Log Analysis (Before Runtime Fix)

Analyzed:

- `gpu_runtime_metrics_20260515_001603.csv`
- `runtime_events_20260515_001603.jsonl`

### Findings

1. **Latency drift is real and severe**
   - infer mean start vs end ~4.5x increase
   - total latency similarly increased
2. `capture_ms` stayed comparatively stable
3. `proc_vms_mib` grew very large over run
4. High correlation between `infer_ms` and `proc_vms_mib` (~0.92)
5. GPU VRAM usage relatively stable; CUDA reserved memory mostly flat
6. Event log had repeated `nvidia_smi_advanced_failed` warnings every sample

### Interpretation

Primary issue looked like **process-side long-run pressure/churn**, not a simple GPU VRAM leak and not “history image folder size” itself.

Likely contributors:
- repeated thread creation in hot path
- repeated heavy allocation/copy patterns
- overhead from repeated warning/event spam

---

## 7) Runtime Fixes Applied (Performance/Drift-Oriented)

Based on the above evidence, targeted fixes were implemented in `app/server.py`.

### A) Remove per-inference thread churn

#### Before
- Inference thread started repeatedly via new `threading.Thread(...)`.
- Per-cycle JPEG encode launched 3 new threads every inference.

#### After
1. Added persistent inference executor:
   - `ThreadPoolExecutor(max_workers=1, thread_name_prefix="hik-infer")`
2. Added persistent JPEG executor:
   - `ThreadPoolExecutor(max_workers=3, thread_name_prefix="hik-encode")`
3. `_start_inference()` now submits job to executor instead of spawning a fresh thread each cycle.
4. Encoding path now uses executor jobs / fallback serial encode, not ad-hoc thread creation each cycle.
5. Added `_is_infer_busy()` logic to work with future/executor and prevent overlapping jobs.

### B) Proper reset/cleanup

During runtime reset:
- executors and futures are cleared and shutdown (`shutdown(..., cancel_futures=True)` when supported)
- runtime timing fields reset

### C) Improve observability for verification

Added per-stage timings to state and CSV:
- `preprocess_ms`
- `build_ms`
- `infer_ms`
- `post_yolo_ms`
- `postprocess_ms`
- `encode_ms`

### D) Stop nvidia-smi warning flood

Advanced nvidia-smi query failure:
- now logs once initially
- marks advanced query unavailable
- quietly falls back to basic query
- basic failure warning also throttled to avoid spam

---

## 8) Clarifications Given During Chat

### About runtime duration
- You can run for 5 hours (not necessarily 12); logs remain useful.

### About EXE packaging
- Converting to Windows `.exe` alone does not usually speed inference.
- Packaging != performance optimization.
- Real improvements come from pipeline/runtime changes (which were implemented).

### About “was it images?”
- Not image file count in folders as primary root cause.
- More likely hot-path processing/thread/allocation behavior around image handling and inference pipeline.

---

## 9) Current Run Command

From project root:

```powershell
conda activate sostra_project
cd c:\Users\RobotComp\sostraproject
uvicorn backend.line_motion_dual_yolo_workspace.app.main:app --host 0.0.0.0 --port 8000
```

---

## 10) Important Paths

- Workspace root:
  - `c:\Users\RobotComp\sostraproject\backend\line_motion_dual_yolo_workspace`
- App code:
  - `...\app\server.py`
  - `...\app\main.py`
  - `...\app\modbus_tcp_send.py`
- History data + index:
  - `...\data\history\`
  - `...\data\history\history_index.sqlite`
- Logs:
  - `...\data\logs\gpu_runtime_metrics_*.csv`
  - `...\data\logs\runtime_events_*.jsonl`
  - `...\data\logs\hardware_info.json`

---

## 11) Suggested Next Validation Plan

1. Run app for 2-5 hours with current fixes.
2. Compare new run vs old problematic run:
   - slope of `infer_ms`
   - slope of `total_ms`
   - slope of `proc_vms_mib`
   - stage drift (`build_ms`, `postprocess_ms`, `encode_ms`)
3. Confirm event log is no longer flooded by nvidia-smi warnings.
4. If drift still exists, next optimization pass should target whichever stage shows strongest slope:
   - `build_ms` -> preprocessing/allocation path
   - `postprocess_ms` -> CPU merge/classifier logic
   - `encode_ms` -> overlay/JPEG path

---

## 12) Summary of What Is Already Done vs Pending

### Done
- Isolated app workspace
- Removed legacy runtime dependency
- DB-backed history indexing
- Detailed telemetry framework
- Root-cause analysis from real logs
- Runtime fixes to reduce thread churn
- Better stage-level metrics for proof

### Pending (optional)
- After your next long run, perform before/after quantitative comparison and decide whether further low-level optimization is needed.

---

If you continue with another AI session, point it to this file first, then to:
- `app/server.py`
- latest `data/logs/gpu_runtime_metrics_*.csv`
- latest `data/logs/runtime_events_*.jsonl`

That will let it resume with full context quickly.

