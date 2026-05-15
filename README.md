# Line Motion Dual YOLO (Isolated App)

This folder is a standalone app workspace. It does not import or depend on old backend app files.

## Structure

- `app/`
  - `main.py`: app entrypoint (`uvicorn ...app.main:app`)
  - `server.py`: standalone FastAPI server logic
  - `modbus_tcp_send.py`: local Modbus pulse helper
- `assets/models/`
  - `yolo/best_finetune1902_v3_beads4x.pt`
  - `classifier/best_classifier_20260219_123823.pt`
- `data/history/`
  - `beads_defect_frames_live/`
  - `beads_defect_frames_live_annotated/`
  - `beads_package_crops_live/`
  - `history_index.sqlite` (auto-created for fast history query)
- `data/logs/`
  - `gpu_runtime_metrics_*.csv` (GPU/runtime telemetry logs)
  - `runtime_events_*.jsonl` (structured warnings/errors/events)
  - `hardware_info.json` (host/GPU/software snapshot)

## Fast defect history

History API now uses local SQLite index (`data/history/history_index.sqlite`) and serves paginated results from DB instead of scanning all images on each request.

## GPU/performance telemetry

- Auto log starts when you run `/start`
- Default duration: 12 hours
- Default interval: 5 seconds
- Log file: `data/logs/gpu_runtime_metrics_YYYYMMDD_HHMMSS.csv`
- Includes: infer/capture/total ms, session counters, torch CUDA alloc/reserved/peaks, GPU util, mem used/total, temperature, power draw, clocks
- Extended: CPU/RAM, process RSS/VMS/CPU/threads/handles, per-camera frame staleness, GC counters, GPU pstate/fan/PCIe/throttle reasons (when available)
- Pipeline stages: `preprocess_ms`, `build_ms`, `infer_ms`, `post_yolo_ms`, `postprocess_ms`, `encode_ms`

Runtime stability hardening included:
- Inference trigger now uses persistent worker executor (no per-trigger thread creation)
- JPEG encoding uses persistent worker pool (no per-inference thread burst)
- `nvidia-smi` advanced query failure is logged once then falls back quietly

Endpoints:
- `POST /metrics/start?hours=12&interval_s=5`
- `GET /metrics/status`

## Run

```powershell
conda activate sostra_project
cd c:\Users\RobotComp\sostraproject
uvicorn backend.line_motion_dual_yolo_workspace.app.main:app --host 0.0.0.0 --port 8000
```

