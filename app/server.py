"""
FastAPI app: motion + dual YOLO (native + 640), no classifier.

Runs YOLO at BOTH native resolution and 640x640. Merge rule:
- Native conf >= 0.47 -> keep (display/save).
- Native conf < 0.47 -> keep only if 640 also detects a matching bbox (IoU overlap).
Streaming and motion logic unchanged. Display shows merged bboxes only.

Run:
  uvicorn backend.line_motion_dual_yolo_workspace.app.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import gc
import shlex
import atexit
import signal
import platform
import threading
import csv
import subprocess
import sqlite3
import json
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from collections import deque
from ctypes import byref, cast, POINTER, c_ubyte, memset, sizeof
from typing import Optional, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from torchvision.ops import box_iou
from torchvision.models import mobilenet_v3_small
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

# Avoid OpenMP duplicate runtime error on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, FileResponse

# ---------------- Config ----------------
APP_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = APP_DIR.parent
ASSETS_DIR = WORKSPACE_DIR / "assets"
DATA_DIR = WORKSPACE_DIR / "data"
DATA_HISTORY_DIR = DATA_DIR / "history"
DATA_LOGS_DIR = DATA_DIR / "logs"
for _p in (DATA_DIR, DATA_HISTORY_DIR, DATA_LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

MOTION_IP = os.getenv("HIK_MOTION_IP", "10.141.13.12").strip()
LINE_IPS = os.getenv("HIK_LINE_IPS", "10.141.13.11,10.141.13.12,10.141.13.13").strip()  # comma-separated IPs (3 cams)

FPS_CAP = float(os.getenv("HIK_FPS_CAP", "15.0"))
RES_SCALE = float(os.getenv("HIK_RES_SCALE", "1.0"))  # full res for capture/inference
EXPOSURE_US = float(os.getenv("HIK_EXPOSURE_US", "1300.0"))
# Explicit per-camera defaults (can still be overridden via /exposure at runtime)
EXPOSURE_US_CAM12 = float(os.getenv("HIK_EXPOSURE_US_CAM12", "1300.0"))
EXPOSURE_US_CAM11_13 = float(os.getenv("HIK_EXPOSURE_US_CAM11_13", "1300.0"))
GEV_SCPD = int(os.getenv("HIK_GEV_SCPD", "1000"))
SIGNAL_INTERVAL_MS = int(os.getenv("HIK_SIGNAL_INTERVAL_MS", "10"))

MODEL_PATH = os.getenv(
    "HIK_YOLO_MODEL",
    r"../assets/models/yolo/best_finetune1902_v3_beads4x.pt",
)
CONF = float(os.getenv("HIK_CONF", "0.38"))
CONF_640_DEFAULT = float(os.getenv("HIK_CONF_640", "0.5"))
SYNC_YOLO_CONF = os.getenv("HIK_SYNC_YOLO_CONF", "0").strip() in ("1", "true", "True")
HALF = os.getenv("HIK_HALF", "1").strip() not in ("0", "false", "False", "")
IMGSZ = int(os.getenv("HIK_IMGSZ", "0"))
NATIVE_IMGSZ = os.getenv("HIK_NATIVE_IMGSZ", "1").strip() in ("1", "true", "True")
# When warmup has no real frame, use this as inference size so YOLO runs at camera res (original)
INFER_HEIGHT = int(os.getenv("HIK_INFER_HEIGHT", "1080"))
INFER_WIDTH = int(os.getenv("HIK_INFER_WIDTH", "1440"))
SHOW_SPEED = os.getenv("HIK_SHOW_SPEED", "1").strip() in ("1", "true", "True")
REQUIRE_CUDA = os.getenv("HIK_REQUIRE_CUDA", "1").strip() in ("1", "true", "True")
WARMUP = os.getenv("HIK_WARMUP", "1").strip() in ("1", "true", "True")
WARMUP_IMGSZ = int(os.getenv("HIK_WARMUP_IMGSZ", "640"))

# Motion detection knobs (simple LK optical flow, matching websocket_stream.py)
DETECTION_FPS = float(os.getenv("HIK_DETECTION_FPS", "15.0"))
VERTICAL_SPEED_THRESHOLD = float(os.getenv("HIK_SPEED_THRESH", "4.5"))  # px/frame below = stopped
STOP_FRAMES_REQUIRED = int(os.getenv("HIK_STOP_FRAMES", "4"))           # consecutive low-speed frames to declare stop
STOP_HOLD_FRAMES = int(os.getenv("HIK_STOP_HOLD", "10"))                # hold stopped state this many frames before triggering inference
SPEED_WINDOW = 5
INFER_TRIGGER_FRAMES = int(os.getenv("HIK_INFER_TRIGGER", "2"))         # frames after stop before inference (matches websocket_stream.py)
INFER_COOLDOWN_MS = int(os.getenv("HIK_INFER_COOLDOWN_MS", "100"))      # min time between inference triggers (matches websocket_stream.py)
REARM_SPEED_MULT = float(os.getenv("HIK_REARM_MULT", "3.0"))            # moving again if speed >= threshold * mult
REARM_FRAMES_REQUIRED = int(os.getenv("HIK_REARM_FRAMES", "3"))         # require N consecutive "fast" frames to re-arm
MOTION_FAIL_SPEED = float(os.getenv("HIK_MOTION_FAIL_SPEED", "0.0"))    # speed value to use when LK tracking fails (0 = treat as stopped)
MOTION_MODE = os.getenv("HIK_MOTION_MODE", "legacy").strip().lower()    # legacy|robust

# Flicker robustness / LK tuning (helps avoid false motion from exposure flicker)
MOTION_EMA_ALPHA = float(os.getenv("HIK_MOTION_EMA", "0.0"))  # 0=off, else 0..0.95 (higher = smoother)
MOTION_BLUR_KSIZE = int(os.getenv("HIK_MOTION_BLUR", "5"))    # 0/1=off, else odd kernel size (e.g. 3/5/7)
# NOTE: per-frame normalization can create artificial motion; default OFF.
MOTION_NORM = os.getenv("HIK_MOTION_NORM", "0").strip() in ("1", "true", "True")  # normalize ROI (experimental)
FLICKER_SKIP = os.getenv("HIK_FLICKER_SKIP", "1").strip() in ("1", "true", "True")
FLICKER_MEAN_THRESH = float(os.getenv("HIK_FLICKER_MEAN", "12.0"))  # mean absdiff in ROI to consider "flicker"
FLICKER_STD_MAX = float(os.getenv("HIK_FLICKER_STD_MAX", "6.0"))     # std(absdiff) small => mostly uniform change
LK_MAX_CORNERS = int(os.getenv("HIK_LK_MAX_CORNERS", "80"))
LK_QUALITY_LEVEL = float(os.getenv("HIK_LK_QUALITY", "0.03"))
LK_MIN_DISTANCE = int(os.getenv("HIK_LK_MIN_DIST", "7"))
LK_WIN_SIZE = int(os.getenv("HIK_LK_WIN", "21"))
LK_MAX_LEVEL = int(os.getenv("HIK_LK_MAX_LEVEL", "2"))
LK_ERR_THRESH = float(os.getenv("HIK_LK_ERR_THRESH", "20.0"))  # <=0 disables filtering
LK_AGREE_FRAC = float(os.getenv("HIK_LK_AGREE_FRAC", "0.4"))    # require this fraction of points to agree with median motion

# Camera robustness guards (helps recover from transient single-camera grab faults)
CAM_STALE_MAX_AGE_MS = float(os.getenv("HIK_CAM_STALE_MAX_AGE_MS", "2500"))
CAM_RECOVER_FAILS = int(os.getenv("HIK_CAM_RECOVER_FAILS", "30"))

# Streaming (optional, low-res)
STREAM_SCALE = float(os.getenv("HIK_STREAM_SCALE", "0.2"))
STREAM_JPEG_QUALITY = int(os.getenv("HIK_STREAM_JPEG_QUALITY", "40"))
STREAM_FPS = float(os.getenv("HIK_STREAM_FPS", "15.0"))

# Display-only brightness (does NOT affect capture/inference; only what we encode for the UI)
DISPLAY_GAIN = float(os.getenv("HIK_DISPLAY_GAIN", "1.0"))   # >1 brighter
DISPLAY_BIAS = float(os.getenv("HIK_DISPLAY_BIAS", "0.0"))   # add offset in [roughly -255..255]
DISPLAY_GAMMA = float(os.getenv("HIK_DISPLAY_GAMMA", "1.0")) # <1 brightens midtones, >1 darkens

# Inference-only top-right masking module (easy to disable/remove later)
INFER_MASK_ENABLE = os.getenv("HIK_INFER_MASK_ENABLE", "1").strip() in ("1", "true", "True")
INFER_MASK_TOPRIGHT_H = int(os.getenv("HIK_INFER_MASK_TOPRIGHT_H", "130"))  # legacy (height now forced to full image)
INFER_MASK_TOPRIGHT_W = int(os.getenv("HIK_INFER_MASK_TOPRIGHT_W", "65"))   # pixels
INFER_MASK_TOP_BAND_H = int(os.getenv("HIK_INFER_MASK_TOP_BAND_H", "60"))  # pixels, full width
INFER_MASK_BOTTOM_BAND_H = int(os.getenv("HIK_INFER_MASK_BOTTOM_BAND_H", "60"))  # pixels, full width
INFER_MASK_SHOW_OVERLAY = os.getenv("HIK_INFER_MASK_SHOW_OVERLAY", "1").strip() in ("1", "true", "True")

# Directory for saving full-frame images when defects are detected
DEFECT_FRAME_OUT_DIR = Path(
    os.getenv(
        "HIK_DEFECT_FRAME_OUT",
        r"../data/history/beads_defect_frames_live",
    )
)
# Additional directory for annotated full-frame images (with drawn bboxes).
# The original DEFECT_FRAME_OUT_DIR remains raw (no drawings) so you can use
# it for training; this one is only for visual inspection.
DEFECT_FRAME_BOX_OUT_DIR = Path(
    os.getenv(
        "HIK_DEFECT_FRAME_BOX_OUT",
        r"../data/history/beads_defect_frames_live_annotated",
    )
)

# Dual-YOLO merge: IoU threshold for matching native and 640 detections
IOU_THRESH = float(os.getenv("HIK_DUAL_IOU_THRESH", "0.35"))  # min IoU to consider "same" bbox

# Crop export (beads-package only) – silica pack is legacy and disabled.
ONLY_CLASS0 = True
SAVE_BEADS_CROPS = False  # no classifier in this app
CROP_SCALE = float(os.getenv("HIK_CROP_SCALE", "2.0"))
DEBUG_MERGE_DECISIONS = os.getenv("HIK_DEBUG_MERGE", "0").strip() in ("1", "true", "True")
MIN_CROP_PX = int(os.getenv("HIK_MIN_CROP_PX", "52"))
MAX_CROPS_PER_CAM = int(os.getenv("HIK_MAX_CROPS_PER_CAM", "50"))
CROP_JPEG_QUALITY = int(os.getenv("HIK_CROP_JPEG_QUALITY", "95"))
CROP_OUT_DIR = Path(os.getenv("HIK_CROP_OUT", r"../data/history/beads_package_crops_live"))
SMALL_BOX_LOG_NAME = os.getenv("HIK_SMALL_BOX_LOG", "small_boxes_removed.csv").strip() or "small_boxes_removed.csv"

# Line signal (Modbus pulse) when any cam sees beads-package
LINE_SIGNAL_ENABLE = os.getenv("HIK_LINE_SIGNAL", "1").strip() in ("1", "true", "True")
LINE_SIGNAL_IP = os.getenv("HIK_LINE_SIGNAL_IP", "10.141.13.181").strip()
LINE_SIGNAL_PORT = int(os.getenv("HIK_LINE_SIGNAL_PORT", "502"))
# Default: 0.1s ON pulse, 2 minute cooldown between pulses
LINE_SIGNAL_HOLD_S = float(os.getenv("HIK_LINE_SIGNAL_HOLD", "0.3"))
LINE_SIGNAL_COOLDOWN_MS = int(os.getenv("HIK_LINE_SIGNAL_COOLDOWN_MS", str(6000)))
LINE_SIGNAL_UNIT = int(os.getenv("HIK_LINE_SIGNAL_UNIT", "1"))
LINE_SIGNAL_REGISTER = int(os.getenv("HIK_LINE_SIGNAL_REGISTER", "0x01D6"), 0)

# ---- Line 1 placeholder config (cameras not yet connected) ----
LINE1_IPS = os.getenv("L1_IPS", "").strip()           # future: "ip1,ip2,ip3"
LINE1_MOTION_IP = os.getenv("L1_MOTION_IP", "").strip()
LINE1_SIGNAL_ENABLE = os.getenv("L1_SIGNAL_ENABLE", "0").strip() in ("1", "true", "True")
LINE1_SIGNAL_IP = os.getenv("L1_SIGNAL_IP", "0.0.0.0").strip()
LINE1_SIGNAL_PORT = int(os.getenv("L1_SIGNAL_PORT", "502"))
LINE1_SIGNAL_REGISTER = int(os.getenv("L1_SIGNAL_REGISTER", "0x0001"), 0)
LINE1_SIGNAL_HOLD_S = float(os.getenv("L1_SIGNAL_HOLD", "0.3"))
LINE1_SIGNAL_COOLDOWN_MS = int(os.getenv("L1_SIGNAL_COOLDOWN_MS", "6000"))

# Maintenance auto-refresh: restart runtime pipeline periodically
# while preserving session timer/counters until manual Stop.
AUTO_REFRESH_ENABLED = os.getenv("HIK_AUTO_REFRESH_ENABLED", "0").strip() in ("1", "true", "True")
AUTO_REFRESH_MINUTES = float(os.getenv("HIK_AUTO_REFRESH_MINUTES", "180"))
AUTO_REFRESH_INTERVAL_S = max(30.0, AUTO_REFRESH_MINUTES * 60.0)

# Freeze watchdog: if camera timestamps go stale while running, perform
# internal stop/start recovery (preserve session counters).
FREEZE_WATCHDOG_ENABLED = os.getenv("HIK_FREEZE_WATCHDOG", "0").strip() in ("1", "true", "True")
FREEZE_WATCHDOG_CHECK_S = max(0.5, float(os.getenv("HIK_FREEZE_WATCHDOG_CHECK_S", "2.0")))
FREEZE_WATCHDOG_STALE_MS = float(os.getenv("HIK_FREEZE_WATCHDOG_STALE_MS", "10000"))
FREEZE_WATCHDOG_RECOVER_COOLDOWN_S = max(
    10.0, float(os.getenv("HIK_FREEZE_WATCHDOG_RECOVER_COOLDOWN_S", "180.0"))
)
FREEZE_WATCHDOG_GRACE_S = max(5.0, float(os.getenv("HIK_FREEZE_WATCHDOG_GRACE_S", "20.0")))
FREEZE_WATCHDOG_INFER_HUNG_S = max(
    10.0, float(os.getenv("HIK_FREEZE_WATCHDOG_INFER_HUNG_S", "90.0"))
)

# Hard relaunch: spawn a fresh process and exit current one.
HARD_RELAUNCH_ENABLED = os.getenv("HIK_HARD_RELAUNCH", "1").strip() in ("1", "true", "True")
# Default ON so scheduled refresh performs a full process restart, which actually
# releases Python/CUDA allocator memory instead of keeping it in-process.
AUTO_REFRESH_HARD_RELAUNCH = os.getenv("HIK_AUTO_REFRESH_HARD_RELAUNCH", "0").strip() in ("1", "true", "True")
HARD_RELAUNCH_CMD = os.getenv("HIK_HARD_RELAUNCH_CMD", "").strip()

# Safe memory cleanup (no process restart, no browser refresh needed).
MEM_CLEANUP_ENABLED = os.getenv("HIK_MEM_CLEANUP", "0").strip() in ("1", "true", "True")
MEM_CLEANUP_INTERVAL_MIN = float(os.getenv("HIK_MEM_CLEANUP_INTERVAL_MIN", "180"))
MEM_CLEANUP_INTERVAL_S = max(30.0, MEM_CLEANUP_INTERVAL_MIN * 60.0)
MEM_CLEANUP_CUDA = os.getenv("HIK_MEM_CLEANUP_CUDA", "0").strip() in ("1", "true", "True")

# In long-running production, the dual-stream YOLO path can increase CUDA
# allocator churn without improving latency on a single consumer GPU.
PARALLEL_YOLO_STREAMS = os.getenv("HIK_PARALLEL_YOLO_STREAMS", "0").strip() in ("1", "true", "True")
CUDA_STATS_EVERY = int(os.getenv("HIK_CUDA_STATS_EVERY", "100"))

# ---------------- SDK Path ----------------
if platform.system() == "Windows":
    sdk = os.getenv("MVCAM_COMMON_RUNENV")
    if sdk:
        sys.path.append(os.path.join(sdk, "Samples", "Python", "MvImport"))

from MvCameraControl_class import (  # noqa: E402
    MvCamera,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_DEVICE_INFO,
    MV_GIGE_DEVICE,
    MV_USB_DEVICE,
    MV_GENTL_GIGE_DEVICE,
    MV_FRAME_OUT,
)
try:
    from PixelType_header import PixelType_Gvsp_Mono8  # noqa: E402
except Exception:
    from CameraParams_header import PixelType_Gvsp_Mono8  # noqa: E402
try:
    from CameraParams_header import MVCC_INTVALUE  # noqa: E402
except Exception:
    MVCC_INTVALUE = None

app = FastAPI()

_state_lock = threading.Lock()
_running = False
_starting = False
_error: Optional[str] = None

_cams: list[MvCamera] = []
_cam_ips: list[str] = []
_motion_idx = 0

_model: Optional[YOLO] = None
_device: str = "cpu"
_class_names: dict[int, str] = {0: "beads-package"}  # silica pack class removed
_class_colors_bgr: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),     # beads-package: blue (BGR)
}

_last_motion_frame: Optional[np.ndarray] = None
_last_motion_ts: float = 0.0
_last_stream_jpeg: Optional[bytes] = None

# Per-camera latest raw frames (continuously drained by background threads)
_latest_cam_raw: list[Optional[np.ndarray]] = [None, None, None]   # Mono8 per camera
_latest_cam_ts: list[float] = [0.0, 0.0, 0.0]

_last_results: list[Optional[bytes]] = [None, None, None]  # pre-encoded JPEGs
_last_capture_ms: Optional[float] = None
_last_infer_ms: Optional[float] = None
_last_total_ms: Optional[float] = None
_last_preprocess_ms: Optional[float] = None
_last_build_ms: Optional[float] = None
_last_post_yolo_ms: Optional[float] = None
_last_postprocess_ms: Optional[float] = None
_last_encode_ms: Optional[float] = None
_last_result_id: int = 0
_last_speed_str: Optional[str] = None
_last_detected: list[str] = ["--", "--", "--"]  # per-cam detection summary text
_last_speed: Optional[float] = None
_last_state: str = "moving"
_last_result_ready_ts: float = 0.0  # wall-clock time when results were stored
_last_line_signal_ms: int = 0
_last_line_signal_rid: int = -1
_last_infer_trigger_ms: int = 0
_infer_started_ts: float = 0.0
_last_stale_warn_ts: float = 0.0

# Per-session defect frame saving
_defect_frame_out_dir: Optional[Path] = None
_last_defect_paths: list[str] = []

# Save first inference images once (3 cams) to live frame folder for confirmation
_first_infer_images_saved: bool = False

# Session statistics (printed on shutdown)
_session_start_ts: float = 0.0       # wall-clock time when /start was called
_session_total_infer: int = 0         # total inference runs
_session_defect_infer: int = 0        # inference runs where any cam had ≥1 beads-package
_maintenance_refreshing: bool = False
_maintenance_msg: str = ""

# UI-controlled signal enable: only when True do we send real Modbus pulse (default ON)
_line_signal_ui_enabled: bool = True

# Per-camera exposure times (µs), runtime-adjustable via API
_exposure_us_per_cam: list[float] = []

# Runtime-adjustable display brightness parameters
DISPLAY_GAIN_RUNTIME: float = DISPLAY_GAIN
DISPLAY_BIAS_RUNTIME: float = DISPLAY_BIAS
DISPLAY_GAMMA_RUNTIME: float = DISPLAY_GAMMA

# Crop classifier (MobileNetV3-small) — always enabled (no UI switch)
CLF_ENABLE_DEFAULT = True
CLF_WEIGHTS = os.getenv(
    "HIK_CROP_CLS_WEIGHTS",
    r"../assets/models/classifier/best_classifier_20260219_123823.pt",
)
CLF_IMGSZ = int(os.getenv("HIK_CROP_CLS_IMGSZ", "128"))
CLF_DEVICE_PREF = os.getenv("HIK_CROP_CLS_DEVICE", "cpu").strip().lower()  # cpu|cuda|same
CLF_THRESH_DEFAULT = float(os.getenv("HIK_CROP_CLS_THRESH", "0.8"))  # classifier accept threshold (fixed at 0.8)
CLF_CONF_TRIGGER_DEFAULT = float(os.getenv("HIK_CLF_CONF_TRIGGER", "0.47"))  # native conf below this -> Y640 gate path
SINGLE_DET_FORCE_640_CONF = float(os.getenv("HIK_SINGLE_DET_FORCE_640_CONF", "0.65"))  # if only one native bbox and conf below this, force 640+clf path
_clf_model: Optional[torch.nn.Module] = None
_clf_device: str = "cpu"
_runtime_conf_native: float = CONF  # Native YOLO confidence threshold (0.0-1.0)
_runtime_conf_640: float = CONF_640_DEFAULT  # 640 YOLO confidence threshold (0.0-1.0)
_runtime_iou_thresh: float = IOU_THRESH  # IoU threshold for matching native and 640 detections
# Classifier runtime settings
_runtime_clf_enable: bool = CLF_ENABLE_DEFAULT  # enable/disable classifier
_runtime_clf_thresh: float = CLF_THRESH_DEFAULT  # classifier accept threshold
_runtime_clf_conf_trigger: float = CLF_CONF_TRIGGER_DEFAULT  # native conf below this -> check 640 first, then classifier

_stop_thread: Optional[threading.Thread] = None
_infer_thread: Optional[threading.Thread] = None
_infer_executor: Optional[ThreadPoolExecutor] = None
_infer_future: Optional[Future] = None
_encode_executor: Optional[ThreadPoolExecutor] = None
_grab_thread: Optional[threading.Thread] = None
_auto_refresh_thread: Optional[threading.Thread] = None
_freeze_watchdog_thread: Optional[threading.Thread] = None
_mem_cleanup_thread: Optional[threading.Thread] = None
_last_auto_refresh_ts: float = 0.0
_last_watchdog_recover_ts: float = 0.0
_hard_relaunch_in_progress: bool = False
_run_generation: int = 0
_last_mem_cleanup_ts: float = 0.0
_last_mem_cleanup_reason: str = ""
_history_sync_interval_s: float = max(0.5, float(os.getenv("HIK_HISTORY_SYNC_INTERVAL_S", "5.0")))
_history_db_path: Path = Path(
    os.getenv("HIK_HISTORY_DB_PATH", str((DATA_HISTORY_DIR / "history_index.sqlite").resolve()))
)
_history_sync_lock = threading.Lock()
_history_last_sync_ts: float = 0.0

METRICS_ENABLE = os.getenv("HIK_METRICS_ENABLE", "1").strip().lower() in ("1", "true", "yes", "on")
METRICS_LOG_INTERVAL_S = max(0.5, float(os.getenv("HIK_METRICS_LOG_INTERVAL_S", "5.0")))
METRICS_DURATION_HOURS = max(0.1, float(os.getenv("HIK_METRICS_DURATION_HOURS", "12")))
METRICS_LOG_PATH = Path(
    os.getenv(
        "HIK_METRICS_LOG_PATH",
        str((DATA_LOGS_DIR / f"gpu_runtime_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv").resolve()),
    )
)
METRICS_EVENT_LOG_PATH = Path(
    os.getenv(
        "HIK_METRICS_EVENT_LOG_PATH",
        str((DATA_LOGS_DIR / f"runtime_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl").resolve()),
    )
)
METRICS_HW_INFO_PATH = Path(
    os.getenv(
        "HIK_METRICS_HW_INFO_PATH",
        str((DATA_LOGS_DIR / "hardware_info.json").resolve()),
    )
)
_metrics_thread: Optional[threading.Thread] = None
_metrics_stop_event = threading.Event()
_metrics_started_ts: float = 0.0
_metrics_end_ts: float = 0.0
_metrics_proc: Any = None
_nvidia_smi_adv_available: Optional[bool] = None
_nvidia_smi_basic_warned: bool = False

# ---- Line 1 state (placeholder — cameras not yet connected) ----
_l1_lock = threading.Lock()
_l1_running: bool = False
_l1_start_ts: float = 0.0
_l1_signal_ui_enabled: bool = True

# Batch inference state (initialised once at warmup, reused for all calls)
_net: Optional[torch.nn.Module] = None   # underlying nn.Module from YOLO
_net_stride: int = 32                     # model stride
_net_imgsz: Tuple[int, int] = (640, 640)  # (H, W) padded to stride
_letterbox: Optional[LetterBox] = None    # reusable letterbox transform (native)
_letterbox_640: Optional[LetterBox] = None  # 640x640 for dual-YOLO cross-check
_cuda_stream_nat: Optional[torch.cuda.Stream] = None
_cuda_stream_640: Optional[torch.cuda.Stream] = None
# Pre-allocated frame buffers to avoid malloc churn (filled in _init_batch_infer)
_frame_bufs: Optional[list[np.ndarray]] = None  # 3 x (H, W, 3) BGR uint8
_host_batch_native_np: Optional[np.ndarray] = None
_host_batch_640_np: Optional[np.ndarray] = None
_host_batch_native_tensor: Optional[torch.Tensor] = None
_host_batch_640_tensor: Optional[torch.Tensor] = None
_dev_batch_native: Optional[torch.Tensor] = None
_dev_batch_640: Optional[torch.Tensor] = None
_crop_worker_slots = threading.BoundedSemaphore(1)  # prevent unbounded crop-writer backlog


def _ip_from_devinfo(dev_info) -> int | None:
    try:
        if dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            return int(dev_info.SpecialInfo.stGigEInfo.nCurrentIp)
    except Exception:
        pass
    return None


def _ip_to_str(ip_int: int | None) -> str:
    if ip_int is None:
        return "unknown"
    return ".".join(str((ip_int >> (8 * i)) & 0xFF) for i in range(4)[::-1])


def _open_camera(dev_info) -> MvCamera:
    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(dev_info)
    if ret != 0:
        raise RuntimeError("CreateHandle failed")
    ret = cam.MV_CC_OpenDevice(3, 0)  # MV_ACCESS_Control = 3
    if ret != 0:
        raise RuntimeError("OpenDevice failed")

    # GigE optimization
    try:
        if dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            pkt = cam.MV_CC_GetOptimalPacketSize()
            if int(pkt) > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(pkt))
            try:
                cam.MV_CC_SetIntValue("GevSCPD", int(GEV_SCPD))
            except Exception:
                pass
    except Exception:
        pass

    cam.MV_CC_SetEnumValue("TriggerMode", 0)
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)

    # True AOI control (apply based on nMax)
    if MVCC_INTVALUE is not None:
        try:
            st_w = MVCC_INTVALUE()
            memset(byref(st_w), 0, sizeof(MVCC_INTVALUE))
            st_h = MVCC_INTVALUE()
            memset(byref(st_h), 0, sizeof(MVCC_INTVALUE))
            if cam.MV_CC_GetIntValue("Width", st_w) == 0 and cam.MV_CC_GetIntValue("Height", st_h) == 0:
                w_max = int(st_w.nMax) if int(st_w.nMax) > 0 else int(st_w.nCurValue)
                h_max = int(st_h.nMax) if int(st_h.nMax) > 0 else int(st_h.nCurValue)
                w_inc = int(st_w.nInc) if int(st_w.nInc) > 0 else 1
                h_inc = int(st_h.nInc) if int(st_h.nInc) > 0 else 1
                w_target = max(2, (int(w_max * RES_SCALE) // w_inc) * w_inc)
                h_target = max(2, (int(h_max * RES_SCALE) // h_inc) * h_inc)
                try:
                    cam.MV_CC_SetIntValue("OffsetX", 0)
                    cam.MV_CC_SetIntValue("OffsetY", 0)
                except Exception:
                    pass
                cam.MV_CC_SetIntValue("Width", int(w_target))
                cam.MV_CC_SetIntValue("Height", int(h_target))
        except Exception:
            pass

    # Frame rate cap
    try:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(FPS_CAP))
    except Exception:
        pass

    # Exposure
    try:
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        cam.MV_CC_SetFloatValue("ExposureTime", float(EXPOSURE_US))
    except Exception:
        pass

    # Use LatestImagesOnly so GetImageBuffer always returns the newest frame
    # and discards old buffered ones (prevents stale-frame / buffer-overflow drops).
    try:
        cam.MV_CC_SetGrabStrategy(1)  # MV_GrabStrategy_LatestImagesOnly
    except Exception:
        pass

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        raise RuntimeError("StartGrabbing failed")
    return cam


def _grab_one(cam: MvCamera) -> tuple:
    st_out = MV_FRAME_OUT()
    for _ in range(3):
        memset(byref(st_out), 0, sizeof(MV_FRAME_OUT))
        ret = cam.MV_CC_GetImageBuffer(st_out, 1000)
        if ret != 0:
            time.sleep(0.002)
            continue
        n = int(st_out.stFrameInfo.nFrameLen)
        h = int(st_out.stFrameInfo.nHeight)
        w = int(st_out.stFrameInfo.nWidth)
        expected = w * h
        if expected <= 0 or n < expected:
            cam.MV_CC_FreeImageBuffer(st_out)
            time.sleep(0.002)
            continue
        buf = cast(st_out.pBufAddr, POINTER(c_ubyte * n))
        img = np.frombuffer(buf.contents, dtype=np.uint8)
        if img.size < expected:
            cam.MV_CC_FreeImageBuffer(st_out)
            time.sleep(0.002)
            continue
        img = img[:expected].reshape(h, w).copy()
        cam.MV_CC_FreeImageBuffer(st_out)
        return img, n
    return None, 0


def _try_restart_grabbing(cam: MvCamera, tag: str) -> None:
    """Best-effort camera stream restart for transient SDK faults."""
    try:
        cam.MV_CC_StopGrabbing()
    except Exception:
        pass
    time.sleep(0.02)
    try:
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[grab][{tag}] restart failed ret={ret}", flush=True)
        else:
            print(f"[grab][{tag}] restart grabbing ok", flush=True)
    except Exception as e:
        print(f"[grab][{tag}] restart error: {e}", flush=True)


def fast_median_3(a: float, b: float, c: float) -> float:
    """Fast median of 3 values without creating lists or using numpy."""
    if a <= b:
        if b <= c:
            return b
        elif a <= c:
            return c
        else:
            return a
    else:
        if a <= c:
            return a
        elif b <= c:
            return c
        else:
            return b


def compute_vertical_speed_legacy(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    prev_pts: Optional[np.ndarray],
) -> Tuple[float, Optional[np.ndarray]]:
    """Original (yesterday) vertical speed via sparse LK optical flow on center ROI.

    Returns (speed_px_per_frame, next_pts).
    """
    if prev_gray is None:
        return 999.0, None

    h, w = curr_gray.shape[:2]
    x0 = int(w * 0.4)
    x1 = int(w * 0.6)
    y0 = int(h * 0.2)
    y1 = int(h * 0.8)

    prev_roi = prev_gray[y0:y1, x0:x1]
    curr_roi = curr_gray[y0:y1, x0:x1]

    if prev_pts is None or len(prev_pts) < 10:
        pts = cv2.goodFeaturesToTrack(
            prev_roi,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=5,
        )
        if pts is None or len(pts) == 0:
            return 999.0, None
        prev_pts = pts

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_roi,
        curr_roi,
        prev_pts,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    if next_pts is None or status is None:
        return 999.0, None

    status = status.reshape(-1)
    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]
    if len(good_prev) < 5:
        return 999.0, None

    dy = good_next[:, 0, 1] - good_prev[:, 0, 1]
    speed = float(np.median(np.abs(dy)))
    return speed, good_next.reshape(-1, 1, 2)


def compute_vertical_speed_robust(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    prev_pts: Optional[np.ndarray],
) -> Tuple[float, Optional[np.ndarray]]:
    """Robust vertical speed via sparse LK optical flow on a center ROI."""
    if prev_gray is None:
        return float(MOTION_FAIL_SPEED), None

    h, w = curr_gray.shape[:2]
    # Narrow vertical strip around the bead lane (center ~20% of width)
    x0 = int(w * 0.4)
    x1 = int(w * 0.6)
    y0 = int(h * 0.2)
    y1 = int(h * 0.8)

    prev_roi = prev_gray[y0:y1, x0:x1]
    curr_roi = curr_gray[y0:y1, x0:x1]

    # Normalize ROI to reduce global brightness flicker effect on LK "brightness constancy"
    if MOTION_NORM:
        prev_roi_f = prev_roi.astype(np.float32)
        curr_roi_f = curr_roi.astype(np.float32)
        pm, ps = float(prev_roi_f.mean()), float(prev_roi_f.std())
        cm, cs = float(curr_roi_f.mean()), float(curr_roi_f.std())
        prev_roi = (prev_roi_f - pm) / (ps + 1e-6)
        curr_roi = (curr_roi_f - cm) / (cs + 1e-6)

    # Reduce sensor noise / flicker-driven gradients a bit (keeps motion edges)
    k = MOTION_BLUR_KSIZE
    if k and k > 1:
        if k % 2 == 0:
            k += 1
        prev_roi = cv2.GaussianBlur(prev_roi, (k, k), 0)
        curr_roi = cv2.GaussianBlur(curr_roi, (k, k), 0)

    # Detect new features if needed
    if prev_pts is None or len(prev_pts) < 10:
        pts = cv2.goodFeaturesToTrack(
            prev_roi,
            maxCorners=LK_MAX_CORNERS,
            qualityLevel=LK_QUALITY_LEVEL,
            minDistance=LK_MIN_DISTANCE,
        )
        if pts is None or len(pts) == 0:
            return float(MOTION_FAIL_SPEED), None
        prev_pts = pts

    # Calculate optical flow (Lucas-Kanade) within ROI
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_roi,
        curr_roi,
        prev_pts,
        None,
        winSize=(LK_WIN_SIZE, LK_WIN_SIZE),
        maxLevel=LK_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    if next_pts is None or status is None:
        return float(MOTION_FAIL_SPEED), None

    status = status.reshape(-1)
    good_mask = status == 1
    if err is not None and LK_ERR_THRESH > 0:
        try:
            e = err.reshape(-1)
            good_mask = good_mask & (e <= LK_ERR_THRESH)
        except Exception:
            pass
    good_prev = prev_pts[good_mask]
    good_next = next_pts[good_mask]

    if len(good_prev) < 5:
        return float(MOTION_FAIL_SPEED), None

    dy = good_next[:, 0, 1] - good_prev[:, 0, 1]
    abs_dy = np.abs(dy)
    med = float(np.median(abs_dy))
    # If many tracks disagree wildly with the median, it's usually flicker/bad LK -> treat as "no motion"
    try:
        agree = float(np.mean(abs_dy <= (med * 2.0 + 0.5)))
        if agree < max(0.0, min(1.0, float(LK_AGREE_FRAC))):
            return float(MOTION_FAIL_SPEED), good_next.reshape(-1, 1, 2)
    except Exception:
        pass
    speed = med

    return speed, good_next.reshape(-1, 1, 2)


def compute_vertical_speed(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    prev_pts: Optional[np.ndarray],
) -> Tuple[float, Optional[np.ndarray]]:
    """Dispatch to legacy/robust motion mode."""
    if MOTION_MODE == "robust":
        return compute_vertical_speed_robust(prev_gray, curr_gray, prev_pts)
    return compute_vertical_speed_legacy(prev_gray, curr_gray, prev_pts)


def _init_batch_infer(model: YOLO, device: str, sample_frame: np.ndarray):
    """Initialise batch-inference: native letterbox + 640 letterbox for dual-YOLO."""
    global _net, _net_stride, _net_imgsz, _letterbox, _letterbox_640
    global _cuda_stream_nat, _cuda_stream_640, _frame_bufs
    global _host_batch_native_np, _host_batch_640_np
    global _host_batch_native_tensor, _host_batch_640_tensor
    global _dev_batch_native, _dev_batch_640

    # Extract the raw nn.Module and move it to the device ONCE
    _net = model.model
    _net.to(device)
    _net.eval()
    if HALF and device.startswith("cuda"):
        _net.half()

    # Model stride
    stride = 32
    try:
        s = getattr(_net, "stride", None)
        if s is not None:
            stride = int(max(s)) if hasattr(s, "__iter__") else int(s)
    except Exception:
        pass
    _net_stride = stride

    # Native: user-facing 1080×1440, but pad to stride-32 (1088 height) internally for model compatibility
    # Model requires stride-32 alignment to avoid concat errors (1080/32 = 33.75, 1088/32 = 34)
    if IMGSZ > 0:
        _net_imgsz = (IMGSZ, IMGSZ)
        _letterbox_stride = _net_stride
    else:
        # Pad both dimensions to stride-32 to match offline script behavior
        # Offline: nh = ((h + stride - 1) // stride) * stride, nw = ((w + stride - 1) // stride) * stride
        target_h = ((INFER_HEIGHT + stride - 1) // stride) * stride  # 1080 -> 1088
        target_w = ((INFER_WIDTH + stride - 1) // stride) * stride  # 1440 -> 1440 (already divisible)
        _net_imgsz = (target_h, target_w)  # (1088, 1440) matches offline script
        _letterbox_stride = stride  # stride-32 padding

    _letterbox = LetterBox(_net_imgsz, auto=True, stride=_letterbox_stride)
    _letterbox_640 = LetterBox((640, 640), auto=True, stride=_net_stride)

    if device.startswith("cuda") and PARALLEL_YOLO_STREAMS:
        _cuda_stream_nat = torch.cuda.Stream()
        _cuda_stream_640 = torch.cuda.Stream()
    else:
        _cuda_stream_nat = None
        _cuda_stream_640 = None

    # Pre-allocate reusable BGR frame buffers (avoids malloc churn in hot path)
    sh = sample_frame.shape
    buf_h = sh[0] if len(sh) >= 2 else INFER_HEIGHT
    buf_w = sh[1] if len(sh) >= 2 else INFER_WIDTH
    _frame_bufs = [np.empty((buf_h, buf_w, 3), dtype=np.uint8) for _ in range(3)]

    batch_size = 3
    native_shape = (batch_size, 3, _net_imgsz[0], _net_imgsz[1])
    low_shape = (batch_size, 3, 640, 640)
    _host_batch_native_np = np.empty(native_shape, dtype=np.float32)
    _host_batch_640_np = np.empty(low_shape, dtype=np.float32)
    _host_batch_native_tensor = torch.from_numpy(_host_batch_native_np)
    _host_batch_640_tensor = torch.from_numpy(_host_batch_640_np)
    infer_dtype = torch.float16 if (HALF and device.startswith("cuda")) else torch.float32
    _dev_batch_native = torch.empty(native_shape, device=device, dtype=infer_dtype)
    _dev_batch_640 = torch.empty(low_shape, device=device, dtype=infer_dtype)

    # Report user-facing resolution (1080×1440) even though internal is 1088×1440 for stride-32
    user_h = INFER_HEIGHT if IMGSZ == 0 else _net_imgsz[0]
    user_w = INFER_WIDTH if IMGSZ == 0 else _net_imgsz[1]
    print(f"[init] batch-infer ready  device={device}  native={_net_imgsz} (user-facing {user_h}×{user_w})  640x640  half={HALF}", flush=True)


def _fill_chw_batch(
    dst_batch: np.ndarray,
    frames_bgr: list[np.ndarray],
    letterbox_fn: LetterBox,
    target_hw: Tuple[int, int],
) -> None:
    """Fill a reusable CHW float32 batch buffer in-place."""
    target_h, target_w = target_hw
    scale = 1.0 / 255.0
    for i, bgr in enumerate(frames_bgr):
        lb = letterbox_fn(image=bgr)
        if lb.shape[0] != target_h or lb.shape[1] != target_w:
            lb = cv2.resize(lb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb_chw = lb[:, :, ::-1].transpose(2, 0, 1)
        np.multiply(rgb_chw, scale, out=dst_batch[i], casting="unsafe")


def _batch_predict(frames: list[np.ndarray], device: str) -> list[tuple[np.ndarray, torch.Tensor]]:
    """Run TRUE batched inference: one GPU forward pass for all frames.

    Returns list of (orig_bgr_image, detections_xyxysc) per frame.
    detections_xyxysc shape is (N, 6) with [x1,y1,x2,y2,conf,cls] in original coords.
    """
    n = len(frames)

    # 1. Letterbox + BGR->RGB + HWC->CHW  (all numpy, very fast)
    orig_bgr: list[np.ndarray] = []
    chw_list: list[np.ndarray] = []

    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f
        orig_bgr.append(bgr)
        lb = _letterbox(image=bgr)
        chw = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        chw_list.append(chw)

    # 2. Stack into batch tensor ONCE  (all images same size after letterbox)
    batch_np = np.stack(chw_list)  # (N, 3, H, W)
    target_dtype = torch.float16 if (HALF and str(device).startswith("cuda")) else torch.float32
    tensor = torch.from_numpy(batch_np).to(device=device, dtype=target_dtype, non_blocking=True)

    # 3. Single batched forward pass + NMS  (one GPU dispatch, not N!)
    with _state_lock:
        current_conf = _runtime_conf_native
    with torch.no_grad():
        preds = _net(tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        dets_list = non_max_suppression(preds, conf_thres=current_conf, iou_thres=0.45)

    # 4. Scale boxes back to original image coordinates
    results: list[tuple[np.ndarray, torch.Tensor]] = []
    for i in range(n):
        det = dets_list[i]
        if det is not None and len(det):
            det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4],
                                     orig_bgr[i].shape[:2]).round()
        results.append((orig_bgr[i], det))

    return results
def _encode_jpeg(img: np.ndarray, quality: int) -> Optional[bytes]:
    try:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


_gamma_lut_cache: dict[float, np.ndarray] = {}


def _apply_display_adjust(img: np.ndarray) -> np.ndarray:
    """Display-only brightness/gamma. Does not touch inference pipeline.

    Note: gain/bias/gamma can be adjusted at runtime via /display API.
    """
    out = img

    try:
        # Use current runtime-controlled values if present
        gain = float(globals().get("DISPLAY_GAIN_RUNTIME", DISPLAY_GAIN))
        bias = float(globals().get("DISPLAY_BIAS_RUNTIME", DISPLAY_BIAS))
        gamma = float(globals().get("DISPLAY_GAMMA_RUNTIME", DISPLAY_GAMMA))

        if gain != 1.0 or bias != 0.0:
            out = cv2.convertScaleAbs(out, alpha=gain, beta=bias)

        g = float(gamma)
        if g != 1.0 and g > 0:
            lut = _gamma_lut_cache.get(g)
            if lut is None:
                # gamma < 1 brightens; gamma > 1 darkens
                lut = np.array([((i / 255.0) ** g) * 255.0 for i in range(256)], dtype=np.uint8)
                _gamma_lut_cache[g] = lut
            out = cv2.LUT(out, lut)
    except Exception:
        return img

    return out


def _apply_infer_topright_mask(img: np.ndarray) -> np.ndarray:
    """Mask side strips + top/bottom bands in a copy, for inference input only."""
    if not INFER_MASK_ENABLE:
        return img
    try:
        h, w = img.shape[:2]
        mh = h  # use full image height
        mw = max(0, min(int(INFER_MASK_TOPRIGHT_W), w))
        top_band_h = max(0, min(int(INFER_MASK_TOP_BAND_H), h))
        bottom_band_h = max(0, min(int(INFER_MASK_BOTTOM_BAND_H), h))
        if mh <= 0 or mw <= 0:
            # Still allow top/bottom-band-only masking if side width is disabled.
            if top_band_h <= 0 and bottom_band_h <= 0:
                return img
            out = img.copy()
            if top_band_h > 0:
                out[0:top_band_h, 0:w] = 0
            if bottom_band_h > 0:
                out[h - bottom_band_h:h, 0:w] = 0
            return out
        out = img.copy()
        # Top-left
        out[0:mh, 0:mw] = 0
        # Top-right
        out[0:mh, w - mw:w] = 0
        # Top band (full width)
        if top_band_h > 0:
            out[0:top_band_h, 0:w] = 0
        # Bottom band (full width)
        if bottom_band_h > 0:
            out[h - bottom_band_h:h, 0:w] = 0
        return out
    except Exception:
        return img


def _draw_infer_mask_overlay(img: np.ndarray) -> np.ndarray:
    """Draw visible overlay of inference-masked regions for operator UI."""
    if not INFER_MASK_ENABLE or not INFER_MASK_SHOW_OVERLAY:
        return img
    try:
        if img is None:
            return img
        out = img.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        h, w = out.shape[:2]
        mh = h  # use full image height
        mw = max(0, min(int(INFER_MASK_TOPRIGHT_W), w))
        top_band_h = max(0, min(int(INFER_MASK_TOP_BAND_H), h))
        bottom_band_h = max(0, min(int(INFER_MASK_BOTTOM_BAND_H), h))
        if mh <= 0 or mw <= 0:
            if top_band_h <= 0 and bottom_band_h <= 0:
                return out
            # Draw top/bottom-band-only overlay
            overlay = out.copy()
            if top_band_h > 0:
                cv2.rectangle(overlay, (0, 0), (w - 1, top_band_h - 1), (0, 0, 0), -1)
            if bottom_band_h > 0:
                cv2.rectangle(overlay, (0, h - bottom_band_h), (w - 1, h - 1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
            if top_band_h > 0:
                cv2.rectangle(out, (0, 0), (w - 1, top_band_h - 1), (255, 255, 0), 2)
            if bottom_band_h > 0:
                cv2.rectangle(out, (0, h - bottom_band_h), (w - 1, h - 1), (255, 255, 0), 2)
            return out
        # Top-left coordinates
        lx1, ly1 = 0, 0
        lx2, ly2 = mw - 1, mh - 1
        # Top-right coordinates
        rx1, ry1 = w - mw, 0
        rx2, ry2 = w - 1, mh - 1
        # Top-band coordinates
        tx1, ty1 = 0, 0
        tx2, ty2 = w - 1, max(0, top_band_h - 1)
        # Bottom-band coordinates
        bx1, by1 = 0, max(0, h - bottom_band_h)
        bx2, by2 = w - 1, h - 1
        # Semi-transparent dark fill so masked area is obvious.
        overlay = out.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
        if top_band_h > 0:
            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
        if bottom_band_h > 0:
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (255, 255, 0), 2)
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        if top_band_h > 0:
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), (255, 255, 0), 2)
        if bottom_band_h > 0:
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
        return out
    except Exception:
        return img


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def _letterbox_clf(image: np.ndarray, new_size: int) -> np.ndarray:
    """Letterbox for crop classifier (keep aspect, pad to new_size x new_size)."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((new_size, new_size, 3), dtype=np.uint8)
    scale = min(new_size / h, new_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return out


def _init_crop_classifier(device: str) -> None:
    """Initialize MobileNetV3-small crop classifier (binary: negative/positive)."""
    global _clf_model, _clf_device
    with _state_lock:
        clf_enable = _runtime_clf_enable
    if not clf_enable:
        print("[clf] disabled", flush=True)
        _clf_model = None
        _clf_device = "cpu"
        return

    weights_path = Path(__file__).resolve().parent / CLF_WEIGHTS
    if not weights_path.exists():
        print(f"[clf] weights not found, disabling classifier: {weights_path}", flush=True)
        _clf_model = None
        _clf_device = "cpu"
        return

    clf_device = str(device)
    pref = CLF_DEVICE_PREF
    if pref == "cpu":
        clf_device = "cpu"
    elif pref == "cuda":
        clf_device = device if str(device).startswith("cuda") else "cpu"
    elif pref == "same":
        clf_device = str(device)
    else:
        clf_device = "cpu"

    try:
        print(f"[clf] loading MobileNetV3-small classifier from {weights_path} on {clf_device}", flush=True)
        base = mobilenet_v3_small(weights=None)
        in_features = base.classifier[0].in_features
        base.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.Hardswish(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 2),
        )
        base.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
        base.to(clf_device)
        base.eval()
        _clf_model = base
        _clf_device = clf_device
        print("[clf] classifier ready", flush=True)
    except Exception as e:
        import traceback
        print(f"[clf] failed to init classifier: {e}", flush=True)
        traceback.print_exc()
        _clf_model = None
        _clf_device = "cpu"


_CLF_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_CLF_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _run_crop_classifier(crops_bgr: list[np.ndarray], device: str) -> list[float]:
    """Run classifier on list of BGR crops. Returns prob of positive class."""
    if _clf_model is None or not crops_bgr:
        return [1.0] * len(crops_bgr)

    imgs = []
    for bgr in crops_bgr:
        lb = _letterbox_clf(bgr, CLF_IMGSZ)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - _CLF_MEAN) / _CLF_STD
        imgs.append(tensor)

    batch = torch.stack(imgs, dim=0)
    dev = torch.device(_clf_device if _clf_device else device)
    if dev.type != "cpu":
        batch = batch.to(dev, non_blocking=True)

    with torch.inference_mode():
        out = _clf_model(batch)  # type: ignore[arg-type]
        probs = torch.softmax(out, dim=1)[:, 1]
        result = probs.detach().cpu().numpy().tolist()

    del batch, out, probs
    return result



def _batch_classifier_accept_indices(
    nat,
    candidate_idxs: list[int],
    orig_img: np.ndarray,
    clf_thresh: float,
    device: str,
) -> set[int]:
    """Run classifier once for multiple candidate boxes and return accepted nat indices."""
    if not candidate_idxs:
        return set()
    h_i, w_i = orig_img.shape[:2]
    crops: list[np.ndarray] = []
    idx_map: list[int] = []
    for j in candidate_idxs:
        row = nat[j]
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * float(CROP_SCALE)
        bh = (y2 - y1) * float(CROP_SCALE)
        left = _clamp_int(cx - bw / 2.0, 0, w_i - 1)
        right = _clamp_int(cx + bw / 2.0, 0, w_i)
        top = _clamp_int(cy - bh / 2.0, 0, h_i - 1)
        bottom = _clamp_int(cy + bh / 2.0, 0, h_i)
        if right <= left + 1 or bottom <= top + 1:
            continue
        crops.append(orig_img[top:bottom, left:right])
        idx_map.append(j)
    if not crops:
        return set()
    probs = _run_crop_classifier(crops, device)
    accepted: set[int] = set()
    for k, p in enumerate(probs):
        if float(p) >= float(clf_thresh):
            accepted.add(idx_map[k])
    return accepted

def _crop_out_root() -> Optional[Path]:
    try:
        if CROP_OUT_DIR.is_absolute():
            return CROP_OUT_DIR
        return (Path(__file__).resolve().parent / CROP_OUT_DIR).resolve()
    except Exception:
        return None


def _log_small_box_rows(rows: list[dict]) -> None:
    out_root = _crop_out_root()
    if out_root is None:
        return
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = out_root / SMALL_BOX_LOG_NAME
        write_header = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "ts_ms",
                "result_id",
                "cam_index",
                "cam_ip",
                "conf",
                "x1",
                "y1",
                "x2",
                "y2",
                "w2x",
                "h2x",
                "reason",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception:
        return


def _filter_det_for_class0_and_min_2x_size(
    det,
    *,
    cam_index: int,
    cam_ip: str,
    result_id: int,
) -> Optional[torch.Tensor]:
    """Keep only class0 (if configured) and drop too-small 2x-expanded boxes; log drops."""
    if det is None or not len(det):
        return det

    d = det
    try:
        # d is (N,6): x1 y1 x2 y2 conf cls
        x1 = d[:, 0]
        y1 = d[:, 1]
        x2 = d[:, 2]
        y2 = d[:, 3]
        conf = d[:, 4]
        cls = d[:, 5].to(torch.int64)

        w2x = (x2 - x1) * float(CROP_SCALE)
        h2x = (y2 - y1) * float(CROP_SCALE)
        small = (w2x < float(MIN_CROP_PX)) | (h2x < float(MIN_CROP_PX))

        if ONLY_CLASS0:
            is_target = cls == 0
            keep = is_target & (~small)
            removed = is_target & small
        else:
            keep = ~small
            removed = small

        # Log removed rows
        try:
            if bool(removed.any()):
                removed_rows = d[removed].detach().cpu().numpy()
                w2x_rows = w2x[removed].detach().cpu().numpy()
                h2x_rows = h2x[removed].detach().cpu().numpy()
                rows = []
                ts_ms = int(time.time() * 1000)
                for r, ww, hh in zip(removed_rows, w2x_rows, h2x_rows):
                    rows.append(
                        {
                            "ts_ms": ts_ms,
                            "result_id": int(result_id),
                            "cam_index": int(cam_index),
                            "cam_ip": cam_ip,
                            "conf": float(r[4]),
                            "x1": float(r[0]),
                            "y1": float(r[1]),
                            "x2": float(r[2]),
                            "y2": float(r[3]),
                            "w2x": float(ww),
                            "h2x": float(hh),
                            "reason": "min_2x_px",
                        }
                    )
                _log_small_box_rows(rows)
        except Exception:
            pass

        return d[keep]
    except Exception:
        return d


def _send_line_signal_pulse(result_id: int) -> None:
    """Send Modbus pulse (ON then OFF) with cooldown / debounce logic.

    Behaviour:
      - First request: send immediately.
      - After a pulse, start a cooldown timer (LINE_SIGNAL_COOLDOWN_MS, default 2 min).
      - If NEW requests arrive during cooldown, we DO NOT send, but we reset the timer
        so that we require a full quiet period of cooldown length before sending again.
    """
    if not LINE_SIGNAL_ENABLE or not LINE_SIGNAL_IP or not _line_signal_ui_enabled:
        return

    now_ms = int(time.time() * 1000)
    with _state_lock:
        global _last_line_signal_ms, _last_line_signal_rid
        # Track last time we saw a *request* (any call with a result_id)
        if _last_line_signal_ms <= 0:
            # First ever request: allow sending immediately
            _last_line_signal_ms = now_ms
            _last_line_signal_rid = int(result_id)
        else:
            elapsed = now_ms - int(_last_line_signal_ms)
            if elapsed < int(LINE_SIGNAL_COOLDOWN_MS):
                # Still within cooldown → reset timer (debounce) and skip sending
                _last_line_signal_ms = now_ms
                _last_line_signal_rid = int(result_id)
                return
            # Cooldown has fully expired → allow sending now and reset timer
            _last_line_signal_ms = now_ms
            _last_line_signal_rid = int(result_id)

    def _worker():
        try:
            from . import modbus_tcp_send  # isolated local module

            modbus_tcp_send.pulse_register(
                LINE_SIGNAL_IP,
                port=int(LINE_SIGNAL_PORT),
                unit_id=int(LINE_SIGNAL_UNIT),
                register=int(LINE_SIGNAL_REGISTER),
                hold_seconds=float(LINE_SIGNAL_HOLD_S),
            )
        except Exception as e:
            print(f"[signal] ERROR: {e}", flush=True)

    threading.Thread(target=_worker, daemon=True).start()


def _save_beads_package_crops(
    *,
    orig_bgr: np.ndarray,
    det,
    cam_index: int,
    result_id: int,
    cam_ip: str,
) -> int:
    """Save 2x expanded crops for class 0 detections."""
    if not SAVE_BEADS_CROPS:
        return 0

    out_root = _crop_out_root()
    if out_root is None:
        return 0
    try:
        out_dir = out_root / f"cam{cam_index}_{cam_ip.replace('.', '-')}"
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return 0

    h, w = orig_bgr.shape[:2]
    saved = 0

    det_cpu = det.cpu().numpy() if hasattr(det, "cpu") else det
    for j, (x1, y1, x2, y2, conf, cls) in enumerate(det_cpu):
        if saved >= max(1, int(MAX_CROPS_PER_CAM)):
            break
        if int(cls) != 0:
            continue

        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        bw = (float(x2) - float(x1)) * float(CROP_SCALE)
        bh = (float(y2) - float(y1)) * float(CROP_SCALE)

        left = _clamp_int(cx - bw / 2.0, 0, w - 1)
        right = _clamp_int(cx + bw / 2.0, 0, w)
        top = _clamp_int(cy - bh / 2.0, 0, h - 1)
        bottom = _clamp_int(cy + bh / 2.0, 0, h)

        if right <= left + 1 or bottom <= top + 1:
            continue

        # If clamping made it too small, skip + log
        crop_w = int(right - left)
        crop_h = int(bottom - top)
        if crop_w < int(MIN_CROP_PX) or crop_h < int(MIN_CROP_PX):
            try:
                _log_small_box_rows(
                    [
                        {
                            "ts_ms": int(time.time() * 1000),
                            "result_id": int(result_id),
                            "cam_index": int(cam_index),
                            "cam_ip": cam_ip,
                            "conf": float(conf),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "w2x": float(bw),
                            "h2x": float(bh),
                            "reason": "clamped_min_px",
                        }
                    ]
                )
            except Exception:
                pass
            continue

        # Extract raw crop (no display adjustments, no annotations) - explicit copy to ensure raw data
        # This crop is already 2x the bbox size (CROP_SCALE=2.0), saved at original resolution
        crop = orig_bgr[top:bottom, left:right].copy()
        ts = int(time.time() * 1000)
        fname = f"{ts}__rid{result_id}__cam{cam_index}__box{j}__conf{float(conf):.2f}.jpg"
        try:
            cv2.imwrite(
                str(out_dir / fname),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(CROP_JPEG_QUALITY)],
            )
            saved += 1
        except Exception:
            continue

    return saved


def _stream_worker(run_generation: int):
    global _last_stream_jpeg
    interval = 1.0 / max(STREAM_FPS, 1.0)
    while True:
        with _state_lock:
            if not _running or _run_generation != run_generation:
                break
            frame = _last_motion_frame
        if frame is not None:
            disp = frame
            if 0 < STREAM_SCALE < 1.0:
                h, w = disp.shape[:2]
                disp = cv2.resize(disp, (max(1, int(w * STREAM_SCALE)), max(1, int(h * STREAM_SCALE))), interpolation=cv2.INTER_AREA)
            disp = _apply_display_adjust(disp)
            jpeg = _encode_jpeg(disp, STREAM_JPEG_QUALITY)
            if jpeg is not None:
                with _state_lock:
                    if _running and _run_generation == run_generation:
                        _last_stream_jpeg = jpeg
        time.sleep(interval)


def _grab_worker(run_generation: int):
    """Continuously grab raw Mono8 frames from the motion camera.

    Stores the raw grayscale frame (no BGR conversion) so _motion_worker
    can use it directly without double-conversion.  BGR conversion only
    happens where actually needed (inference, streaming).
    Also stores in _latest_cam_raw[_motion_idx] so inference can read it.
    """
    global _last_motion_frame, _last_motion_ts
    fail_count = 0
    while True:
        with _state_lock:
            if not _running or _run_generation != run_generation:
                break
            cam = _cams[_motion_idx] if _cams else None
            midx = _motion_idx
        if cam is None:
            time.sleep(0.01)
            continue

        try:
            frame, _ = _grab_one(cam)
        except Exception as e:
            fail_count += 1
            if fail_count in (1, 10):
                print(f"[grab][motion] exception: {e}", flush=True)
            if CAM_RECOVER_FAILS > 0 and fail_count % CAM_RECOVER_FAILS == 0:
                _try_restart_grabbing(cam, "motion")
            time.sleep(0.01)
            continue

        if frame is None:
            fail_count += 1
            if CAM_RECOVER_FAILS > 0 and fail_count % CAM_RECOVER_FAILS == 0:
                _try_restart_grabbing(cam, "motion")
            time.sleep(0.002)
            continue

        fail_count = 0

        # Store raw Mono8 (grayscale) -- no conversion needed
        ts = time.perf_counter()
        with _state_lock:
            if not _running or _run_generation != run_generation:
                break
            _last_motion_frame = frame
            _last_motion_ts = ts
            _latest_cam_raw[midx] = frame
            _latest_cam_ts[midx] = ts


def _cam_drain_worker(cam_idx: int, run_generation: int):
    """Continuously drain frames from a non-motion camera.

    Keeps the SDK buffer empty so that frames are always fresh.
    Stores the latest raw Mono8 frame in _latest_cam_raw[cam_idx].
    """
    fail_count = 0
    while True:
        with _state_lock:
            if not _running or _run_generation != run_generation:
                break
            cam = _cams[cam_idx] if cam_idx < len(_cams) else None
        if cam is None:
            time.sleep(0.01)
            continue

        try:
            frame, _ = _grab_one(cam)
        except Exception as e:
            fail_count += 1
            if fail_count in (1, 10):
                print(f"[grab][cam{cam_idx}] exception: {e}", flush=True)
            if CAM_RECOVER_FAILS > 0 and fail_count % CAM_RECOVER_FAILS == 0:
                _try_restart_grabbing(cam, f"cam{cam_idx}")
            time.sleep(0.01)
            continue

        if frame is None:
            fail_count += 1
            if CAM_RECOVER_FAILS > 0 and fail_count % CAM_RECOVER_FAILS == 0:
                _try_restart_grabbing(cam, f"cam{cam_idx}")
            time.sleep(0.002)
            continue

        fail_count = 0

        ts = time.perf_counter()
        with _state_lock:
            if not _running or _run_generation != run_generation:
                break
            _latest_cam_raw[cam_idx] = frame
            _latest_cam_ts[cam_idx] = ts


def _motion_worker(run_generation: int):
    """Motion detection — matches websocket_stream.py logic.

    State machine (simple, proven):
      moving  --(speed < threshold for N frames)--> stopped & schedule inference
      stopped --(speed >= threshold*2 OR hold expired)--> moving (re-arm, instant)
    """
    global _last_motion_frame, _last_state, _last_speed
    prev_gray: Optional[np.ndarray] = None
    prev_pts: Optional[np.ndarray] = None
    speed_history: deque = deque(maxlen=SPEED_WINDOW)
    movement_state = "moving"
    stop_hold_counter = 0
    low_counter = 0
    last_detection_time = 0.0
    infer_scheduled_for_this_stop = False

    detection_interval = 1.0 / max(DETECTION_FPS, 1.0)

    while True:
        with _state_lock:
            if not _running or _run_generation != run_generation:
                break
            frame = _last_motion_frame
        if frame is None:
            time.sleep(0.002)
            continue

        now = time.perf_counter()
        if now - last_detection_time < detection_interval:
            time.sleep(0.001)
            continue

        # Frame is already Mono8 (grayscale) from _grab_worker -- use directly
        gray = frame

        speed, prev_pts = compute_vertical_speed(prev_gray, gray, prev_pts)
        prev_gray = gray

        # Smooth speed with median filter
        speed_history.append(speed)
        if len(speed_history) >= 3:
            speed_med = fast_median_3(speed_history[-3], speed_history[-2], speed_history[-1])
        elif len(speed_history) == 2:
            speed_med = (speed_history[-2] + speed_history[-1]) / 2.0
        else:
            speed_med = speed

        # State machine (matches websocket_stream.py exactly)
        if movement_state == "moving":
            if speed_med <= VERTICAL_SPEED_THRESHOLD:
                low_counter += 1
            else:
                low_counter = 0

            if low_counter >= STOP_FRAMES_REQUIRED:
                movement_state = "stopped"
                stop_hold_counter = 0
                low_counter = 0
                infer_scheduled_for_this_stop = False

        else:  # stopped
            stop_hold_counter += 1

            # Trigger inference after INFER_TRIGGER_FRAMES in stopped state
            if stop_hold_counter == max(1, int(INFER_TRIGGER_FRAMES)) and not infer_scheduled_for_this_stop:
                _start_inference()
                infer_scheduled_for_this_stop = True

            # Re-arm: hold expired OR speed >= threshold*2 (instant, no multi-frame requirement)
            if stop_hold_counter >= STOP_HOLD_FRAMES or speed_med >= VERTICAL_SPEED_THRESHOLD * 2.0:
                movement_state = "moving"

        last_detection_time = now
        _last_state = movement_state
        _last_speed = float(speed_med)


def _is_infer_busy() -> bool:
    try:
        if _infer_future is not None and not _infer_future.done():
            return True
    except Exception:
        pass
    try:
        return _infer_thread is not None and _infer_thread.is_alive()
    except Exception:
        return False


def _run_inference_job(run_generation: int) -> None:
    """Executor wrapper so we can track busy state without thread churn."""
    global _infer_thread
    cur = threading.current_thread()
    with _state_lock:
        _infer_thread = cur
    try:
        _inference_worker(run_generation)
    finally:
        with _state_lock:
            if _infer_thread is cur:
                _infer_thread = None


def _start_inference():
    global _infer_thread, _infer_executor, _infer_future, _last_infer_trigger_ms, _infer_started_ts
    # Global cooldown (prevents spam during flicker/stop)
    try:
        now_ms = int(time.time() * 1000)
        if now_ms - int(_last_infer_trigger_ms) < int(INFER_COOLDOWN_MS):
            return
        _last_infer_trigger_ms = now_ms
    except Exception:
        pass
    if _is_infer_busy():
        return
    with _state_lock:
        run_generation = _run_generation
    _infer_started_ts = time.time()
    if _infer_executor is None:
        _infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hik-infer")
    try:
        _infer_future = _infer_executor.submit(_run_inference_job, run_generation)
    except Exception:
        # Fallback for rare executor failures.
        _infer_thread = threading.Thread(target=_inference_worker, args=(run_generation,), daemon=True)
        _infer_thread.start()


def _grab_cam_thread(cam, result_slot, time_slot, index):
    """Grab a single camera frame in a thread (for parallel capture)."""
    t_start = time.perf_counter()
    frame, nbytes = _grab_one(cam)
    if frame is not None:
        result_slot[index] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    time_slot[index] = (time.perf_counter() - t_start) * 1000.0


def _render_and_encode_result(orig_img: np.ndarray, det, quality: int) -> Optional[bytes]:
    """Draw overlays and encode one JPEG result."""
    try:
        # IMPORTANT: always draw/adjust on a copy so we never mutate the raw frame.
        # Otherwise (when display adjust is effectively a no-op) rectangles can be
        # drawn into orig_img, which then leaks into saved crops/full frames.
        img = _apply_display_adjust(orig_img.copy())
        img = _draw_infer_mask_overlay(img)
        if det is not None and len(det):
            det_cpu = det.cpu().numpy() if hasattr(det, "cpu") else det
            for x1, y1, x2, y2, conf, cls in det_cpu:
                c = int(cls)
                if ONLY_CLASS0 and c != 0:
                    continue
                color = _class_colors_bgr.get(c, (0, 255, 0))
                # Draw box only – no label text
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        return _encode_jpeg(img, quality)
    except Exception:
        return None


def _inference_worker(run_generation: int):
    global _last_results, _last_capture_ms, _last_infer_ms, _last_total_ms
    global _last_preprocess_ms, _last_build_ms, _last_post_yolo_ms, _last_postprocess_ms, _last_encode_ms
    global _last_result_id, _last_speed_str, _last_result_ready_ts, _last_detected
    global _session_total_infer, _session_defect_infer, _first_infer_images_saved
    global _last_stale_warn_ts, _infer_started_ts
    try:
        with _state_lock:
            if not _running or _run_generation != run_generation:
                return
            device = _device
            motion_idx = _motion_idx
            n_cams = len(_cams)
            # Read pre-fetched latest frames from continuous drain threads (zero blocking)
            # Copy frames quickly while holding lock (needed for thread safety)
            raw_frames_refs = list(_latest_cam_raw)
            cam_ages: list[float] = [
                (time.perf_counter() - ts) * 1000.0 if ts > 0 else 9999.0
                for ts in _latest_cam_ts
            ]
        
        # Copy frames into pre-allocated buffers to avoid malloc churn.
        # Falls back to .copy() if buffers aren't available or shape mismatches.
        raw_frames: list[Optional[np.ndarray]] = [None, None, None]
        for i, f in enumerate(raw_frames_refs):
            if f is None:
                continue
            raw_frames[i] = f.copy()

        if _net is None or n_cams < 3:
            return

        t0 = time.perf_counter()

        # ---- CAPTURE (from pre-fetched latest frames — zero grab latency) ----
        # Convert to BGR, reusing pre-allocated buffers when shape matches.
        frames: list[Optional[np.ndarray]] = [None, None, None]
        for i in range(3):
            raw = raw_frames[i] if i < len(raw_frames) else None
            if raw is None:
                continue
            if raw.ndim == 2:
                bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            else:
                bgr = raw
            buf = _frame_bufs[i] if _frame_bufs is not None and i < len(_frame_bufs) else None
            if buf is not None and buf.shape == bgr.shape and buf.dtype == bgr.dtype:
                np.copyto(buf, bgr)
                frames[i] = buf
            else:
                frames[i] = bgr.copy()

        if any(f is None for f in frames):
            ages_str = "  ".join(f"c{i}={cam_ages[i]:.0f}ms" for i in range(3))
            print(f"[infer] ERROR: missing pre-fetched frame(s) (ages: {ages_str})", flush=True)
            return
        if CAM_STALE_MAX_AGE_MS > 0 and any(age > CAM_STALE_MAX_AGE_MS for age in cam_ages[:3]):
            now_ts = time.time()
            if now_ts - _last_stale_warn_ts >= 2.0:
                ages_str = "  ".join(f"c{i}={cam_ages[i]:.0f}ms" for i in range(3))
                print(
                    f"[infer] WARN: stale camera frame(s) ages=[{ages_str}] "
                    f"(limit={CAM_STALE_MAX_AGE_MS:.0f}ms), skip cycle",
                    flush=True,
                )
                _last_stale_warn_ts = now_ts
            return

        t1 = time.perf_counter()
        capture_ms = (t1 - t0) * 1000.0

        # ---- PREPROCESS: build orig_bgr ----
        n = len(frames)
        orig_bgr: list[np.ndarray] = []
        for f in frames:
            bgr = f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            orig_bgr.append(bgr)
        infer_bgr: list[np.ndarray] = [_apply_infer_topright_mask(bgr) for bgr in orig_bgr]

        t2 = time.perf_counter()
        with _state_lock:
            current_conf_native = _runtime_conf_native
            current_conf_640 = _runtime_conf_640

        # ---- Build both batches into reusable host/device buffers ----
        if (
            _host_batch_native_np is None
            or _host_batch_640_np is None
            or _host_batch_native_tensor is None
            or _host_batch_640_tensor is None
            or _dev_batch_native is None
            or _dev_batch_640 is None
        ):
            raise RuntimeError("inference buffers are not initialized")

        _fill_chw_batch(_host_batch_native_np, infer_bgr, _letterbox, _net_imgsz)
        _fill_chw_batch(_host_batch_640_np, infer_bgr, _letterbox_640, (640, 640))
        tensor_native = _dev_batch_native[:n]
        tensor_640 = _dev_batch_640[:n]
        tensor_native.copy_(_host_batch_native_tensor[:n], non_blocking=device.startswith("cuda"))
        tensor_640.copy_(_host_batch_640_tensor[:n], non_blocking=device.startswith("cuda"))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t_build_done = time.perf_counter()

        # ---- Run both resolutions on GPU ----
        with torch.no_grad():
            if device.startswith("cuda") and PARALLEL_YOLO_STREAMS and _cuda_stream_nat is not None:
                with torch.cuda.stream(_cuda_stream_nat):
                    preds_nat = _net(tensor_native)
                with torch.cuda.stream(_cuda_stream_640):
                    preds_640 = _net(tensor_640)
                torch.cuda.synchronize()
            else:
                preds_nat = _net(tensor_native)
                preds_640 = _net(tensor_640)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
        t_forward_done = time.perf_counter()

        if isinstance(preds_nat, (list, tuple)):
            preds_nat = preds_nat[0]
        if isinstance(preds_640, (list, tuple)):
            preds_640 = preds_640[0]
        dets_native = non_max_suppression(preds_nat, conf_thres=current_conf_native, iou_thres=0.45)
        dets_640 = non_max_suppression(preds_640, conf_thres=current_conf_640, iou_thres=0.45)
        native_shape = tuple(int(v) for v in tensor_native.shape[2:])
        low_shape = tuple(int(v) for v in tensor_640.shape[2:])

        for i in range(n):
            det = dets_native[i]
            if det is not None:
                det_cpu = det.float().cpu()
                if len(det):
                    det_cpu[:, :4] = scale_boxes(
                        native_shape, det_cpu[:, :4], orig_bgr[i].shape[:2]
                    ).round()
                dets_native[i] = det_cpu
        for i in range(n):
            det = dets_640[i]
            if det is not None:
                det_cpu = det.float().cpu()
                if len(det):
                    det_cpu[:, :4] = scale_boxes(
                        low_shape, det_cpu[:, :4], orig_bgr[i].shape[:2]
                    ).round()
                dets_640[i] = det_cpu

        # Drop local refs promptly; backing buffers themselves stay allocated and are reused.
        del tensor_native, tensor_640, preds_nat, preds_640

        t3 = time.perf_counter()
        preprocess_ms = (t2 - t1) * 1000.0
        build_ms = (t_build_done - t2) * 1000.0
        forward_ms = (t_forward_done - t_build_done) * 1000.0
        post_yolo_ms = (t3 - t_forward_done) * 1000.0

        try:
            rid_next = _last_result_id + 1
        except Exception:
            rid_next = 0

        # ---- MERGE: low-conf flow = 640 check first, then classifier ----
        # Get runtime classifier settings
        with _state_lock:
            clf_enable = _runtime_clf_enable
            clf_thresh = _runtime_clf_thresh
            clf_conf_trigger = _runtime_clf_conf_trigger

        # Step 2: merge by requested flow:
        # - conf >= trigger: accept native directly
        # - conf < trigger: must pass 640 IoU/class first, then classifier
        dets_list: list[Optional[torch.Tensor]] = [None] * n
        for i in range(n):
            nat = dets_native[i]
            low = dets_640[i]
            if nat is None or len(nat) == 0:
                dets_list[i] = nat if nat is not None else torch.empty((0, 6), dtype=torch.float32)
                continue

            keep: list[bool] = []
            dbg_high_keep = 0
            dbg_low_reject_no640 = 0
            dbg_low_reject_clf = 0
            dbg_low_reject_iou = 0
            dbg_low_keep_iou = 0
            if low is None or len(low) == 0:
                for j in range(len(nat)):
                    row = nat[j]
                    cls = int(row[5])
                    if ONLY_CLASS0 and cls != 0:
                        keep.append(True)
                        continue
                    conf = float(row[4])
                    # High confidence native accepted directly, except single-det guard
                    # (if only one native box and conf < 0.7, force 640+clf path)
                    single_det_force_path = (len(nat) == 1 and conf < SINGLE_DET_FORCE_640_CONF)
                    if conf >= clf_conf_trigger and not single_det_force_path:
                        keep.append(True)
                        dbg_high_keep += 1
                        continue
                    # Low confidence requires 640 confirmation; none present -> reject
                    keep.append(False)
                    dbg_low_reject_no640 += 1
            else:
                low_boxes = low[:, :4]
                low_classes = low[:, 5].to(torch.int64)
                clf_candidate_idxs: list[int] = []
                for j in range(len(nat)):
                    row = nat[j]
                    conf = float(row[4])
                    cls = int(row[5])
                    if ONLY_CLASS0 and cls != 0:
                        keep.append(True)
                        continue
                    single_det_force_path = (len(nat) == 1 and conf < SINGLE_DET_FORCE_640_CONF)
                    if conf >= clf_conf_trigger and not single_det_force_path:
                        keep.append(True)
                        dbg_high_keep += 1
                        continue

                    # Low confidence native: must pass 640 first
                    one_box = row[:4].unsqueeze(0)
                    iou = box_iou(one_box, low_boxes).squeeze(0)  # [M]
                    same_class_mask = (low_classes == cls)
                    if not same_class_mask.any():
                        keep.append(False)
                        dbg_low_reject_iou += 1
                        continue

                    max_iou_same_class = iou[same_class_mask].max().item()
                    if max_iou_same_class < _runtime_iou_thresh:
                        keep.append(False)
                        dbg_low_reject_iou += 1
                        continue

                    # Passed 640 IoU gate; defer classifier to one batched call per camera.
                    if clf_enable and _clf_model is not None:
                        keep.append(False)
                        clf_candidate_idxs.append(j)
                        continue
                    elif clf_enable:
                        keep.append(False)
                        dbg_low_reject_clf += 1
                        continue

                    keep.append(True)
                    dbg_low_keep_iou += 1

                if clf_enable and _clf_model is not None and clf_candidate_idxs:
                    accepted = _batch_classifier_accept_indices(
                        nat=nat,
                        candidate_idxs=clf_candidate_idxs,
                        orig_img=orig_bgr[i],
                        clf_thresh=clf_thresh,
                        device=device,
                    )
                    for idx in clf_candidate_idxs:
                        if idx in accepted:
                            keep[idx] = True
                            dbg_low_keep_iou += 1
                        else:
                            dbg_low_reject_clf += 1

            keep_t = torch.tensor(keep, dtype=torch.bool)
            dets_list[i] = nat[keep_t]
            if DEBUG_MERGE_DECISIONS:
                try:
                    print(
                        f"[merge][cam{i}] nat={len(nat)} low={(0 if low is None else len(low))} "
                        f"high_keep={dbg_high_keep} low_no640={dbg_low_reject_no640} "
                        f"low_clf_reject={dbg_low_reject_clf} low_iou_keep={dbg_low_keep_iou} "
                        f"low_iou_reject={dbg_low_reject_iou}",
                        flush=True,
                    )
                except Exception:
                    pass

        del dets_native, dets_640

        # Summarize detections per camera (for UI/status)
        det_texts: list[str] = []
        any_beads = False
        cams_with_beads: list[int] = []
        for i in range(n):
            det = dets_list[i]
            if det is None or not len(det):
                det_texts.append("none")
                continue
            det_cpu = det.cpu().numpy() if hasattr(det, "cpu") else det
            counts: dict[int, int] = {}
            max_conf: dict[int, float] = {}
            for *_xyxy, _conf, cls in det_cpu:
                c = int(cls)
                if ONLY_CLASS0 and c != 0:
                    continue
                counts[c] = counts.get(c, 0) + 1
                cf = float(_conf)
                prev = max_conf.get(c)
                if prev is None or cf > prev:
                    max_conf[c] = cf
            parts = []
            for c in sorted(counts.keys()):
                m = max_conf.get(c)
                if m is None:
                    parts.append(f"{_class_names.get(c, f'cls{c}')} x{counts[c]}")
                else:
                    parts.append(f"{_class_names.get(c, f'cls{c}')} x{counts[c]} (max {m:.2f})")
            if counts.get(0, 0) >= 1:
                any_beads = True
                cams_with_beads.append(i)
            det_texts.append(", ".join(parts) if parts else "none")

        with _state_lock:
            if not _running or _run_generation != run_generation:
                return

        # If any camera saw >=1 beads-package -> pulse line signal and save full frames
        if any_beads:
            _send_line_signal_pulse(rid_next)

            # Save full-frame RAW images for cameras with beads for later analysis
            try:
                out_root = (
                    DEFECT_FRAME_OUT_DIR
                    if DEFECT_FRAME_OUT_DIR.is_absolute()
                    else (Path(__file__).resolve().parent / DEFECT_FRAME_OUT_DIR).resolve()
                )
            except Exception:
                out_root = None

            if out_root is not None:
                try:
                    out_root.mkdir(parents=True, exist_ok=True)
                except Exception:
                    out_root = None

            if out_root is not None:
                ts_ms = int(time.time() * 1000)
                saved_paths: list[str] = []
                for cam_idx in cams_with_beads:
                    try:
                        ip = _cam_ips[cam_idx] if cam_idx < len(_cam_ips) else f"cam{cam_idx}"
                    except Exception:
                        ip = f"cam{cam_idx}"
                    safe_ip = str(ip).replace(".", "-")
                    fname = f"{ts_ms}__rid{rid_next}__cam{cam_idx}_{safe_ip}.jpg"
                    fpath = out_root / fname
                    try:
                        cv2.imwrite(str(fpath), orig_bgr[cam_idx])
                        saved_paths.append(str(fpath))
                    except Exception:
                        continue

                if saved_paths:
                    with _state_lock:
                        # Keep only last 10 paths to avoid unbounded growth
                        global _last_defect_paths, _defect_frame_out_dir
                        _defect_frame_out_dir = out_root
                        _last_defect_paths.extend(saved_paths)
                        if len(_last_defect_paths) > 30:
                            _last_defect_paths = _last_defect_paths[-30:]

            # Additionally save annotated full-frame images (with bboxes) to a separate folder
            try:
                box_root = (
                    DEFECT_FRAME_BOX_OUT_DIR
                    if DEFECT_FRAME_BOX_OUT_DIR.is_absolute()
                    else (Path(__file__).resolve().parent / DEFECT_FRAME_BOX_OUT_DIR).resolve()
                )
            except Exception:
                box_root = None

            if box_root is not None:
                try:
                    box_root.mkdir(parents=True, exist_ok=True)
                except Exception:
                    box_root = None

            if box_root is not None:
                ts_ms_box = int(time.time() * 1000)
                for cam_idx in cams_with_beads:
                    try:
                        ip = _cam_ips[cam_idx] if cam_idx < len(_cam_ips) else f"cam{cam_idx}"
                    except Exception:
                        ip = f"cam{cam_idx}"
                    safe_ip = str(ip).replace(".", "-")
                    fname_box = f"{ts_ms_box}__rid{rid_next}__cam{cam_idx}_{safe_ip}_annotated.jpg"
                    fpath_box = box_root / fname_box
                    try:
                        img_annot = orig_bgr[cam_idx].copy()
                        det = dets_list[cam_idx]
                        # Draw all merged bboxes (all classes) with confidence labels before saving
                        if det is not None and len(det):
                            det_cpu = det.cpu().numpy() if hasattr(det, "cpu") else det
                            for x1, y1, x2, y2, conf, cls in det_cpu:
                                c = int(cls)
                                color = _class_colors_bgr.get(c, (0, 255, 0))
                                x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                                # Draw bounding box
                                cv2.rectangle(
                                    img_annot,
                                    (x1_int, y1_int),
                                    (x2_int, y2_int),
                                    color,
                                    2,
                                )
                                # Draw confidence label on top-left corner
                                conf_str = f"{float(conf):.2f}"
                                (text_w, text_h), baseline = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                # Draw background rectangle for text readability
                                cv2.rectangle(
                                    img_annot,
                                    (x1_int, y1_int - text_h - 4),
                                    (x1_int + text_w + 4, y1_int),
                                    color,
                                    -1,
                                )
                                cv2.putText(
                                    img_annot,
                                    conf_str,
                                    (x1_int + 2, y1_int - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),  # white text
                                    1,
                                )
                        cv2.imwrite(str(fpath_box), img_annot)
                    except Exception:
                        continue

        t4 = time.perf_counter()
        postprocess_ms = (t4 - t3) * 1000.0

        # ---- ENCODE JPEGs (persistent worker pool, no per-infer thread churn) ----
        encoded: list[Optional[bytes]] = [None, None, None]
        if _encode_executor is not None:
            futures = []
            for i in range(n):
                futures.append(_encode_executor.submit(_render_and_encode_result, orig_bgr[i], dets_list[i], 85))
            for i, fut in enumerate(futures):
                try:
                    encoded[i] = fut.result()
                except Exception:
                    encoded[i] = None
        else:
            for i in range(n):
                encoded[i] = _render_and_encode_result(orig_bgr[i], dets_list[i], 85)

        t5 = time.perf_counter()
        encode_ms = (t5 - t4) * 1000.0
        total_ms = (t5 - t0) * 1000.0

        # ---- STORE RESULTS (client can fetch from this instant) ----
        ready_ts = time.time()
        with _state_lock:
            if not _running or _run_generation != run_generation:
                return
            _last_results = encoded[:3]
            _last_capture_ms = capture_ms
            _last_preprocess_ms = preprocess_ms
            _last_build_ms = build_ms
            _last_infer_ms = forward_ms
            _last_post_yolo_ms = post_yolo_ms
            _last_postprocess_ms = postprocess_ms
            _last_encode_ms = encode_ms
            _last_total_ms = total_ms
            _last_result_id += 1
            _last_result_ready_ts = ready_ts
            _last_detected = det_texts[:3]
            _last_speed_str = (
                f"pre {preprocess_ms:.1f}  build {build_ms:.1f}  fwd {forward_ms:.1f}  "
                f"nms {post_yolo_ms:.1f}  "
                f"post {postprocess_ms:.1f}  enc {encode_ms:.1f} ms"
            )
            _session_total_infer += 1
            if any_beads:
                _session_defect_infer += 1
            session_total_infer = _session_total_infer

        # Save first 3 inference images (one per cam) to live frame folder once: native frame + merged bboxes
        with _state_lock:
            do_first_save = not _first_infer_images_saved
            if do_first_save:
                _first_infer_images_saved = True
        if do_first_save:
            try:
                out_root = (
                    DEFECT_FRAME_OUT_DIR
                    if DEFECT_FRAME_OUT_DIR.is_absolute()
                    else (Path(__file__).resolve().parent / DEFECT_FRAME_OUT_DIR).resolve()
                )
                out_root.mkdir(parents=True, exist_ok=True)
                for i in range(min(3, n)):
                    img = orig_bgr[i].copy()
                    det = dets_list[i]
                    if det is not None and len(det):
                        det_cpu = det.cpu().numpy() if hasattr(det, "cpu") else det
                        for x1, y1, x2, y2, conf, cls in det_cpu:
                            c = int(cls)
                            if ONLY_CLASS0 and c != 0:
                                continue
                            color = _class_colors_bgr.get(c, (0, 255, 0))
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    path = out_root / f"first_infer_cam{i}.jpg"
                    cv2.imwrite(str(path), img)
                print(f"[infer] Saved first 3 inference images (merged bboxes) to {out_root}", flush=True)
            except Exception as e:
                print(f"[infer] Failed to save first-infer images: {e}", flush=True)

        # Save beads-package crops asynchronously (do not block UI update).
        # Snapshot GPU tensors to CPU numpy so the background thread holds no GPU refs.
        if SAVE_BEADS_CROPS:
            try:
                cam_ips_local = list(_cam_ips)
            except Exception:
                cam_ips_local = []

            dets_cpu_snap = []
            for i in range(n):
                d = dets_list[i]
                if d is not None and len(d):
                    dets_cpu_snap.append(d.detach().cpu().numpy())
                else:
                    dets_cpu_snap.append(None)

            if _crop_worker_slots.acquire(blocking=False):
                def _crop_worker(_dets_snap=dets_cpu_snap, _orig=orig_bgr,
                                 _ips=cam_ips_local, _rid=rid_next, _n=n):
                    try:
                        for i in range(_n):
                            det = _dets_snap[i]
                            if det is None or not len(det):
                                continue
                            try:
                                cam_ip = _ips[i] if i < len(_ips) else f"cam{i}"
                            except Exception:
                                cam_ip = f"cam{i}"
                            _save_beads_package_crops(
                                orig_bgr=_orig[i],
                                det=det,
                                cam_index=i,
                                result_id=_rid,
                                cam_ip=cam_ip,
                            )
                    except Exception:
                        pass
                    finally:
                        try:
                            _crop_worker_slots.release()
                        except Exception:
                            pass

                threading.Thread(target=_crop_worker, daemon=True).start()

        del dets_list

        age_str = "  ".join(f"c{i}={cam_ages[i]:.0f}ms" for i in range(3))
        print(f"[infer] ages=[{age_str}]  cvt={capture_ms:.1f}ms  "
              f"pre={preprocess_ms:.1f}ms  build={build_ms:.1f}ms  "
              f"fwd={forward_ms:.1f}ms  nms={post_yolo_ms:.1f}ms  "
              f"post={postprocess_ms:.1f}ms  enc={encode_ms:.1f}ms  "
              f"total={total_ms:.1f}ms", flush=True)
        if device.startswith("cuda") and CUDA_STATS_EVERY > 0 and (session_total_infer % CUDA_STATS_EVERY) == 0:
            try:
                alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
                print(
                    f"[cuda] infer_count={session_total_infer}  alloc={alloc_mb:.1f}MB  "
                    f"reserved={reserved_mb:.1f}MB  peak_reserved={peak_reserved_mb:.1f}MB",
                    flush=True,
                )
            except Exception:
                pass
    except Exception as e:
        import traceback
        print(f"[infer] ERROR: {e}", flush=True)
        traceback.print_exc()
    finally:
        _infer_started_ts = 0.0


def _open_cameras() -> tuple[list[MvCamera], list[str], int]:
    MvCamera.MV_CC_Initialize()
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_GIGE_DEVICE, device_list
    )
    if ret != 0 or device_list.nDeviceNum == 0:
        raise RuntimeError("No Hikrobot cameras found")

    devs: list[tuple[str, MV_CC_DEVICE_INFO]] = []
    for i in range(device_list.nDeviceNum):
        dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        ip_str = _ip_to_str(_ip_from_devinfo(dev_info))
        devs.append((ip_str, dev_info))

    # If LINE_IPS provided, use those in that order
    if LINE_IPS:
        wanted = [s.strip() for s in LINE_IPS.split(",") if s.strip()]
        if len(wanted) != 3:
            raise RuntimeError("HIK_LINE_IPS must contain exactly 3 IPs")
        ordered = []
        for ip in wanted:
            match = next((d for d in devs if d[0] == ip), None)
            if match is None:
                raise RuntimeError(f"Camera IP not found: {ip}")
            ordered.append(match)
    else:
        # Default: sort by IP and take first 3
        ordered = sorted(devs, key=lambda x: x[0])[:3]

    cams: list[MvCamera] = []
    ips: list[str] = []
    for ip_str, dev_info in ordered:
        cam = _open_camera(dev_info)
        cams.append(cam)
        ips.append(ip_str)

    # Per-camera exposure defaults:
    #   - Cam 12 (..12): 585 µs
    #   - Cams 11/13 (..11, ..13): 885 µs
    # Falls back to EXPOSURE_US for any other camera/IP.
    # Still runtime-adjustable via /exposure API.
    global _exposure_us_per_cam
    _exposure_us_per_cam = []
    for cam, ip in zip(cams, ips):
        try:
            last_octet = ip.split(".")[-1]
        except Exception:
            last_octet = ""
        try:
            if last_octet == "12":
                exp = float(EXPOSURE_US_CAM12)
            elif last_octet in ("11", "13"):
                exp = float(EXPOSURE_US_CAM11_13)
            else:
                exp = float(EXPOSURE_US)
            cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            cam.MV_CC_SetFloatValue("ExposureTime", exp)
            _exposure_us_per_cam.append(exp)
        except Exception:
            _exposure_us_per_cam.append(float(EXPOSURE_US))

    # Motion index: prefer MOTION_IP if present
    motion_idx = 0
    if MOTION_IP:
        for i, ip in enumerate(ips):
            if ip == MOTION_IP:
                motion_idx = i
                break
    return cams, ips, motion_idx


def _close_cameras():
    global _cams
    for cam in _cams:
        try:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass
    _cams = []
    try:
        MvCamera.MV_CC_Finalize()
    except Exception:
        pass


def _reset_runtime_resources():
    """Deep reset of runtime resources so stop/start is closer to relaunch."""
    global _model, _net, _clf_model, _clf_device, _letterbox, _letterbox_640
    global _net_stride, _net_imgsz, _cuda_stream_nat, _cuda_stream_640, _frame_bufs
    global _host_batch_native_np, _host_batch_640_np
    global _host_batch_native_tensor, _host_batch_640_tensor
    global _dev_batch_native, _dev_batch_640
    global _last_motion_frame, _last_motion_ts, _last_stream_jpeg
    global _latest_cam_raw, _latest_cam_ts
    global _last_results, _last_capture_ms, _last_infer_ms, _last_total_ms
    global _last_preprocess_ms, _last_build_ms, _last_post_yolo_ms, _last_postprocess_ms, _last_encode_ms
    global _last_result_id, _last_speed_str, _last_detected, _last_result_ready_ts
    global _last_infer_trigger_ms, _infer_thread, _grab_thread, _stop_thread, _infer_started_ts
    global _infer_executor, _infer_future, _encode_executor
    global _nvidia_smi_adv_available, _nvidia_smi_basic_warned

    infer_exec: Optional[ThreadPoolExecutor] = None
    encode_exec: Optional[ThreadPoolExecutor] = None

    with _state_lock:
        _model = None
        _net = None
        _clf_model = None
        _clf_device = "cpu"
        _letterbox = None
        _letterbox_640 = None
        _cuda_stream_nat = None
        _cuda_stream_640 = None
        _frame_bufs = None
        _host_batch_native_np = None
        _host_batch_640_np = None
        _host_batch_native_tensor = None
        _host_batch_640_tensor = None
        _dev_batch_native = None
        _dev_batch_640 = None
        _net_stride = 32
        _net_imgsz = (640, 640)

        _last_motion_frame = None
        _last_motion_ts = 0.0
        _last_stream_jpeg = None
        _latest_cam_raw = [None, None, None]
        _latest_cam_ts = [0.0, 0.0, 0.0]

        _last_results = [None, None, None]
        _last_capture_ms = None
        _last_infer_ms = None
        _last_total_ms = None
        _last_preprocess_ms = None
        _last_build_ms = None
        _last_post_yolo_ms = None
        _last_postprocess_ms = None
        _last_encode_ms = None
        _last_result_id = 0
        _last_speed_str = None
        _last_detected = ["--", "--", "--"]
        _last_result_ready_ts = 0.0
        _last_infer_trigger_ms = 0
        _infer_started_ts = 0.0

        _infer_thread = None
        _grab_thread = None
        _stop_thread = None
        infer_exec = _infer_executor
        encode_exec = _encode_executor
        _infer_executor = None
        _infer_future = None
        _encode_executor = None
        _nvidia_smi_adv_available = None
        _nvidia_smi_basic_warned = False

    for _ex in (infer_exec, encode_exec):
        if _ex is None:
            continue
        try:
            _ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            try:
                _ex.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass

    # Best-effort memory cleanup (helps long-run latency creep).
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available() and MEM_CLEANUP_CUDA:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _cleanup_memory(reason: str = "manual") -> dict:
    """Best-effort memory cleanup without stopping runtime."""
    global _last_mem_cleanup_ts, _last_mem_cleanup_reason
    now = time.time()
    gc_collected = 0
    cuda_cache_cleared = False
    try:
        gc_collected = int(gc.collect())
    except Exception:
        gc_collected = 0
    try:
        if torch.cuda.is_available() and MEM_CLEANUP_CUDA:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
                cuda_cache_cleared = True
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
                cuda_cache_cleared = True or cuda_cache_cleared
            except Exception:
                pass
    except Exception:
        pass
    with _state_lock:
        _last_mem_cleanup_ts = now
        _last_mem_cleanup_reason = str(reason)
    return {
        "ts": now,
        "gc_collected": gc_collected,
        "cuda_cache_cleared": bool(cuda_cache_cleared),
        "reason": str(reason),
    }


def _memory_cleanup_worker():
    while True:
        time.sleep(MEM_CLEANUP_INTERVAL_S)
        if not MEM_CLEANUP_ENABLED:
            continue
        try:
            with _state_lock:
                if not _running or _starting or _maintenance_refreshing or _hard_relaunch_in_progress:
                    continue
            info = _cleanup_memory(reason="auto_interval")
            print(
                f"[mem] cleanup done (reason={info.get('reason')}, "
                f"gc={info.get('gc_collected')}, cuda={info.get('cuda_cache_cleared')})",
                flush=True,
            )
        except Exception as e:
            print(f"[mem] cleanup failed: {e}", flush=True)


@app.get("/")
def index():
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI Defect Detector — Line 1 &amp; Line 2</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f1115; color: #e6e6e6; font-family: Arial, sans-serif;
           height: 100vh; overflow: hidden; display: flex; flex-direction: column; }

    /* Header */
    .hdr { flex-shrink: 0; background: #13171f; border-bottom: 1px solid #1e242e;
           padding: 5px 14px; display: flex; align-items: center; gap: 16px; }
    .hdr-title { font-size: 13px; font-weight: 700; white-space: nowrap; }
    .hdr-line { font-size: 12px; white-space: nowrap; color: #9ba8bc; }
    .hdr-sys { display: flex; gap: 5px; margin-left: auto; }

    /* Main layout */
    .main { flex: 1; min-height: 0; display: flex; gap: 8px; padding: 7px 8px 7px; }

    /* Left panel */
    .left { width: 330px; flex-shrink: 0; display: flex; flex-direction: column;
            gap: 7px; overflow-y: auto; }
    .lcard { background: #151922; border-radius: 8px; padding: 9px 10px;
             display: flex; flex-direction: column; gap: 6px; }
    .lcard { border-top: 1px solid #252b38; }
    .line-title { font-size: 12px; font-weight: 700; color: #c9cdd6; letter-spacing: .5px; }

    /* Buttons */
    .btn-row { display: flex; gap: 4px; flex-wrap: wrap; }
    .btn { padding: 4px 10px; border: none; border-radius: 5px; background: #2a2f3a;
           color: #fff; cursor: pointer; font-size: 11px; transition: background .12s; }
    .btn:hover { background: #353d4d; }
    .btn.active-start { background: #1d7f4f; }
    .btn.active-stop  { background: #a04a4a; }
    .btn.power        { background: #6a2020; }
    .btn.power.cancel { background: #4a3a18; }

    /* Status */
    .status-line { font-size: 13px; font-weight: 700; }
    .status-line.moving  { color: #7bd88f; }
    .status-line.stopped { color: #f0a35a; }
    .status-line.idle    { color: #666; }

    /* Toggle switch */
    .sw-row { display: flex; align-items: center; gap: 5px; }
    .sw-label { font-size: 10px; color: #999; }
    .sw { position: relative; display: inline-block; width: 36px; height: 20px; }
    .sw input { opacity: 0; width: 0; height: 0; }
    .sw-track { position: absolute; cursor: pointer; inset: 0; background: #444;
                border-radius: 20px; transition: .15s; }
    .sw-track:before { content: ""; position: absolute; height: 14px; width: 14px;
                       left: 3px; bottom: 3px; background: #fff; border-radius: 50%;
                       transition: .15s; }
    input:checked + .sw-track { background: #2ecc71; }
    input:checked + .sw-track:before { transform: translateX(16px); }

    /* Text */
    .mtext { font-size: 10px; color: #999; white-space: pre-wrap; line-height: 1.4; }
    .timer-line { font-size: 11px; color: #c9cdd6; font-weight: 600; }

    /* Sliders */
    .sliders-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px 8px; }
    .sl-row { display: flex; flex-direction: column; gap: 1px; }
    .sl-lbl { font-size: 10px; color: #888; }
    .sl-inner { display: flex; align-items: center; gap: 3px; }
    .sl-inner input[type=range] { flex: 1; height: 4px; border-radius: 2px;
                                  background: #2a2f3a; -webkit-appearance: none; outline: none; }
    .sl-inner input[type=range]::-webkit-slider-thumb { -webkit-appearance: none;
      width: 12px; height: 12px; border-radius: 50%; background: #2ecc71; cursor: pointer; }
    .sl-inner input[type=range]::-moz-range-thumb { width: 12px; height: 12px;
      border-radius: 50%; background: #2ecc71; cursor: pointer; border: none; }
    .sl-val { font-size: 10px; color: #ccc; min-width: 34px; text-align: right; }

    /* Exposure */
    .exp-row { display: flex; gap: 3px; align-items: center; }
    .exp-lbl { font-size: 9px; color: #888; min-width: 55px; flex-shrink: 0; }
    .exp-inp { flex: 1; min-width: 0; background: #0f1115; color: #e6e6e6;
               border: 1px solid #2a2f3a; border-radius: 3px; padding: 2px 4px; font-size: 10px; }

    /* Camera reorder row */
    .cam-order-row { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
    .cam-order-lbl { font-size: 9px; color: #666; white-space: nowrap; min-width: 55px; flex-shrink: 0; }
    .cam-slot { font-size: 9px; color: #9ba8bc; background: #1a1f2a; border: 1px solid #252b38;
                border-radius: 3px; padding: 2px 5px; min-width: 36px; text-align: center; }
    .swap-btn { font-size: 10px; color: #555; background: none; border: none;
                cursor: pointer; padding: 0 1px; line-height: 1; transition: color .12s; }
    .swap-btn:hover { color: #9ba8bc; }

    /* Stream thumbnail */
    .stream-thumb { width: 100%; border-radius: 4px; background: #0a0c10;
                    display: block; max-height: 90px; object-fit: contain; }

    /* Right panel */
    .right { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
    .line-group { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 4px; }
    .lg-hdr { flex-shrink: 0; font-size: 10px; font-weight: 700; letter-spacing: 1px;
              padding: 3px 6px; border-radius: 3px; }
    .lg-hdr { color: #6b7a94; background: #111520; }
    .cams-row { flex: 1; min-height: 0; display: flex; gap: 6px; }
    .cam-cell { flex: 1; min-width: 0; background: #151922; border-radius: 7px;
                overflow: hidden; display: flex; flex-direction: column; }
    .cam-cell { border-top: 1px solid #252b38; }
    .cam-img-wrap { flex: 1; min-height: 0; background: #0a0c10;
                    display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .cam-img-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
    .cam-footer { flex-shrink: 0; padding: 3px 8px; display: flex;
                  justify-content: space-between; align-items: center; min-height: 20px; }
    .cam-label { font-size: 10px; color: #777; }
    .det-text { font-size: 10px; font-weight: 700; }
    .det-text.ok   { color: #7bd88f; }
    .det-text.bad  { color: #ff6b6b; }
    .det-text.idle { color: #555; }
  </style>
</head>
<body>

<div class="hdr">
  <span class="hdr-title">AI Defect Detector</span>
  <span class="hdr-line" id="hdrL2">Линия № 2: Idle</span>
  <span class="hdr-line" id="hdrL1">Линия № 1: Idle</span>
  <div class="hdr-sys">
    <button class="btn" onclick="openHistory()" style="font-size:10px;">History</button>
    <button class="btn power" onclick="shutdownComputer()" style="font-size:10px;">Shutdown</button>
    <button class="btn power cancel" onclick="cancelShutdown()" style="font-size:10px;">Cancel</button>
  </div>
</div>

<div class="main">

  <!-- Left panel -->
  <div class="left">

    <!-- LINE 2 control card -->
    <div class="lcard">
      <div class="line-title">Линия № 2</div>
      <div class="btn-row">
        <button class="btn" id="startBtn" onclick="startL2()">Start</button>
        <button class="btn" id="stopBtn"  onclick="stopL2()">Stop</button>
      </div>
      <div class="sw-row">
        <label class="sw"><input type="checkbox" id="signalToggle" onclick="setSignalEnabled(this.checked)"><span class="sw-track"></span></label>
        <span class="sw-label">Send signal to line</span>
      </div>
      <div class="status-line idle" id="statusL2">Idle</div>
      <div class="mtext" id="metricsL2">--</div>
      <div class="mtext" id="inferBreakdown">--</div>
      <div class="timer-line" id="timerL2">--</div>
      <div class="mtext" id="systemMsg">--</div>
      <img class="stream-thumb" id="stream" alt="motion cam">
      <div class="sliders-grid" style="margin-top:3px;">
        <div class="sl-row">
          <span class="sl-lbl">Native YOLO conf</span>
          <div class="sl-inner">
            <input type="range" id="yoloConfNativeSlider" min="0" max="100" value="38" oninput="updateYoloConfNative(this.value)">
            <span class="sl-val" id="yoloConfNativeValue">--</span>
          </div>
        </div>
        <div class="sl-row">
          <span class="sl-lbl">640 YOLO conf</span>
          <div class="sl-inner">
            <input type="range" id="yoloConf640Slider" min="0" max="100" value="38" oninput="updateYoloConf640(this.value)">
            <span class="sl-val" id="yoloConf640Value">--</span>
          </div>
        </div>
        <div class="sl-row">
          <span class="sl-lbl">640 IoU min</span>
          <div class="sl-inner">
            <input type="range" id="iouThreshSlider" min="0" max="100" value="50" oninput="updateIouThresh(this.value)">
            <span class="sl-val" id="iouThreshValue">0.50</span>
          </div>
        </div>
        <div class="sl-row">
          <span class="sl-lbl">Brightness</span>
          <div class="sl-inner">
            <input type="range" id="displayGainSlider" min="50" max="300" value="150" oninput="updateDisplayBrightness(this.value)">
            <span class="sl-val" id="displayGainValue">1.00x</span>
          </div>
        </div>
        <div class="sl-row">
          <span class="sl-lbl">Y640 trigger</span>
          <div class="sl-inner">
            <input type="range" id="clfConfTriggerSlider" min="0" max="100" value="55" oninput="updateClfConfTrigger(this.value)">
            <span class="sl-val" id="clfConfTriggerValue">0.55</span>
          </div>
        </div>
        <div class="sl-row">
          <span class="sl-lbl">CLF threshold</span>
          <div class="sl-inner">
            <input type="range" id="clfThreshSlider" min="0" max="100" value="80" oninput="updateClfThresh(this.value)">
            <span class="sl-val" id="clfThreshValue">0.80</span>
          </div>
        </div>
      </div>
      <div class="exp-row" style="margin-top:3px;">
        <span class="exp-lbl">Exposure (us)</span>
        <input type="number" id="expCam0" min="50" max="1000000" step="10" placeholder="Cam 11" class="exp-inp">
        <input type="number" id="expCam1" min="50" max="1000000" step="10" placeholder="Cam 13" class="exp-inp">
        <input type="number" id="expCam2" min="50" max="1000000" step="10" placeholder="Cam 12" class="exp-inp">
      </div>
      <div class="cam-order-row" style="margin-top:3px;">
        <span class="cam-order-lbl">Cam slots</span>
        <span class="cam-slot" id="l2s0">--</span>
        <button class="swap-btn" onclick="swapCams(2,0,1)" title="Swap slots 1&amp;2">&#8644;</button>
        <span class="cam-slot" id="l2s1">--</span>
        <button class="swap-btn" onclick="swapCams(2,1,2)" title="Swap slots 2&amp;3">&#8644;</button>
        <span class="cam-slot" id="l2s2">--</span>
      </div>
    </div>

    <!-- LINE 1 control card -->
    <div class="lcard">
      <div class="line-title">Линия № 1 <span style="font-size:9px;color:#555;font-weight:400;">(not connected)</span></div>
      <div class="btn-row">
        <button class="btn" id="startBtnL1" onclick="startL1()">Start</button>
        <button class="btn" id="stopBtnL1"  onclick="stopL1()">Stop</button>
      </div>
      <div class="sw-row">
        <label class="sw"><input type="checkbox" id="signalToggleL1" onclick="setSignalEnabledL1(this.checked)"><span class="sw-track"></span></label>
        <span class="sw-label">Send signal to line</span>
      </div>
      <div class="status-line idle" id="statusL1">Idle</div>
      <div class="timer-line" id="timerL1">--</div>
      <div class="mtext" style="color:#3a4a5a;margin-top:2px;">Cameras: placeholder - IPs not set</div>
      <div class="mtext" style="color:#3a4a5a;">Relay signal: placeholder - register not set</div>
      <div class="cam-order-row" style="margin-top:3px;">
        <span class="cam-order-lbl">Cam slots</span>
        <span class="cam-slot" id="l1s0">4</span>
        <button class="swap-btn" onclick="swapCams(1,0,1)" title="Swap slots 1&amp;2">&#8644;</button>
        <span class="cam-slot" id="l1s1">5</span>
        <button class="swap-btn" onclick="swapCams(1,1,2)" title="Swap slots 2&amp;3">&#8644;</button>
        <span class="cam-slot" id="l1s2">6</span>
      </div>
    </div>

  </div>

  <!-- Right panel (6 cameras) -->
  <div class="right">

    <div class="line-group">
      <div class="lg-hdr">Линия № 2 &mdash; Камеры 1 &middot; 2 &middot; 3</div>
      <div class="cams-row">
        <div class="cam-cell">
          <div class="cam-img-wrap"><img id="cam0" alt=""></div>
          <div class="cam-footer">
            <span class="cam-label" id="label0">Cam --</span>
            <span class="det-text idle" id="det0">--</span>
          </div>
        </div>
        <div class="cam-cell">
          <div class="cam-img-wrap"><img id="cam1" alt=""></div>
          <div class="cam-footer">
            <span class="cam-label" id="label1">Cam --</span>
            <span class="det-text idle" id="det1">--</span>
          </div>
        </div>
        <div class="cam-cell">
          <div class="cam-img-wrap"><img id="cam2" alt=""></div>
          <div class="cam-footer">
            <span class="cam-label" id="label2">Cam --</span>
            <span class="det-text idle" id="det2">--</span>
          </div>
        </div>
      </div>
    </div>

    <div class="line-group">
      <div class="lg-hdr">Линия № 1 &mdash; Камеры 4 &middot; 5 &middot; 6</div>
      <div class="cams-row">
        <div class="cam-cell">
          <div class="cam-img-wrap"><img id="cam3" alt=""></div>
          <div class="cam-footer">
            <span class="cam-label" id="label3">Cam 4</span>
            <span class="det-text idle" id="det3">--</span>
          </div>
        </div>
        <div class="cam-cell">
          <div class="cam-img-wrap"><img id="cam4" alt=""></div>
          <div class="cam-footer">
            <span class="cam-label" id="label4">Cam 5</span>
            <span class="det-text idle" id="det4">--</span>
          </div>
        </div>
        <div class="cam-cell">
          <div class="cam-img-wrap"><img id="cam5" alt=""></div>
          <div class="cam-footer">
            <span class="cam-label" id="label5">Cam 6</span>
            <span class="det-text idle" id="det5">--</span>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
let running = false, _l1Running = false;
let _lastResultId = null;
let _streamBusy = false, _loadingResults = false, _lastStreamMs = 0;
const STREAM_INTERVAL_MS = 120;
let _signalEnabled = true, _signalEnabledL1 = true;
let _l2CamOrder = [0, 2, 1];  // display slot -> data index (default: cam11, cam13, cam12)
let _l1CamOrder = [0, 1, 2];  // placeholder
let _l2IpsList = [];          // cached IP list for label updates

function fetchTimeout(url, ms) {
  const c = new AbortController(), t = setTimeout(() => c.abort(), ms);
  return fetch(url, { signal: c.signal }).finally(() => clearTimeout(t));
}

// LINE 2
function startL2() {
  setRunButtons(true);
  fetch('/start').then(() => { running = true; poll(); }).catch(() => setRunButtons(false));
}
function stopL2() {
  setRunButtons(false);
  fetch('/stop').then(() => { running = false; }).catch(() => {});
}
function start() { startL2(); }
function stop()  { stopL2(); }

function swapCams(line, slotA, slotB) {
  const arr = line === 2 ? _l2CamOrder : _l1CamOrder;
  const tmp = arr[slotA]; arr[slotA] = arr[slotB]; arr[slotB] = tmp;
  updateCamOrderUI();
  // force result reload on next poll tick
  if (line === 2) _lastResultId = null;
}

function updateCamOrderUI() {
  // Line 2 slot labels
  if (_l2IpsList.length === 3) {
    ['l2s0','l2s1','l2s2'].forEach((id, i) => {
      const el = document.getElementById(id);
      if (el) el.textContent = _l2IpsList[_l2CamOrder[i]];
    });
    // also update footer labels
    ['label0','label1','label2'].forEach((id, i) => {
      const el = document.getElementById(id);
      if (el) el.textContent = 'Cam ' + _l2IpsList[_l2CamOrder[i]];
    });
  }
  // Line 1 slot labels (placeholder numbers)
  const l1Labels = ['4','5','6'];
  ['l1s0','l1s1','l1s2'].forEach((id, i) => {
    const el = document.getElementById(id);
    if (el) el.textContent = l1Labels[_l1CamOrder[i]];
  });
}

function setRunButtons(on) {
  const s = document.getElementById('startBtn'), t = document.getElementById('stopBtn');
  if (s) s.classList.toggle('active-start', !!on);
  if (t) t.classList.toggle('active-stop', !on);
}

async function refreshSignalState() {
  try {
    const r = await fetch('/signal'); const s = await r.json();
    _signalEnabled = !!s.enabled;
    const el = document.getElementById('signalToggle'); if (el) el.checked = _signalEnabled;
  } catch(e) {}
}
async function setSignalEnabled(v) {
  _signalEnabled = !!v;
  const el = document.getElementById('signalToggle'); if (el) el.checked = _signalEnabled;
  try { await fetch('/signal?enabled=' + (_signalEnabled ? 1 : 0), { method: 'POST' }); } catch(e) {}
}

async function poll() {
  while (running) {
    try {
      const r = await fetchTimeout('/status', 2000);
      const s = await r.json();
      if (!s.running && !s.maintenance_refreshing) { running = false; setRunButtons(false); break; }
      setRunButtons(true);

      const state = (s.state || 'moving').toLowerCase();
      const stEl = document.getElementById('statusL2');
      stEl.textContent = state === 'stopped' ? 'STOPPED' : 'MOVING';
      stEl.className = 'status-line ' + (state === 'stopped' ? 'stopped' : 'moving');
      document.getElementById('hdrL2').textContent = 'Линия № 2: ' + (state === 'stopped' ? 'STOPPED' : 'MOVING');

      const fmt = v => v == null ? '--' : Number(v).toFixed(1);
      document.getElementById('metricsL2').textContent =
        'inf=' + fmt(s.infer_ms) + 'ms  cap=' + fmt(s.capture_ms) + 'ms  spd=' + fmt(s.speed);
      document.getElementById('inferBreakdown').textContent = s.speed_str || '--';
      const maintMsg = (s.maintenance_msg || '').trim();
      if (maintMsg) document.getElementById('systemMsg').textContent = maintMsg;
      if (s.running) {
        document.getElementById('timerL2').textContent =
          (s.runtime_str || '00:00:00') + ' | Defects: ' + (s.defect_count || 0);
      }

      if (Array.isArray(s.ips_list) && s.ips_list.length === 3) {
        _l2IpsList = s.ips_list;
        updateCamOrderUI();
      }

      if (Array.isArray(s.detected) && s.detected.length === 3) {
        const setDet = (id, v) => {
          const el = document.getElementById(id);
          if (s.infer_busy) { el.textContent = ' '; el.className = 'det-text'; return; }
          const val = (v || '').trim().toLowerCase();
          if (!val || val === 'none' || val === '--') { el.textContent = 'OK'; el.className = 'det-text ok'; }
          else { el.textContent = 'DEFECT: ' + v; el.className = 'det-text bad'; }
        };
        setDet('det0', s.detected[_l2CamOrder[0]]);
        setDet('det1', s.detected[_l2CamOrder[1]]);
        setDet('det2', s.detected[_l2CamOrder[2]]);
      }

      const syncSl = (slId, valId, sv) => {
        if (sv == null) return;
        const e = document.getElementById(slId), v = document.getElementById(valId);
        if (!e || !v) return;
        const n = Math.round(sv * 100);
        if (parseInt(e.value) !== n) { e.value = n; v.textContent = sv.toFixed(2); }
      };
      syncSl('yoloConfNativeSlider', 'yoloConfNativeValue', s.yolo_conf_native);
      syncSl('yoloConf640Slider',    'yoloConf640Value',    s.yolo_conf_640);
      syncSl('iouThreshSlider',      'iouThreshValue',      s.iou_thresh);
      syncSl('clfConfTriggerSlider', 'clfConfTriggerValue', s.clf_conf_trigger);
      syncSl('clfThreshSlider',      'clfThreshValue',      s.clf_thresh);

      if (Array.isArray(s.exposure_us) && s.exposure_us.length >= 3) {
        const ids = ['expCam0','expCam1','expCam2'];
        for (let i = 0; i < 3; i++) {
          const el = document.getElementById(ids[i]);
          if (el && !el.matches(':focus')) el.value = Math.round(s.exposure_us[_l2CamOrder[i]]);
        }
      }

      if (s.has_results && s.result_id !== _lastResultId) {
        _lastResultId = s.result_id;
        const ts = Date.now();
        _loadingResults = true;
        const loadImg = (el, url) => new Promise(res => { el.onload = el.onerror = () => res(); el.src = url; });
        await Promise.all([
          loadImg(document.getElementById('cam0'), '/result/' + _l2CamOrder[0] + '?ts=' + ts),
          loadImg(document.getElementById('cam1'), '/result/' + _l2CamOrder[1] + '?ts=' + ts),
          loadImg(document.getElementById('cam2'), '/result/' + _l2CamOrder[2] + '?ts=' + ts),
        ]);
        _loadingResults = false;
      }

      const nowMs = Date.now();
      if (!_streamBusy && (nowMs - _lastStreamMs) >= STREAM_INTERVAL_MS) {
        _streamBusy = true; _lastStreamMs = nowMs;
        const si = document.getElementById('stream');
        si.onload = si.onerror = () => { _streamBusy = false; };
        si.src = '/stream?ts=' + nowMs;
      }
    } catch(e) {}
    await new Promise(r => setTimeout(r, 100));
  }
}

// LINE 1
function startL1() {
  setRunButtonsL1(true);
  fetch('/l1/start').then(() => { _l1Running = true; pollL1(); }).catch(() => setRunButtonsL1(false));
}
function stopL1() {
  setRunButtonsL1(false);
  fetch('/l1/stop').then(() => { _l1Running = false; }).catch(() => {});
}
function setRunButtonsL1(on) {
  const s = document.getElementById('startBtnL1'), t = document.getElementById('stopBtnL1');
  if (s) s.classList.toggle('active-start', !!on);
  if (t) t.classList.toggle('active-stop', !on);
}

async function refreshSignalStateL1() {
  try {
    const r = await fetch('/l1/signal'); const s = await r.json();
    _signalEnabledL1 = !!s.enabled;
    const el = document.getElementById('signalToggleL1'); if (el) el.checked = _signalEnabledL1;
  } catch(e) {}
}
async function setSignalEnabledL1(v) {
  _signalEnabledL1 = !!v;
  const el = document.getElementById('signalToggleL1'); if (el) el.checked = _signalEnabledL1;
  try { await fetch('/l1/signal?enabled=' + (_signalEnabledL1 ? 1 : 0), { method: 'POST' }); } catch(e) {}
}

let _l1PlaceholderLoaded = false;
async function pollL1() {
  while (_l1Running) {
    try {
      const r = await fetchTimeout('/l1/status', 2000);
      const s = await r.json();
      if (!s.running) {
        _l1Running = false; setRunButtonsL1(false);
        document.getElementById('statusL1').textContent = 'Idle';
        document.getElementById('statusL1').className = 'status-line idle';
        document.getElementById('hdrL1').textContent = 'Линия № 1: Idle';
        break;
      }
      document.getElementById('statusL1').textContent = 'Running';
      document.getElementById('statusL1').className = 'status-line moving';
      document.getElementById('hdrL1').textContent = 'Линия № 1: Running';
      if (s.runtime_str) document.getElementById('timerL1').textContent = s.runtime_str + ' | Defects: 0';

      if (!_l1PlaceholderLoaded) {
        _l1PlaceholderLoaded = true;
        const ts = Date.now();
        const loadImg = (el, url) => new Promise(res => { el.onload = el.onerror = () => res(); el.src = url; });
        await Promise.all([
          loadImg(document.getElementById('cam3'), '/l1/result/0?ts=' + ts),
          loadImg(document.getElementById('cam4'), '/l1/result/1?ts=' + ts),
          loadImg(document.getElementById('cam5'), '/l1/result/2?ts=' + ts),
        ]);
      }
    } catch(e) {}
    await new Promise(r => setTimeout(r, 500));
  }
}

// Controls
let _threshDebounce = null, _displayDebounce = null;
function updateYoloConfNative(v) {
  const n = parseInt(v)/100; document.getElementById('yoloConfNativeValue').textContent = n.toFixed(2);
  if (_threshDebounce) clearTimeout(_threshDebounce);
  _threshDebounce = setTimeout(() => fetch('/thresholds?yolo_conf_native='+n,{method:'POST'}).catch(()=>{}), 100);
}
function updateYoloConf640(v) {
  const n = parseInt(v)/100; document.getElementById('yoloConf640Value').textContent = n.toFixed(2);
  if (_threshDebounce) clearTimeout(_threshDebounce);
  _threshDebounce = setTimeout(() => fetch('/thresholds?yolo_conf_640='+n,{method:'POST'}).catch(()=>{}), 100);
}
function updateIouThresh(v) {
  const n = parseInt(v)/100; document.getElementById('iouThreshValue').textContent = n.toFixed(2);
  if (_threshDebounce) clearTimeout(_threshDebounce);
  _threshDebounce = setTimeout(() => fetch('/thresholds?iou_thresh='+n,{method:'POST'}).catch(()=>{}), 100);
}
function updateClfConfTrigger(v) {
  const n = parseInt(v)/100; document.getElementById('clfConfTriggerValue').textContent = n.toFixed(2);
  if (_threshDebounce) clearTimeout(_threshDebounce);
  _threshDebounce = setTimeout(() => fetch('/classifier?conf_trigger='+n,{method:'POST'}).catch(()=>{}), 100);
}
function updateClfThresh(v) {
  const n = parseInt(v)/100; document.getElementById('clfThreshValue').textContent = n.toFixed(2);
  if (_threshDebounce) clearTimeout(_threshDebounce);
  _threshDebounce = setTimeout(() => fetch('/classifier?thresh='+n,{method:'POST'}).catch(()=>{}), 100);
}
function updateDisplayBrightness(v) {
  const gain = parseInt(v)/100; document.getElementById('displayGainValue').textContent = gain.toFixed(2)+'x';
  if (_displayDebounce) clearTimeout(_displayDebounce);
  _displayDebounce = setTimeout(() => fetch('/display?gain='+gain,{method:'POST'}).catch(()=>{}), 120);
}
function attachExposureHandlers() {
  ['expCam0','expCam1','expCam2'].forEach((id, idx) => {
    const el = document.getElementById(id); if (!el) return;
    el.addEventListener('change', () => {
      const v = parseFloat(el.value); if (!isFinite(v)) return;
      fetch('/exposure?cam_index='+idx+'&exposure_us='+v,{method:'POST'}).catch(()=>{});
    });
  });
}

function openHistory() { window.open('/history','_blank'); }
async function shutdownComputer() {
  if (!confirm('Shutdown computer in 15 seconds?')) return;
  try {
    const r = await fetch('/system/shutdown?confirm=1&delay_sec=15',{method:'POST'});
    const j = await r.json();
    document.getElementById('systemMsg').textContent = j.ok ? 'Shutdown sent (15s).' : 'Shutdown failed.';
  } catch(e) { document.getElementById('systemMsg').textContent = 'Shutdown error.'; }
}
async function cancelShutdown() {
  try {
    const r = await fetch('/system/shutdown-cancel',{method:'POST'});
    const j = await r.json();
    document.getElementById('systemMsg').textContent = j.ok ? 'Shutdown cancelled.' : 'Cancel failed.';
  } catch(e) { document.getElementById('systemMsg').textContent = 'Cancel error.'; }
}

window.addEventListener('load', () => {
  setRunButtons(false); setRunButtonsL1(false);
  refreshSignalState(); refreshSignalStateL1();

  fetch('/thresholds').then(r=>r.json()).then(t => {
    const set = (slId, valId, v) => {
      if (v == null) return;
      const e = document.getElementById(slId), vl = document.getElementById(valId);
      if (e) e.value = Math.round(v*100); if (vl) vl.textContent = v.toFixed(2);
    };
    set('yoloConfNativeSlider','yoloConfNativeValue', t.yolo_conf_native);
    set('yoloConf640Slider',   'yoloConf640Value',   t.yolo_conf_640);
    set('iouThreshSlider',     'iouThreshValue',      t.iou_thresh);
  }).catch(()=>{});

  fetch('/classifier').then(r=>r.json()).then(c => {
    const set = (slId, valId, v) => {
      if (v == null) return;
      const e = document.getElementById(slId), vl = document.getElementById(valId);
      if (e) e.value = Math.round(v*100); if (vl) vl.textContent = v.toFixed(2);
    };
    set('clfConfTriggerSlider','clfConfTriggerValue', c.conf_trigger);
    set('clfThreshSlider',     'clfThreshValue',      c.thresh);
  }).catch(()=>{});

  fetch('/display').then(r=>r.json()).then(d => {
    if (d && d.gain != null) {
      document.getElementById('displayGainSlider').value = Math.round(d.gain*100);
      document.getElementById('displayGainValue').textContent = d.gain.toFixed(2)+'x';
    }
  }).catch(()=>{});

  fetch('/exposure').then(r=>r.json()).then(e => {
    if (Array.isArray(e.exposure_us) && e.exposure_us.length >= 3) {
      ['expCam0','expCam1','expCam2'].forEach((id, i) => {
        const el = document.getElementById(id); if (el) el.value = Math.round(e.exposure_us[i]);
      });
    }
  }).catch(()=>{});

  attachExposureHandlers();
  startL2();
});
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/start")
def start(preserve_session: int = 0):
    global _running, _starting, _error, _model, _device, _cams, _cam_ips, _motion_idx
    global _class_names, _session_start_ts, _session_total_infer, _session_defect_infer
    global _runtime_conf_native, _runtime_conf_640, _runtime_iou_thresh
    global _last_auto_refresh_ts, _run_generation, _infer_executor, _encode_executor
    keep_session = bool(int(preserve_session))
    with _state_lock:
        if _running or _starting:
            return {"ok": True}
        _starting = True
        _error = None
        if not keep_session:
            _session_start_ts = time.time()
            _session_total_infer = 0
            _session_defect_infer = 0
            # Reset defect-frame tracking each session
            global _defect_frame_out_dir, _last_defect_paths
            _defect_frame_out_dir = None
            _last_defect_paths = []
            # Reset first-infer one-shot save marker for a fresh session
            global _first_infer_images_saved
            _first_infer_images_saved = False
            # Initialize runtime thresholds (dual-YOLO merge)
            _runtime_conf_native = CONF
            _runtime_conf_640 = CONF_640_DEFAULT
            _runtime_iou_thresh = IOU_THRESH
            # Initialize classifier runtime settings
            global _runtime_clf_enable, _runtime_clf_thresh, _runtime_clf_conf_trigger
            _runtime_clf_enable = CLF_ENABLE_DEFAULT
            _runtime_clf_thresh = CLF_THRESH_DEFAULT
            _runtime_clf_conf_trigger = CLF_CONF_TRIGGER_DEFAULT
        else:
            if _session_start_ts <= 0:
                _session_start_ts = time.time()

    try:
        cams, ips, motion_idx = _open_cameras()
        if torch.cuda.is_available():
            _device = "cuda:0"
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
        else:
            _device = "cpu"
        if REQUIRE_CUDA and not _device.startswith("cuda"):
            raise RuntimeError("CUDA not available; set HIK_REQUIRE_CUDA=0 to allow CPU.")
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found: {model_path}")
        _model = YOLO(model_path)
        try:
            # Keep model class names available for drawing/UI
            if hasattr(_model, "names") and isinstance(_model.names, dict) and _model.names:
                _class_names = {int(k): str(v) for k, v in _model.names.items()}
        except Exception:
            pass
        try:
            _model.fuse()
        except Exception:
            pass

        # ---- Init batch inference (extract nn.Module, compute stride/imgsz) ----
        try:
            warmup_frames = []
            for _c in cams:
                _wf, _ = _grab_one(_c)
                if _wf is not None:
                    warmup_frames.append(_wf)
            if not warmup_frames:
                sz = IMGSZ if IMGSZ > 0 else WARMUP_IMGSZ
                warmup_frames = [np.zeros((sz, sz), dtype=np.uint8)]
            while len(warmup_frames) < 3:
                warmup_frames.append(warmup_frames[0].copy())

            _init_batch_infer(_model, _device, warmup_frames[0])
            
            # Initialize classifier if enabled
            _init_crop_classifier(_device)

            # Warmup: run 3 batched forward passes to prime CUDA caches & cuDNN
            if WARMUP:
                for _wi in range(3):
                    _batch_predict(warmup_frames, _device)
                if _device.startswith("cuda"):
                    torch.cuda.synchronize()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                print(f"[init] warmup done (3 batched runs)", flush=True)
        except Exception as e:
            import traceback
            print(f"[init] warmup error: {e}", flush=True)
            traceback.print_exc()

        with _state_lock:
            _run_generation += 1
            run_generation = _run_generation
            _cams = cams
            _cam_ips = ips
            _motion_idx = motion_idx
            _running = True
            _starting = False
            _last_auto_refresh_ts = time.time()
            if _infer_executor is None:
                _infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hik-infer")
            if _encode_executor is None:
                _encode_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="hik-encode")

        t = threading.Thread(target=_motion_worker, args=(run_generation,), daemon=True)
        t.start()
        g = threading.Thread(target=_grab_worker, args=(run_generation,), daemon=True)
        g.start()
        s = threading.Thread(target=_stream_worker, args=(run_generation,), daemon=True)
        s.start()
        # Drain threads for non-motion cameras (keep their buffers fresh)
        for _ci in range(len(cams)):
            if _ci == motion_idx:
                continue  # motion cam already drained by _grab_worker
            d = threading.Thread(target=_cam_drain_worker, args=(_ci, run_generation), daemon=True)
            d.start()
        _start_metrics_logger(force_restart=False)
        return {"ok": True, "ips": ips, "motion_idx": motion_idx}
    except Exception as e:
        with _state_lock:
            _starting = False
            _running = False
            _error = str(e)
        _close_cameras()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stop")
def stop():
    _stop_runtime(clear_maintenance=True)
    _stop_metrics_logger()
    return {"ok": True}


def _stop_runtime(clear_maintenance: bool = True) -> None:
    global _running, _last_auto_refresh_ts, _maintenance_refreshing, _maintenance_msg
    global _hard_relaunch_in_progress, _run_generation
    with _state_lock:
        _running = False
        _run_generation += 1
        _last_auto_refresh_ts = 0.0
        if clear_maintenance:
            _maintenance_refreshing = False
            _maintenance_msg = ""
            _hard_relaunch_in_progress = False
    _close_cameras()
    _reset_runtime_resources()


def _build_hard_relaunch_cmd() -> list[str]:
    if HARD_RELAUNCH_CMD:
        return shlex.split(HARD_RELAUNCH_CMD, posix=(platform.system() != "Windows"))
    if len(sys.argv) >= 2:
        stem = Path(sys.argv[0]).stem.lower()
        if "uvicorn" in stem:
            return [sys.executable, "-m", "uvicorn", *sys.argv[1:]]
        return [sys.executable, *sys.argv]
    return []


def _spawn_detached_process(cmd: list[str], startup_delay: float = 3.0) -> None:
    cwd = os.getcwd()
    if platform.system() == "Windows":
        bat_path = os.path.join(cwd, "_relaunch.bat")
        cmd_str = subprocess.list2cmdline(cmd)
        # ping -n N waits ~(N-1) seconds — reliable delay on Windows
        ping_n = max(2, int(startup_delay) + 1)
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write("@echo off\n")
            f.write(f"echo [relaunch] waiting {startup_delay}s for port release...\n")
            f.write(f"ping -n {ping_n} 127.0.0.1 >nul\n")
            f.write(f"echo [relaunch] starting uvicorn...\n")
            f.write(f"{cmd_str}\n")
        subprocess.Popen(
            ["cmd", "/c", bat_path],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=cwd,
        )
    else:
        shell_script = f"sleep {int(startup_delay)} && {subprocess.list2cmdline(cmd)}"
        subprocess.Popen(
            ["bash", "-c", shell_script],
            start_new_session=True,
            close_fds=True,
            cwd=cwd,
        )


def _request_hard_relaunch(reason: str, delay_sec: float = 1.0) -> bool:
    global _maintenance_refreshing, _maintenance_msg, _hard_relaunch_in_progress
    if not HARD_RELAUNCH_ENABLED:
        return False
    with _state_lock:
        if _hard_relaunch_in_progress or not _running:
            return False
        _hard_relaunch_in_progress = True
        _maintenance_refreshing = True
        _maintenance_msg = "Выполняется полный перезапуск приложения... Пожалуйста, подождите."

    def _worker():
        global _maintenance_refreshing, _maintenance_msg, _hard_relaunch_in_progress
        try:
            _stop_runtime(clear_maintenance=False)
            time.sleep(max(0.2, float(delay_sec)))
            cmd = _build_hard_relaunch_cmd()
            if not cmd:
                raise RuntimeError("cannot build relaunch command")
            print(f"[relaunch] spawning fresh process, reason={reason}, cmd={' '.join(cmd)}", flush=True)
            _spawn_detached_process(cmd)
            os._exit(0)
        except Exception as e:
            print(f"[relaunch] failed: {e} -> falling back to soft refresh with retries", flush=True)
            # Fall back: try to restart in-process so we don't stay dead.
            last_err = None
            for attempt in range(1, _AUTO_REFRESH_MAX_RETRIES + 1):
                try:
                    start(preserve_session=1)
                    print(f"[relaunch] fallback soft refresh succeeded (attempt {attempt})", flush=True)
                    last_err = None
                    break
                except Exception as retry_err:
                    last_err = retry_err
                    print(
                        f"[relaunch] fallback attempt {attempt}/{_AUTO_REFRESH_MAX_RETRIES} failed: {retry_err}",
                        flush=True,
                    )
                    if attempt < _AUTO_REFRESH_MAX_RETRIES:
                        time.sleep(_AUTO_REFRESH_RETRY_DELAY_S)
            if last_err is not None:
                print(f"[relaunch] all fallback attempts failed, system stopped", flush=True)
            with _state_lock:
                _hard_relaunch_in_progress = False
                _maintenance_refreshing = False
                _maintenance_msg = ""

    threading.Thread(target=_worker, daemon=True).start()
    return True


_AUTO_REFRESH_MAX_RETRIES = 5
_AUTO_REFRESH_RETRY_DELAY_S = 10.0


def _auto_refresh_worker():
    """Periodic maintenance refresh that preserves current tracking session."""
    global _maintenance_refreshing, _maintenance_msg, _last_auto_refresh_ts
    while True:
        time.sleep(1.0)
        if not AUTO_REFRESH_ENABLED:
            continue
        try:
            with _state_lock:
                running = _running
                starting = _starting
                session_start = _session_start_ts
                last_refresh = _last_auto_refresh_ts
            if not running or starting or session_start <= 0:
                continue
            now = time.time()
            baseline = last_refresh if last_refresh > 0 else session_start
            if (now - baseline) < AUTO_REFRESH_INTERVAL_S:
                continue

            with _state_lock:
                _last_auto_refresh_ts = now
                _maintenance_msg = "Идет автообновление системы... Пожалуйста, подождите."
            print(
                f"[maint] scheduled cleanup triggered ({AUTO_REFRESH_MINUTES:.2f} min interval) -> cleanup only (no app restart)",
                flush=True,
            )
            _cleanup_memory(reason="scheduled_auto_refresh")
            with _state_lock:
                _maintenance_msg = ""
            continue

            # Try hard relaunch first (spawns fresh process, truly frees RAM).
            # On success os._exit(0) kills us instantly; on failure the
            # relaunch worker's own fallback does soft refresh with retries.
            if AUTO_REFRESH_HARD_RELAUNCH:
                if _request_hard_relaunch("scheduled_auto_refresh", delay_sec=0.5):
                    continue

            # Soft refresh path with retries so transient camera/model errors
            # don't leave the system permanently stopped.
            _stop_runtime(clear_maintenance=False)
            time.sleep(1.0)

            last_err = None
            for attempt in range(1, _AUTO_REFRESH_MAX_RETRIES + 1):
                try:
                    start(preserve_session=1)
                    print(f"[maint] soft refresh succeeded (attempt {attempt})", flush=True)
                    last_err = None
                    break
                except Exception as retry_err:
                    last_err = retry_err
                    print(
                        f"[maint] soft refresh attempt {attempt}/{_AUTO_REFRESH_MAX_RETRIES} failed: {retry_err}",
                        flush=True,
                    )
                    if attempt < _AUTO_REFRESH_MAX_RETRIES:
                        time.sleep(_AUTO_REFRESH_RETRY_DELAY_S)

            if last_err is not None:
                print(
                    f"[maint] all {_AUTO_REFRESH_MAX_RETRIES} soft refresh attempts failed, "
                    f"will retry next cycle in {AUTO_REFRESH_INTERVAL_S:.0f}s",
                    flush=True,
                )

            with _state_lock:
                _maintenance_refreshing = False
                _maintenance_msg = ""
        except Exception as e:
            with _state_lock:
                _maintenance_refreshing = False
                _maintenance_msg = ""
            print(f"[maint] auto-refresh skipped due to error: {e}", flush=True)


def _freeze_watchdog_worker():
    """Best-effort watchdog without restarting the whole application."""
    global _last_watchdog_recover_ts
    while True:
        time.sleep(FREEZE_WATCHDOG_CHECK_S)
        if not FREEZE_WATCHDOG_ENABLED:
            continue
        try:
            now_perf = time.perf_counter()
            now_wall = time.time()
            with _state_lock:
                running = _running
                starting = _starting
                in_maint = _maintenance_refreshing
                last_started_ts = _last_auto_refresh_ts
                latest_ts = list(_latest_cam_ts[:3])
                last_recover_ts = _last_watchdog_recover_ts
                infer_thread_alive = _is_infer_busy()
                infer_started_ts = _infer_started_ts
                cams_local = list(_cams)
                ips_local = list(_cam_ips)
            if not running or starting or in_maint:
                continue
            if last_started_ts > 0 and (now_wall - last_started_ts) < FREEZE_WATCHDOG_GRACE_S:
                continue
            if (now_wall - last_recover_ts) < FREEZE_WATCHDOG_RECOVER_COOLDOWN_S:
                continue

            ages_ms: list[float] = []
            for ts in latest_ts:
                if ts <= 0:
                    ages_ms.append(99999.0)
                else:
                    ages_ms.append((now_perf - ts) * 1000.0)
            stale = any(age > FREEZE_WATCHDOG_STALE_MS for age in ages_ms)
            infer_hung = (
                infer_thread_alive
                and infer_started_ts > 0
                and (now_wall - infer_started_ts) > FREEZE_WATCHDOG_INFER_HUNG_S
            )
            if not stale and not infer_hung:
                continue

            with _state_lock:
                _maintenance_msg = "Обнаружено зависание камеры... Выполняется авто-восстановление."
                _last_watchdog_recover_ts = now_wall
            ages_str = "  ".join(f"c{i}={ages_ms[i]:.0f}ms" for i in range(len(ages_ms)))
            if infer_hung:
                print(
                    f"[watchdog] inference appears hung "
                    f"(alive_for={now_wall - infer_started_ts:.1f}s, limit={FREEZE_WATCHDOG_INFER_HUNG_S:.1f}s) "
                    f"-> cleanup only (no app restart)",
                    flush=True,
                )
            else:
                print(
                    f"[watchdog] stale camera timestamps detected ({ages_str}) "
                    f"-> restart grabbing only (no app restart)",
                    flush=True,
                )
            if infer_hung:
                _cleanup_memory(reason="watchdog_infer_hung")
            else:
                for i, cam in enumerate(cams_local):
                    try:
                        cam_ip = ips_local[i] if i < len(ips_local) else f"cam{i}"
                    except Exception:
                        cam_ip = f"cam{i}"
                    _try_restart_grabbing(cam, cam_ip)
                _cleanup_memory(reason="watchdog_stale_camera")
        except Exception as e:
            print(f"[watchdog] recovery skipped due to error: {e}", flush=True)


@app.post("/system/shutdown")
def system_shutdown(confirm: int = 0, delay_sec: int = 15):
    if int(confirm) != 1:
        raise HTTPException(status_code=400, detail="confirmation required")
    delay = max(0, min(int(delay_sec), 3600))
    try:
        if platform.system() == "Windows":
            subprocess.run(["shutdown", "/s", "/t", str(delay), "/f"], check=True)
        elif platform.system() == "Linux":
            minutes = max(1, (delay + 59) // 60)
            subprocess.run(["shutdown", "-h", f"+{minutes}"], check=True)
        elif platform.system() == "Darwin":
            minutes = max(1, (delay + 59) // 60)
            subprocess.run(["shutdown", "-h", f"+{minutes}"], check=True)
        else:
            raise RuntimeError("unsupported OS")
        return {"ok": True, "delay_sec": delay}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"shutdown failed: {e}")


@app.post("/system/shutdown-cancel")
def system_shutdown_cancel():
    try:
        if platform.system() == "Windows":
            subprocess.run(["shutdown", "/a"], check=True)
        elif platform.system() in ("Linux", "Darwin"):
            subprocess.run(["shutdown", "-c"], check=True)
        else:
            raise RuntimeError("unsupported OS")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cancel failed: {e}")


@app.post("/system/relaunch")
def system_relaunch(confirm: int = 0, delay_sec: float = 1.0):
    if int(confirm) != 1:
        raise HTTPException(status_code=400, detail="confirmation required")
    if not HARD_RELAUNCH_ENABLED:
        raise HTTPException(status_code=400, detail="hard relaunch disabled")
    ok = _request_hard_relaunch("api_relaunch", delay_sec=max(0.2, float(delay_sec)))
    if not ok:
        return {"ok": False, "detail": "relaunch already in progress or app not running"}
    return {"ok": True, "mode": "hard", "delay_sec": max(0.2, float(delay_sec))}


@app.post("/system/cleanup-memory")
def system_cleanup_memory():
    info = _cleanup_memory(reason="api_manual")
    return {"ok": True, **info}


@app.get("/signal")
def get_signal():
    with _state_lock:
        enabled = _line_signal_ui_enabled
    return {"enabled": enabled}


@app.post("/signal")
def set_signal(enabled: int):
    global _line_signal_ui_enabled
    val = bool(int(enabled))
    with _state_lock:
        _line_signal_ui_enabled = val
    return {"enabled": _line_signal_ui_enabled}


@app.get("/exposure")
def get_exposure():
    """Return current per-camera exposure times in microseconds."""
    with _state_lock:
        return {
            "exposure_us": list(_exposure_us_per_cam),
            "ips": list(_cam_ips),
        }


@app.post("/exposure")
def set_exposure(cam_index: int, exposure_us: float):
    """Set exposure time (µs) for a given camera index (0..2)."""
    global _exposure_us_per_cam
    if cam_index < 0 or cam_index >= 3:
        raise HTTPException(status_code=400, detail="bad cam index")
    exp = max(50.0, float(exposure_us))  # basic clamp
    with _state_lock:
        if cam_index >= len(_cams):
            raise HTTPException(status_code=400, detail="camera not ready")
        cam = _cams[cam_index]
        try:
            cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            cam.MV_CC_SetFloatValue("ExposureTime", exp)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to set exposure: {e}")
        # Ensure list has 3 entries
        if len(_exposure_us_per_cam) < 3:
            _exposure_us_per_cam = list(_exposure_us_per_cam) + [
                float(EXPOSURE_US)
            ] * (3 - len(_exposure_us_per_cam))
        _exposure_us_per_cam[cam_index] = exp
    return {"cam_index": cam_index, "exposure_us": exp}


@app.get("/display")
def get_display():
    """Get current display-only brightness settings."""
    return {
        "gain": float(globals().get("DISPLAY_GAIN_RUNTIME", DISPLAY_GAIN)),
        "bias": float(globals().get("DISPLAY_BIAS_RUNTIME", DISPLAY_BIAS)),
        "gamma": float(globals().get("DISPLAY_GAMMA_RUNTIME", DISPLAY_GAMMA)),
    }


@app.post("/display")
def set_display(
    gain: Optional[float] = None,
    bias: Optional[float] = None,
    gamma: Optional[float] = None,
):
    """Adjust display-only brightness (does not affect inference)."""
    global DISPLAY_GAIN_RUNTIME, DISPLAY_BIAS_RUNTIME, DISPLAY_GAMMA_RUNTIME
    if gain is not None:
        DISPLAY_GAIN_RUNTIME = max(0.1, min(5.0, float(gain)))
    if bias is not None:
        DISPLAY_BIAS_RUNTIME = max(-255.0, min(255.0, float(bias)))
    if gamma is not None:
        DISPLAY_GAMMA_RUNTIME = max(0.2, min(5.0, float(gamma)))
    return {
        "gain": DISPLAY_GAIN_RUNTIME,
        "bias": DISPLAY_BIAS_RUNTIME,
        "gamma": DISPLAY_GAMMA_RUNTIME,
    }


@app.get("/thresholds")
def get_thresholds():
    with _state_lock:
        return {
            "yolo_conf_native": _runtime_conf_native,
            "yolo_conf_640": _runtime_conf_640,
            "iou_thresh": _runtime_iou_thresh,
            "clf_enabled": _runtime_clf_enable,
            "clf_conf_trigger": _runtime_clf_conf_trigger,
            "clf_thresh": _runtime_clf_thresh,
        }


@app.post("/thresholds")
def set_thresholds(
    yolo_conf_native: Optional[float] = None,
    yolo_conf_640: Optional[float] = None,
    iou_thresh: Optional[float] = None,
):
    global _runtime_conf_native, _runtime_conf_640, _runtime_iou_thresh
    with _state_lock:
        if SYNC_YOLO_CONF:
            # Offline script uses one shared YOLO conf for native and 640.
            # Keep backend aligned by forcing both to the same runtime value.
            shared_conf = None
            if yolo_conf_native is not None:
                shared_conf = float(yolo_conf_native)
            elif yolo_conf_640 is not None:
                shared_conf = float(yolo_conf_640)
            if shared_conf is not None:
                shared_conf = max(0.0, min(1.0, shared_conf))
                _runtime_conf_native = shared_conf
                _runtime_conf_640 = shared_conf
        else:
            if yolo_conf_native is not None:
                _runtime_conf_native = max(0.0, min(1.0, float(yolo_conf_native)))
            if yolo_conf_640 is not None:
                _runtime_conf_640 = max(0.0, min(1.0, float(yolo_conf_640)))
        if iou_thresh is not None:
            _runtime_iou_thresh = max(0.0, min(1.0, float(iou_thresh)))
    return {
        "yolo_conf_native": _runtime_conf_native,
        "yolo_conf_640": _runtime_conf_640,
        "iou_thresh": _runtime_iou_thresh,
    }


@app.get("/status")
def status():
    with _state_lock:
        ips_list = []
        for ip in _cam_ips:
            try:
                ips_list.append(ip.split(".")[-1])
            except Exception:
                ips_list.append(ip)
        ready_text = ""
        if len(ips_list) == 3:
            ready_text = f"Cam {ips_list[0]} ready | Cam {ips_list[1]} ready | Cam {ips_list[2]} ready"
        infer_busy = _is_infer_busy()
        
        # Calculate runtime timer and defect count
        runtime_seconds = 0
        runtime_str = "00:00:00"
        if _session_start_ts > 0 and _running:
            runtime_seconds = int(time.time() - _session_start_ts)
            hrs = runtime_seconds // 3600
            mins = (runtime_seconds % 3600) // 60
            secs = runtime_seconds % 60
            runtime_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        
        # Defect frame output directory and latest saved paths
        defect_dir_str = ""
        if _defect_frame_out_dir is not None:
            try:
                defect_dir_str = str(_defect_frame_out_dir)
            except Exception:
                defect_dir_str = ""

        return {
            "running": _running,
            "starting": _starting,
            "maintenance_refreshing": _maintenance_refreshing,
            "maintenance_msg": _maintenance_msg,
            "hard_relaunch_in_progress": _hard_relaunch_in_progress,
            "last_mem_cleanup_ts": _last_mem_cleanup_ts,
            "last_mem_cleanup_reason": _last_mem_cleanup_reason,
            "error": _error,
            "state": _last_state,
            "speed": _last_speed,
            "capture_ms": _last_capture_ms,
            "preprocess_ms": _last_preprocess_ms,
            "build_ms": _last_build_ms,
            "infer_ms": _last_infer_ms,
            "post_yolo_ms": _last_post_yolo_ms,
            "postprocess_ms": _last_postprocess_ms,
            "encode_ms": _last_encode_ms,
            "total_ms": _last_total_ms,
            "speed_str": _last_speed_str,
            "detected": list(_last_detected),
            "infer_busy": infer_busy,
            "has_results": any(r is not None and r for r in _last_results),
            "result_id": _last_result_id,
            "result_ready_ts": _last_result_ready_ts,
            "server_ts": time.time(),
            "ips": ", ".join(_cam_ips),
            "ips_list": ips_list,
            "motion_idx": _motion_idx,
            "device": _device,
            "ready_text": ready_text,
            "runtime_str": runtime_str,
            "defect_count": _session_defect_infer,
            "yolo_conf_native": _runtime_conf_native,
            "yolo_conf_640": _runtime_conf_640,
            "iou_thresh": _runtime_iou_thresh,
            "clf_enabled": _runtime_clf_enable,
            "clf_conf_trigger": _runtime_clf_conf_trigger,
            "clf_thresh": _runtime_clf_thresh,
            "classes": {0: _class_names.get(0, "beads-package")} if ONLY_CLASS0 else dict(_class_names),
            "defect_frame_dir": defect_dir_str,
            "last_defect_paths": list(_last_defect_paths),
            "exposure_us": list(_exposure_us_per_cam),
            "metrics_log_path": str(METRICS_LOG_PATH),
            "metrics_event_log_path": str(METRICS_EVENT_LOG_PATH),
            "metrics_hw_info_path": str(METRICS_HW_INFO_PATH),
            "metrics_active": bool(_metrics_thread is not None and _metrics_thread.is_alive()),
            "history_index_db": str(_history_db_path),
        }


@app.get("/classifier")
def get_classifier():
    with _state_lock:
        return {
            "enabled": _runtime_clf_enable,
            "conf_trigger": _runtime_clf_conf_trigger,
            "thresh": _runtime_clf_thresh,
        }


@app.post("/classifier")
def set_classifier(
    enabled: Optional[int] = None,
    conf_trigger: Optional[float] = None,
    thresh: Optional[float] = None,
):
    global _runtime_clf_enable, _runtime_clf_conf_trigger, _runtime_clf_thresh, _clf_model
    reinit_device: Optional[str] = None
    with _state_lock:
        if enabled is not None:
            _runtime_clf_enable = bool(enabled)
            # Reinitialize classifier if enabled changed
            if _device:
                reinit_device = _device
        if conf_trigger is not None:
            _runtime_clf_conf_trigger = max(0.0, min(1.0, float(conf_trigger)))
        if thresh is not None:
            _runtime_clf_thresh = max(0.0, min(1.0, float(thresh)))
    if reinit_device:
        _init_crop_classifier(reinit_device)
    return {
        "enabled": _runtime_clf_enable,
        "conf_trigger": _runtime_clf_conf_trigger,
        "thresh": _runtime_clf_thresh,
    }


_no_cache_headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}

# ---- Line 1 placeholder JPEG (generated once on first request) ----
_PLACEHOLDER_JPEG: Optional[bytes] = None

def _get_placeholder_jpeg() -> bytes:
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is None:
        img = np.zeros((360, 480, 3), dtype=np.uint8)
        img[:] = (25, 28, 35)
        cv2.putText(img, "Not Connected", (60, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (55, 75, 105), 2)
        cv2.putText(img, "LINE 1", (175, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 60, 90), 1)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        _PLACEHOLDER_JPEG = bytes(buf) if ok else b""
    return _PLACEHOLDER_JPEG


# ---- Line 1 routes (placeholder — cameras not yet connected) ----

@app.get("/l1/start")
def l1_start():
    global _l1_running, _l1_start_ts
    with _l1_lock:
        if not _l1_running:
            _l1_running = True
            _l1_start_ts = time.time()
    return {"ok": True, "running": _l1_running}


@app.get("/l1/stop")
def l1_stop():
    global _l1_running
    with _l1_lock:
        _l1_running = False
    return {"ok": True, "running": False}


@app.get("/l1/status")
def l1_status():
    with _l1_lock:
        runtime_str = "00:00:00"
        if _l1_running and _l1_start_ts > 0:
            s = int(time.time() - _l1_start_ts)
            runtime_str = f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        return {
            "running": _l1_running,
            "state": "idle",
            "runtime_str": runtime_str if _l1_running else "00:00:00",
            "signal_enabled": _l1_signal_ui_enabled,
            "cam_ips": LINE1_IPS,
        }


@app.get("/l1/signal")
def l1_get_signal():
    with _l1_lock:
        return {"enabled": _l1_signal_ui_enabled}


@app.post("/l1/signal")
def l1_set_signal(enabled: int = 1):
    global _l1_signal_ui_enabled
    with _l1_lock:
        _l1_signal_ui_enabled = bool(enabled)
    return {"enabled": _l1_signal_ui_enabled}


@app.get("/l1/result/{i}")
def l1_result(i: int):
    if i < 0 or i > 2:
        raise HTTPException(status_code=404, detail="bad index")
    return Response(_get_placeholder_jpeg(), media_type="image/jpeg", headers=_no_cache_headers)


@app.get("/l1/stream")
def l1_stream():
    return Response(_get_placeholder_jpeg(), media_type="image/jpeg", headers=_no_cache_headers)


def _resolve_annotated_root() -> Optional[Path]:
    try:
        root = (
            DEFECT_FRAME_BOX_OUT_DIR
            if DEFECT_FRAME_BOX_OUT_DIR.is_absolute()
            else (Path(__file__).resolve().parent / DEFECT_FRAME_BOX_OUT_DIR).resolve()
        )
        return root
    except Exception:
        return None


def _parse_hhmm_to_minutes(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    parts = txt.split(":")
    if len(parts) != 2:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
    except Exception:
        return None
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return None
    return hh * 60 + mm


def _within_time_interval(minute_of_day: int, start_min: Optional[int], end_min: Optional[int]) -> bool:
    if start_min is None and end_min is None:
        return True
    if start_min is None:
        return minute_of_day <= int(end_min)
    if end_min is None:
        return minute_of_day >= int(start_min)
    if start_min <= end_min:
        return start_min <= minute_of_day <= end_min
    # Interval across midnight, e.g. 23:00 -> 02:00
    return minute_of_day >= start_min or minute_of_day <= end_min


def _history_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_history_db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS defect_history (
            filename TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            day TEXT NOT NULL,
            minute_of_day INTEGER NOT NULL,
            shift_key TEXT NOT NULL,
            shift_label TEXT NOT NULL,
            mtime_ns INTEGER NOT NULL
        )
        """
    )
    # Migration: add line column if missing; all existing records are from Line 2
    existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(defect_history)").fetchall()]
    if "line" not in existing_cols:
        conn.execute("ALTER TABLE defect_history ADD COLUMN line INTEGER NOT NULL DEFAULT 2")
        conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_defect_history_ts ON defect_history(ts_ms DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_defect_history_day ON defect_history(day)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_defect_history_shift ON defect_history(shift_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_defect_history_line ON defect_history(line)")
    return conn


def _sync_history_index(root: Path, force: bool = False) -> None:
    global _history_last_sync_ts
    now = time.time()
    if not force and (now - _history_last_sync_ts) < _history_sync_interval_s:
        return
    if not root.exists() or not root.is_dir():
        return
    with _history_sync_lock:
        now = time.time()
        if not force and (now - _history_last_sync_ts) < _history_sync_interval_s:
            return
        conn = _history_connect()
        try:
            known = dict(conn.execute("SELECT filename, mtime_ns FROM defect_history"))
            seen: set[str] = set()
            upserts: list[tuple] = []
            for p in root.glob("*_annotated.jpg"):
                try:
                    st = p.stat()
                    mtime_ns = int(st.st_mtime_ns)
                except Exception:
                    continue
                name = p.name
                seen.add(name)
                if known.get(name) == mtime_ns:
                    continue
                try:
                    ts_ms = int(name.split("__", 1)[0])
                except Exception:
                    ts_ms = int(st.st_mtime * 1000)
                dt = datetime.fromtimestamp(ts_ms / 1000.0)
                day = dt.strftime("%Y-%m-%d")
                minute_of_day = dt.hour * 60 + dt.minute
                is_day_shift = (8 * 60) <= minute_of_day < (19 * 60 + 30)
                shift_key = "day" if is_day_shift else "night"
                shift_label = "Дневная смена" if is_day_shift else "Ночная смена"
                upserts.append((name, ts_ms, day, minute_of_day, shift_key, shift_label, mtime_ns, 2))
            if upserts:
                conn.executemany(
                    """
                    INSERT INTO defect_history(filename, ts_ms, day, minute_of_day, shift_key, shift_label, mtime_ns, line)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(filename) DO UPDATE SET
                        ts_ms=excluded.ts_ms,
                        day=excluded.day,
                        minute_of_day=excluded.minute_of_day,
                        shift_key=excluded.shift_key,
                        shift_label=excluded.shift_label,
                        mtime_ns=excluded.mtime_ns,
                        line=excluded.line
                    """,
                    upserts,
                )
            stale = [(k,) for k in known.keys() if k not in seen]
            if stale:
                conn.executemany("DELETE FROM defect_history WHERE filename = ?", stale)
            conn.commit()
            _history_last_sync_ts = now
        finally:
            conn.close()


def _history_where_clause(
    day_from: Optional[str],
    day_to: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    shift: Optional[str],
    line: Optional[str] = None,
) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if day_from:
        clauses.append("day >= ?")
        params.append(day_from)
    if day_to:
        clauses.append("day <= ?")
        params.append(day_to)
    shift_norm = (shift or "").strip().lower()
    if shift_norm in ("day", "night"):
        clauses.append("shift_key = ?")
        params.append(shift_norm)
    try:
        line_int = int(line) if line and line.strip() else None
    except (ValueError, TypeError):
        line_int = None
    if line_int is not None:
        clauses.append("line = ?")
        params.append(line_int)
    start_min = _parse_hhmm_to_minutes(start_time)
    end_min = _parse_hhmm_to_minutes(end_time)
    if start_min is not None and end_min is not None:
        if start_min <= end_min:
            clauses.append("minute_of_day BETWEEN ? AND ?")
            params.extend([start_min, end_min])
        else:
            clauses.append("(minute_of_day >= ? OR minute_of_day <= ?)")
            params.extend([start_min, end_min])
    elif start_min is not None:
        clauses.append("minute_of_day >= ?")
        params.append(start_min)
    elif end_min is not None:
        clauses.append("minute_of_day <= ?")
        params.append(end_min)
    where = ""
    if clauses:
        where = " WHERE " + " AND ".join(clauses)
    return where, params


def _fetch_history_page(
    root: Path,
    *,
    day_from: Optional[str],
    day_to: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    shift: Optional[str],
    line: Optional[str] = None,
    page: int,
    page_size: int,
) -> dict:
    _sync_history_index(root)
    where, params = _history_where_clause(day_from, day_to, start_time, end_time, shift, line)
    conn = _history_connect()
    try:
        total_items = int(conn.execute(f"SELECT COUNT(*) FROM defect_history{where}", params).fetchone()[0])
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        safe_page = max(1, min(int(page), total_pages))
        offset = (safe_page - 1) * page_size
        rows = conn.execute(
            f"""
            SELECT filename, ts_ms, day, shift_label, shift_key, line
            FROM defect_history
            {where}
            ORDER BY ts_ms DESC
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset],
        ).fetchall()
        stats = conn.execute(
            f"""
            SELECT
              SUM(CASE WHEN shift_key='day' THEN 1 ELSE 0 END) AS day_count,
              SUM(CASE WHEN shift_key='night' THEN 1 ELSE 0 END) AS night_count
            FROM defect_history
            {where}
            """,
            params,
        ).fetchone()
    finally:
        conn.close()
    items = []
    for filename, ts_ms, day, shift_label, shift_key, line_num in rows:
        dt = datetime.fromtimestamp(int(ts_ms) / 1000.0)
        items.append(
            {
                "filename": filename,
                "ts_ms": int(ts_ms),
                "date": day,
                "time": dt.strftime("%H:%M:%S"),
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "ru_datetime": dt.strftime("%d.%m.%Y %H:%M:%S"),
                "shift": shift_label,
                "shift_key": shift_key,
                "line": int(line_num) if line_num is not None else 2,
            }
        )
    day_count = int((stats[0] or 0) if stats else 0)
    night_count = int((stats[1] or 0) if stats else 0)
    return {
        "items": items,
        "total_items": total_items,
        "total_pages": total_pages,
        "page": safe_page,
        "day_count": day_count,
        "night_count": night_count,
    }


def _log_runtime_event(level: str, event: str, **payload) -> None:
    try:
        METRICS_EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts_iso": datetime.now().isoformat(timespec="seconds"),
            "epoch_s": round(time.time(), 3),
            "level": str(level).lower(),
            "event": str(event),
            "payload": payload or {},
        }
        with METRICS_EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return


def _parse_num(v: str) -> Optional[float]:
    txt = str(v).strip()
    if not txt or txt.upper() == "N/A":
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _query_nvidia_smi() -> dict:
    global _nvidia_smi_adv_available, _nvidia_smi_basic_warned
    out = {
        "gpu_util_pct": None,
        "gpu_mem_util_pct": None,
        "gpu_mem_used_mib": None,
        "gpu_mem_total_mib": None,
        "gpu_temp_c": None,
        "gpu_power_w": None,
        "gpu_power_limit_w": None,
        "gpu_sm_clock_mhz": None,
        "gpu_mem_clock_mhz": None,
        "gpu_fan_pct": None,
        "gpu_pstate": None,
        "gpu_pcie_gen": None,
        "gpu_pcie_width": None,
        "gpu_throttle_power_cap": None,
        "gpu_throttle_hw_thermal": None,
        "gpu_throttle_sw_thermal": None,
        "gpu_throttle_hw_slowdown": None,
    }
    fields_adv = (
        "utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,"
        "power.limit,clocks.sm,clocks.mem,fan.speed,pstate,pcie.link.gen.current,pcie.link.width.current,"
        "clocks_throttle_reasons.power_cap,clocks_throttle_reasons.hw_thermal_slowdown,"
        "clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_slowdown"
    )
    if _nvidia_smi_adv_available is not False:
        try:
            cmd = ["nvidia-smi", f"--query-gpu={fields_adv}", "--format=csv,noheader,nounits"]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
            if cp.returncode != 0:
                raise RuntimeError((cp.stderr or "").strip() or "nvidia-smi failed")
            line = (cp.stdout or "").strip().splitlines()
            if not line:
                return out
            parts = [p.strip() for p in line[0].split(",")]
            if len(parts) >= 17:
                _nvidia_smi_adv_available = True
                out["gpu_util_pct"] = _parse_num(parts[0])
                out["gpu_mem_util_pct"] = _parse_num(parts[1])
                out["gpu_mem_used_mib"] = _parse_num(parts[2])
                out["gpu_mem_total_mib"] = _parse_num(parts[3])
                out["gpu_temp_c"] = _parse_num(parts[4])
                out["gpu_power_w"] = _parse_num(parts[5])
                out["gpu_power_limit_w"] = _parse_num(parts[6])
                out["gpu_sm_clock_mhz"] = _parse_num(parts[7])
                out["gpu_mem_clock_mhz"] = _parse_num(parts[8])
                out["gpu_fan_pct"] = _parse_num(parts[9])
                out["gpu_pstate"] = parts[10] if parts[10] and parts[10] != "N/A" else None
                out["gpu_pcie_gen"] = _parse_num(parts[11])
                out["gpu_pcie_width"] = _parse_num(parts[12])
                out["gpu_throttle_power_cap"] = parts[13] if parts[13] and parts[13] != "N/A" else None
                out["gpu_throttle_hw_thermal"] = parts[14] if parts[14] and parts[14] != "N/A" else None
                out["gpu_throttle_sw_thermal"] = parts[15] if parts[15] and parts[15] != "N/A" else None
                out["gpu_throttle_hw_slowdown"] = parts[16] if parts[16] and parts[16] != "N/A" else None
                return out
            raise RuntimeError(f"unexpected advanced nvidia-smi columns: {len(parts)}")
        except Exception as e:
            if _nvidia_smi_adv_available is None:
                _log_runtime_event("warn", "nvidia_smi_advanced_failed", error=str(e))
            _nvidia_smi_adv_available = False
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if cp.returncode != 0:
            return out
        line = (cp.stdout or "").strip().splitlines()
        if not line:
            return out
        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) >= 7:
            out["gpu_util_pct"] = _parse_num(parts[0])
            out["gpu_mem_used_mib"] = _parse_num(parts[1])
            out["gpu_mem_total_mib"] = _parse_num(parts[2])
            out["gpu_temp_c"] = _parse_num(parts[3])
            out["gpu_power_w"] = _parse_num(parts[4])
            out["gpu_sm_clock_mhz"] = _parse_num(parts[5])
            out["gpu_mem_clock_mhz"] = _parse_num(parts[6])
    except Exception as e:
        if not _nvidia_smi_basic_warned:
            _log_runtime_event("warn", "nvidia_smi_basic_failed", error=str(e))
            _nvidia_smi_basic_warned = True
    return out


def _write_hardware_info() -> None:
    try:
        info = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": sys.version,
            },
            "torch": {
                "version": getattr(torch, "__version__", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda": getattr(torch.version, "cuda", None),
                "cudnn_available": bool(torch.backends.cudnn.is_available()),
                "cudnn_version": torch.backends.cudnn.version(),
            },
            "metrics": {
                "log_path": str(METRICS_LOG_PATH),
                "event_log_path": str(METRICS_EVENT_LOG_PATH),
            },
        }
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu_torch"] = {
                    "name": props.name,
                    "total_memory_mib": round(props.total_memory / (1024 * 1024), 2),
                    "multi_processor_count": int(props.multi_processor_count),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            except Exception:
                pass
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                info["host_memory"] = {
                    "total_gib": round(float(vm.total) / (1024 ** 3), 2),
                }
                info["cpu"] = {
                    "logical_count": int(psutil.cpu_count(logical=True) or 0),
                    "physical_count": int(psutil.cpu_count(logical=False) or 0),
                }
            except Exception:
                pass
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,driver_version,vbios_version,pstate,pci.bus_id",
                "--format=csv,noheader",
            ]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
            if cp.returncode == 0 and (cp.stdout or "").strip():
                info["gpu_nvidia_smi"] = cp.stdout.strip().splitlines()[0]
        except Exception:
            pass
        METRICS_HW_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_HW_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        _log_runtime_event("warn", "hardware_info_write_failed", error=str(e))


def _metrics_snapshot() -> dict:
    now = time.time()
    with _state_lock:
        latest_cam_age_ms = []
        for ts in _latest_cam_ts:
            if ts > 0:
                latest_cam_age_ms.append(max(0.0, (now - ts) * 1000.0))
            else:
                latest_cam_age_ms.append(None)
        snap = {
            "running": int(bool(_running)),
            "state": _last_state or "",
            "result_id": int(_last_result_id),
            "capture_ms": float(_last_capture_ms) if _last_capture_ms is not None else None,
            "preprocess_ms": float(_last_preprocess_ms) if _last_preprocess_ms is not None else None,
            "build_ms": float(_last_build_ms) if _last_build_ms is not None else None,
            "infer_ms": float(_last_infer_ms) if _last_infer_ms is not None else None,
            "post_yolo_ms": float(_last_post_yolo_ms) if _last_post_yolo_ms is not None else None,
            "postprocess_ms": float(_last_postprocess_ms) if _last_postprocess_ms is not None else None,
            "encode_ms": float(_last_encode_ms) if _last_encode_ms is not None else None,
            "total_ms": float(_last_total_ms) if _last_total_ms is not None else None,
            "session_total_infer": int(_session_total_infer),
            "session_defect_infer": int(_session_defect_infer),
            "run_generation": int(_run_generation),
            "result_age_ms": max(0.0, (now - _last_result_ready_ts) * 1000.0) if _last_result_ready_ts > 0 else None,
            "cam0_age_ms": latest_cam_age_ms[0] if len(latest_cam_age_ms) > 0 else None,
            "cam1_age_ms": latest_cam_age_ms[1] if len(latest_cam_age_ms) > 1 else None,
            "cam2_age_ms": latest_cam_age_ms[2] if len(latest_cam_age_ms) > 2 else None,
            "python_active_threads": int(threading.active_count()),
            "gc_gen0_count": int(gc.get_count()[0]),
            "gc_gen1_count": int(gc.get_count()[1]),
            "gc_gen2_count": int(gc.get_count()[2]),
        }
    if torch.cuda.is_available():
        try:
            dev = torch.device("cuda:0")
            snap["torch_cuda_alloc_mib"] = torch.cuda.memory_allocated(dev) / (1024 * 1024)
            snap["torch_cuda_reserved_mib"] = torch.cuda.memory_reserved(dev) / (1024 * 1024)
            snap["torch_cuda_max_alloc_mib"] = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
            snap["torch_cuda_max_reserved_mib"] = torch.cuda.max_memory_reserved(dev) / (1024 * 1024)
        except Exception as e:
            _log_runtime_event("warn", "torch_cuda_metrics_failed", error=str(e))
            snap["torch_cuda_alloc_mib"] = None
            snap["torch_cuda_reserved_mib"] = None
            snap["torch_cuda_max_alloc_mib"] = None
            snap["torch_cuda_max_reserved_mib"] = None
    else:
        snap["torch_cuda_alloc_mib"] = None
        snap["torch_cuda_reserved_mib"] = None
        snap["torch_cuda_max_alloc_mib"] = None
        snap["torch_cuda_max_reserved_mib"] = None

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            snap["system_cpu_pct"] = float(psutil.cpu_percent(interval=None))
            snap["system_ram_pct"] = float(vm.percent)
            snap["system_ram_used_gib"] = float(vm.used) / (1024 ** 3)
            snap["system_ram_total_gib"] = float(vm.total) / (1024 ** 3)
            proc = _metrics_proc if _metrics_proc is not None else psutil.Process(os.getpid())
            snap["proc_cpu_pct"] = float(proc.cpu_percent(interval=None))
            mem = proc.memory_info()
            snap["proc_rss_mib"] = float(mem.rss) / (1024 * 1024)
            snap["proc_vms_mib"] = float(mem.vms) / (1024 * 1024)
            snap["proc_threads"] = int(proc.num_threads())
            if hasattr(proc, "num_handles"):
                try:
                    snap["proc_handles"] = int(proc.num_handles())
                except Exception:
                    snap["proc_handles"] = None
            else:
                snap["proc_handles"] = None
        except Exception as e:
            _log_runtime_event("warn", "psutil_metrics_failed", error=str(e))
    else:
        snap["system_cpu_pct"] = None
        snap["system_ram_pct"] = None
        snap["system_ram_used_gib"] = None
        snap["system_ram_total_gib"] = None
        snap["proc_cpu_pct"] = None
        snap["proc_rss_mib"] = None
        snap["proc_vms_mib"] = None
        snap["proc_threads"] = None
        snap["proc_handles"] = None

    snap.update(_query_nvidia_smi())
    return snap


def _metrics_logger_worker(run_until_ts: float) -> None:
    fields = [
        "ts_iso",
        "epoch_s",
        "running",
        "state",
        "run_generation",
        "result_id",
        "capture_ms",
        "preprocess_ms",
        "build_ms",
        "infer_ms",
        "post_yolo_ms",
        "postprocess_ms",
        "encode_ms",
        "total_ms",
        "result_age_ms",
        "session_total_infer",
        "session_defect_infer",
        "cam0_age_ms",
        "cam1_age_ms",
        "cam2_age_ms",
        "python_active_threads",
        "gc_gen0_count",
        "gc_gen1_count",
        "gc_gen2_count",
        "system_cpu_pct",
        "system_ram_pct",
        "system_ram_used_gib",
        "system_ram_total_gib",
        "proc_cpu_pct",
        "proc_rss_mib",
        "proc_vms_mib",
        "proc_threads",
        "proc_handles",
        "torch_cuda_alloc_mib",
        "torch_cuda_reserved_mib",
        "torch_cuda_max_alloc_mib",
        "torch_cuda_max_reserved_mib",
        "gpu_util_pct",
        "gpu_mem_util_pct",
        "gpu_mem_used_mib",
        "gpu_mem_total_mib",
        "gpu_temp_c",
        "gpu_power_w",
        "gpu_power_limit_w",
        "gpu_sm_clock_mhz",
        "gpu_mem_clock_mhz",
        "gpu_fan_pct",
        "gpu_pstate",
        "gpu_pcie_gen",
        "gpu_pcie_width",
        "gpu_throttle_power_cap",
        "gpu_throttle_hw_thermal",
        "gpu_throttle_sw_thermal",
        "gpu_throttle_hw_slowdown",
    ]
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        while not _metrics_stop_event.is_set():
            now = time.time()
            if now >= run_until_ts:
                break
            row = _metrics_snapshot()
            row["ts_iso"] = datetime.fromtimestamp(now).isoformat(timespec="seconds")
            row["epoch_s"] = round(now, 3)
            writer.writerow(row)
            f.flush()
            _metrics_stop_event.wait(METRICS_LOG_INTERVAL_S)


def _start_metrics_logger(force_restart: bool = False) -> None:
    global _metrics_thread, _metrics_started_ts, _metrics_end_ts, _metrics_proc
    if not METRICS_ENABLE:
        return
    if _metrics_thread is not None and _metrics_thread.is_alive():
        if not force_restart:
            return
        _metrics_stop_event.set()
        _metrics_thread.join(timeout=2.0)
    if psutil is not None:
        try:
            _metrics_proc = psutil.Process(os.getpid())
            _metrics_proc.cpu_percent(interval=None)
        except Exception:
            _metrics_proc = None
    _metrics_stop_event.clear()
    _metrics_started_ts = time.time()
    _metrics_end_ts = _metrics_started_ts + METRICS_DURATION_HOURS * 3600.0
    _write_hardware_info()
    _log_runtime_event(
        "info",
        "metrics_started",
        duration_hours=METRICS_DURATION_HOURS,
        interval_s=METRICS_LOG_INTERVAL_S,
        csv_path=str(METRICS_LOG_PATH),
        events_path=str(METRICS_EVENT_LOG_PATH),
    )
    t = threading.Thread(target=_metrics_logger_worker, args=(_metrics_end_ts,), daemon=True)
    t.start()
    _metrics_thread = t


def _stop_metrics_logger() -> None:
    if _metrics_thread is not None and _metrics_thread.is_alive():
        _metrics_stop_event.set()
        _log_runtime_event("info", "metrics_stop_requested")


@app.post("/metrics/start")
def metrics_start(hours: Optional[float] = None, interval_s: Optional[float] = None):
    global METRICS_DURATION_HOURS, METRICS_LOG_INTERVAL_S
    if hours is not None:
        METRICS_DURATION_HOURS = max(0.1, float(hours))
    if interval_s is not None:
        METRICS_LOG_INTERVAL_S = max(0.5, float(interval_s))
    _start_metrics_logger(force_restart=True)
    return {
        "ok": True,
        "log_path": str(METRICS_LOG_PATH),
        "event_log_path": str(METRICS_EVENT_LOG_PATH),
        "hardware_info_path": str(METRICS_HW_INFO_PATH),
        "duration_hours": METRICS_DURATION_HOURS,
        "interval_s": METRICS_LOG_INTERVAL_S,
    }


@app.get("/metrics/status")
def metrics_status():
    alive = bool(_metrics_thread is not None and _metrics_thread.is_alive())
    return {
        "enabled": METRICS_ENABLE,
        "active": alive,
        "log_path": str(METRICS_LOG_PATH),
        "event_log_path": str(METRICS_EVENT_LOG_PATH),
        "hardware_info_path": str(METRICS_HW_INFO_PATH),
        "started_ts": _metrics_started_ts,
        "end_ts": _metrics_end_ts,
        "interval_s": METRICS_LOG_INTERVAL_S,
        "duration_hours": METRICS_DURATION_HOURS,
    }


@app.post("/history/reindex")
def history_reindex():
    root = _resolve_annotated_root()
    if root is None:
        raise HTTPException(status_code=500, detail="history directory unavailable")
    _sync_history_index(root, force=True)
    conn = _history_connect()
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM defect_history").fetchone()[0])
    finally:
        conn.close()
    return {"ok": True, "indexed_items": total, "db_path": str(_history_db_path)}


@app.get("/result/{i}")
def result(i: int):
    if i < 0 or i > 2:
        raise HTTPException(status_code=404, detail="bad index")
    with _state_lock:
        jpeg = _last_results[i]
    if jpeg is None:
        raise HTTPException(status_code=404, detail="no result")
    return Response(jpeg, media_type="image/jpeg", headers=_no_cache_headers)


@app.get("/stream")
def stream():
    with _state_lock:
        jpeg = _last_stream_jpeg
    if jpeg is None:
        raise HTTPException(status_code=404, detail="no stream")
    return Response(jpeg, media_type="image/jpeg", headers=_no_cache_headers)


@app.get("/history")
def history_page():
    html = """
<!DOCTYPE html>
<html>
<head>
  <title>История дефектов</title>
  <style>
    body { background: #0f1115; color: #e6e6e6; font-family: Arial, sans-serif; margin: 20px; }
    .card { background: #151922; padding: 12px; border-radius: 10px; margin-bottom: 12px; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; }
    .field { display: flex; flex-direction: column; gap: 4px; }
    .field label { font-size: 12px; color: #c9cdd6; }
    select, input { background: #0f1115; color: #e6e6e6; border: 1px solid #2a2f3a; border-radius: 6px; padding: 6px 8px; }
    .picker-wrap { position: relative; display: flex; align-items: center; }
    .picker-input { width: 100%; padding-right: 34px; }
    .picker-btn {
      position: absolute;
      right: 5px;
      width: 24px;
      height: 24px;
      border: none;
      border-radius: 6px;
      background: #2a2f3a;
      color: #ffffff;
      font-size: 14px;
      cursor: pointer;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .picker-btn:hover { background: #3a4150; }
    input[type="date"], input[type="time"] { color-scheme: dark; }
    input[type="date"]::-webkit-calendar-picker-indicator,
    input[type="time"]::-webkit-calendar-picker-indicator {
      opacity: 0;
      cursor: pointer;
    }
    .btn { padding: 8px 12px; border: none; border-radius: 6px; background: #2a2f3a; color: #fff; cursor: pointer; }
    .meta { font-size: 13px; color: #c9cdd6; }
    .shift-legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 13px; color: #c9cdd6; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; margin-left: 6px; }
    .pill.day { background: #2a6fdb; color: #fff; }
    .pill.night { background: #7b4ad9; color: #fff; }
    .pill.line { background: #2a3a4a; color: #9ba8bc; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }
    .item { background: #111722; border-radius: 8px; padding: 8px; }
    .item img { width: 100%; height: auto; border-radius: 6px; background: #000; cursor: zoom-in; }
    .row { font-size: 13px; margin-bottom: 6px; }
    .zoom-modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.85);
      align-items: center;
      justify-content: center;
      z-index: 9999;
      cursor: zoom-out;
      padding: 20px;
    }
    .zoom-modal img {
      max-width: calc(100vw - 40px);
      max-height: calc(100vh - 40px);
      width: auto;
      height: auto;
      border-radius: 8px;
      box-shadow: 0 0 24px rgba(0,0,0,0.6);
    }
    .zoom-meta {
      position: absolute;
      top: 24px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(15, 17, 21, 0.7); /* 70% opacity */
      color: #ffffff;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.2px;
      text-align: center;
      white-space: nowrap;
    }
    .pager { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
    .pager-info { font-size: 13px; color: #c9cdd6; }
  </style>
</head>
<body>
  <h2>История дефектов</h2>

  <div class="card">
    <div class="controls">
      <div class="field">
        <label>Период: с даты</label>
        <div class="picker-wrap">
          <input id="dateFromFilter" class="picker-input" type="date" />
          <button class="picker-btn" type="button" onclick="openPicker('dateFromFilter')" aria-label="Открыть календарь">📅</button>
        </div>
      </div>
      <div class="field">
        <label>по дату</label>
        <div class="picker-wrap">
          <input id="dateToFilter" class="picker-input" type="date" />
          <button class="picker-btn" type="button" onclick="openPicker('dateToFilter')" aria-label="Открыть календарь">📅</button>
        </div>
      </div>
      <div class="field">
        <label>Интервал: c</label>
        <div class="picker-wrap">
          <input id="startTimeFilter" class="picker-input" type="time" />
          <button class="picker-btn" type="button" onclick="openPicker('startTimeFilter')" aria-label="Открыть время">🕒</button>
        </div>
      </div>
      <div class="field">
        <label>по</label>
        <div class="picker-wrap">
          <input id="endTimeFilter" class="picker-input" type="time" />
          <button class="picker-btn" type="button" onclick="openPicker('endTimeFilter')" aria-label="Открыть время">🕒</button>
        </div>
      </div>
      <div class="field">
        <label>Смена</label>
        <select id="shiftFilter">
          <option value="">Все</option>
          <option value="day">Дневная (08:00 - 19:30)</option>
          <option value="night">Ночная (19:30 - 08:00)</option>
        </select>
      </div>
      <div class="field">
        <label>Линия</label>
        <select id="lineFilter">
          <option value="">Все линии</option>
          <option value="2">Линия № 2</option>
          <option value="1">Линия № 1</option>
        </select>
      </div>
      <button class="btn" onclick="loadHistory()">Показать</button>
      <button class="btn" onclick="resetFilters()">Сброс</button>
    </div>
    <div id="meta" class="meta" style="margin-top:8px;">--</div>
    <div id="shiftLegend" class="shift-legend">--</div>
    <div class="pager">
      <button class="btn" onclick="prevPage()">Назад</button>
      <button class="btn" onclick="nextPage()">Вперед</button>
      <div id="pagerInfo" class="pager-info">Страница -- / --</div>
    </div>
  </div>

  <div id="grid" class="grid"></div>
  <div id="zoomModal" class="zoom-modal" onclick="closeZoom()">
    <div id="zoomMeta" class="zoom-meta" onclick="event.stopPropagation()">--</div>
    <img id="zoomImg" src="" alt="zoom" onclick="event.stopPropagation()" />
  </div>

<script>
const PAGE_SIZE = 100;
let _currentPage = 1;
let _totalPages = 1;
let _loadingHistory = false;
let _loadedItems = 0;
let _zoomItems = [];
let _zoomIndex = -1;

function esc(s) {
  return (s || '').toString().replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function openPicker(id) {
  const el = document.getElementById(id);
  if (!el) return;
  if (typeof el.showPicker === 'function') {
    try { el.showPicker(); return; } catch (e) {}
  }
  el.focus();
}

function getFilters() {
  const dayFrom = document.getElementById('dateFromFilter').value || '';
  const dayTo = document.getElementById('dateToFilter').value || '';
  const startTime = document.getElementById('startTimeFilter').value || '';
  const endTime = document.getElementById('endTimeFilter').value || '';
  const shift = document.getElementById('shiftFilter').value || '';
  const line = document.getElementById('lineFilter').value || '';
  const page = _currentPage;
  const params = new URLSearchParams();
  if (dayFrom) params.set('day_from', dayFrom);
  if (dayTo) params.set('day_to', dayTo);
  if (startTime) params.set('start_time', startTime);
  if (endTime) params.set('end_time', endTime);
  if (shift) params.set('shift', shift);
  if (line) params.set('line', line);
  params.set('page', String(page));
  params.set('page_size', String(PAGE_SIZE));
  return params;
}

function resetFilters() {
  document.getElementById('dateFromFilter').value = '';
  document.getElementById('dateToFilter').value = '';
  document.getElementById('startTimeFilter').value = '';
  document.getElementById('endTimeFilter').value = '';
  document.getElementById('shiftFilter').value = '';
  document.getElementById('lineFilter').value = '';
  _currentPage = 1;
  loadHistory();
}

function prevPage() {
  if (_currentPage <= 1) return;
  _currentPage -= 1;
  loadHistory(false);
}

function nextPage() {
  if (_currentPage >= _totalPages) return;
  _currentPage += 1;
  loadHistory(false);
}

function openZoomByIndex(index) {
  if (!Array.isArray(_zoomItems) || !_zoomItems.length) return;
  if (index < 0 || index >= _zoomItems.length) return;
  _zoomIndex = index;
  renderZoom();
}

function renderZoom() {
  const modal = document.getElementById('zoomModal');
  const img = document.getElementById('zoomImg');
  const meta = document.getElementById('zoomMeta');
  if (_zoomIndex < 0 || _zoomIndex >= _zoomItems.length) return;
  const item = _zoomItems[_zoomIndex];
  img.src = item.src;
  meta.textContent = item.ru_datetime || '--';
  modal.style.display = 'flex';
}

function closeZoom() {
  const modal = document.getElementById('zoomModal');
  const img = document.getElementById('zoomImg');
  const meta = document.getElementById('zoomMeta');
  modal.style.display = 'none';
  img.src = '';
  if (meta) meta.textContent = '--';
  _zoomIndex = -1;
}

function isZoomOpen() {
  const modal = document.getElementById('zoomModal');
  return !!modal && modal.style.display === 'flex';
}

function zoomPrev() {
  if (!isZoomOpen()) return;
  if (_zoomIndex <= 0) return;
  _zoomIndex -= 1;
  renderZoom();
}

function zoomNext() {
  if (!isZoomOpen()) return;
  if (_zoomIndex >= _zoomItems.length - 1) return;
  _zoomIndex += 1;
  renderZoom();
}

async function loadHistory(append = false) {
  if (_loadingHistory) return;
  _loadingHistory = true;
  const metaEl = document.getElementById('meta');
  const shiftLegendEl = document.getElementById('shiftLegend');
  const pagerInfoEl = document.getElementById('pagerInfo');
  const grid = document.getElementById('grid');
  try {
    const params = getFilters();
    const r = await fetch('/history-data?' + params.toString());
    const data = await r.json();

    const items = Array.isArray(data.items) ? data.items : [];
    const dayCount = Number(data.day_count || 0);
    const nightCount = Number(data.night_count || 0);
    const page = Number(data.page || 1);
    const totalPages = Number(data.total_pages || 1);
    const total = Number(data.total_items || items.length);
    const pageSize = Number(data.page_size || PAGE_SIZE);
    _currentPage = page;
    _totalPages = totalPages > 0 ? totalPages : 1;
    if (!append) {
      _zoomItems = [];
    }
    const renderedParts = [];
    for (const it of items) {
      const src = '/history-image/' + encodeURIComponent(it.filename);
      const idx = _zoomItems.length;
      _zoomItems.push({
        src: src,
        ru_datetime: (it.ru_datetime || it.datetime || '--').toString(),
      });
      const lineNum = it.line || 2;
      renderedParts.push(
        '<div class="item">' +
          '<div class="row"><b>' + esc(it.ru_datetime || it.datetime) + '</b>' +
          '<span class="pill ' + esc(it.shift_key || '') + '">' + esc(it.shift || '') + '</span>' +
          '<span class="pill line">Линия № ' + lineNum + '</span></div>' +
          '<img loading="lazy" src="' + src + '" onclick="openZoomByIndex(' + idx + ')" />' +
        '</div>'
      );
    }
    const rendered = renderedParts.join('');

    if (append) {
      if (rendered) {
        grid.insertAdjacentHTML('beforeend', rendered);
      }
      _loadedItems += items.length;
    } else {
      _loadedItems = items.length;
      if (!items.length) {
        grid.innerHTML = '<div class="card">Нет записей для выбранных фильтров.</div>';
      } else {
        grid.innerHTML = rendered;
      }
    }

    const endIdx = Math.min(_loadedItems, total);
    pagerInfoEl.textContent = 'Страница ' + _currentPage + ' / ' + _totalPages + ' | Загружено: ' + endIdx + ' из ' + total;
    metaEl.textContent = 'Найдено всего: ' + total;
    shiftLegendEl.innerHTML =
      'Дневная смена (08:00 - 19:30): <b>' + dayCount + '</b>' +
      ' | Ночная смена (19:30 - 08:00): <b>' + nightCount + '</b>';
  } catch (e) {
    metaEl.textContent = 'Ошибка загрузки истории';
    shiftLegendEl.textContent = '--';
    if (pagerInfoEl) pagerInfoEl.textContent = 'Страница -- / --';
    grid.innerHTML = '<div class="card">Не удалось загрузить данные.</div>';
  } finally {
    _loadingHistory = false;
  }
}

async function autoLoadNextPageOnScroll() {
  if (_loadingHistory) return;
  if (_currentPage >= _totalPages) return;
  const nearBottom = (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 220);
  if (!nearBottom) return;
  _currentPage += 1;
  await loadHistory(true);
}

window.addEventListener('load', async () => {
  await loadHistory(false);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeZoom();
      return;
    }
    if (e.key === 'ArrowLeft') {
      zoomPrev();
      return;
    }
    if (e.key === 'ArrowRight') {
      zoomNext();
      return;
    }
  });
  window.addEventListener('scroll', () => { autoLoadNextPageOnScroll(); }, { passive: true });
});
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/history-data")
def history_data(
    day_from: Optional[str] = None,
    day_to: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    shift: Optional[str] = None,
    line: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
):
    fixed_page_size = 100
    try:
        page_size = fixed_page_size if int(page_size) != fixed_page_size else int(page_size)
    except Exception:
        page_size = fixed_page_size
    safe_page = max(1, int(page))
    root = _resolve_annotated_root()
    if root is None:
        return {
            "items": [],
            "root_dir": "",
            "sort": "latest_first",
            "page": safe_page,
            "page_size": page_size,
            "total_items": 0,
            "total_pages": 1,
            "day_count": 0,
            "night_count": 0,
        }
    data = _fetch_history_page(
        root,
        day_from=day_from,
        day_to=day_to,
        start_time=start_time,
        end_time=end_time,
        shift=shift,
        line=line,
        page=safe_page,
        page_size=page_size,
    )
    return {
        "items": data["items"],
        "root_dir": str(root),
        "sort": "latest_first",
        "page": data["page"],
        "page_size": page_size,
        "total_items": data["total_items"],
        "total_pages": data["total_pages"],
        "day_count": data["day_count"],
        "night_count": data["night_count"],
    }


@app.get("/history-image/{filename}")
def history_image(filename: str):
    # Prevent path traversal; we only serve files directly from annotated directory.
    if "/" in filename or "\\" in filename or not filename:
        raise HTTPException(status_code=400, detail="bad filename")
    root = _resolve_annotated_root()
    if root is None:
        raise HTTPException(status_code=404, detail="history directory unavailable")
    fpath = (root / filename).resolve()
    if not str(fpath).startswith(str(root.resolve())):
        raise HTTPException(status_code=400, detail="bad filename")
    if not fpath.exists() or not fpath.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(fpath), media_type="image/jpeg", headers=_no_cache_headers)


# ---- Shutdown summary (Ctrl+C) ----
_shutdown_printed = False


def _print_session_summary():
    global _shutdown_printed
    if _shutdown_printed:
        return
    _shutdown_printed = True

    total = _session_total_infer
    defects = _session_defect_infer
    clean = total - defects
    start = _session_start_ts

    if start > 0:
        elapsed = time.time() - start
        hrs = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        dur_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
    else:
        dur_str = "N/A (never started)"

    pct = (defects / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 56, flush=True)
    print("  SESSION SUMMARY", flush=True)
    print("=" * 56, flush=True)
    print(f"  Run time           : {dur_str}", flush=True)
    print(f"  Total inferences   : {total}", flush=True)
    print(f"  Defects detected   : {defects}  ({pct:.1f}%)", flush=True)
    print(f"  Clean (no defect)  : {clean}", flush=True)
    print("=" * 56 + "\n", flush=True)


atexit.register(_print_session_summary)


def _sigint_handler(sig, frame):
    _print_session_summary()
    raise SystemExit(0)


signal.signal(signal.SIGINT, _sigint_handler)

if AUTO_REFRESH_ENABLED:
    _auto_refresh_thread = threading.Thread(target=_auto_refresh_worker, daemon=True)
    _auto_refresh_thread.start()

if FREEZE_WATCHDOG_ENABLED:
    _freeze_watchdog_thread = threading.Thread(target=_freeze_watchdog_worker, daemon=True)
    _freeze_watchdog_thread.start()

if MEM_CLEANUP_ENABLED:
    _mem_cleanup_thread = threading.Thread(target=_memory_cleanup_worker, daemon=True)
    _mem_cleanup_thread.start()
