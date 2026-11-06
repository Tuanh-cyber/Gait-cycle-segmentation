"""Hằng số, đường dẫn và bản đồ frame dùng chung cho Bước 2 (subject tracking)."""
import json, os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2]            # ROOT/data
OUT = DATA / "step2_tracking"
DIRS = {k: OUT / k for k in ("all_tracks", "appearance", "selection", "runner", "keypoints_v2",
                             "qc", "review", "logs", "models")}

# ---- 2.1 detection / tracking
WEIGHTS = "yolo11m.pt"
CONF = 0.25
IMGSZ = 640   # KHÔNG tăng: 1280 làm người gần camera vỡ thành 2 box từng phần (cắt chân 25%, cắt đầu 34% frame)
TRACKER_YAML = Path(__file__).with_name("botsort_reid.yaml")

# ---- 2.2 chấm điểm track
W_PROX, W_LEN, W_MOTION, W_SIZE = 0.25, 0.25, 0.35, 0.15
V_REF = 1.5           # (chiều cao bbox / s thật) mức vận tốc nền coi là "đang chạy"
SIZE_REF = 0.25       # chiều cao bbox / H coi là đủ lớn
AR_MIN, AR_MAX = 1.0, 4.5
CAM_OUTLIER = 60.0    # |cam_dx| >= 60 px (ảnh 480) là lỗi ước lượng; cùng quy ước với step0_audit/code/merge.py

# ---- 2.3 nối tracklet + nội suy
JOIN_MAX_GAP_S = 0.5
JOIN_MAX_CENTER = 0.5  # × chiều cao bbox
JOIN_MAX_DH = 0.30
JOIN_MAX_APPEAR = 0.5  # khoảng cách Bhattacharyya histogram HSV vùng thân
JOIN_MAX_OVERLAP_S = 0.5    # YOLO phát hiện 2 lần trên 1 người: 2 track chồng nhau ngắn (C20: 6 frame)
JOIN_MAX_CENTER_OVERLAP = 0.25
JOIN_TOUCH_FRAMES = 2       # chồng <= chừng này frame coi như gap = 0, dùng ngưỡng nối thường (C38_8: chồng đúng 1 frame)
JOIN_WINDOW = 5        # số quan sát cuối/đầu dùng để ước lượng vận tốc và appearance
INTERP_MAX_FRAMES = 3

# ---- 2.5 cờ QC
COVERAGE_MIN = 0.90
MARGIN_LOW = 0.15
MARGIN_TIE = 0.05
FOOT_SCORE_MIN = 0.50
KP_OUT_SCORE = 0.50    # đo bbox bị cắt: đầu/bàn chân tin cậy nằm ngoài bbox
KP_OUT_TOL = 0.03      # × chiều cao bbox
KP_OUT_MAX = 0.10
HEAD_IDX = [0, 17]
MOTION_FULL_S = 1.0    # track ngắn hơn chừng này (s thật) thì s_motion giảm tỉ lệ: 3 frame có thể đạt vận tốc tối đa vì nhiễu
FOOT_VIS_THR = 0.30
FOOT_VIS_MIN = 4
FOOT_IDX = [15, 16, 20, 21, 22, 23, 24, 25]
LR_PAIRS = [(13, 14), (15, 16), (20, 21), (22, 23), (24, 25)]
CENTER_JUMP = 0.5

SKELETON = [(0, 18), (18, 19), (18, 5), (18, 6), (5, 7), (7, 9), (6, 8), (8, 10), (19, 11), (19, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (15, 24), (15, 20), (20, 22), (16, 25), (16, 21), (21, 23)]


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def load_meta():
    m = pd.read_csv(DATA / "step0_audit" / "step0_metadata.csv", encoding="utf-8-sig")
    return m.set_index("video_id", drop=False)


def excluded_shots(value):
    if pd.isna(value):
        return set()
    return {int(s.strip().replace("shot", "")) for s in str(value).split(",") if s.strip().startswith("shot")}


@dataclass
class Video:
    """Bản đồ frame của một video. `use` = frame thực sự được xử lý."""
    vid: str
    path: Path
    n_frames: int
    s_final: int
    fr: pd.DataFrame          # 1 dòng/frame giải mã, thêm cột use, inherited_from
    aw: int                   # kích thước ảnh phân tích của Bước 0
    ah: int


def analysis_size(width, height):
    """Công thức của step0_audit.frames(): 480 × h, h tính từ width/height ffprobe (CHƯA xoay)."""
    return 480, int(round(height * 480 / width / 2) * 2)


def load_video(vid, meta=None):
    m = (load_meta() if meta is None else meta).loc[vid]
    fr = pd.read_csv(DATA / "step0_audit" / "frames" / f"{vid}_frames.csv")
    for c in ("is_dup", "is_edit_freeze"):
        fr[c] = fr[c].astype(bool)
    excl = excluded_shots(m.excluded_shots)
    in_excl = fr.shot_id.isin(excl)
    # Quy tắc 1 + loại frame continuity_id = -1: frame lẻ giữa đoạn freeze, không có t_real_s và không thuộc khối nào.
    fr["use"] = ~fr.is_dup & ~fr.is_edit_freeze & ~in_excl & (fr.continuity_id >= 0)
    src = fr.frame.where(~fr.is_dup).ffill()
    inh = np.full(len(fr), -1, dtype=np.int64)
    for i in np.flatnonzero(fr.is_dup.values):
        if np.isnan(src.iat[i]):
            continue
        j = int(src.iat[i])
        if (fr.use.iat[j] and not fr.is_edit_freeze.iat[i] and not in_excl.iat[i]
                and fr.continuity_id.iat[i] == fr.continuity_id.iat[j]):
            inh[i] = j
    fr["inherited_from"] = inh
    aw, ah = analysis_size(m.width, m.height)
    folder = "Correct" if m.label == "CORRECT" else "False"
    return Video(vid, DATA / folder / m.video, int(m.n_frames), int(m.s_final), fr, aw, ah)


def atomic_csv(df, path, **kw):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def update_timing(vid, **kw):
    p = DIRS["logs"] / f"timing_{vid}.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(kw)
    atomic_json(p, d)
