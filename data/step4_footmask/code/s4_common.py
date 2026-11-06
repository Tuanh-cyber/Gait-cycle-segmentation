"""Hằng số, nạp dữ liệu, cửa sổ crop, RLE cho Bước 4 (foot mask bằng SAM2)."""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2]
OUT = DATA / "step4_footmask"
DIRS = {k: OUT / k for k in ("masks", "review", "qc", "logs", "_tmp")}
MODELS = OUT / "models"
CKPT = {"small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
        "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml")}
CHECKPOINT = "base_plus"
IMAGE_SIZE = 768         # SAM2 mặc định 1024

# chỉ số Halpe26: thứ tự trong mỗi chân là ankle, big toe, small toe, heel
FOOT = {1: [15, 20, 22, 24], 2: [16, 21, 23, 25]}     # object 1 = slot A, object 2 = slot B
ANK, BIG, SMALL, HEEL = 0, 1, 2, 3
KNEE = {1: 13, 2: 14}
MIN_PTS = 3

CROP_PAD = 0.15          # mỗi phía; tổng cộng nới 30%
JPEG_Q = 95
BOX_PAD = 0.35           # × L_foot

COV_MIN = 0.5
COV_TOL = 0.15           # × L_foot: điểm cách mask trong khoảng này vẫn tính là trúng khi xét trôi/xung đột; keypoint hay nằm sát mép giày
AREA_LO, AREA_HI = 0.4, 2.5
BAD_RUN = 3
PASS_MIN_RUN = (3, 3, 1) # độ dài đoạn lỗi tối thiểu mỗi vòng sửa; vòng cuối dọn cả đoạn 1–2 frame. Không giới hạn 5/khối (xem README)
OVERLAP_NO_PROMPT = 0.1  # IoU box 2 bàn chân trên mức này: không re-prompt, vì prompt sẽ dính bàn chân che phía trước
OCC_AREA, PART_AREA, PART_IOU = 0.25, 0.6, 0.3
COLLAPSE_IOU = 0.8       # 2 mask gần trùng mà keypoint chân này không nằm trong mask: mask đang ở chân kia, tính occluded
ST_VISIBLE, ST_PARTIAL, ST_OCCLUDED = 0, 1, 2
STATE_NAMES = ["visible", "partial", "occluded"]
AXIS_MAX_GAP = 3
REGION_EDGES = (-0.15, 0.33, 0.66, 1.15)
REGIONS = ("heel", "mid", "toe")
PERP_MAX = 0.6           # × L_foot: pixel xa trục hơn mức này không thuộc vùng nào (ống chân chiếu lên trục sẽ rơi vào vùng gót)
CONFLICT_SEP = 0.5       # × L_foot: 2 bàn chân gần hơn mức này thì không xét xung đột ID
CONFLICT_RUN = 3
COMP_MIN_FRAC, COMP_MIN_PX = 0.05, 20  # mảnh mask rời nhỏ hơn mức này không xét là mảnh riêng
COMP_MAX_DIST = 1.0      # × L_foot: mảnh rời có tâm xa bàn chân của nó hơn mức này thì bỏ (vệt trên nền)
LEAK_OUTSIDE = 0.3       # tỉ lệ pixel ngoài 3 vùng; trên mức này tính là mask lan ra ngoài bàn chân (cẳng chân, nền)

GATE_NEAR_FAIL = 0.6
GATE_CONFLICT_FAIL = 0.10
GATE_FAR_WARN = 0.4
GATE_REPROMPT_WARN = 3

# near/far không tin cậy từ review tay (đề bài Bước 4): {video: tập continuity_id, None = cả video}
# F10_2: đề bài ghi "khối cuối (12 frame)" nhưng khối 12 frame là continuity_id=1 (f523–583, góc ngang, lệch metadata),
# còn đoạn quay lưng nằm ở cuối continuity_id=2 (49 frame). Chưa rõ ý định nên bỏ nhãn cả hai, tránh gán nhầm near/far.
UNRELIABLE_BLOCKS = {"C27": None, "C04_3": None, "F10_2": {1, 2}}


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def load_meta():
    return pd.read_csv(DATA / "step0_audit" / "step0_metadata.csv", dtype=str, keep_default_na=False)


@dataclass
class Clip:
    vid: str
    frame: np.ndarray
    t: np.ndarray
    cont: np.ndarray
    uid: np.ndarray
    kp: np.ndarray
    valid: np.ndarray
    near: np.ndarray
    near_conf: np.ndarray
    reliable: np.ndarray    # theo frame: nhãn near/far dùng được
    bbox: np.ndarray
    W: int
    H: int
    path: Path

    @property
    def n(self):
        return len(self.frame)


def load_clip(vid, meta):
    m = meta.set_index("video_id").loc[vid]
    z = np.load(DATA / "step3_footkp" / "footkp" / f"{vid}_footkp.npz", allow_pickle=True)
    k = np.load(DATA / "step2_tracking" / "keypoints_v2" / f"{vid}_kp.npz", allow_pickle=True)
    fr = pd.read_csv(DATA / "step0_audit" / "frames" / f"{vid}_frames.csv").set_index("frame")
    f = z["frame"].astype(int)
    if not np.array_equal(f, k["frame"].astype(int)):
        raise ValueError(f"{vid}: frame Bước 2 và Bước 3 khác nhau")
    sub = fr.loc[f]
    if sub.is_dup.any() or sub.is_edit_freeze.any():
        raise ValueError(f"{vid}: có frame trùng/freeze")
    t = z["t_real_s_v2"] if "t_real_s_v2" in z.files else z["t_real_s"]
    cont = z["continuity_id"].astype(int)
    rel = np.full(len(f), bool(z["near_track_reliable"]))
    if vid in UNRELIABLE_BLOCKS:
        bl = UNRELIABLE_BLOCKS[vid]
        rel = np.zeros(len(f), bool) if bl is None else rel & ~np.isin(cont, list(bl))
    return Clip(vid=vid, frame=f, t=t.astype(float), cont=cont, uid=sub.unique_id.values.astype(int), kp=z["kp_clean"].astype(float),
                valid=z["valid_mask"].astype(bool), near=z["near_track"].astype(str), near_conf=z["near_conf"].astype(float),
                reliable=rel, bbox=k["bbox"].astype(float), W=int(k["W"]), H=int(k["H"]),
                path=DATA / ("Correct" if m.label == "CORRECT" else "False") / m.video)


def foot_pts(c, i, obj):
    """Điểm hợp lệ của 1 bàn chân ở frame i (K×2)."""
    idx = [j for j in FOOT[obj] if c.valid[i, j]]
    return c.kp[i, idx]


def foot_ok(c, idx, obj):
    return c.valid[np.ix_(idx, FOOT[obj])].sum(1) >= MIN_PTS


def l_foot(c, idx, obj):
    f = FOOT[obj]
    ok = c.valid[idx, f[HEEL]] & c.valid[idx, f[BIG]]
    d = np.linalg.norm(c.kp[idx, f[HEEL]] - c.kp[idx, f[BIG]], axis=1)[ok]
    return float(np.median(d)) if len(d) else np.nan


def crop_window(c, idx):
    """Cửa sổ cố định của 1 khối: hợp bbox người + điểm bàn chân, nới 15% mỗi phía. Trả về x0, y0, x1, y1 nguyên."""
    pts = [c.bbox[idx].reshape(-1, 2)]
    pts += [c.kp[idx][:, FOOT[o]].reshape(-1, 2) for o in (1, 2)]
    p = np.vstack(pts)
    p = p[np.isfinite(p).all(1)]
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    w, h = x1 - x0, y1 - y0
    return (int(max(0, np.floor(x0 - CROP_PAD * w))), int(max(0, np.floor(y0 - CROP_PAD * h))),
            int(min(c.W, np.ceil(x1 + CROP_PAD * w))), int(min(c.H, np.ceil(y1 + CROP_PAD * h))))


def rle_encode(mask):
    """Row-major, bắt đầu bằng số 0. Trả về mảng uint32 độ dài các đoạn."""
    a = np.asarray(mask, bool).ravel()
    if a.size == 0:
        return np.zeros(0, np.uint32)
    ch = np.flatnonzero(a[1:] != a[:-1]) + 1
    runs = np.diff(np.r_[0, ch, a.size])
    if a[0]:
        runs = np.r_[0, runs]
    return runs.astype(np.uint32)


def rle_decode(runs, shape):
    a = np.zeros(int(np.prod(shape)), bool)
    p = 0
    for i, r in enumerate(runs):
        if i % 2:
            a[p:p + int(r)] = True
        p += int(r)
    return a.reshape(shape)


def atomic_npz(path, **arr):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, path)
