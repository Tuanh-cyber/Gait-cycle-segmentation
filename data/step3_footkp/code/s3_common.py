"""Hằng số, đường dẫn và nạp dữ liệu cho Bước 3 (kiểm định & làm sạch keypoint bàn chân).

v2: mọi tính toán thời gian dùng t_real_s_v2 (Bước 1b), bỏ luật (d) ngoài bbox."""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2]            # ROOT/data
OUT = DATA / "step3_footkp"
DIRS = {k: OUT / k for k in ("footkp", "review", "plots", "qc")}
KP_DIR = DATA / "step2_tracking" / "keypoints_v2"

# ---- chỉ số Halpe26
L_LEG = [11, 13, 15, 20, 22, 24]      # hip knee ankle bigtoe smalltoe heel
R_LEG = [12, 14, 16, 21, 23, 25]
HIP, KNEE, ANKLE, BIG, SMALL, HEEL = 0, 1, 2, 3, 4, 5   # vị trí trong L_LEG / R_LEG
FOOT_IDX = [15, 16, 20, 21, 22, 23, 24, 25]
L_FOOT, R_FOOT = [15, 20, 22, 24], [16, 21, 23, 25]

# ---- 3.1 Viterbi
SWITCH_COST = 0.30       # phạt mỗi lần đổi trạng thái (đơn vị: tổng sai lệch chuẩn hoá L_leg / 12 điểm)
DIST_CAP = 1.0          # chặn sai lệch mỗi điểm (× L_leg): một điểm hỏng không đủ gây đổi nhãn
W_PHASE = 1.0           # trọng số prior lệch pha trên chênh x cổ chân
PHASE_R_MIN = 0.80      # độ tuần hoàn d² tối thiểu để dùng prior lệch pha (sin thuần = 1). Dưới mức này thường là tốc độ đổi trong cùng speed part
PHASE_R_MIN_RAMP = 0.70  # riêng part ramp đã sửa thời gian (F26: nếu không có prior, f0–f8 bị hoán đổi nhầm)
STRIDE_S_MIN, STRIDE_S_MAX = 0.35, 4.0   # dải quét chu kỳ stride theo t_real_s (rộng vì k có thể sai)
MAX_GAP_FRAMES = 3      # quy tắc đặc tả: không nối/nội suy qua khoảng trống > 3 frame unique

# ---- 3.3 làm sạch
SCORE_MIN = 0.30
HEEL_TOE_DEV = 0.40
VEL_PCT = 99.5
SG_WINDOW = 7
SG_ORDER = 3

# ---- 3.4 gate
FOOT_MIN_PTS = 3         # >= 3/4 điểm (ankle, big toe, small toe, heel) mỗi chân
LEG_MIN_PTS = 4          # bản đúng chữ của đặc tả: >= 4/6 điểm của tập "chân" (3.1)
GATE_NEAR_FAIL = 0.70
GATE_SWAP_FAIL = 0.05
GATE_FAR_WARN = 0.50
NEAR_UNRELIABLE_MANUAL = ("C27",)   # người kiểm tra: màu near/far đổi chân từ f83 tới hết clip
N_PERTURB = 30
PERTURB_SCALES = (0.9, 1.1)


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def load_meta():
    return pd.read_csv(DATA / "step0_audit" / "step0_metadata.csv", dtype=str, keep_default_na=False)


@dataclass
class Clip:
    vid: str
    frame: np.ndarray
    t: np.ndarray         # t_real_s_v2
    t_old: np.ndarray     # t_real_s của Bước 1, chỉ để so sánh
    pts: np.ndarray
    time_model: np.ndarray
    cont: np.ndarray
    part: np.ndarray
    k_conf: np.ndarray
    unique_id: np.ndarray
    kp: np.ndarray       # N×26×2, float64, NaN nếu không có keypoint
    score: np.ndarray
    bbox: np.ndarray
    status: np.ndarray
    W: int
    H: int
    meta: pd.Series

    @property
    def n(self):
        return len(self.frame)


def load_clip(vid, meta):
    m = meta.set_index("video_id").loc[vid]
    z = np.load(KP_DIR / f"{vid}_kp.npz", allow_pickle=True)
    fr = pd.read_csv(DATA / "step0_audit" / "frames" / f"{vid}_frames.csv").set_index("frame")
    f = z["frame"].astype(int)
    sub = fr.loc[f]
    if sub.is_dup.any() or sub.is_edit_freeze.any():
        raise ValueError(f"{vid}: keypoints_v2 có frame trùng/freeze")
    if not np.array_equal(sub.continuity_id.values, z["continuity_id"]):
        raise ValueError(f"{vid}: continuity_id không khớp step0")
    if not np.isfinite(sub.t_real_s_v2.values).all():
        raise ValueError(f"{vid}: frame hợp lệ thiếu t_real_s_v2")
    return Clip(vid=vid, frame=f, t=sub.t_real_s_v2.values.astype(float), t_old=z["t_real_s"].astype(float),
                pts=z["pts_s"].astype(float), time_model=sub.time_model.values.astype(str), cont=z["continuity_id"].astype(int),
                part=sub.speed_part_id.values.astype(int), k_conf=sub.k_confidence.values.astype(str),
                unique_id=sub.unique_id.values.astype(int), kp=z["kp"].astype(float),
                score=np.nan_to_num(z["score"].astype(float)), bbox=z["bbox"].astype(float),
                status=z["track_status"].astype(str), W=int(z["W"]), H=int(z["H"]), meta=m)


def video_path(m):
    return DATA / ("Correct" if m.label == "CORRECT" else "False") / m.video


def leg_length(kp):
    """L_leg của 1 khối = trung vị |hip−ankle| gộp 2 bên. Không đổi khi hoán đổi nhãn cả chân."""
    a = np.linalg.norm(kp[:, 11] - kp[:, 15], axis=1)
    b = np.linalg.norm(kp[:, 12] - kp[:, 16], axis=1)
    v = np.r_[a, b]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


def leg_length_t(kp, t, half_window_s=0.5):
    """L_leg theo thời gian: trung vị trượt ±0.5 s, cho người lớn dần trong khối (F10_1: bbox đổi gần 2 lần)."""
    a = np.linalg.norm(kp[:, 11] - kp[:, 15], axis=1)
    b = np.linalg.norm(kp[:, 12] - kp[:, 16], axis=1)
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        w = np.abs(t - t[i]) <= half_window_s
        v = np.r_[a[w], b[w]]
        v = v[np.isfinite(v)]
        if len(v):
            out[i] = np.median(v)
    return out


def atomic_npz(path, **arr):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, path)



# ---- 3.3 vận tốc: chuẩn hoá theo tốc độ cục bộ (k của Bước 1 có thể sai hoặc đổi trong cùng part)
VEL_LOCAL_HALF = 15      # ± số frame unique để lấy trung vị tốc độ cục bộ
PLATEAU_MAX = 5          # 2 bước nhảy cách nhau <= chừng này frame và điểm quay lại quỹ đạo cũ: các frame giữa là outlier
COLLAPSE_DIST = 0.15     # 2 cổ chân cách nhau < chừng này × L_leg
COLLAPSE_REF = 0.40      # trong khi tham chiếu pha nói 2 chân phải cách nhau > chừng này
R_SCORE, R_RATIO, R_VEL, R_COLLAPSE = 1, 2, 4, 16     # bit 8 từng là luật (d), đã bỏ ở v2
REASONS = {"score": R_SCORE, "heel_toe": R_RATIO, "velocity": R_VEL, "collapse": R_COLLAPSE}
VEL_JSON = OUT / "qc" / "velocity_threshold.json"


def segments(c, valid=None, max_gap=MAX_GAP_FRAMES):
    """Đoạn liên tục: cùng continuity_id + speed_part_id, khoảng trống <= max_gap frame unique. Trả về list chỉ số."""
    idx = np.arange(c.n) if valid is None else np.flatnonzero(valid)
    if not len(idx):
        return []
    brk = (np.diff(c.cont[idx]) != 0) | (np.diff(c.part[idx]) != 0) | (np.diff(c.unique_id[idx]) - 1 > max_gap)
    return np.split(idx, np.flatnonzero(brk) + 1)

GATE_NEAR_CONF = 0.6     # WARN: bằng chứng near/far yếu (khối conf <= 0.6 chỉ khớp metadata 2/3)
GATE_PHASE_MIN = 0.5     # WARN: dưới 50% frame có chu kỳ rõ: không kiểm chéo được bằng tham chiếu pha
GATE_PERTURB = 0.05      # WARN: heel/toe dịch trung vị > 5% L_leg khi đổi bbox ±10%
GATE_RAMP = 2.0          # WARN: khoảng cách giữa 2 lần chéo chân (p90/p10) > chừng này trong 1 part: k của Bước 1 sai cục bộ


def config_hash():
    import hashlib
    keys = [k for k in globals() if k.isupper() and isinstance(globals()[k], (int, float, str, tuple))]
    keys = [k for k in keys if k not in ("VEL_JSON",)]
    return hashlib.sha1(repr(sorted((k, globals()[k]) for k in keys)).encode()).hexdigest()[:10]
