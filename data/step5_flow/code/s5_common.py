"""Hằng số, nạp dữ liệu Bước 3–4, cửa sổ crop cho Bước 5 (optical flow theo vùng bàn chân)."""
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2]
OUT = DATA / "step5_flow" / "_trial" if os.environ.get("S5_TRIAL") else DATA / "step5_flow"     # S5_TRIAL: kiểm chứng trước, không đụng output chính
DIRS = {k: OUT / k for k in ("features", "plots", "review", "qc", "logs")}
GMA_ROOT = DATA.parent / ".third_party" / "GMA"
CHECKPOINT = GMA_ROOT / "checkpoints" / "gma-sintel.pth"

sys.path.insert(0, str(DATA / "step4_footmask" / "code"))
import s4_common as S4      # noqa: E402  chỉ đọc
import s4_report as S4R     # noqa: E402  chỉ đọc

MODEL = "gma-sintel"
ITERS = 12

# crop
BOX_PAD = 0.25              # × chiều cao bbox, 2 bên
BOX_PAD_BOTTOM = 0.15       # × chiều cao bbox, phía dưới
MIN_L_FOOT = 48             # px trên lưới flow
MAX_EDGE = 640              # cạnh dài của crop lớn nhất trong khối
MAX_EDGE_HARD = 800         # được vượt MAX_EDGE khi cần giữ L_foot >= MIN_L_FOOT

# nhất quán forward/backward
FB_ABS, FB_REL = 1.0, 0.05  # px trên lưới flow; × |flow|

# bù camera
MASK_DILATE = 0.15          # × L_foot, nới mask chân khi loại khỏi mặt đất
CAM_MOVING = 3.0            # px trên lưới flow: trên mức này là camera đang chuyển động

# bù trực tiếp trên dải đất quanh bàn chân
GROUND_ABOVE, GROUND_BELOW = 0.5, 1.0   # × L_foot quanh điểm chân thấp nhất
GROUND_LEG_RADIUS = 0.35    # × L_foot, bán kính ống chân loại khỏi dải đất
GROUND_TEX, GROUND_TEX_LOW = 1e-3, 2e-4     # trị riêng nhỏ của structure tensor; pixel có chi tiết theo cả 2 hướng, thử mức cao trước
GROUND_STATIC_DENSITY = 0.02    # pixel chi tiết đứng yên chiếm ít nhất chừng này dải đất thì mới coi là camera đứng yên, ít hơn là chữ
GROUND_CLUSTER_FRAC = 0.3   # cụm độ dịch nhỏ hơn mức này so với cụm lớn nhất thì bỏ
GROUND_ALL_FRAC = 0.2       # dùng cả dải đất khi cụm chiếm ít nhất chừng này
GROUND_STATIC = 2.0         # px: nhỏ hơn mức này là đứng yên (chữ gắn trên ảnh, hoặc camera đứng yên)
GROUND_TOL = 3.0            # px, bán kính đồng thuận
GROUND_MIN_POINTS = 15
GROUND_MIN_FRAC = 0.1       # số inlier tối thiểu so với số pixel có chi tiết
GROUND_WIN = 9              # cặp
GROUND_MOVING_SHARE = 0.5   # đa số cặp xung quanh đo được chuyển động thì cặp "đứng yên" là do chữ
GROUND_NONE, GROUND_MOVING, GROUND_STATIC_KIND = 2, 0, 1
OVERLAY_CELL = 8            # px ảnh gốc, ô của mặt nạ chữ
OVERLAY_MIN_PAIRS = 3       # ô có pixel chi tiết đứng yên ở ít nhất chừng này cặp thì là chữ gắn trên ảnh
OVERLAY_MIN_PIXELS = 3      # số pixel chi tiết đứng yên tối thiểu trong 1 ô, ở 1 cặp
OVERLAY_STATIC_PX = 0.5     # px: đứng yên tuyệt đối
OVERLAY_DILATE = 2          # ô
CAM_STATIC_SHARE = 0.8      # chừng này cặp có nền chi tiết đứng yên dày đặc thì camera đứng yên, không có khái niệm chữ
GROUND_MIN_ALONG = 0.5      # độ dịch chiếu lên trung vị trượt, dưới mức này là ngoại lai
SPARSE_SHARE = 0.5          # tỉ lệ cặp đo được dưới mức này: chuỗi thưa, lọc ngoại lai bằng trung vị trượt
SPARSE_WIN = 5              # số điểm đo

# phần nền còn thiếu sau khi bù trực tiếp: chỉ thành phần biến thiên chậm hơn 1 stride
CAM_UNDER_VX = -2.0         # P10 của vx cả bàn chân trên toàn video; dưới mức này là bù thiếu, dùng cho cờ và gate
SPLINE_CLIP = 1             # mỗi lần đổi cách tính bước B, tăng số này để feature cũ không bị coi là đã xong
VARIANTS = (("v1", -1.0), ("v2", -2.0))     # (tên, ngưỡng kích hoạt bước B). Biến thể đầu là mặc định
COMP_SLOW_BAD, COMP_SLOW_BAD_REL = 6.0, 2.0 # L_leg/s; × trung vị của khối: |comp_slow| vượt cả 2 mức là bù trực tiếp sai cục bộ, không phải nền trôi chậm
STANCE_MERGE_FRAMES, STANCE_MERGE_S = 1, 0.03   # gộp 2 đoạn chống cách nhau không quá chừng này frame unique, hoặc chừng này giây thật. Chỉ dùng cho chỉ số QC
STANCE_THR = 0.5            # slow_ratio gót trên mức này là đang chống
STANCE_BRIDGE = 2           # frame: đoạn không hợp lệ ngắn hơn hoặc bằng chừng này, kẹp giữa 2 frame chống, không làm đứt đoạn chống
DUTY_RANGE = (0.25, 0.40)   # kỳ vọng sinh lý khi chạy
STANCE_S_RANGE = (0.15, 0.30)
N_STANCE_RANGE = (0.75, 1.25)
KP_MIN_P10_MAX = 1.0        # L_leg/s
STRIDE_T_DEFAULT = 0.75     # giây, khi Bước 3 không có stride_T

# feature
REGIONS = ("heel", "mid", "toe", "foot")
MIN_PIXELS = 10
SLOW_MAIN = 2.0             # L_leg/s, slow_ratio chính
SLOW_EXTRA = (1.0, 1.5, 3.0)
HIST_EDGES = np.r_[np.arange(0, 5, 0.05), np.arange(5, 61, 1.0)]    # L_leg/s, để đổi ngưỡng không phải tính lại flow
OTSU_BINS = 256

# Δt: pts của nhiều video là VFR giả (bước ảnh đều mà pts xen kẽ, ví dụ 33/50 ms), chia cho Δt gốc sinh răng cưa
DT_REL_WIN = 9              # trung vị trượt để bỏ biến thiên chậm (pha chạy, ramp) trước khi so với Δt
DT_JITTER_MIN = 0.05        # độ lệch chuẩn của Δt / trung vị trượt; dưới mức này làm mượt không có tác dụng, giữ Δt gốc
DT_RHO_MIN = 0.4            # rho(độ dịch, Δt) lớn nhất của camera và keypoint thô; dưới mức này Δt không tỉ lệ với chuyển động
DT_WIN = 5

# cờ fit camera theo frame
CAM_BAD_INLIER = 0.15       # cụm được chọn theo độ dịch lớn nhất nên không nhất thiết là cụm đông nhất
BG_BAD_FLOOR = 5.0          # px: bg_residual > P95 của video chỉ tính là xấu khi cũng trên mức này, nếu không mọi video đều mất 5% frame tốt
BG_BAD_REL = 0.25           # và trên chừng này × độ dịch camera của cặp đó
TAIL_HI, TAIL_LO = 15.0, 0.2    # L_leg/s, cả 2 chân cùng ở đuôi khi camera đang pan
PAN_WIN = 9
CF_INLIER, CF_BG, CF_TAIL, CF_BG_BELOW_FLOOR, CF_SLOW = 1, 2, 4, 8, 16   # bit 8 chỉ để tham khảo, không làm flow_valid = False
CF_BAD = CF_INLIER | CF_BG | CF_TAIL | CF_SLOW

# kiểm tra
LS_F_MIN, LS_F_MAX, LS_DF = 0.9, 2.5, 0.005     # Hz, tần số bước đi của 1 chân (stride)
LS_MIN_FRAMES = 8
LS_MIN_SPAN = 0.4           # giây
MIN_PROMINENCE = 1.5        # L_leg/s, độ nổi tối thiểu của 1 cực tiểu |v|
SLOW_HIGH = 0.5             # cả 2 chân slow_ratio trên mức này: cùng tiếp đất

# gate
FAIL_RHO, WARN_RHO = 0.4, 0.7
FAIL_CADENCE = 0.15
WARN_FB = 0.7
WARN_BG, WARN_BG_REL = 3.0, 0.25    # px trên lưới flow; × trung vị độ dịch camera

def config_hash():
    d = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple))}
    return hashlib.sha1(repr(sorted((k, repr(v)) for k, v in d.items())).encode()).hexdigest()[:10]


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class Video:
    vid: str
    c: object               # S4.Clip: frame, t, cont, kp (kp_clean), valid, bbox, W, H, path
    z: object               # masks_post
    shot: np.ndarray
    l_leg: np.ndarray
    s_final: int
    cadence: float
    cadence_source: str
    abs_time_ok: str
    part: np.ndarray
    kp_raw: np.ndarray
    stride_T: float
    part_T: dict

    @property
    def n(self):
        return self.c.n


def part_periods(vid):
    """Chu kỳ stride theo speed_part từ Bước 3: {part: T}. Trung vị qua các khối."""
    b = pd.read_csv(DATA / "step3_footkp" / "qc" / f"{vid}_blocks.csv")
    out = {}
    for txt in b.periods.dropna():
        for k, w in json.loads(txt).items():
            if w.get("T") and np.isfinite(w["T"]) and w["T"] > 0:
                out.setdefault(int(k), []).append(float(w["T"]))
    return {k: float(np.median(w)) for k, w in out.items()}


def load_video(vid, meta):
    c = S4.load_clip(vid, meta)
    z = np.load(S4.OUT / "masks_post" / f"{vid}.npz", allow_pickle=True)
    if not np.array_equal(z["frame"], c.frame):
        raise ValueError(f"{vid}: frame của masks_post khác Bước 3")
    k = np.load(DATA / "step3_footkp" / "footkp" / f"{vid}_footkp.npz", allow_pickle=True)
    fr = pd.read_csv(DATA / "step0_audit" / "frames" / f"{vid}_frames.csv").set_index("frame")
    m = meta.set_index("video_id").loc[vid]
    s = int(m.s_final)
    if s not in (-1, 1):
        raise ValueError(f"{vid}: s_final = {m.s_final}")
    cad, src = (float(m.cadence_real_spm_v2), "cadence_real_spm_v2") if m.cadence_real_spm_v2 != "" else (np.nan, "none")
    if not np.isfinite(cad):
        q = pd.read_csv(DATA / "step3_footkp" / "qc_footkp.csv").set_index("video_id")
        T = float(q.loc[vid, "stride_T_median_s"])
        if np.isfinite(T) and T > 0:
            cad, src = 120.0 / T, "stride_T_median_s"
    q3 = pd.read_csv(DATA / "step3_footkp" / "qc_footkp.csv").set_index("video_id")
    T3 = float(q3.loc[vid, "stride_T_median_s"])
    stride = T3 if np.isfinite(T3) and T3 > 0 else (120.0 / cad if np.isfinite(cad) else STRIDE_T_DEFAULT)
    return Video(vid, c, z, fr.loc[c.frame].shot_id.values.astype(int), k["L_leg"].astype(float), s, cad, src, str(m.abs_time_ok),
                 k["speed_part_id"].astype(int), k["kp_raw"].astype(float), float(stride), part_periods(vid))


def pairs(v):
    """Chỉ số a của các cặp (a, a+1): 2 frame unique liền nhau trong list Bước 3, cùng continuity_id và shot_id."""
    c = v.c
    same = (c.cont[:-1] == c.cont[1:]) & (v.shot[:-1] == v.shot[1:]) & (c.t[1:] > c.t[:-1])
    return np.flatnonzero(same)


def crop_box(v, a):
    """Nửa dưới cơ thể của 2 frame trong cặp, nới rộng, gồm cả keypoint bàn chân. Trả về (x0, y0, x1, y1) nguyên, hoặc None."""
    c = v.c
    boxes = c.bbox[[a, a + 1]]
    boxes = boxes[np.isfinite(boxes).all(1)]
    pts = []
    for b in (a, a + 1):
        for o in (1, 2):
            p = S4.foot_pts(c, b, o)
            if len(p):
                pts.append(p)
    if not len(boxes):
        return None
    h = np.median(boxes[:, 3] - boxes[:, 1])
    x0 = boxes[:, 0].min() - BOX_PAD * h
    x1 = boxes[:, 2].max() + BOX_PAD * h
    y0 = (boxes[:, 1] + boxes[:, 3]).min() / 2
    y1 = boxes[:, 3].max() + BOX_PAD_BOTTOM * h
    if pts:
        p = np.vstack(pts)
        x0, x1 = min(x0, p[:, 0].min() - 0.1 * h), max(x1, p[:, 0].max() + 0.1 * h)
        y0, y1 = min(y0, p[:, 1].min() - 0.1 * h), max(y1, p[:, 1].max() + 0.1 * h)
    x0, y0, x1, y1 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0))), int(min(c.W, np.ceil(x1))), int(min(c.H, np.ceil(y1)))
    return (x0, y0, x1, y1) if x1 - x0 >= 16 and y1 - y0 >= 16 else None


def block_scales(v, idx, boxes):
    """Hệ số resize cố định theo khối: L_foot >= MIN_L_FOOT, cạnh dài <= MAX_EDGE (được tới MAX_EDGE_HARD nếu cần giữ L_foot). Trả về {block: s}."""
    out = {}
    blocks = v.c.cont[idx] * 100000 + v.shot[idx]
    for b in np.unique(blocks):
        w = idx[blocks == b]
        lf = np.nanmin(np.nanmedian(v.z["l_foot"][w], axis=0))
        edge = max(max(boxes[a][2] - boxes[a][0], boxes[a][3] - boxes[a][1]) for a in w if boxes[a] is not None) if any(boxes[a] is not None for a in w) else 0
        s = max(1.0, MIN_L_FOOT / lf) if np.isfinite(lf) else 1.0
        if edge * s > MAX_EDGE:
            s = max(MAX_EDGE / edge, MIN_L_FOOT / lf) if np.isfinite(lf) else MAX_EDGE / edge
            s = min(s, MAX_EDGE_HARD / edge)
        out[int(b)] = float(s)
    return out, blocks


def atomic_npz(path, **arr):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arr)
    tmp.replace(path)
