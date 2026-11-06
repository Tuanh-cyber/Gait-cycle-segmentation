"""Bước 1b: phát hiện bước chạy trên trục thời gian VIDEO, dùng chung cho dò ramp và kiểm chứng."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2]
RATIO_MIN = 1.8         # đề bài: max(d)/min(d) > 1.8
MIN_STEPS = 4
HYST = 0.15             # bộ kích Schmitt: |d| phải vượt 15% biên độ p95 của đoạn mới tính là đổi phía
MAX_GAP_FRAMES = 3


def footkp_path(vid):
    """Bước 1b đọc keypoint v1 (đã sao lưu) để kết quả không đổi khi Bước 3 chạy lại."""
    p = DATA / "step3_footkp" / "footkp_v1" / f"{vid}_footkp.npz"
    return p if p.exists() else DATA / "step3_footkp" / "footkp" / f"{vid}_footkp.npz"


def load_signal(vid):
    """Trả về DataFrame frame-unique: frame, pts_s, cont, part, uid, k, kconf, d (near − far, × L_leg)."""
    z = np.load(footkp_path(vid))
    fr = pd.read_csv(DATA / "step0_audit" / "frames" / f"{vid}_frames.csv").set_index("frame")
    f = z["frame"].astype(int)
    sub = fr.loc[f]
    kp, near, L = z["kp_clean"].astype(float), z["near_track"].astype(str), z["L_leg"].astype(float)
    x_a, x_b = kp[:, 15, 0], kp[:, 16, 0]
    d = np.where(near == "A", x_a - x_b, x_b - x_a) / L
    return pd.DataFrame(dict(frame=f, pts_s=sub.pts_s.values, cont=sub.continuity_id.values, part=sub.speed_part_id.values,
                             uid=sub.unique_id.values, k=sub.speed_k.values, kconf=sub.k_confidence.values, d=d))


def crossings(seg, hyst=HYST):
    """Lần chéo chân trong 1 đoạn (cùng khối + cùng part). Trả về list chuỗi thời điểm: khoảng trống dài cắt chuỗi."""
    v = seg[np.isfinite(seg.d)]
    if len(v) < 4:
        return []
    h = hyst * np.percentile(np.abs(v.d), 95)
    t, d, u = v.pts_s.values, v.d.values, v.uid.values
    chains, cur, state, last_side_i = [], [], 0, None
    for i in range(len(d)):
        if i and u[i] - u[i - 1] - 1 > MAX_GAP_FRAMES:      # khoảng trống dài: không có bước nào vắt qua
            if len(cur) >= 2:
                chains.append(cur)
            cur, state, last_side_i = [], 0, None
        side = 1 if d[i] > h else -1 if d[i] < -h else 0
        if side == 0:
            continue
        if state != 0 and side != state:
            # thời điểm chéo: lần đổi dấu thật cuối cùng giữa 2 cực, nội suy tuyến tính dưới mức frame
            a, b = last_side_i, i
            for j in range(b, a, -1):
                if d[j - 1] * d[j] <= 0:
                    t0 = t[j - 1] + (t[j] - t[j - 1]) * abs(d[j - 1]) / max(abs(d[j - 1]) + abs(d[j]), 1e-9)
                    cur.append(float(t0))
                    break
        state, last_side_i = side, i
    if len(cur) >= 2:
        chains.append(cur)
    return chains


def steps_of(sig):
    """Mỗi khoảng giữa 2 lần chéo = 1 bước. Trả về DataFrame cont, part, chain, t0, t1, d."""
    rows = []
    for (c, p), seg in sig.groupby(["cont", "part"], sort=True):
        for n, ch in enumerate(crossings(seg)):
            for a, b in zip(ch[:-1], ch[1:]):
                rows.append(dict(cont=c, part=p, chain=n, t0=a, t1=b, d=b - a))
    return pd.DataFrame(rows, columns=["cont", "part", "chain", "t0", "t1", "d"])
