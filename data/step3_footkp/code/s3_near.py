"""3.2 Xác định chân gần camera theo bằng chứng hình học, không dùng nhãn L/R của model.

Sau 3.1, slot L_LEG là chân A, slot R_LEG là chân B của khối. Mỗi bằng chứng cho số dương nếu A gần hơn:
  foot_size   ảnh bàn chân dài hơn (heel->toe)
  leg_size    đùi + cẳng chân dài hơn trong ảnh (phối cảnh, ít nhiễu hơn bàn chân)
  ground_y    camera cao hơn mặt đất: bàn chân gần chạm đất ở vị trí thấp hơn trong ảnh
  occlusion   lúc 2 chân chồng lấn, chân xa bị che nên score thấp hơn
"""
import numpy as np

import s3_common as C
from s3_common import ANKLE, BIG, HEEL, HIP, KNEE, L_LEG, R_LEG, SMALL

CUES = ("foot_size", "leg_size", "ground_y", "occlusion")
CHUNK_FRAMES = 8        # gom frame thành khúc, vì frame liền kề không độc lập
N_BOOT = 1000
OVERLAP = 0.25          # |chênh x cổ chân| < chừng này × L_leg coi là chồng lấn


def _foot_len(kp, leg, valid):
    a = np.linalg.norm(kp[:, leg[HEEL]] - kp[:, leg[BIG]], axis=1)
    b = np.linalg.norm(kp[:, leg[HEEL]] - kp[:, leg[SMALL]], axis=1)
    a[~(valid[:, leg[HEEL]] & valid[:, leg[BIG]])] = np.nan
    b[~(valid[:, leg[HEEL]] & valid[:, leg[SMALL]])] = np.nan
    return np.nanmax(np.stack([a, b]), axis=0)


def _leg_len(kp, leg, valid):
    a = np.linalg.norm(kp[:, leg[HIP]] - kp[:, leg[KNEE]], axis=1) + np.linalg.norm(kp[:, leg[KNEE]] - kp[:, leg[ANKLE]], axis=1)
    a[~(valid[:, leg[HIP]] & valid[:, leg[KNEE]] & valid[:, leg[ANKLE]])] = np.nan
    return a


def per_frame_cues(kp, sc, valid, L):
    """Trả về dict cue -> mảng N (NaN nếu frame không có bằng chứng)."""
    n = len(kp)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {"foot_size": np.log(_foot_len(kp, L_LEG, valid) / _foot_len(kp, R_LEG, valid)),
               "leg_size": np.log(_leg_len(kp, L_LEG, valid) / _leg_len(kp, R_LEG, valid))}
        pts = [ANKLE, HEEL, BIG, SMALL]
        ya = np.nanmean(np.where(valid[:, [L_LEG[i] for i in pts]], kp[:, [L_LEG[i] for i in pts], 1], np.nan), axis=1)
        yb = np.nanmean(np.where(valid[:, [R_LEG[i] for i in pts]], kp[:, [R_LEG[i] for i in pts], 1], np.nan), axis=1)
        # chỉ lấy khi cả 2 chân thấp (gần pha chống/chạm đất): so vị trí chạm đất, không so pha đu chân
        low = np.nanpercentile(np.r_[ya, yb], 60) if np.isfinite(np.r_[ya, yb]).any() else np.nan
        g = np.where((ya > low) | (yb > low), np.nan, np.nan)
        out["ground_y"] = _ground(ya, yb, L)
        d = np.abs(kp[:, 15, 0] - kp[:, 16, 0]) / L
        ov = d < OVERLAP
        m = [KNEE, ANKLE, HEEL, BIG, SMALL]
        sa = np.clip(sc[:, [L_LEG[i] for i in m]], 0, 1).mean(1)
        sb = np.clip(sc[:, [R_LEG[i] for i in m]], 0, 1).mean(1)
        out["occlusion"] = np.where(ov, sa - sb, np.nan)
    return out


def _ground(ya, yb, L):
    """Mức thấp nhất mỗi chân đạt được trong cửa sổ ±12 frame (xấp xỉ vị trí chạm đất), so 2 chân, chuẩn hoá L_leg."""
    n = len(ya)
    out = np.full(n, np.nan)
    for i in range(n):
        a, b = ya[max(0, i - 12):i + 13], yb[max(0, i - 12):i + 13]
        if np.isfinite(a).sum() >= 8 and np.isfinite(b).sum() >= 8:
            out[i] = (np.nanpercentile(a, 90) - np.nanpercentile(b, 90)) / L[i]
    return out


# thang của từng cue: |giá trị| điển hình, cố định để cue không lấn nhau (đo trên toàn dataset, không dùng metadata)
CUE_SCALE = dict(foot_size=0.10, leg_size=0.05, ground_y=0.05, occlusion=0.05)
CUE_WEIGHT = dict(foot_size=1.0, leg_size=1.0, ground_y=1.0, occlusion=1.0)


def block_decision(cues, seed=0):
    """cues: dict cue -> N frame của 1 khối. Trả về near ('A'/'B'), conf ∈ [0,1], tóm tắt từng cue."""
    n = len(next(iter(cues.values())))
    n_chunk = max(1, n // CHUNK_FRAMES)
    edges = np.linspace(0, n, n_chunk + 1).astype(int)
    M = np.full((n_chunk, len(CUES)), np.nan)
    for q, k in enumerate(CUES):
        for i in range(n_chunk):
            x = cues[k][edges[i]:edges[i + 1]]
            x = x[np.isfinite(x)]
            if len(x):
                M[i, q] = np.clip(np.median(x) / CUE_SCALE[k], -3, 3) * CUE_WEIGHT[k]
    per = {k: (float(np.nanmedian(cues[k])) if np.isfinite(cues[k]).any() else np.nan) for k in CUES}
    if not np.isfinite(M).any():
        return "A", 0.0, per, 0.5
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_chunk, (N_BOOT, n_chunk))
    s = np.nansum(np.nanmean(np.where(np.isfinite(M[pick]), M[pick], np.nan), axis=1), axis=1)
    p = float((s > 0).mean())
    near = "A" if p >= 0.5 else "B"
    return near, abs(2 * p - 1), per, p


def expected_from_meta(near_leg):
    """A = chân mang nhãn L của model ở đa số frame (3.1). 'left' -> A; 'right' -> B; khác -> None."""
    return {"left": "A", "right": "B"}.get(near_leg)
