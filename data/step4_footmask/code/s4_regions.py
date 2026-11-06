"""4.3–4.5: trạng thái che khuất, trục bàn chân, chia vùng, nhãn near/far, xung đột ID."""
import numpy as np

import s4_common as C


def states(area, ratio, iou, cov_tol):
    """N×2. occluded: ratio < 0.25, rỗng, hoặc mask đã dính sang chân kia; partial: ratio 0.25–0.6 hoặc IoU > 0.3; còn lại visible."""
    s = np.full(area.shape, C.ST_VISIBLE, np.uint8)
    s[(ratio < C.PART_AREA) | (iou[:, None] > C.PART_IOU)] = C.ST_PARTIAL
    collapsed = (iou[:, None] > C.COLLAPSE_IOU) & (cov_tol < C.COV_MIN)
    s[(ratio < C.OCC_AREA) | (area == 0) | ~np.isfinite(ratio) | collapsed] = C.ST_OCCLUDED
    return s


def foot_axes(c, idx, obj, M_obj, off):
    """origin (heel), vector heel->giữa 2 ngón. Thiếu điểm thì mượn trục frame hợp lệ gần nhất <= 3 frame unique.
    Trả về origin N×2, axis N×2 (chưa chuẩn hoá), borrowed N (bool)."""
    f = C.FOOT[obj]
    n = len(idx)
    good = c.valid[idx, f[C.HEEL]] & c.valid[idx, f[C.BIG]] & c.valid[idx, f[C.SMALL]]
    origin, axis = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    heel = c.kp[idx, f[C.HEEL]]
    toe = (c.kp[idx, f[C.BIG]] + c.kp[idx, f[C.SMALL]]) / 2
    axis[good], origin[good] = (toe - heel)[good], heel[good]
    borrowed = np.zeros(n, bool)
    uid = c.uid[idx]
    pos = np.flatnonzero(good)
    if not len(pos):
        return origin, axis, borrowed
    for a in np.flatnonzero(~good):
        d = np.abs(uid[pos] - uid[a])
        j = pos[int(np.argmin(d))]
        if d.min() > C.AXIS_MAX_GAP:
            continue
        axis[a] = axis[j]
        borrowed[a] = True
        if c.valid[idx[a], f[C.HEEL]]:
            origin[a] = heel[a]
        else:
            # gót thiếu: dời gót của frame mượn theo độ dịch trọng tâm mask
            ya, xa = np.nonzero(M_obj[a])
            yj, xj = np.nonzero(M_obj[j])
            origin[a] = heel[j] + (np.array([xa.mean() - xj.mean(), ya.mean() - yj.mean()]) if len(xa) and len(xj) else 0.0)
    return origin, axis, borrowed


def region_split(mask, origin, axis, Lf, off):
    """mask (H×W, toạ độ crop) -> 3 mask vùng. Trả về (list 3 mask, tỉ lệ pixel ngoài cả 3 vùng, độ rộng vuông góc × L_foot)."""
    empty = [np.zeros_like(mask) for _ in C.REGIONS]
    if not mask.any() or not np.isfinite(axis).all() or not np.isfinite(origin).all():
        return empty, np.nan, np.nan
    nrm = np.linalg.norm(axis)
    if nrm < 1e-6:
        return empty, np.nan, np.nan
    u = axis / nrm
    ys, xs = np.nonzero(mask)
    dx, dy = xs + off[0] - origin[0], ys + off[1] - origin[1]
    p = (dx * u[0] + dy * u[1]) / Lf
    q = (-dx * u[1] + dy * u[0]) / Lf
    e = C.REGION_EDGES
    out = []
    near_axis = np.abs(q) <= C.PERP_MAX
    for a, b, first in ((e[0], e[1], True), (e[1], e[2], False), (e[2], e[3], False)):
        sel = ((p >= a) if first else (p > a)) & (p <= b) & near_axis
        m = np.zeros_like(mask)
        m[ys[sel], xs[sel]] = True
        out.append(m)
    return out, float(((p < e[0]) | (p > e[3]) | ~near_axis).mean()), float(np.percentile(q, 95) - np.percentile(q, 5))


def conflicts(c, idx, met, Lf):
    """Mask một chân chủ yếu nằm trên keypoint chân kia, khi 2 bàn chân đủ tách để phân biệt. N×2 bool."""
    n = len(idx)
    out = np.zeros((n, 2), bool)
    for a, i in enumerate(idx):
        pa, pb = C.foot_pts(c, i, 1), C.foot_pts(c, i, 2)
        if not len(pa) or not len(pb):
            continue
        sep = np.linalg.norm(pa.mean(0) - pb.mean(0))
        if sep < C.CONFLICT_SEP * np.nanmean([Lf[1], Lf[2]]):
            continue
        for o in (0, 1):
            own, oth = met["cov_tol"][a, o], met["other"][a, o]
            if np.isfinite(oth) and oth >= 0.5 and (not np.isfinite(own) or oth > own):
                out[a, o] = True
    return out


def runs(flag, min_len):
    """Các đoạn True liên tiếp >= min_len. Trả về list (a, b)."""
    out, a = [], None
    for i, v in enumerate(np.r_[flag, False]):
        if v and a is None:
            a = i
        if not v and a is not None:
            if i - a >= min_len:
                out.append((a, i - 1))
            a = None
    return out


def labels(c, idx):
    """N×2 nhãn. Khối không tin cậy dùng A/B."""
    L = np.empty((len(idx), 2), "U4")
    for a, i in enumerate(idx):
        if c.reliable[i]:
            L[a] = ("near", "far") if c.near[i] == "A" else ("far", "near")
        else:
            L[a] = ("A", "B")
    return L
