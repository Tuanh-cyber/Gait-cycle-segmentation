"""Pixel theo frame và chân, quyết định Δt, feature theo vùng, cờ fit camera, ngưỡng slow_ratio."""
import numpy as np
import pandas as pd

import cv2
import s5_common as C

STATS = ("v_med", "v_p25", "v_p75", "vx_med", "vy_med")
FOOT_POINTS = {o: [C.S4.FOOT[o][j] for j in (C.S4.HEEL, C.S4.BIG, C.S4.SMALL)] for o in (1, 2)}


def region_masks(v, a, s, box, shape):
    """R[o][k] bool trên lưới flow; foot = hợp 3 vùng (không gồm phần mask ngoài vùng); all = hợp mask 2 chân gốc."""
    x0, y0, x1, y1 = box
    fs = (v.c.H, v.c.W)
    h, w = shape
    R, allm = {}, np.zeros(shape, bool)

    def rs(m):
        return cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    for o in (0, 1):
        regs = [rs(C.S4R.decode(v.z, a, o, k, fs)) for k in (1, 2, 3)]
        R[o] = regs + [regs[0] | regs[1] | regs[2]]
        allm |= rs(C.S4R.decode(v.z, a, o, 0, fs))
    return R, allm


def collect(res, fb, R, s_final, scale, l_leg):
    """Độ dịch residual của pixel trên từng chân, đơn vị L_leg, chưa chia Δt. dx đã nhân s_final."""
    out = []
    k = 1.0 / scale / l_leg
    for o in (0, 1):
        m = R[o][3]
        reg = np.select([R[o][0][m], R[o][1][m]], [0, 1], 2).astype(np.uint8)
        out.append(dict(dx=(res[..., 0][m] * k * s_final).astype(np.float32), dy=(res[..., 1][m] * k).astype(np.float32), reg=reg, fb=fb[m]))
    return out


def stats(Px, n, dt, ok, use_fb):
    """Feature N×2×4. ok N×2: chân được dùng. use_fb: chỉ lấy pixel qua kiểm tra forward/backward."""
    F = {k: np.full((n, 2, 4), np.nan, np.float32) for k in STATS}
    F["n_pixels"] = np.zeros((n, 2, 4), np.int32)
    F["v_hist"] = np.zeros((n, 2, 4, len(C.HIST_EDGES) - 1), np.uint16)
    if not use_fb:
        F["frac_fb_valid"] = np.full((n, 2, 4), np.nan, np.float32)
    pool = []
    for a, feet in Px.items():
        for o in (0, 1):
            if not ok[a, o]:
                continue
            p = feet[o]
            if not len(p["dx"]):
                continue
            vx, vy = p["dx"] / dt[a], p["dy"] / dt[a]
            sp = np.hypot(vx, vy)
            for r in range(4):
                base = np.ones(len(sp), bool) if r == 3 else p["reg"] == r
                if not base.any():
                    continue
                if not use_fb:
                    F["frac_fb_valid"][a, o, r] = p["fb"][base].mean()
                sel = base & p["fb"] if use_fb else base
                k = int(sel.sum())
                F["n_pixels"][a, o, r] = k
                if k < C.MIN_PIXELS:
                    continue
                x = sp[sel]
                F["v_med"][a, o, r] = np.median(x)
                F["v_p25"][a, o, r], F["v_p75"][a, o, r] = np.percentile(x, 25), np.percentile(x, 75)
                F["vx_med"][a, o, r], F["vy_med"][a, o, r] = np.median(vx[sel]), np.median(vy[sel])
                F["v_hist"][a, o, r] = np.minimum(np.histogram(np.minimum(x, C.HIST_EDGES[-1] - 1e-3), C.HIST_EDGES)[0], 65535)
                if r == 3:
                    pool.append((a, x))
    return F, pool


def otsu_threshold(x):
    """Otsu trên giá trị v (L_leg/s). Cắt đuôi ở P99.5 để vài pixel lỗi không kéo giãn trục."""
    if len(x) < 100:
        return np.nan
    hi = np.percentile(x, 99.5)
    if hi <= 0:
        return np.nan
    h, e = np.histogram(np.minimum(x, hi), C.OTSU_BINS, (0, hi))
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h, dtype=float)
    m0 = np.cumsum(h * c)
    tot, mt = w0[-1], m0[-1]
    w1 = tot - w0
    with np.errstate(all="ignore"):
        between = w0 * w1 * (m0 / w0 - (mt - m0) / w1) ** 2
    return float(e[int(np.nanargmax(between)) + 1])


def ratio_from_hist(hist, n, thr):
    """Tỉ lệ pixel v < thr, nội suy tuyến tính trong bin. hist: ...×B, n: ... ."""
    e = C.HIST_EDGES
    thr = min(thr, e[-1])       # Otsu có thể vượt trục histogram (C09_2: 67 L_leg/s); v đã bị chặn ở e[-1] khi đếm
    j = int(np.clip(np.searchsorted(e, thr, side="right") - 1, 0, len(e) - 2))
    below = hist[..., :j].sum(-1).astype(float) + hist[..., j] * (thr - e[j]) / (e[j + 1] - e[j])
    with np.errstate(all="ignore"):
        return np.where(n >= C.MIN_PIXELS, below / np.maximum(n, 1), np.nan).astype(np.float32)


def keypoint_disp(v, a, s_final):
    """Độ dịch tâm 3 điểm bàn chân trong ảnh, đơn vị L_leg, chưa trừ camera, chưa chia Δt. Trả về (clean 2×2, raw 2×2), theo object, x đã nhân s_final."""
    c = v.c
    clean, raw = np.full((2, 2), np.nan), np.full((2, 2), np.nan)
    L = v.l_leg[a] if np.isfinite(v.l_leg[a]) else np.nanmedian(v.l_leg)
    sign = np.array([s_final, 1.0])
    for o in (1, 2):
        pts = FOOT_POINTS[o]
        ok = c.valid[a, pts] & c.valid[a + 1, pts]
        if ok.any():
            clean[o - 1] = (c.kp[a + 1, pts][ok] - c.kp[a, pts][ok]).mean(0) / L * sign
        p0, p1 = v.kp_raw[a, pts], v.kp_raw[a + 1, pts]
        g = np.isfinite(p0).all(1) & np.isfinite(p1).all(1)
        if g.any():
            raw[o - 1] = (p1[g] - p0[g]).mean(0) / L * sign
    return clean, raw


def shifted(Px, sx, sy):
    """Px sau khi trừ độ dịch theo cặp (đơn vị L_leg). sx, sy: mảng N."""
    out = {}
    for a, feet in Px.items():
        out[a] = [dict(p, dx=p["dx"] - sx[a], dy=p["dy"] - sy[a]) for p in feet]
    return out


def _relative(x, group):
    """x / trung vị trượt trong từng nhóm, để chỉ còn biến thiên nhanh theo frame."""
    out = np.full(len(x), np.nan)
    for g in np.unique(group[np.isfinite(x)]):
        w = np.flatnonzero((group == g) & np.isfinite(x))
        if len(w) >= C.DT_REL_WIN:
            s = pd.Series(x[w])
            with np.errstate(all="ignore"):
                out[w] = (s / s.rolling(C.DT_REL_WIN, center=True, min_periods=3).median()).values
    return out


def _rho(a, b):
    from scipy.stats import spearmanr
    w = np.isfinite(a) & np.isfinite(b)
    if w.sum() < 8 or np.ptp(a[w]) == 0 or np.ptp(b[w]) == 0:
        return np.nan
    return float(spearmanr(a[w], b[w])[0])


def decide_dt(v, dt, cam_px):
    """Trả về (dt_mode, lý do, jitter, rho_cam, rho_kp). Dùng kp_raw: kp_clean đã lọc trên trục t_real nên độ dịch của nó tỉ lệ với Δt theo cấu tạo."""
    group = v.c.cont * 100000 + v.shot
    n = v.n
    rel = _relative(dt, group)
    jitter = float(np.nanstd(rel)) if np.isfinite(rel).any() else 0.0
    L = np.where(np.isfinite(v.l_leg), v.l_leg, np.nanmedian(v.l_leg))
    disp = np.full(n, np.nan)
    pts = [20, 21, 22, 23, 24, 25]
    a = np.flatnonzero(np.isfinite(dt))
    if len(a):
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                disp[a] = np.nanmean(np.linalg.norm(v.kp_raw[a + 1][:, pts] - v.kp_raw[a][:, pts], axis=2), axis=1) / L[a]
    rho_kp = _rho(rel, _relative(disp, group))
    moving = np.nanmedian(cam_px) > C.CAM_MOVING if np.isfinite(cam_px).any() else False
    rho_cam = _rho(rel, _relative(cam_px, group)) if moving else np.nan
    if jitter < C.DT_JITTER_MIN:
        return "raw", "khong_rung", jitter, rho_cam, rho_kp
    best = np.nanmax([rho_cam, rho_kp]) if np.isfinite([rho_cam, rho_kp]).any() else np.nan
    if np.isfinite(best) and best >= C.DT_RHO_MIN:
        return "raw", "ti_le_voi_chuyen_dong", jitter, rho_cam, rho_kp
    return "smooth", "khong_ti_le", jitter, rho_cam, rho_kp


def smooth_dt(v, dt):
    """Trung bình trượt DT_WIN cặp trong từng (khối, shot, part), rồi chia lại để giữ nguyên tổng thời gian của nhóm."""
    out = dt.copy()
    group = (v.c.cont * 100000 + v.shot) * 1000 + v.part
    for g in np.unique(group[np.isfinite(dt)]):
        w = np.flatnonzero((group == g) & np.isfinite(dt))
        h = C.DT_WIN // 2
        x = np.array([dt[w[max(0, i - h):i + h + 1]].mean() for i in range(len(w))])
        out[w] = x * dt[w].sum() / x.sum()
    return out


def camera_flags(v, has_flow, kind, inlier, bg, cam_grid, foot_any, slow=None):
    # bg_residual đo trên mọi pixel chi tiết, gồm cả bóng đổ và pixel flow bị làm trơn, nên sàn phải tỉ lệ với độ dịch camera
    """Cờ fit camera theo frame. foot_any N×2: median v cả bàn chân, không lọc, theo Δt đã chọn. Trả về (bad N, reason N uint8, P95).
    Cặp không đo được đã được nội suy nên không tự động là xấu: mặt đường trơn vốn không có chi tiết."""
    n = v.n
    reason = np.zeros(n, np.uint8)
    low = has_flow & (kind == C.GROUND_MOVING) & (inlier < C.CAM_BAD_INLIER)
    reason[low] |= C.CF_INLIER
    p95 = float(np.nanpercentile(bg[has_flow], 95)) if np.isfinite(bg[has_flow]).any() else np.nan
    over = has_flow & (bg > p95)
    floor = np.maximum(C.BG_BAD_FLOOR, C.BG_BAD_REL * cam_grid)
    reason[over & (bg > floor)] |= C.CF_BG
    reason[over & ~(bg > floor)] |= C.CF_BG_BELOW_FLOOR
    group = v.c.cont * 100000 + v.shot
    pan = np.zeros(n, bool)
    for g in np.unique(group[has_flow]):
        w = np.flatnonzero((group == g) & has_flow)
        pan[w] = pd.Series(cam_grid[w]).rolling(C.PAN_WIN, center=True, min_periods=1).median().values > C.CAM_MOVING
    both = np.isfinite(foot_any).all(1)
    tail = both & pan & (((foot_any > C.TAIL_HI).all(1)) | ((foot_any < C.TAIL_LO).all(1)))
    reason[tail] |= C.CF_TAIL
    if slow is not None:
        a = np.abs(slow)
        for g in np.unique(group[has_flow]):
            w = np.flatnonzero((group == g) & has_flow)
            reason[w[(a[w] > C.COMP_SLOW_BAD) & (a[w] > C.COMP_SLOW_BAD_REL * np.median(a[w]))]] |= C.CF_SLOW
    return (reason & C.CF_BAD) != 0, reason, p95
