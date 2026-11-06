"""Bù camera trực tiếp: tịnh tiến của mặt đất ngay quanh bàn chân, chỉ từ pixel có chi tiết.

Mặt đường ít chi tiết làm flow nằm gần 0 dù camera đang bám theo người chạy: vạch dọc đường bị khẩu độ, nền mờ bị làm trơn. Chỉ pixel có cả 2 hướng gradient
(vạch ngang, mép dấu sơn, sỏi) mới cho độ dịch thật. Chữ gắn trên ảnh cũng có chi tiết nhưng đứng yên, nên được tách bằng độ dịch."""
import cv2
import numpy as np
import pandas as pd

import s5_common as C


def leg_capsules(v, a, s, crop, sbox, shape, lf):
    """Mặt nạ ống chân (cổ chân → gối) trên hộp kho, để không lấy pixel chân làm mặt đất."""
    x0, y0, x1, y1 = sbox
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    org = np.array(crop[:2], float)
    th = max(3, int(C.GROUND_LEG_RADIUS * lf * 2))
    for o in (1, 2):
        ank, knee = C.S4.FOOT[o][C.S4.ANK], C.S4.KNEE[o]
        for t in (a, a + 1):
            p, q = v.c.kp[t, ank], v.c.kp[t, knee]
            if np.isfinite(p).all() and np.isfinite(q).all():
                P, Q = (p - org) * s - [x0, y0], (q - org) * s - [x0, y0]
                cv2.line(m, tuple(np.round(P).astype(int).tolist()), tuple(np.round(Q).astype(int).tolist()), 1, th)
    return m > 0


def strip_mask(v, a, s, crop, sbox, shape, feet):
    """Dải đất trên hộp kho: từ trên điểm chân thấp nhất GROUND_ABOVE × L_foot tới dưới GROUND_BELOW × L_foot, trừ mask chân đã nới và ống chân. Không lấy vùng xa hơn."""
    x0, y0, x1, y1 = sbox
    pts = [C.S4.foot_pts(v.c, a, o) for o in (1, 2)]
    pts = [p for p in pts if len(p)]
    out = np.zeros((y1 - y0, x1 - x0), bool)
    if not pts:
        return out
    lf = float(np.nanmin(v.z["l_foot"][a])) * s
    yg = (np.vstack(pts)[:, 1].max() - crop[1]) * s - y0
    out[int(max(0, yg - C.GROUND_ABOVE * lf)):int(max(0, yg + C.GROUND_BELOW * lf))] = True
    k = max(3, 2 * int(C.MASK_DILATE * lf) + 1)
    out &= cv2.dilate(feet[y0:y1, x0:x1].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) == 0
    out &= ~leg_capsules(v, a, s, crop, sbox, shape, lf)
    return out


def _cluster(fx, fy, rng):
    n = len(fx)
    best, arg = -1, 0
    for i in rng.choice(n, min(n, 200), replace=False):
        k = int(((fx - fx[i]) ** 2 + (fy - fy[i]) ** 2 <= C.GROUND_TOL ** 2).sum())
        if k > best:
            best, arg = k, i
    return (fx - fx[arg]) ** 2 + (fy - fy[arg]) ** 2 <= C.GROUND_TOL ** 2


def consensus(fx, fy, rng):
    """Tối đa 3 cụm độ dịch; trong các cụm đủ lớn, lấy cụm có độ dịch lớn nhất, vì làm trơn của flow chỉ kéo mặt đất về 0, không đẩy ra xa. Trả về (tx, ty, số inlier)."""
    found, rest = [], np.ones(len(fx), bool)
    for _ in range(3):
        if rest.sum() < C.GROUND_MIN_POINTS:
            break
        idx = np.flatnonzero(rest)
        w = _cluster(fx[idx], fy[idx], rng)
        found.append((float(np.median(fx[idx][w])), float(np.median(fy[idx][w])), int(w.sum())))
        rest[idx[w]] = False
    if not found:
        return 0.0, 0.0, 0
    top = max(f[2] for f in found)
    ok = [f for f in found if f[2] >= max(C.GROUND_MIN_POINTS, C.GROUND_CLUSTER_FRAC * top)]
    return max(ok, key=lambda f: np.hypot(f[0], f[1])) if ok else max(found, key=lambda f: f[2])


def overlay_map(records, H, W):
    """Mặt nạ chữ gắn trên ảnh, trong toạ độ ảnh gốc, chia ô. Ô có pixel chi tiết đứng yên lặp lại ở nhiều cặp là chữ.
    records: danh sách (gray, flow, sbox, crop, scale) trên hộp kho. Trả về (mặt nạ, camera_static). Camera đứng yên thì mọi nền đều đứng yên, không tách được, trả về rỗng."""
    cell = C.OVERLAY_CELL
    count = np.zeros((H // cell + 1, W // cell + 1), np.int32)
    dense = []
    for gray, flow, strip, sb, crop, s in records:
        me = cv2.cornerMinEigenVal(gray, 5, ksize=3)
        mag = np.hypot(flow[..., 0], flow[..., 1])
        n = int(strip.sum())
        dense.append(n > 0 and ((me > C.GROUND_TEX_LOW) & (mag < C.GROUND_STATIC) & strip).sum() >= C.GROUND_STATIC_DENSITY * n)
        # chữ có độ tương phản cao và đứng yên tuyệt đối; mặt đất ít chi tiết có flow nhỏ do làm trơn không được tính
        ys, xs = np.nonzero((me > C.GROUND_TEX) & (mag < C.OVERLAY_STATIC_PX))
        if not len(xs):
            continue
        cx = ((xs + sb[0]) / s + crop[0]).astype(int) // cell
        cy = ((ys + sb[1]) / s + crop[1]).astype(int) // cell
        here = np.zeros_like(count)
        np.add.at(here, (np.clip(cy, 0, count.shape[0] - 1), np.clip(cx, 0, count.shape[1] - 1)), 1)
        count += here >= C.OVERLAY_MIN_PIXELS
    static_cam = bool(dense) and np.mean(dense) >= C.CAM_STATIC_SHARE
    if static_cam:
        return np.zeros_like(count, bool), True
    m = count >= C.OVERLAY_MIN_PAIRS
    k = 2 * C.OVERLAY_DILATE + 1
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0, False


def overlay_on_box(overlay, sb, crop, s, shape):
    """Chiếu mặt nạ chữ lên hộp kho của 1 cặp."""
    cell = C.OVERLAY_CELL
    h, w = shape
    xs = (((np.arange(w) + sb[0]) / s + crop[0]).astype(int) // cell).clip(0, overlay.shape[1] - 1)
    ys = (((np.arange(h) + sb[1]) / s + crop[1]).astype(int) // cell).clip(0, overlay.shape[0] - 1)
    return overlay[np.ix_(ys, xs)]


def estimate_pair(gray, flow, strip, rng):
    """Ước lượng của 1 cặp. Trả về dict: kind 'moving', 'static' hoặc 'none'; tx, ty (lưới flow); n_tex, n_inlier, source."""
    me = cv2.cornerMinEigenVal(gray, 5, ksize=3)
    n_strip = int(strip.sum())
    none = dict(kind="none", tx=np.nan, ty=np.nan, n_tex=0, n_inlier=0, source="none", inlier_frac=np.nan)
    if not n_strip:
        return none
    mag = np.hypot(flow[..., 0], flow[..., 1])
    low = strip & (me > C.GROUND_TEX_LOW)
    n_static = int((low & (mag < C.GROUND_STATIC)).sum())
    static_dense = n_static >= C.GROUND_STATIC_DENSITY * n_strip     # chữ gắn trên ảnh chỉ chiếm phần rất nhỏ của dải đất
    for th, name in ((C.GROUND_TEX, "tex_high"), (C.GROUND_TEX_LOW, "tex_low")):
        sel = strip & (me > th) & (mag >= C.GROUND_STATIC)
        if int(sel.sum()) >= C.GROUND_MIN_POINTS:
            tx, ty, k = consensus(flow[..., 0][sel], flow[..., 1][sel], rng)
            if k >= C.GROUND_MIN_POINTS and not (static_dense and k < C.GROUND_MIN_FRAC * n_static):
                return dict(kind="moving", tx=tx, ty=ty, n_tex=int(low.sum()), n_inlier=k, source=name, inlier_frac=k / int(sel.sum()))
    if static_dense:
        return dict(kind="static", tx=0.0, ty=0.0, n_tex=int(low.sum()), n_inlier=n_static, source="static", inlier_frac=1.0)
    # ít chi tiết: dùng cả dải nếu flow trên đó đang chuyển động nhất quán (C13_2); nếu gần 0 thì không tin được
    fx, fy = flow[..., 0][strip], flow[..., 1][strip]
    if np.median(np.hypot(fx, fy)) >= C.GROUND_STATIC:
        tx, ty, k = consensus(fx, fy, rng)
        if k >= C.GROUND_ALL_FRAC * n_strip:
            return dict(kind="moving", tx=tx, ty=ty, n_tex=int(low.sum()), n_inlier=k, source="all_strip", inlier_frac=k / n_strip)
    return dict(none, n_tex=int(low.sum()))


def temporal(est, t, group):
    """Điền các cặp không đo được bằng nội suy tuyến tính theo thời gian trong từng khối.
    est: dict mảng kind (0 moving, 1 static, 2 none), tx, ty. Trả về tx, ty đã điền và source (0 đo được, 1 điền từ lân cận, 2 không có nguồn nào)."""
    kind, tx, ty = est["kind"], est["tx"].copy(), est["ty"].copy()
    src = np.full(len(kind), 2, np.uint8)
    for g in np.unique(group[np.isfinite(t)]):
        w = np.flatnonzero((group == g) & np.isfinite(t))
        if not len(w):
            continue
        k = kind[w]
        moving = k == 0
        # video có nền đứng yên thật thì đa số cặp là static; chỉ coi static là chữ khi đa số cặp xung quanh đang đo được chuyển động
        share = pd.Series(moving.astype(float)).rolling(C.GROUND_WIN, center=True, min_periods=1).mean().values
        keep = moving | ((k == 1) & (share < C.GROUND_MOVING_SHARE))
        tt = t[w]
        # camera đổi hướng hay giảm hơn nửa tốc độ trong vài frame là không hợp lý; đó là cụm bám vào nền xa hoặc flow bị làm trơn.
        # Không loại phía trên: frame bị bỏ làm độ dịch nhảy gấp đôi là thật.
        if moving.sum() >= 3:
            sx, sy = tx[w].copy(), ty[w].copy()
            sx[~moving], sy[~moving] = np.nan, np.nan
            mx = pd.Series(sx).rolling(C.GROUND_WIN, center=True, min_periods=1).median().values
            my = pd.Series(sy).rolling(C.GROUND_WIN, center=True, min_periods=1).median().values
            big = np.hypot(mx, my) > C.GROUND_STATIC
            with np.errstate(all="ignore"):
                along = (sx * mx + sy * my) / np.maximum(mx * mx + my * my, 1e-6)
            drop = moving & big & (along < C.GROUND_MIN_ALONG)
            moving = moving & ~drop
            k = np.where(drop, C.GROUND_NONE, k)
        # chuỗi đo thưa: 1 điểm đo sai sẽ lan ra cả đoạn nội suy, nên lọc bằng trung vị các điểm đo lân cận. Chuỗi dày thì giữ nguyên, vì bước nhảy thật (frame bị bỏ) có ở đó.
        if moving.mean() < C.SPARSE_SHARE and moving.sum() >= 3:
            j = w[moving]
            for arr in (tx, ty):
                arr[j] = pd.Series(arr[j]).rolling(C.SPARSE_WIN, center=True, min_periods=1).median().values
        for arr in (tx, ty):
            y = arr[w].copy()
            if keep.any():
                # nội suy tuyến tính theo thời gian, giữ giá trị 2 đầu
                y[~keep] = np.interp(tt[~keep], tt[keep], y[keep])
                arr[w] = y
            else:
                arr[w] = np.nan
        src[w] = np.where(keep, 0, 1 if keep.any() else 2)
    return tx, ty, src


def background_residual(flow, gray, strip, tx, ty, moving):
    """Trung vị |flow − tịnh tiến| trên pixel có chi tiết của dải đất; khi camera chuyển động thì bỏ pixel đứng yên (chữ). NaN nếu không có pixel."""
    sel = strip & (cv2.cornerMinEigenVal(gray, 5, ksize=3) > C.GROUND_TEX_LOW)
    if moving:
        sel &= np.hypot(flow[..., 0], flow[..., 1]) >= C.GROUND_STATIC
    if not sel.any():
        return np.nan
    return float(np.median(np.hypot(flow[..., 0][sel] - tx, flow[..., 1][sel] - ty)))


def slow_component(t, vx_all, frames, T, blocks):
    """Thành phần nền còn thiếu, biến thiên chậm hơn 1 stride: spline bậc 3 có nút cách T, fit theo P10 của vx trong cửa sổ T.
    t: N; vx_all: N×2 vx cả bàn chân, NaN ở chân không dùng được; frames: chỉ số frame có flow; blocks: nhóm khối. Trả về c N (L_leg/s), NaN ở frame không có flow."""
    from scipy.interpolate import LSQUnivariateSpline
    c = np.full(len(t), np.nan)
    for b in np.unique(blocks[frames]):
        w = frames[blocks[frames] == b]
        tt = t[w]
        y = np.full(len(w), np.nan)
        for i, ti in enumerate(tt):
            x = vx_all[w[np.abs(tt - ti) <= T / 2]]
            x = x[np.isfinite(x)]
            if len(x) >= 4:
                y[i] = np.percentile(x, 10)
        g = np.isfinite(y)
        if not g.any():
            continue
        span = tt.max() - tt.min()
        knots = np.arange(tt[g].min() + T, tt[g].max() - T / 2, T)
        # cần đủ điểm và thoả điều kiện Schoenberg–Whitney; không thì dùng hằng số của khối, vẫn không nhanh hơn 1 stride
        if span >= 2 * T and g.sum() >= len(knots) + 5:
            try:
                # spline có thể vượt xa dữ liệu ở mép khối; không cho hiệu chỉnh ra ngoài khoảng P10 đã đo
                c[w] = np.clip(LSQUnivariateSpline(tt[g], y[g], knots, k=3)(tt), y[g].min(), y[g].max())
                continue
            except ValueError:
                pass
        c[w] = float(np.median(y[g]))
    return c
