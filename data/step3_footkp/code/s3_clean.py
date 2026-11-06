"""3.3 Làm sạch điểm bàn chân: đánh dấu outlier theo lý do, NaN hoá, lọc đa thức cục bộ trên trục t_real_s_v2."""
import json

import numpy as np

import s3_common as C
from s3_common import FOOT_IDX, L_LEG, R_LEG, HEEL, BIG


def swap_scores(score, flag):
    out = score.copy()
    out[np.ix_(flag, L_LEG)], out[np.ix_(flag, R_LEG)] = score[np.ix_(flag, R_LEG)], score[np.ix_(flag, L_LEG)]
    return out


def L_of(c, kp):
    """L_leg theo thời gian, tính riêng từng khối."""
    L = np.full(c.n, np.nan)
    for b in np.unique(c.cont):
        w = c.cont == b
        L[w] = C.leg_length_t(kp[w], c.t[w])
        if not np.isfinite(L[w]).all():
            L[w] = np.where(np.isfinite(L[w]), L[w], C.leg_length(kp[w]))
    return L


def foot_velocity(c, kp, L):
    """Tốc độ từng bước (frame i-1 -> i) của 8 điểm bàn chân, trừ tâm hông, đơn vị L_leg/s thật.
    Trả về v (N×8, NaN qua ranh giới khối/part/khoảng trống) và tốc độ cục bộ M (N)."""
    hip = (kp[:, 11] + kp[:, 12]) / 2
    r = kp[:, FOOT_IDX] - hip[:, None]
    v = np.full((c.n, 8), np.nan)
    for seg in C.segments(c):
        if len(seg) < 2:
            continue
        a, b = seg[:-1], seg[1:]
        dt = c.t[b] - c.t[a]
        ok = dt > 0
        v[b[ok]] = np.linalg.norm(r[b[ok]] - r[a[ok]], axis=-1) / dt[ok, None] / L[b[ok], None]
    M = np.full(c.n, np.nan)
    for seg in C.segments(c):
        for pos, i in enumerate(seg):
            w = seg[max(0, pos - C.VEL_LOCAL_HALF):pos + C.VEL_LOCAL_HALF + 1]
            x = v[w].ravel()
            x = x[np.isfinite(x)]
            if len(x) >= 8:
                M[i] = np.median(x)
    return v, M


def velocity_stats(clips):
    """Bước 1 (toàn dataset): trả về ngưỡng + giá trị đối chiếu. clips: (c, kp_fixed, score_fixed)."""
    allv, alln, kc, ms, ramp, abso = [], [], [], [], [], []
    for c, kp, sc in clips:
        L = L_of(c, kp)
        v, M = foot_velocity(c, kp, L)
        good = sc[:, FOOT_IDX] >= C.SCORE_MIN
        cao = np.repeat((c.k_conf == "cao")[:, None], 8, 1)
        rp = np.repeat((c.time_model == "ramp")[:, None], 8, 1)
        ok_abs = np.full((c.n, 8), c.meta.abs_time_ok == "True")
        ms.append(M[c.k_conf == "cao"])
        allv.append(v[good])
        alln.append(np.where(np.isfinite(M[:, None]), v / M[:, None], np.nan)[good])
        kc.append(cao[good])
        ramp.append(rp[good])
        abso.append(ok_abs[good])
    allv, alln, kc, ramp, abso = map(np.concatenate, (allv, alln, kc, ramp, abso))
    Mref = float(np.nanmedian(np.concatenate(ms)))
    pct = C.VEL_PCT
    vn = alln * Mref
    d = dict(percentile=pct, M_ref_L_per_s=round(Mref, 3),
             thr_L_per_s=round(float(np.nanpercentile(allv, pct)), 2),
             thr_neu_chuan_hoa_cuc_bo_L_per_s=round(float(np.nanpercentile(vn, pct)), 2),
             literal_k_cao_L_per_s=round(float(np.nanpercentile(allv[kc], pct)), 2),
             literal_k_khac_L_per_s=round(float(np.nanpercentile(allv[~kc], pct)), 2),
             literal_ramp_L_per_s=round(float(np.nanpercentile(allv[ramp], pct)), 2),
             literal_abs_true_L_per_s=round(float(np.nanpercentile(allv[abso], pct)), 2),
             literal_abs_false_khong_ramp_L_per_s=round(float(np.nanpercentile(allv[~abso & ~ramp], pct)), 2),
             n_values=int(np.isfinite(vn).sum()))
    C.VEL_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return d


def mark_outliers(c, kp, sc, dref, thr):
    """kp, sc đã sửa swap. Trả về reason (N×26 uint8) chỉ cho 8 điểm bàn chân."""
    reason = np.zeros((c.n, 26), np.uint8)
    F = np.array(FOOT_IDX)
    finite = np.isfinite(kp[:, :, 0])
    L = L_of(c, kp)

    # (a) score
    reason[:, F] |= ((sc[:, F] < C.SCORE_MIN) & finite[:, F]).astype(np.uint8) * C.R_SCORE

    # (b) độ dài heel->big toe so với trung vị của chân đó trong khối, chuẩn hoá L_leg(t) vì người lớn dần trong khối
    for leg in (L_LEG, R_LEG):
        h, b = leg[HEEL], leg[BIG]
        r = np.linalg.norm(kp[:, h] - kp[:, b], axis=1) / L
        for blk in np.unique(c.cont):
            w = c.cont == blk
            good = w & np.isfinite(r) & (sc[:, h] >= C.SCORE_MIN) & (sc[:, b] >= C.SCORE_MIN)
            if good.sum() < 5:
                continue
            bad = w & np.isfinite(r) & (np.abs(r / np.median(r[good]) - 1) > C.HEEL_TOE_DEV)
            reason[bad, h] |= C.R_RATIO
            reason[bad, b] |= C.R_RATIO

    # (c) vận tốc: spike/plateau = 2 bước nhảy gần nhau; bước nhảy đơn lẻ không đánh dấu (có thể là đổi trạng thái thật)
    vn, _ = foot_velocity(c, kp, L)       # v2: t_real_s_v2 đã đồng nhất thang thời gian, so trực tiếp L_leg/s
    hip = (kp[:, 11] + kp[:, 12]) / 2
    for seg in C.segments(c):
        for q, j in enumerate(FOOT_IDX):
            jump = np.flatnonzero(vn[seg, q] > thr)
            rel = kp[seg, j] - hip[seg]
            for a, b in zip(jump[:-1], jump[1:]):
                # đi rồi quay lại: rời quỹ đạo ở bước a, trở về ở bước b. Hai cú nhảy độc lập không tính.
                if b - a <= C.PLATEAU_MAX:
                    away = np.linalg.norm(rel[a] - rel[a - 1])
                    back = np.linalg.norm(rel[b] - rel[a - 1])
                    if back < 0.5 * away:
                        reason[seg[a:b], j] |= C.R_VEL
            # bước nhảy sát đầu/cuối đoạn: frame mép không có bước còn lại để xác nhận
            if len(jump) and jump[0] <= 1:
                reason[seg[:jump[0]], j] |= C.R_VEL
            if len(jump) and jump[-1] >= len(seg) - 2:
                reason[seg[jump[-1]:], j] |= C.R_VEL

    return reason


def collapse_frames(c, kp, dref):
    """2 chân chập vào 1: 2 cổ chân gần trùng trong khi tham chiếu pha nói phải cách xa. Không hoán đổi nào sửa được."""
    L = L_of(c, kp)
    a = np.linalg.norm(kp[:, 15] - kp[:, 16], axis=1) / L
    return (a < C.COLLAPSE_DIST) & (np.abs(dref) > C.COLLAPSE_REF)


def smooth_segment(t, y):
    """Savitzky–Golay tổng quát cho trục không đều: mỗi điểm khớp đa thức bậc SG_ORDER trên SG_WINDOW điểm gần nhất."""
    n = len(t)
    out = np.empty_like(y)
    w = C.SG_WINDOW
    for i in range(n):
        a = min(max(0, i - w // 2), n - w)
        tt = t[a:a + w]
        s = tt[-1] - tt[0]
        x = (tt - t[i]) / s
        A = np.vander(x, C.SG_ORDER + 1, increasing=True)
        coef = np.linalg.lstsq(A, y[a:a + w], rcond=None)[0]
        out[i] = coef[0]                                       # x = 0 tại t[i]
    return out


def clean_and_filter(c, kp, valid):
    """kp đã sửa swap, valid N×26. Lọc từng điểm từng đoạn; nối khoảng trống ngắn (<=3 frame) bằng tuyến tính, đánh dấu riêng."""
    out = np.full_like(kp, np.nan)
    interp = np.zeros(valid.shape, bool)
    sm = np.zeros(valid.shape, bool)
    for j in range(26):
        for seg in C.segments(c, valid[:, j]):
            y = kp[seg, j]
            out[seg, j] = y
            if len(seg) >= C.SG_WINDOW:
                out[seg, j] = smooth_segment(c.t[seg], y)
                sm[seg, j] = True
            # khoảng trống <= 3 frame trong đoạn: nội suy trên giá trị đã lọc, valid vẫn False
            for a, b in zip(seg[:-1], seg[1:]):
                if b - a > 1:
                    tt = (c.t[a + 1:b] - c.t[a]) / (c.t[b] - c.t[a])
                    out[a + 1:b, j] = out[a, j] + tt[:, None] * (out[b, j] - out[a, j])
                    interp[a + 1:b, j] = True
    return out, interp, sm
