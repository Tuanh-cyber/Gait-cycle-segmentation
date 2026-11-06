"""3.1 Sửa đảo nhãn L/R theo thời gian bằng Viterbi 2 trạng thái {giữ, hoán đổi}.

Sai lệch dự đoán dùng toạ độ trừ tâm hông (tâm hông không đổi khi hoán đổi nhãn, nên loại được chuyển
động camera + người chạy mà không phụ thuộc vào nhãn). Dự đoán tuyến tính cần 2 frame trước, nên trạng
thái Viterbi là cặp (S[t-1], S[t]): nghiệm tối ưu thật, không phải xấp xỉ tham lam.

Ranh giới speed_part_id: không dùng vận tốc băng qua ranh giới (dự đoán = vị trí frame trước).

Prior lệch pha: chênh x cổ chân d(t) đổi dấu 2 lần/stride, nên d² có chu kỳ = 1 bước và KHÔNG đổi khi hoán
đổi nhãn. Giải điều chế d² cho pha θ(t) mà không dùng nhãn nào; d_ref(t) = A(t)·cos θ(t) là tham chiếu dấu
cho từng frame. Dấu tổng thể (θ hay θ+π) lấy theo lượt Viterbi đầu, tính riêng mỗi speed part.
"""
import numpy as np

from s3_common import (ANKLE, DIST_CAP, L_LEG, MAX_GAP_FRAMES, PHASE_R_MIN, PHASE_R_MIN_RAMP, R_LEG, STRIDE_S_MAX, STRIDE_S_MIN, SWITCH_COST,
                       W_PHASE, leg_length_t)


def _dist(a, b, L):
    d = np.linalg.norm(a - b, axis=-1) / L
    return np.minimum(np.nan_to_num(d, nan=0.0), DIST_CAP)


def _steps(c, idx):
    """Với dãy chỉ số frame có keypoint trong 1 khối: bước t cần prev, prev2 và tỉ lệ thời gian (0 = chỉ vị trí)."""
    t, part, uid = c.t, c.part, c.unique_id
    steps = []
    for a in range(1, len(idx)):
        i, p = idx[a], idx[a - 1]
        gap = uid[i] - uid[p] - 1
        r, p2 = 0.0, -1
        if a >= 2:
            q = idx[a - 2]
            ok = part[i] == part[p] == part[q] and gap <= MAX_GAP_FRAMES and uid[p] - uid[q] - 1 <= MAX_GAP_FRAMES
            if ok and t[p] > t[q]:
                r, p2 = (t[i] - t[p]) / (t[p] - t[q]), q
        steps.append((i, p, p2, r))
    return steps


def viterbi_block(c, idx, run_sign, ref=None, switch=SWITCH_COST, w_phase=W_PHASE):
    """idx: chỉ số frame (có keypoint) của 1 khối. Trả về S (0/1) theo idx và chi phí từng frame."""
    n = len(idx)
    if n < 3:
        return np.zeros(n, int), np.zeros(n)
    kp, score = c.kp, c.score
    hip = (kp[:, 11] + kp[:, 12]) / 2
    Lt = leg_length_t(kp[idx], c.t[idx])
    Lt = np.where(np.isfinite(Lt) & (Lt > 1), Lt, np.nanmedian(Lt))
    Lmap = dict(zip(idx.tolist(), Lt))
    # P[i, s, leg, j, xy]: chân thứ `leg` ở trạng thái s, trừ tâm hông
    L, R = kp[:, L_LEG] - hip[:, None], kp[:, R_LEG] - hip[:, None]
    Ws = np.minimum(np.clip(score[:, L_LEG], 0, 1), np.clip(score[:, R_LEG], 0, 1))
    leg = lambda i, s: (L[i], R[i]) if s == 0 else (R[i], L[i])

    def unary(a, i, s):
        """Prior lệch pha: dấu chênh x cổ chân khớp tham chiếu d_ref."""
        if ref is None or not np.isfinite(ref[a]):
            return 0.0
        d = run_sign * (kp[i, 15, 0] - kp[i, 16, 0]) / Lmap[i] * (1 - 2 * s)
        w = min(Ws[i, ANKLE], 1.0)
        return w_phase * w * abs(d - ref[a]) if np.isfinite(d) else 0.0

    INF = 1e12
    # trạng thái (s_prev, s_cur) -> chi phí; khởi tạo: frame đầu giữ nhãn (S chỉ có nghĩa tương đối)
    cost = np.full((n, 2, 2), INF)
    back = np.zeros((n, 2, 2), int)
    for s0 in (0, 1):
        for s1 in (0, 1):
            cost[1, s0, s1] = unary(0, idx[0], s0) + unary(1, idx[1], s1) + (switch if s0 != s1 else 0.0)
    step = _steps(c, idx)
    for a, (i, p, p2, r) in enumerate(step[1:], start=2):
        Li = Lmap[i]
        w = np.minimum(Ws[i], Ws[p])
        if p2 >= 0:
            w = np.minimum(w, Ws[p2])
        for s_prev in (0, 1):
            Ap, Bp = leg(p, s_prev)
            for s_cur in (0, 1):
                At, Bt = leg(i, s_cur)
                best, arg = INF, 0
                for s_pp in (0, 1):
                    if cost[a - 1, s_pp, s_prev] >= INF:
                        continue
                    if p2 >= 0 and r > 0:
                        App, Bpp = leg(p2, s_pp)
                        PA, PB = Ap + r * (Ap - App), Bp + r * (Bp - Bpp)
                    else:
                        PA, PB = Ap, Bp
                    obs = (w * (_dist(At, PA, Li) + _dist(Bt, PB, Li))).sum() / 12.0
                    tot = cost[a - 1, s_pp, s_prev] + obs + unary(a, i, s_cur) + (switch if s_cur != s_prev else 0.0)
                    if tot < best:
                        best, arg = tot, s_pp
                cost[a, s_prev, s_cur] = best
                back[a, s_prev, s_cur] = arg
    S = np.zeros(n, int)
    s_prev, s_cur = np.unravel_index(np.argmin(cost[n - 1]), (2, 2))
    S[n - 1], S[n - 2] = s_cur, s_prev
    for a in range(n - 1, 1, -1):
        s_pp = back[a, S[a - 1], S[a]]
        S[a - 2] = s_pp
    return S, cost[n - 1].min()


def phase_reference(c, idx, run_sign):
    """Tham chiếu pha mỗi speed part. Trả về (|d_ref| có dấu theo θ, R độ tuần hoàn, chu kỳ stride mỗi part)."""
    Lt = leg_length_t(c.kp[idx], c.t[idx])
    d = run_sign * (c.kp[idx, 15, 0] - c.kp[idx, 16, 0]) / Lt
    ref, R_all, info = np.full(len(idx), np.nan), np.zeros(len(idx)), {}
    grid = np.geomspace(STRIDE_S_MIN, STRIDE_S_MAX, 240)
    for p in np.unique(c.part[idx]):
        w = np.flatnonzero((c.part[idx] == p) & np.isfinite(d))
        if len(w) < 12:
            continue
        t, e = c.t[idx][w], d[w] ** 2
        e0 = e - e.mean()
        grid_p = grid[grid <= (t[-1] - t[0]) * 1.5]          # đoạn phải dài >= 2/3 stride
        if not len(grid_p) or e0.std() < 1e-6:
            continue
        P = np.abs(np.exp(-4j * np.pi * np.outer(1 / grid_p, t)) @ e0)
        T = grid_p[np.argmax(P)]
        R = 2 * P.max() / len(t) / (e0.std() * np.sqrt(2))   # sóng sin thuần cho R = 1
        # part ramp đã được Bước 1b làm đều thời gian và đã kiểm CV, nên R thấp ở đó là do nhiễu, không phải do tốc độ đổi
        used = R >= (PHASE_R_MIN_RAMP if c.time_model[idx][w[0]] == "ramp" else PHASE_R_MIN)
        info[int(p)] = dict(T=float(T), R=float(R), used=bool(used))
        R_all[w] = R
        if not used:
            continue
        sig = T                                               # cửa sổ Gauss σ = 1 stride: pha chỉ trôi chậm
        g = np.exp(-0.5 * ((t[:, None] - t[None]) / sig) ** 2)
        z = g @ (e0 * np.exp(-4j * np.pi * t / T))
        th2 = 4 * np.pi * t / T + np.unwrap(np.angle(z))
        A = np.sqrt(np.maximum(2 * (g @ e) / g.sum(1), 1e-9))
        ref[w] = A * np.cos(th2 / 2)
    return ref, R_all, info


def _orient(c, idx, ref, S, run_sign):
    """Chọn dấu tổng thể của tham chiếu từng part theo lượt đầu."""
    Lt = leg_length_t(c.kp[idx], c.t[idx])
    d = run_sign * (c.kp[idx, 15, 0] - c.kp[idx, 16, 0]) / Lt * (1 - 2 * S)
    out = ref.copy()
    for p in np.unique(c.part[idx]):
        w = (c.part[idx] == p) & np.isfinite(ref) & np.isfinite(d)
        if w.any() and np.sum(d[w] * ref[w]) < 0:
            out[(c.part[idx] == p)] *= -1
    return out


def fix_swaps(c, run_sign, use_phase=True, **kw):
    """swap_flag từng frame. Mỗi khối chuẩn hoá để đa số frame giữ nhãn gốc: A = chân mang nhãn L đa số.
    Trả về (flag, d_ref theo định hướng cuối, thông tin từng khối)."""
    flag = np.zeros(c.n, bool)
    dref = np.full(c.n, np.nan)
    has = np.isfinite(c.kp[:, L_LEG + R_LEG, 0]).any(1)
    info = {}
    for b in np.unique(c.cont):
        idx = np.flatnonzero((c.cont == b) & has)
        if len(idx) == 0:
            continue
        S, _ = viterbi_block(c, idx, run_sign, **kw)
        ref, R_all, pinfo = phase_reference(c, idx, run_sign) if len(idx) >= 12 else (np.full(len(idx), np.nan), None, {})
        if use_phase and np.isfinite(ref).any():
            ref = _orient(c, idx, ref, S, run_sign)
            S, _ = viterbi_block(c, idx, run_sign, ref=ref, **kw)
            ref = _orient(c, idx, ref, S, run_sign)
        if S.mean() > 0.5:
            S = 1 - S
            ref = -ref
        flag[idx] = S.astype(bool)
        dref[idx] = ref
        info[int(b)] = pinfo
    return flag, dref, info


def apply_swap(kp, flag):
    out = kp.copy()
    out[np.ix_(flag, L_LEG)], out[np.ix_(flag, R_LEG)] = kp[np.ix_(flag, R_LEG)], kp[np.ix_(flag, L_LEG)]
    return out
