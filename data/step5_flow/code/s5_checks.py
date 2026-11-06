"""Mục 7: Spearman với tốc độ keypoint, Lomb–Scargle so với cadence, near/far luân phiên; gate."""
import numpy as np
from scipy.signal import find_peaks, lombscargle
from scipy.stats import spearmanr

import s5_common as C

HEEL, FOOT = 0, 3

def _rho(a, b):
    w = np.isfinite(a) & np.isfinite(b)
    if w.sum() < 10 or np.ptp(a[w]) == 0 or np.ptp(b[w]) == 0:
        return np.nan, int(w.sum())
    return float(spearmanr(a[w], b[w])[0]), int(w.sum())


def slots_by_label(label):
    """Vị trí object theo frame: (first, second) N, first = near hoặc A."""
    first = np.where(np.isin(label[:, 0], ("near", "A")), 0, 1)
    return first, 1 - first


def periodogram(t, y, block):
    """Công suất Lomb–Scargle chuẩn hoá, tính riêng từng khối rồi cộng theo số frame, vì pha không liên tục qua các khối."""
    f = np.arange(C.LS_F_MIN, C.LS_F_MAX + 1e-9, C.LS_DF)
    P, W = np.zeros_like(f), 0
    for b in np.unique(block):
        w = (block == b) & np.isfinite(y)
        if w.sum() < C.LS_MIN_FRAMES or t[w].max() - t[w].min() < C.LS_MIN_SPAN or np.ptp(y[w]) == 0:
            continue
        P += w.sum() * lombscargle(t[w], y[w] - y[w].mean(), 2 * np.pi * f, normalize=True)
        W += w.sum()
    return f, (P / W if W else None), W


def cadence_parts(d, prefix, cadence, part_T):
    """Lomb–Scargle của slow_ratio gót riêng từng speed_part. Video nhiều part so với chu kỳ của part đó ở Bước 3; video 1 part so với tham chiếu của video."""
    first, second = slots_by_label(d["label"])
    i = np.arange(len(first))
    slow = d[prefix + "slow_ratio"][:, :, HEEL].copy()
    slow[d["exclude_time_features"]] = np.nan
    t, block, part = d["t_real_s_v2"], d["continuity_id"] * 100000 + d["shot_id"], d["speed_part_id"]
    ids = np.unique(part)
    out = []
    for p in ids:
        w = part == p
        P, W = None, 0
        for o in (first, second):
            y = np.where(w, slow[i, o], np.nan)
            f, q, n = periodogram(t, y, block)
            if q is not None:
                P = q * n if P is None else P + q * n
                W += n
        spans = [np.ptp(t[(block == b) & w]) for b in np.unique(block[w]) if ((block == b) & w).sum() >= 2]
        span = float(max(spans)) if spans else 0.0
        ref = cadence
        src = "video"
        if len(ids) > 1 and int(p) in part_T:
            ref, src = 120.0 / part_T[int(p)], "part_T"
        row = dict(part=int(p), n=int(W), span_s=span, ref=ref, ref_source=src, cadence_flow=np.nan, ls_power=np.nan, dev=np.nan, dev_spm=np.nan, resolution_spm=120 / span if span > 0 else np.nan)
        if P is not None:
            P = P / W
            interior = np.flatnonzero((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])) + 1
            if len(interior):
                k = int(interior[np.argmax(P[interior])])
                cf = float(120 * f[k])
                row.update(cadence_flow=cf, ls_power=float(P[k]))
                if np.isfinite(ref):
                    row.update(dev=abs(cf - ref) / ref, dev_spm=abs(cf - ref))
        out.append(row)
    return out


def minima_per_stride(d, prefix, first, second, cadence, part_T):
    """Số cực tiểu của |v| cả bàn chân trên mỗi stride, trên mỗi chân; và công suất tại 2f so với tại f. Dùng chu kỳ tham chiếu, không dùng flow."""
    from scipy.signal import lombscargle as ls
    T_ref = 120.0 / cadence if np.isfinite(cadence) else np.nan
    if not np.isfinite(T_ref):
        return np.nan, np.nan
    t, block = d["t_real_s_v2"], d["continuity_id"] * 100000 + d["shot_id"]
    v = d[prefix + "v_med"][:, :, FOOT]
    i = np.arange(len(t))
    count, total = 0, 0.0
    pf, p2 = 0.0, 0.0
    for o in (first, second):
        s = v[i, o]
        for b in np.unique(block):
            w = np.flatnonzero((block == b) & ~d["exclude_time_features"])
            if len(w) < 8:
                continue
            g = np.isfinite(s[w])
            if g.sum() < 8:
                continue
            tt, y = t[w], np.interp(t[w], t[w][g], s[w][g])
            span = tt[-1] - tt[0]
            if span < 0.5 * T_ref:
                continue
            uni = np.arange(tt[0], tt[-1], T_ref / 16)
            yy = np.interp(uni, tt, y)
            ker = np.ones(3) / 3
            yy = np.convolve(yy, ker, "same")
            pk, _ = find_peaks(-yy, prominence=C.MIN_PROMINENCE, distance=max(1, int(0.25 * 16)))
            count += len(pk)
            total += span / T_ref
            z = y - y.mean()
            pw = ls(tt, z, 2 * np.pi * np.array([1 / T_ref, 2 / T_ref]), normalize=True)
            pf += pw[0] * g.sum()
            p2 += pw[1] * g.sum()
    if total <= 0:
        return np.nan, np.nan
    return float(count / total), float(p2 / pf) if pf > 0 else np.nan


def evaluate(d, cadence, prefix="", kp_key="kp_speed", part_T=None):
    """prefix "" là feature chính (không lọc FB), "fb_" là bản lọc FB. Feature đã là NaN ở frame không dùng được."""
    q = {}
    part_T = part_T or {}
    v_foot = d[prefix + "v_med"][:, :, FOOT]
    usable = d["frame_valid_mask"] & d["flow_valid"][:, None]
    kp = np.where(usable, d[kp_key], np.nan)
    q["spearman_flow_kp"], q["n_spearman"] = _rho(v_foot.ravel(), kp.ravel())
    first, second = slots_by_label(d["label"])
    i = np.arange(len(first))
    q["spearman_first"], _ = _rho(v_foot[i, first], kp[i, first])
    q["spearman_second"], _ = _rho(v_foot[i, second], kp[i, second])

    # 7b: tính riêng từng speed_part, so với tham chiếu cùng part
    parts = cadence_parts(d, prefix, cadence, part_T)
    q["cadence_parts"] = parts
    use = [p for p in parts if np.isfinite(p["cadence_flow"])]
    q["cadence_ref"], q["ls_span_s"] = cadence, float(max((p["span_s"] for p in parts), default=np.nan))
    if use:
        wt = np.array([p["n"] for p in use], float)
        avg = lambda k: float(np.average([p[k] for p in use], weights=wt))
        q.update(cadence_flow=avg("cadence_flow"), ls_power=avg("ls_power"), cadence_resolution_spm=avg("resolution_spm"))
        have = [p for p in use if np.isfinite(p["dev"])]
        if have:
            wh = np.array([p["n"] for p in have], float)
            q.update(cadence_dev=float(np.average([p["dev"] for p in have], weights=wh)), cadence_dev_spm=float(np.average([p["dev_spm"] for p in have], weights=wh)),
                     cadence_ref_used=float(np.average([p["ref"] for p in have], weights=wh)))
        else:
            q.update(cadence_dev=np.nan, cadence_dev_spm=np.nan, cadence_ref_used=np.nan)
    else:
        q.update(cadence_flow=np.nan, ls_power=np.nan, cadence_resolution_spm=float(120 / q["ls_span_s"]) if parts and q["ls_span_s"] > 0 else np.nan, cadence_dev=np.nan, cadence_dev_spm=np.nan, cadence_ref_used=np.nan)

    # cực tiểu |v| mỗi stride trên 1 chân: bù đúng thì 1, nền trôi thì gần 2
    q["minima_per_stride"], q["power_ratio_2f"] = minima_per_stride(d, prefix, first, second, cadence, part_T)

    # 7c: chỉ để tham khảo
    for key, reg in (("alternation_rho", FOOT), ("alternation_rho_heel", HEEL)):
        s = d[prefix + "slow_ratio"][:, :, reg]
        q[key], _ = _rho(s[i, first], s[i, second])
    s = d[prefix + "slow_ratio"][:, :, FOOT]
    a, b = s[i, first], s[i, second]
    both = np.isfinite(a) & np.isfinite(b)
    q["frac_both_high"] = float(((a > C.SLOW_HIGH) & (b > C.SLOW_HIGH))[both].mean()) if both.any() else np.nan
    q["frac_both_low"] = float(((a <= C.SLOW_HIGH) & (b <= C.SLOW_HIGH))[both].mean()) if both.any() else np.nan
    return q


def _old_runs(x, dt, block):
    """Định nghĩa cũ: đoạn liên tục x > 0,5; đoạn không hợp lệ <= STANCE_BRIDGE frame kẹp giữa không làm đứt. Trả về (mặt nạ frame chống, mặt nạ frame tính thời gian, đoạn [(chỉ số, bị cắt)])."""
    good = np.isfinite(x) & np.isfinite(dt)
    st = good & (x > C.STANCE_THR)
    runs = []
    for b in np.unique(block[np.isfinite(dt)]):
        w = np.flatnonzero((block == b) & np.isfinite(dt))
        k = 0
        while k < len(w):
            if not st[w[k]]:
                k += 1
                continue
            j, last, gap = k, k, 0
            while j + 1 < len(w) and gap <= C.STANCE_BRIDGE:
                j += 1
                if st[w[j]]:
                    last, gap = j, 0
                elif good[w[j]]:
                    break
                else:
                    gap += 1
            runs.append((w[k:last + 1], k == 0 or last == len(w) - 1))
            k = last + 1
    return st, good, runs


def _new_runs(x, dt, t, block):
    """Định nghĩa mới, chỉ cho QC: gộp 2 đoạn cách nhau <= max(STANCE_MERGE_FRAMES frame, STANCE_MERGE_S giây), rồi bỏ đoạn chỉ có 1 frame (sau khi gộp, đoạn đó không còn sát đoạn nào)."""
    good = np.isfinite(x) & np.isfinite(dt)
    raw = good & (x > C.STANCE_THR)
    st = np.zeros(len(x), bool)
    runs = []
    for b in np.unique(block[np.isfinite(dt)]):
        w = np.flatnonzero((block == b) & np.isfinite(dt))
        on = raw[w]
        segs = []
        k = 0
        while k < len(w):
            if on[k]:
                e = k
                while e + 1 < len(w) and on[e + 1]:
                    e += 1
                segs.append([k, e])
                k = e + 1
            else:
                k += 1
        merged = []
        for s0, e0 in segs:
            if merged:
                ps, pe = merged[-1]
                gap_frames = s0 - pe - 1
                gap_s = t[w[s0]] - (t[w[pe]] + dt[w[pe]])
                if gap_frames <= C.STANCE_MERGE_FRAMES or gap_s <= C.STANCE_MERGE_S:
                    merged[-1][1] = e0
                    continue
            merged.append([s0, e0])
        for s0, e0 in merged:
            if e0 == s0:
                continue
            m = w[s0:e0 + 1]
            st[m] = True
            runs.append((m, s0 == 0 or e0 == len(w) - 1))
    return st, good | st, runs


def _summary(st, base, runs, dt, T):
    tot = np.nansum(dt[base])
    whole = [m for m, cut in runs if not cut]
    use = whole if whole else [m for m, _ in runs]
    strides = np.nansum((dt / T)[base])
    return dict(duty_factor=float(np.nansum(dt[st]) / tot) if tot > 0 else np.nan,
                stance_run_median_s=float(np.median([np.nansum(dt[m]) for m in use])) if use else np.nan,
                stance_run_median_frames=float(np.median([len(m) for m in use])) if use else np.nan,
                n_stance_per_stride=float(len(runs) / strides) if strides > 0 else np.nan)


def stance_metrics(d, cadence, part_T):
    """Chỉ để kiểm tra, không vào gate, không đổi feature. first = near hoặc A, second = far hoặc B.
    Không tiền tố: chống khi max(slow_ratio gót, giữa, mũi) > 0,5, có gộp khoảng hở. heel_: như vậy nhưng chỉ gót. old_: định nghĩa cũ, chỉ gót, không gộp theo thời gian."""
    first, second = slots_by_label(d["label"])
    i = np.arange(len(first))
    t_ok = ~d["exclude_time_features"]
    dt, t = d["dt_real"], d["t_real_s_v2"]
    block = d["continuity_id"] * 100000 + d["shot_id"]
    T = np.full(len(i), 120.0 / cadence if np.isfinite(cadence) else np.nan)
    if len(np.unique(d["speed_part_id"])) > 1:
        for p, val in part_T.items():
            T[d["speed_part_id"] == p] = val
    usable = d["frame_valid_mask"] & d["flow_valid"][:, None]
    kp = np.where(usable, d["kp_speed"], np.nan)
    h = d["slow_ratio"]
    out = {}
    for name, o in (("first", first), ("second", second)):
        heel = h[i, o, 0].astype(float).copy()
        top = np.fmax(np.fmax(h[i, o, 0], h[i, o, 1]), h[i, o, 2]).astype(float)
        heel[~t_ok] = np.nan
        top[~t_ok] = np.nan
        for pre, res in (("", _new_runs(top, dt, t, block)), ("heel_", _new_runs(heel, dt, t, block)), ("old_", _old_runs(heel, dt, block))):
            for k, val in _summary(*res, dt, T).items():
                out[f"{pre}{k}_{name}"] = val
        p10 = kp[i, o][np.isfinite(kp[i, o])]
        out[f"kp_speed_min_p10_{name}"] = float(np.percentile(p10, 10)) if len(p10) else np.nan
    return out


def extra_qc(d, q):
    """Dấu hiệu bù camera thiếu và tỉ số cadence. Ghi vào q."""
    vx = d["vx_med"][:, :, FOOT]
    x = vx[np.isfinite(vx)]
    q["vx_p10"] = float(np.percentile(x, 10)) if len(x) else np.nan
    q["frac_foot_backward"] = float((x < -1).mean()) if len(x) else np.nan
    q["cam_undercomp"] = bool(np.isfinite(q["vx_p10"]) and q["vx_p10"] < C.CAM_UNDER_VX)
    ok = np.isfinite(q["cadence_flow"]) and np.isfinite(q.get("cadence_ref_used", np.nan))
    q["cadence_ratio"] = float(q["cadence_flow"] / q["cadence_ref_used"]) if ok else np.nan
    q["cadence_below_resolution"] = bool(np.isfinite(q["cadence_dev"]) and q["cadence_dev"] > C.FAIL_CADENCE and not cadence_fails(q))


def cadence_fails(q):
    """Lệch quá ngưỡng và cũng lớn hơn độ phân giải tần số của video."""
    return bool(np.isfinite(q["cadence_dev"]) and q["cadence_dev"] > C.FAIL_CADENCE and not (np.isfinite(q["cadence_resolution_spm"]) and q["cadence_dev_spm"] < q["cadence_resolution_spm"]))


def gate(q):
    fail, warn = [], []
    if np.isfinite(q["spearman_flow_kp"]):
        if q["spearman_flow_kp"] < C.FAIL_RHO:
            fail.append(f"spearman<{C.FAIL_RHO}")
        elif q["spearman_flow_kp"] < C.WARN_RHO:
            warn.append(f"spearman<{C.WARN_RHO}")
    else:
        warn.append("khong_tinh_duoc_spearman")
    if cadence_fails(q):
        fail.append(f"lech_cadence>{C.FAIL_CADENCE:.0%}")
    if q["bg_residual_median"] > max(C.WARN_BG, C.WARN_BG_REL * q.get("cam_grid_median", 0.0)):
        warn.append("bg_residual_cao")
    if q["frac_fb_valid"] < C.WARN_FB:
        warn.append(f"frac_fb_valid<{C.WARN_FB}")
    if q.get("cam_undercomp"):
        warn.append("cam_undercomp")
    return ("FAIL" if fail else "WARN" if warn else "PASS"), "; ".join(fail + warn)
