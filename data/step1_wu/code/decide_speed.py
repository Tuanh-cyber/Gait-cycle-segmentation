"""Chốt hệ số tốc độ phát k (thời gian thật = thời gian video × k) theo từng PHẦN của clip.
1) Mỗi đoạn liên tục (cadence_segments) -> cửa sổ 2 s, bước 0.25 s; nhãn HIGH nếu cadence video ≥ 100 spm.
   Run-length (run < 1.0 s bị gộp vào lân cận) -> các phần cùng tốc độ.
2) Cadence từng phần bằng Lomb-Scargle trên toàn phần.
3) Đoạn có cả phần nhanh & chậm: phần nhanh k=1 (nếu cadence 130–215), phần chậm k = cad_chậm / cad_nhanh
   (cùng VĐV, cùng buổi) -> snap về {1/2,1/3,1/4,1/5,1/6,1/8} nếu lệch <15%.
4) Phần đơn tốc độ: ứng viên K sao cho cadence/k ∈ [140,215]; chọn ứng viên gần Wu nhất (log); không có Wu -> k lớn nhất hợp lệ.
5) Cờ: mâu thuẫn Wu (>1.6×, chỉ xét đoạn đơn tốc độ), không ứng viên hợp lệ, peak yếu, phần quá ngắn."""
import numpy as np, pandas as pd, json, glob, os
exec(open("step1_wu/code/cadence.py").read().split("rows = []")[0])
K = np.array([1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/8]); KN = ["1","1/2","1/3","1/4","1/5","1/6","1/8"]
LO, HI = 140, 215
man = pd.read_csv("step1_wu/manifest.csv")
segs = pd.read_csv("step1_wu/cadence_segments.csv")

def wu_for(v, t0, t1):
    c = man[(man.video_id == v) & (man.status == "ok")]
    for r in c.itertuples():
        if r.t0_src_s - 0.05 <= t0 and t1 <= r.t1_src_s + 0.05: return r.wu_speed, (r.n_change_windows or 0)
    return np.nan, np.nan

parts = []
for s in segs.itertuples():
    z = np.load(f"keypoints/{s.video_id}_kp.npz")
    # cửa sổ 2 s, bước 0.25 s -> cadence log; tách 2 cụm nếu tâm cụm lệch >= 1.6x
    W = []
    if s.dur_video_s >= 3.0:
        for a in np.arange(s.t0, s.t1 - 2.0 + 1e-6, 0.25):
            r = signals(z, a, a + 2.0)
            if r is not None: W.append((a + 1.0, np.log(peak(*r)[0])))
    bounds = [(s.t0, s.t1)]
    if len(W) >= 6:
        c = np.array([w[0] for w in W]); x = np.array([w[1] for w in W])
        lo, hi = np.percentile(x, 15), np.percentile(x, 85)
        for _ in range(20):
            lab = np.abs(x - hi) < np.abs(x - lo)
            if lab.all() or (~lab).all(): break
            lo, hi = x[~lab].mean(), x[lab].mean()
        if hi - lo >= np.log(1.6) and 0 < lab.sum() < len(lab):
            # median filter nhãn (5 cửa sổ) rồi run-length, bỏ run < 0.5 s
            lab = np.array([np.median(lab[max(0,i-2):i+3]) >= 0.5 for i in range(len(lab))])
            runs = []
            for ci, h in zip(c, lab):
                if runs and runs[-1][2] == h: runs[-1][1] = ci
                else: runs.append([ci, ci, h])
            runs = [r for r in runs if r[1] - r[0] >= 0.5] or runs
            merged = []
            for r in runs:
                if merged and merged[-1][2] == r[2]: merged[-1][1] = r[1]
                else: merged.append(list(r))
            if len(merged) > 1:
                cuts = [(merged[i][1] + merged[i+1][0]) / 2 for i in range(len(merged) - 1)]
                edges = [s.t0] + cuts + [s.t1]; bounds = list(zip(edges[:-1], edges[1:]))
    wu, nch = wu_for(s.video_id, s.t0, s.t1)
    def part_cad(b0, b1):
        r = signals(z, b0, b1); return peak(*r) if r is not None else (np.nan, 0)
    while len(bounds) > 1:
        cads = [part_cad(*b) for b in bounds]
        cm = max(c for c, q in cads); hi = [c >= cm / 1.6 for c, q in cads]
        bad = [i for i, (c, q) in enumerate(cads) if hi[i] and (c < 120 or q < 50)]
        bad += [i for i in range(len(bounds)-1) if hi[i] == hi[i+1]]       # 2 phần liền kề cùng loại -> gộp
        if not bad: break
        i = bad[0]; j = i + 1 if i + 1 < len(bounds) else i - 1
        a, b = sorted([i, j]); bounds[a] = (bounds[a][0], bounds[b][1]); bounds.pop(b)
    P = []
    for b0, b1 in bounds:
        r = signals(z, b0, b1)
        cad, q = peak(*r) if r is not None else (np.nan, 0)
        P.append(dict(video_id=s.video_id, seg=s.seg, shot_id=s.shot_id, t0=round(b0, 3), t1=round(b1, 3),
                      dur_video_s=round(b1 - b0, 2), cadence_video_spm=round(cad, 1), peak_quality=round(q, 1),
                      wu_speed=wu, wu_change_windows=nch, n_parts_in_seg=len(bounds)))
    cmax = max(p["cadence_video_spm"] for p in P)
    fast = [p for p in P if p["cadence_video_spm"] >= cmax / 1.6]
    slow = [p for p in P if p["cadence_video_spm"] < cmax / 1.6]
    for p in P:
        cad = p["cadence_video_spm"]; flags = []
        if len(P) > 1 and fast and slow:
            ref = np.median([f["cadence_video_spm"] for f in fast])
            if p in fast:
                k = 1.0; method = "phần real-time trong clip đổi tốc độ"
                if not (130 <= cad <= LO + 75): flags.append("cadence phần nhanh ngoài 130–215")
            else:
                kc = cad / ref; i = int(np.argmin(np.abs(np.log(K / kc))))
                k = K[i] if abs(np.log(K[i] / kc)) < np.log(1.15) else kc
                method = f"tỉ lệ cadence chậm/nhanh = {kc:.3f}"
        else:
            ok = (cad / K >= LO) & (cad / K <= HI)
            if ok.any():
                cand = K[ok]
                if np.isfinite(p["wu_speed"]):
                    k = cand[int(np.argmin(np.abs(np.log(cand / p["wu_speed"]))))]; method = "cadence hợp lệ + gần Wu nhất"
                else:
                    k = cand.max(); method = "cadence hợp lệ (không có Wu)"
                if len(cand) > 1: flags.append("nhiều ứng viên k: " + ",".join(KN[list(K).index(c)] for c in cand))
            else:
                d = np.minimum(np.abs(np.log(cad / K / LO)), np.abs(np.log(cad / K / HI)))
                k = K[int(np.argmin(d))]; method = "không ứng viên trong 140–215, lấy gần nhất"; flags.append("cadence ngoài khoảng sinh lý")
            if np.isfinite(p["wu_speed"]) and abs(np.log(k / p["wu_speed"])) > np.log(1.6) and not (p["wu_change_windows"] or 0):
                flags.append(f"mâu thuẫn Wu ({p['wu_speed']:.2f})")
        if p["dur_video_s"] * cad / 60 < 4: flags.append("< 4 bước trong phần này")
        if p["peak_quality"] < 50: flags.append("peak cadence yếu")
        nsteps = p["dur_video_s"] * cad / 60
        if k == 1 and 130 <= cad <= 215 and nsteps >= 4 and p["peak_quality"] >= 50: conf = "cao"
        elif "tỉ lệ" in method: conf = "trung bình" if nsteps >= 4 else "thấp"
        elif k == 1: conf = "trung bình"
        else: conf = "trung bình" if (not any(f.startswith(("nhiều","mâu","cadence ngoài")) for f in flags) and nsteps >= 4) else "thấp"
        p["k_confidence"] = conf
        kn = next((KN[i] for i, kk in enumerate(K) if abs(kk - k) < 1e-9), f"{k:.3f}")
        p.update(k=round(float(k), 4), k_label=kn, cadence_real_spm=round(cad / k, 1), method=method, flags="; ".join(flags))
        parts.append(p)
out = pd.DataFrame(parts); out.to_csv("step1_wu/speed_parts.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(out[["video_id","seg","t0","t1","cadence_video_spm","wu_speed","k_label","cadence_real_spm","k_confidence","flags"]].to_string())
print(out.k_confidence.value_counts())
