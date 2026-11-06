#!/usr/bin/env python3
"""Bước 1b: hệ số tốc độ k(t) cho slow-mo dạng ramp. Chạy từ ROOT/data:

  ../.venv-track/bin/python step1_wu/code/ramp_time.py --dry-run     # chỉ tính và in, không ghi gì
  ../.venv-track/bin/python step1_wu/code/ramp_time.py               # sao lưu rồi ghi

Giả định: trong 1 clip ngắn, thời lượng bước THỰC gần hằng (T_step), nên k_i = T_step / d_i (d_i: thời lượng
bước theo thời gian video). Giả định này xoá biến thiên cadence thật trong part ramp.

Đọc: step3_footkp/footkp_v1 (keypoint trước khi chạy lại Bước 3). Không đổi speed_k, t_real_s cũ.
"""
import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import ramp_common as RC

DATA = RC.DATA
FR = DATA / "step0_audit" / "frames"
META = DATA / "step0_audit" / "step0_metadata.csv"
SNAP = np.array([1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 8])
SNAP_TOL = 0.15
D_MIN_S = 0.25          # 1 bước < 0.25 s video (> 240 bước/phút, trên dải sinh lý 140–215 của Bước 1): chéo giả, không làm mốc
EDGE = 0.15             # k_i giữ phẳng trong bước, nội suy PCHIP ở 15% hai đầu bước: tích phân mỗi bước vẫn gần T_step
FAST_BAND = 1.30        # "các bước nhỏ nhất": d <= 1.3 × d nhỏ nhất hợp lệ
CV_MAX = 0.15


def snap(k):
    k = min(k, 1.0)      # không có mốc nào cho thấy video bị tua nhanh
    i = int(np.argmin(np.abs(np.log(SNAP) - np.log(k))))
    return float(SNAP[i]) if abs(k / SNAP[i] - 1) <= SNAP_TOL else float(k)


def realtime_T(fr, st):
    """(a) Mốc thật: median d_i của phần speed_k = 1, k_confidence cao, không phải part ramp."""
    good = fr[(fr.speed_k == 1) & (fr.k_confidence == "cao")].speed_part_id.unique()
    d = st[st.part.isin(good)].d.values
    return float(np.median(d)) if len(d) >= 2 else None


def analyse_video(vid):
    sig = RC.load_signal(vid)
    fr = pd.read_csv(FR / f"{vid}_frames.csv")
    st = RC.steps_of(sig)
    tested = []
    for p in sorted(x for x in fr.speed_part_id.unique() if x >= 0):
        d = st[st.part == p].d.values
        tested.append(dict(video=vid, part=int(p), n_steps=len(d), ratio=float(d.max() / d.min()) if len(d) else np.nan,
                           testable=len(d) >= RC.MIN_STEPS,
                           is_ramp=bool(len(d) >= RC.MIN_STEPS and d.max() / d.min() > RC.RATIO_MIN)))
    ramp = [t["part"] for t in tested if t["is_ramp"]]
    return sig, fr, st, tested, ramp


def build_k(fr, st, ramp):
    """Trả về speed_k_t theo frame, T_step, nguồn, chi tiết từng part."""
    k_t = fr.speed_k.values.astype(float).copy()
    T_rt = realtime_T(fr[~fr.speed_part_id.isin(ramp)], st)
    if T_rt is not None:
        T, src = T_rt, "realtime_ref"
    else:
        d = st[st.part.isin(ramp) & (st.d >= D_MIN_S)].d.values
        T, src = float(np.median(d[d <= FAST_BAND * d.min()])), "fastest_steps"
    info = {}
    for p in ramp:
        g = st[(st.part == p) & (st.d >= D_MIN_S)]
        ki = np.array([snap(T / x) for x in g.d.values])
        tm = np.ravel(np.column_stack([g.t0 + EDGE * g.d, g.t1 - EDGE * g.d]))
        kn = np.repeat(ki, 2)
        f = fr.speed_part_id == p
        x = fr.pts_s[f].values
        k_x = PchipInterpolator(tm, kn, extrapolate=False)(x)
        k_x = np.where(x < tm[0], kn[0], np.where(x > tm[-1], kn[-1], k_x))
        k_t[f.values] = k_x
        info[p] = dict(ki=ki, tm=tm, k_min=float(ki.min()), k_max=float(ki.max()))
    return k_t, T, src, info


def integrate(fr, k_t, ramp):
    """Cùng công thức write_time.py (cộng dồn dt×k, reset theo logic cũ), hình thang trong part ramp."""
    pid, cont, pts = fr.speed_part_id.values, fr.continuity_id.values, fr.pts_s.values
    t = np.full(len(fr), np.nan)
    acc, prev, prev_c, prev_p, prev_k = 0.0, None, None, None, None
    rs = set(ramp)
    for i in range(len(fr)):
        if pid[i] < 0:
            continue
        if prev is not None and cont[i] == prev_c:
            kk = (k_t[i] + prev_k) / 2 if pid[i] in rs and prev_p == pid[i] else k_t[i]
            acc += (pts[i] - prev) * kk
        t[i] = acc
        prev, prev_c, prev_p, prev_k = pts[i], cont[i], pid[i], k_t[i]
    return np.round(t, 5)


def step_stats(fr, st, tcol, parts):
    """Thời lượng từng bước theo 1 trục thời gian, nội suy tại thời điểm chéo. Bỏ bước bị loại là chéo giả."""
    rows = []
    for p in parts:
        f = fr[(fr.speed_part_id == p) & np.isfinite(fr[tcol])]
        g = st[(st.part == p) & (st.d >= D_MIN_S)]
        a, b = np.interp(g.t0, f.pts_s, f[tcol]), np.interp(g.t1, f.pts_s, f[tcol])
        rows.append(pd.DataFrame(dict(part=p, t0=g.t0.values, t1=g.t1.values, d=g.d.values, dur=b - a)))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["part", "t0", "t1", "d", "dur"])


def cv(x):
    return float(np.std(x) / np.mean(x)) if len(x) else np.nan


def image_rate_corr(fr, st, k_t, p):
    """Kiểm chứng độc lập keypoint: cường độ đổi ảnh theo giây video phải tỉ lệ với k trong 1 part."""
    f = fr[fr.speed_part_id == p].assign(_k=k_t[fr.speed_part_id.values == p])
    r, kk = [], []
    for g in st[st.part == p].itertuples():
        w = f[(f.pts_s >= g.t0) & (f.pts_s < g.t1) & ~f.is_dup]
        if len(w) > 1:
            r.append(np.nanmedian(w.diff_prev.replace(np.inf, np.nan)) / np.diff(w.pts_s).mean())
            kk.append(w._k.mean())
    r, kk = np.array(r), np.array(kk)
    ok = np.isfinite(r) & (r > 0) & (kk > 0)
    if ok.sum() < 3 or np.ptp(np.log(kk[ok])) < 1e-6:
        return np.nan
    return float(np.corrcoef(np.log(kk[ok]), np.log(r[ok]))[0, 1])


def plot(vid, fr, st, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid, new, old = "#3A3A38", "#E4E2DC", "#167A3C", "#7A4FB5"
    parts = res["ramp"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
    for j, p in enumerate(parts):
        a = res["old"][res["old"].part == p]
        b = res["new"][res["new"].part == p]
        tm = (a.t0 + a.t1) / 2
        ax[0].plot(tm, a.dur, color=old, lw=1.8, marker="o", ms=5, label="t_real_s cũ" if j == 0 else None)
        ax[0].plot(tm, b.dur, color=new, lw=1.8, marker="o", ms=5, label="t_real_s_v2" if j == 0 else None)
        f = fr[fr.speed_part_id == p]
        k = res["k_t"][fr.speed_part_id.values == p]
        ax[1].plot(f.pts_s, f.speed_k, color=old, lw=1.8, label="speed_k cũ" if j == 0 else None)
        ax[1].plot(f.pts_s, k, color=new, lw=1.8, label="speed_k_t" if j == 0 else None)
    ax[0].set_ylabel("Thời lượng bước (s)", fontsize=9, color=ink)
    ax[0].set_ylim(bottom=0)
    ax[0].set_title(f"{vid}: thời lượng bước trước và sau sửa, T_step = {res['T']:.3f} s ({res['src']})", fontsize=10, color=ink, loc="left")
    ax[1].set_ylabel("k", fontsize=9, color=ink)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_xlabel("pts_s (thời gian video)", fontsize=9, color=ink)
    for a in ax:
        a.grid(axis="y", color=grid, lw=.6)
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(grid)
        a.tick_params(colors=ink, labelsize=8)
        a.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, fontsize=8, ncol=2, borderaxespad=0)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def backup(stamp):
    dst = DATA / "step0_audit" / f"frames_backup_{stamp}"
    if not dst.exists():
        shutil.copytree(FR, dst)
        print(f"Đã sao lưu frames -> {dst.name} ({len(list(dst.glob('*.csv')))} file)")
    for src in (META, META.with_suffix(".xlsx")):
        d = src.with_name(f"{src.stem}_backup_{stamp}{src.suffix}")
        if not d.exists():
            shutil.copy2(src, d)
    s3 = DATA / "step3_footkp"
    if not (s3 / "footkp_v1").exists():
        shutil.copytree(s3 / "footkp", s3 / "footkp_v1")
        shutil.copy2(s3 / "qc_footkp.csv", s3 / "qc_footkp_v1.csv")
        shutil.copy2(s3 / "qc" / "velocity_threshold.json", s3 / "qc" / "velocity_threshold_v1.json")
        print("Đã giữ bản Bước 3 v1 -> footkp_v1, qc_footkp_v1.csv")


def write_meta(S):
    """Chỉ thêm cột và hạ abs_time_ok; đọc mọi cột dạng chuỗi để không đổi giá trị cũ."""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    m = pd.read_csv(META, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    L = S.set_index("video_id")
    for c in ("time_model", "n_ramp_parts", "cadence_real_spm_v2"):
        m[c] = [("" if pd.isna(L.loc[v, c]) else str(L.loc[v, c])) for v in m.video_id]
    m["abs_time_ok"] = [("False" if L.loc[v, "abs_time_force_false"] else a) for v, a in zip(m.video_id, m.abs_time_ok)]
    m.to_csv(META, index=False, encoding="utf-8-sig")
    x = META.with_suffix(".xlsx")
    wb = openpyxl.load_workbook(x)
    ws = wb["metadata"]
    hdr = [c.value for c in ws[1]]
    vcol = hdr.index("video_id") + 1
    red, green = PatternFill("solid", fgColor="F8D7DA"), PatternFill("solid", fgColor="D4EDDA")
    for c in ("time_model", "n_ramp_parts", "cadence_real_spm_v2", "abs_time_ok"):
        if c in hdr:
            col = hdr.index(c) + 1
        else:
            col = ws.max_column + 1
            h = ws.cell(1, col, c)
            hdr.append(c)
            h.font = Font(bold=True, color="FFFFFF")
            h.fill = PatternFill("solid", fgColor="1F4E78")
            h.alignment = Alignment(wrap_text=True, vertical="top")
        for r in range(2, ws.max_row + 1):
            v = ws.cell(r, vcol).value
            val = m.set_index("video_id").loc[v, c]
            if c == "abs_time_ok":
                val = val == "True"
            elif c == "n_ramp_parts":
                val = int(val)
            elif c == "cadence_real_spm_v2":
                val = float(val) if val else None
            cell = ws.cell(r, col, val)
            if c == "abs_time_ok":
                cell.fill = green if val else red
    rd = wb["README"]
    done = {row[0].value for row in rd.iter_rows()}
    for a, b in [("time_model", "constant: k hằng trong mỗi part. ramp: có part đã dùng k(t), xem step1_wu/ramp_parts.csv và step1_wu/code/ramp_time.py"),
                 ("cadence_real_spm_v2", "Video ramp: 60/T_step. Giả định T_step hằng xoá biến thiên cadence thật trong part ramp; chỉ dùng cadence của video abs_time_ok = True"),
                 ("frames CSV (Bước 1b)", "speed_k_t, t_real_s_v2, time_model. speed_k, t_real_s giữ nguyên để so sánh")]:
        if a not in done:
            rd.append([a, b])
    wb.save(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    m = pd.read_csv(META, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    allp, parts, summ, out = [], [], [], {}
    for vid in m.video_id:
        sig, fr, st, tested, ramp = analyse_video(vid)
        allp += tested
        row = m.set_index("video_id").loc[vid]
        if ramp:
            k_t, T, src, info = build_k(fr, st, ramp)
            t2 = integrate(fr, k_t, ramp)
            fr2 = fr.assign(_v2=t2)
            old, new = step_stats(fr, st, "t_real_s", ramp), step_stats(fr2, st, "_v2", ramp)
            res = dict(ramp=ramp, T=T, src=src, old=old, new=new, k_t=k_t)
            for p in ramp:
                d = st[st.part == p]
                bad = (d.d < D_MIN_S).sum()
                parts.append(dict(video=vid, part=p, n_steps=len(d), d_min=round(d.d.min(), 3), d_max=round(d.d.max(), 3),
                                  ratio=round(d.d.max() / d.d.min(), 2), T_step=round(T, 3), T_source=src,
                                  k_min=round(info[p]["k_min"], 4), k_max=round(info[p]["k_max"], 4),
                                  n_steps_rejected=int(bad),
                                  cv_old=round(cv(old[old.part == p].dur), 3), cv_v2=round(cv(new[new.part == p].dur), 3),
                                  corr_k_image_rate=round(image_rate_corr(fr, st, k_t, p), 3),
                                  cv_pass=bool(cv(new[new.part == p].dur) < CV_MAX)))
            out[vid] = (fr, k_t, t2, ramp, st, res)
            force_false = src != "realtime_ref"
            cad = round(60 / T, 1)
        else:
            k_t = fr.speed_k.values.astype(float)
            t2 = integrate(fr, k_t, [])
            ok = np.isfinite(fr.t_real_s.values)
            if not np.allclose(t2[ok], fr.t_real_s.values[ok], atol=2e-5) or (np.isfinite(t2) != ok).any():
                raise RuntimeError(f"{vid}: công thức khác write_time.py")
            out[vid] = (fr, k_t, fr.t_real_s.values, [], st, None)
            force_false = False
            cad = row.cadence_real_spm
            cad = float(cad) if cad else np.nan
        summ.append(dict(video_id=vid, time_model="ramp" if ramp else "constant", n_ramp_parts=len(ramp),
                         cadence_real_spm_v2=cad, abs_time_force_false=force_false,
                         abs_old=row.abs_time_ok))
    P, A, S = pd.DataFrame(parts), pd.DataFrame(allp), pd.DataFrame(summ)
    pd.set_option("display.width", 230, "display.max_rows", 200)
    print(P.to_string(index=False))
    print(f"\nĐã kiểm {len(A)} part / {A.video.nunique()} video; {int(A.testable.sum())} đủ >= {RC.MIN_STEPS} bước; ramp: {int(A.is_ramp.sum())} part ở {A[A.is_ramp].video.nunique()} video")
    print("Video ramp:", ", ".join(sorted(A[A.is_ramp].video.unique())))
    sus = A[~A.testable & (A.ratio > RC.RATIO_MIN)]
    print("Part < 4 bước nhưng ratio > 1.8 (không áp dụng được):", sus[["video", "part", "n_steps", "ratio"]].round(2).to_dict("records"))
    print("Part < 2 bước:", len(A[A.n_steps < 2]), A[A.n_steps < 2][["video", "part"]].to_dict("records"))
    print("abs_time_ok True -> False:", S[S.abs_time_force_false & (S.abs_old == "True")].video_id.tolist())
    print("CV không đạt:", P[~P.cv_pass][["video", "part", "cv_v2"]].to_dict("records") if len(P) else [])
    if a.dry_run:
        return
    stamp = date.today().strftime("%Y%m%d")
    backup(stamp)
    (DATA / "step1_wu" / "plots_ramp").mkdir(exist_ok=True)
    P.to_csv(DATA / "step1_wu" / "ramp_parts.csv", index=False)
    A.to_csv(DATA / "step1_wu" / "ramp_tested_parts.csv", index=False)
    for vid, (fr, k_t, t2, ramp, st, res) in out.items():
        fr = fr.copy()
        fr["speed_k_t"] = np.round(k_t, 5)
        fr["t_real_s_v2"] = t2
        fr["time_model"] = np.where(fr.speed_part_id.isin(ramp), "ramp", "constant")
        fr.to_csv(FR / f"{vid}_frames.csv", index=False)
        if ramp:
            plot(vid, fr, st, res, DATA / "step1_wu" / "plots_ramp" / f"{vid}.png")
    write_meta(S)
    print("Đã ghi frames CSV, metadata, ramp_parts.csv, plots_ramp/")


if __name__ == "__main__":
    main()
