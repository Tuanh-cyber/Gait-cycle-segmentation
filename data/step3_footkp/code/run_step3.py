#!/usr/bin/env python3
"""Bước 3: kiểm định & làm sạch keypoint bàn chân. Chạy từ ROOT/data:

  ../.venv-track/bin/python step3_footkp/code/run_step3.py                       # cả 66 video, resume
  ../.venv-track/bin/python step3_footkp/code/run_step3.py --videos C30,F30      # danh sách
  ... --force                                                                    # làm lại các video được chọn
  ... --pose-device cpu
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

import s3_clean as K
import s3_common as C
import s3_near as N
import s3_perturb as P
import s3_report as R
import s3_swap as W

QC_COLS = ["video_id", "gate", "gate_reasons", "n_frames", "n_blocks",
           "foot_valid_frac_near", "foot_valid_frac_far", "leg_valid_frac_near", "leg_valid_frac_far",
           "swap_rate", "n_swap_events", "swap_rate_resid", "resid_phase", "resid_jump",
           "E_flip_ankle", "E_flip_heel", "E_flip_toe", "E_flip_ankle_raw", "E_flip_heel_raw", "E_flip_toe_raw",
           "scale_perturb_instability", "scale_perturb_instability_heel", "scale_perturb_instability_toe",
           "scale_perturb_instability_ankle", "scale_perturb_p90", "scale_perturb_flip_frac",
           "near_far_conf", "near_track_reliable", "near_match_metadata", "near_match_frames", "near_track_main", "near_leg_metadata", "time_model",
           "phase_periodic_frac", "stride_T_median_s", "sign_changes_per_stride", "step_time_ratio",
           "sg_rms_L", "ankle_amp_retained", "outlier_any_frac",
           *[f"outlier_{k}_frac" for k in C.REASONS],
           *[f"outlier_frac_{j}" for j in C.FOOT_IDX], "config"]


def analyse(c, thr):
    """Toàn bộ 3.1–3.3 cho 1 video, không ghi file."""
    s = int(c.meta.s_final)
    flag, dref, info = W.fix_swaps(c, s)
    kp = W.apply_swap(c.kp, flag)
    sc = K.swap_scores(c.score, flag)
    reason = K.mark_outliers(c, kp, sc, dref, thr)
    col = K.collapse_frames(c, kp, dref)
    reason[np.ix_(col, C.FOOT_IDX)] |= C.R_COLLAPSE
    finite = np.isfinite(kp[:, :, 0])
    valid = finite & (sc >= C.SCORE_MIN) & (reason == 0)
    kp_clean, interp, smoothed = K.clean_and_filter(c, kp, valid)
    L = K.L_of(c, kp)

    near_track, near_conf = np.full(c.n, "A", "U1"), np.zeros(c.n)
    blocks = []
    for b in np.unique(c.cont):
        w = c.cont == b
        cues = N.per_frame_cues(kp[w], sc[w], valid[w], L[w])
        near, conf, per, p = N.block_decision(cues)
        near_track[w], near_conf[w] = near, conf
        exp = N.expected_from_meta(c.meta.near_leg_camera)
        blocks.append(dict(video_id=c.vid, continuity_id=int(b), n=int(w.sum()), near_track=near, near_conf=round(conf, 3),
                           p_A_near=round(p, 3), expected_from_metadata=exp or "", match=(near == exp) if exp else "",
                           near_leg_anat=c.meta.near_leg_camera if exp and near == exp else "",
                           swap_rate=round(float(flag[w].mean()), 4), **{k: round(v, 4) for k, v in per.items()},
                           periods=json.dumps(info.get(int(b), {}))))
    conf_video = float(np.average([b["near_conf"] for b in blocks], weights=[b["n"] for b in blocks]))
    reliable = conf_video >= C.GATE_NEAR_CONF and c.vid not in C.NEAR_UNRELIABLE_MANUAL
    return dict(near_track_reliable=reliable, swap_flag=flag, dref=dref, info=info, kp_fixed=kp, score_fixed=sc, reason=reason, valid=valid,
                kp_clean=kp_clean, interp=interp, smoothed=smoothed, near_track=near_track, near_conf=near_conf,
                L=L, blocks=blocks)


def qc_row(c, res, pert):
    F = C.FOOT_IDX
    near_L = res["near_track"] == "A"
    n_valid_L = res["valid"][:, C.L_FOOT].sum(1) >= C.FOOT_MIN_PTS
    n_valid_R = res["valid"][:, C.R_FOOT].sum(1) >= C.FOOT_MIN_PTS
    leg_L = res["valid"][:, C.L_LEG].sum(1) >= C.LEG_MIN_PTS
    leg_R = res["valid"][:, C.R_LEG].sum(1) >= C.LEG_MIN_PTS
    resid, ra, rb, ev = R.residual_swap(c, res["kp_clean"], res["dref"], res["L"])
    pts = dict(ankle=([15], [16]), heel=([24], [25]), toe=([20, 22], [21, 23]))
    q = dict(video_id=c.vid, n_frames=c.n, n_blocks=len(np.unique(c.cont)),
             foot_valid_frac_near=float(np.where(near_L, n_valid_L, n_valid_R).mean()),
             foot_valid_frac_far=float(np.where(near_L, n_valid_R, n_valid_L).mean()),
             leg_valid_frac_near=float(np.where(near_L, leg_L, leg_R).mean()),
             leg_valid_frac_far=float(np.where(near_L, leg_R, leg_L).mean()),
             swap_rate=float(res["swap_flag"].mean()), n_swap_events=int(len(R.swap_events(c, res["swap_flag"]))),
             swap_rate_resid=float(resid[ev].mean()) if ev.any() else 1.0,
             resid_phase=float(ra[ev].mean()) if ev.any() else 0.0, resid_jump=float(rb[ev].mean()) if ev.any() else 0.0)
    for k, (a, b) in pts.items():
        q[f"E_flip_{k}"] = R.e_flip(c, res["kp_clean"], a, b)
        q[f"E_flip_{k}_raw"] = R.e_flip(c, c.kp, a, b)
    q.update(P.summarize(pert))
    # near/far: độ tin cậy và khớp metadata, gia quyền theo số frame từng khối
    bl = pd.DataFrame(res["blocks"])
    q["near_far_conf"] = float(np.average(bl.near_conf, weights=bl.n))
    q["near_track_main"] = bl.groupby("near_track").n.sum().idxmax()
    q["near_leg_metadata"] = c.meta.near_leg_camera
    q["near_match_metadata"] = float(np.average(bl.match.astype(bool), weights=bl.n)) if (bl.expected_from_metadata != "").any() else np.nan
    q["near_match_frames"] = int(bl.n[bl.match.astype(bool)].sum()) if (bl.expected_from_metadata != "").any() else np.nan
    q["near_track_reliable"] = bool(res["near_track_reliable"])
    q["time_model"] = "ramp" if (c.time_model == "ramp").any() else "constant"
    # tuần hoàn
    q["phase_periodic_frac"] = float(np.isfinite(res["dref"]).mean())
    good_T = [p["T"] for b in res["info"].values() for p in b.values() if p["used"]]
    q["stride_T_median_s"] = float(np.median(good_T)) if good_T else np.nan
    s = int(c.meta.s_final)
    d = s * (res["kp_clean"][:, 15, 0] - res["kp_clean"][:, 16, 0]) / res["L"]
    n_chg, dur = 0, 0.0
    for seg in C.segments(c):
        x = d[seg]
        if not np.isfinite(res["dref"][seg]).all():      # chỉ tính phần có chu kỳ rõ: T không có nghĩa ở đoạn tốc độ đổi
            continue
        x = x[np.isfinite(x)]
        if len(x) > 3:
            n_chg += int((np.diff(np.sign(x)) != 0).sum())
            dur += float(c.t[seg[-1]] - c.t[seg[0]])
    q["sign_changes_per_stride"] = n_chg / (dur / q["stride_T_median_s"]) if dur > 0 and good_T else np.nan
    # thời gian 1 bước theo t_real_s (khoảng cách giữa 2 lần chéo chân) trong cùng part: phải gần như không đổi
    gaps = []
    for seg in C.segments(c):
        x = d[seg]
        ok = np.isfinite(x)
        ss, tt = np.sign(x[ok]), c.t[seg][ok]
        z = tt[1:][ss[1:] != ss[:-1]]
        gaps += list(np.diff(z)[np.diff(z) > 0])
    q["step_time_ratio"] = float(np.percentile(gaps, 90) / np.percentile(gaps, 10)) if len(gaps) >= 4 else np.nan
    # tác động của bộ lọc: lọc có làm cụt biên độ cổ chân không
    v = res["valid"] & res["smoothed"]
    diff = np.linalg.norm(res["kp_clean"] - res["kp_fixed"], axis=2) / res["L"][:, None]
    q["sg_rms_L"] = float(np.sqrt(np.nanmean(diff[:, F][v[:, F]] ** 2))) if v[:, F].any() else np.nan
    d_raw = s * (res["kp_fixed"][:, 15, 0] - res["kp_fixed"][:, 16, 0]) / res["L"]
    both = np.isfinite(d) & np.isfinite(d_raw)
    q["ankle_amp_retained"] = float(np.percentile(np.abs(d[both]), 95) / max(np.percentile(np.abs(d_raw[both]), 95), 1e-6)) if both.any() else np.nan
    # outlier
    r = res["reason"][:, F]
    q["outlier_any_frac"] = float((r > 0).mean())
    for k, b in C.REASONS.items():
        q[f"outlier_{k}_frac"] = float(((r & b) > 0).mean())
    for q_i, j in enumerate(F):
        q[f"outlier_frac_{j}"] = float((r[:, q_i] > 0).mean())
    q["config"] = C.config_hash()
    q["gate"], q["gate_reasons"] = R.gate(q)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}


def threshold(meta, force=False):
    if C.VEL_JSON.exists() and not force:
        d = json.loads(C.VEL_JSON.read_text())
        if d.get("config") == C.config_hash():
            return d
    t0 = time.time()
    clips = []
    for vid in meta.video_id:
        c = C.load_clip(vid, meta)
        flag, _, _ = W.fix_swaps(c, int(c.meta.s_final))
        clips.append((c, W.apply_swap(c.kp, flag), K.swap_scores(c.score, flag)))
    d = K.velocity_stats(clips)
    d["config"] = C.config_hash()
    C.VEL_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"ngưỡng vận tốc (toàn dataset, {time.time() - t0:.1f}s): {d}", flush=True)
    return d


def done(vid):
    p = C.DIRS["qc"] / f"{vid}_qc.json"
    if not (p.exists() and (C.DIRS["footkp"] / f"{vid}_footkp.npz").exists()):
        return False
    return json.loads(p.read_text()).get("config") == C.config_hash()


def process(vid, meta, thr, pose):
    t0 = time.time()
    c = C.load_clip(vid, meta)
    res = analyse(c, thr)
    t1 = time.time()
    pert = P.perturb_video(c, pose, res["L"])
    pert.to_csv(C.DIRS["qc"] / f"{vid}_perturb.csv", index=False)
    t2 = time.time()
    C.atomic_npz(C.DIRS["footkp"] / f"{vid}_footkp.npz", frame=c.frame, pts_s=c.pts, t_real_s_v2=c.t, t_real_s=c.t_old,
                 time_model=c.time_model, near_track_reliable=res["near_track_reliable"], continuity_id=c.cont, speed_part_id=c.part,
                 kp_raw=c.kp.astype(np.float32), kp_clean=res["kp_clean"].astype(np.float32), valid_mask=res["valid"],
                 swap_flag=res["swap_flag"], near_track=res["near_track"], near_conf=res["near_conf"].astype(np.float32),
                 outlier_reason=res["reason"], interp_mask=res["interp"], score_fixed=res["score_fixed"].astype(np.float32),
                 L_leg=res["L"].astype(np.float32), d_ref=res["dref"].astype(np.float32), config=C.config_hash())
    pd.DataFrame(res["blocks"]).to_csv(C.DIRS["qc"] / f"{vid}_blocks.csv", index=False)
    q = qc_row(c, res, pert)
    # ảnh: chỉ tạo lại khi gate đổi hoặc video ramp; ảnh còn lại giữ nguyên bản người kiểm tra đã xem
    v1 = C.OUT / "qc_footkp_v1.csv"
    old_gate = pd.read_csv(v1, dtype=str, keep_default_na=False).set_index("video_id").gate.get(vid) if v1.exists() else None
    if q["time_model"] == "ramp" or q["gate"] != old_gate or not (C.DIRS["review"] / f"{vid}.jpg").exists():
        R.review_grid(c, res, C.DIRS["review"] / f"{vid}.jpg")
        R.review_swaps(c, res)
        R.ankle_plot(c, res, C.DIRS["plots"] / f"{vid}_ankle.png")
        q["images_regenerated"] = True
    else:
        q["images_regenerated"] = False
    t3 = time.time()
    q["time_analyse_s"], q["time_pose_s"], q["time_images_s"] = round(t1 - t0, 2), round(t2 - t1, 2), round(t3 - t2, 2)
    (C.DIRS["qc"] / f"{vid}_qc.json").write_text(json.dumps(q, ensure_ascii=False, indent=1, default=float))
    print(f"  {vid}: {c.n} frame, {t3 - t0:.1f}s (phân tích {t1 - t0:.1f}, pose {t2 - t1:.1f}, ảnh {t3 - t2:.1f}) "
          f"gate={q['gate']} {q['gate_reasons']}", flush=True)


def write_table(meta):
    rows = []
    for vid in meta.video_id:
        p = C.DIRS["qc"] / f"{vid}_qc.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df[QC_COLS].to_csv(C.OUT / "qc_footkp.csv", index=False)
    return df


def summary(df):
    pd.set_option("display.width", 220, "display.max_rows", 100, "display.max_colwidth", 60)
    print("\nGate:", df.gate.value_counts().to_dict())
    bad = df[df.gate != "PASS"]
    if len(bad):
        print(bad[["video_id", "gate", "gate_reasons", "foot_valid_frac_near", "foot_valid_frac_far", "swap_rate", "swap_rate_resid", "near_far_conf", "near_match_metadata"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", help="danh sách video, cách nhau bởi dấu cách hoặc dấu phẩy")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pose-device", default="mps")
    a = ap.parse_args()
    C.ensure_dirs()
    meta = C.load_meta()
    vids = meta.video_id.tolist()
    if a.videos:
        vids = [v for x in a.videos for v in x.split(",") if v]
        unknown = sorted(set(vids) - set(meta.video_id))
        if unknown:
            raise SystemExit(f"video không có trong metadata: {unknown}")
    d = threshold(meta)
    todo = [v for v in vids if a.force or not done(v)]
    print(f"{len(vids)} video, {len(vids) - len(todo)} đã xong, {len(todo)} cần chạy. Ngưỡng vận tốc {d['thr_L_per_s']} L_leg/s", flush=True)
    pose = P.PoseModel(a.pose_device) if todo else None
    t0 = time.time()
    for i, v in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}]", end="", flush=True)
        process(v, meta, d["thr_L_per_s"], pose)
    R.write_review_csv(meta)
    df = write_table(meta)
    if df is not None and len(df) == len(meta) and (C.OUT / "qc_footkp_v1.csv").exists():
        R.write_diff(meta, df)
    if todo:
        print(f"Xong {len(todo)} video trong {time.time() - t0:.1f}s", flush=True)
    if df is not None:
        summary(df[df.video_id.isin(vids)])


if __name__ == "__main__":
    main()
