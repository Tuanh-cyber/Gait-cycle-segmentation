#!/usr/bin/env python3
"""Bước 5: optical flow GMA, bù camera trên dải đất, feature theo vùng bàn chân. Chạy từ ROOT/data:

  ../.venv-track/bin/python step5_flow/code/run_step5.py --videos C17,F30      # vài video
  caffeinate -i ../.venv-track/bin/python step5_flow/code/run_step5.py         # cả 66 video, resume
  ... --force  ... --device cpu
  ... --store-only                                                             # chỉ chạy GMA vào flow_store
  S5_TRIAL=1 ... --videos F29                                                  # ghi vào _trial/, không đụng output chính

GMA chỉ chạy khi kho flow_store/ chưa có video đó. Đổi cách bù camera thì feature được tính lại từ kho.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import cv2
import numpy as np
import pandas as pd
import torch

import s5_checks as K
import s5_common as C
import s5_features as X
import s5_flow as W
import s5_ground as G
import s5_report as R
import s5_store as T

STANCE_BASE = ("duty_factor", "stance_run_median_s", "stance_run_median_frames", "n_stance_per_stride")
STANCE_COLS = [f"{p}{m}_{f}" for p in ("", "heel_", "old_") for m in STANCE_BASE for f in ("first", "second")] + [f"kp_speed_min_p10_{f}" for f in ("first", "second")]
VARIANT_COLS = ["gate", "gate_reasons", "comp_method", "comp_slow_median", "comp_slow_max", "vx_p10", "cam_undercomp", "spearman_flow_kp", "spearman_first", "spearman_second", "cadence_flow", "cadence_dev", "cadence_dev_spm", "cadence_below_resolution",
                "minima_per_stride", "power_ratio_2f", "alternation_rho", "alternation_rho_heel", "slow_ratio_median", "n_cam_fit_bad", "n_cam_bad_slow", "frac_flow_valid", "frac_fb_valid"] + STANCE_COLS
QC_COLS = ["video_id", "gate", "gate_reasons", "n_frames", "n_pairs", "n_pairs_no_crop", "comp_method", "comp_slow_median", "comp_slow_max", "comp_direct_v_median",
           "frac_comp_measured", "frac_comp_static", "frac_comp_filled", "frac_comp_none", "n_cam_fit_bad", "n_cam_bad_inlier", "n_cam_bad_bg", "n_cam_bad_tail", "n_cam_bad_slow", "n_cam_bg_below_floor", "frac_flow_valid",
           "scale_min", "scale_max", "l_foot_grid", "crop_edge_max", "frac_fb_valid", "bg_residual_median", "bg_residual_p95", "bg_residual_v", "bg_inlier_frac", "cam_flow_median", "cam_grid_median", "overlay_frac", "cam_static",
           "cam_dx_rho", "cam_dx_ratio", "dt_mode", "dt_mode_reason", "dt_jitter", "dt_rho_cam", "dt_rho_kp", "dt_cv",
           "spearman_flow_kp", "spearman_first", "spearman_second", "n_spearman", "spearman_fb", "spearman_rawdt", "spearman_smoothdt", "spearman_direct",
           "cadence_ref", "cadence_ref_source", "cadence_ref_used", "cadence_flow", "cadence_dev", "cadence_dev_spm", "cadence_resolution_spm", "cadence_below_resolution", "cadence_ratio", "n_cadence_parts", "ls_power", "ls_span_s",
           "cadence_fb", "cadence_flow_direct", "cadence_dev_direct", "vx_p10", "vx_p10_direct", "frac_foot_backward", "cam_undercomp", "minima_per_stride", "minima_per_stride_direct", "power_ratio_2f", "power_ratio_2f_direct",
           "alternation_rho", "alternation_rho_heel", "alternation_rho_heel_direct", "frac_both_high", "frac_both_low", "frac_both_still",
           "otsu_thr", "fb_otsu_thr", "v_foot_p10", "v_foot_p50", "v_foot_p90", "slow_ratio_median", "slow_ratio_median_direct",
           "frac_usable_first", "frac_usable_second", "abs_time_ok", "device", "model", "time_s", "fps", "config", "default_variant", "foot_labels"] + STANCE_COLS + [f"v2_{k}" for k in VARIANT_COLS]
SAVE_STATS = ("v_med", "v_p25", "v_p75", "vx_med", "vy_med", "n_pixels", "v_hist")
KIND = {"moving": C.GROUND_MOVING, "static": C.GROUND_STATIC_KIND, "none": C.GROUND_NONE}


def features_of(Px, n, dt_used, dt_raw, dt_smooth, ok, otsu_ok, label_keys):
    """Mọi feature cho một bộ pixel đã bù."""
    F, pool = X.stats(Px, n, dt_used, ok, False)
    FB, pool_fb = X.stats(Px, n, dt_used, ok, True)
    Fr, _ = X.stats(Px, n, dt_raw, ok, False)
    Fs, _ = X.stats(Px, n, dt_smooth, ok, False)
    join = lambda P: np.concatenate([x for a, x in P if otsu_ok[a]]) if any(otsu_ok[a] for a, _ in P) else np.zeros(0, np.float32)
    otsu, otsu_fb = X.otsu_threshold(join(pool)), X.otsu_threshold(join(pool_fb))
    empty = np.full(F["n_pixels"].shape, np.nan, np.float32)
    d = {k: F[k] for k in SAVE_STATS}
    d["frac_fb_valid"] = F["frac_fb_valid"]
    d.update({"fb_" + k: FB[k] for k in SAVE_STATS})
    d["slow_ratio"] = X.ratio_from_hist(F["v_hist"], F["n_pixels"], C.SLOW_MAIN)
    for thr in C.SLOW_EXTRA:
        d[f"slow_ratio_{thr:.1f}".replace(".", "p")] = X.ratio_from_hist(F["v_hist"], F["n_pixels"], thr)
    d["slow_ratio_otsu"] = X.ratio_from_hist(F["v_hist"], F["n_pixels"], otsu) if np.isfinite(otsu) else empty
    d["fb_slow_ratio"] = X.ratio_from_hist(FB["v_hist"], FB["n_pixels"], C.SLOW_MAIN)
    d["fb_slow_ratio_otsu"] = X.ratio_from_hist(FB["v_hist"], FB["n_pixels"], otsu_fb) if np.isfinite(otsu_fb) else empty
    d.update(v_rawdt=Fr["v_med"], v_smoothdt=Fs["v_med"], slow_ratio_rawdt=X.ratio_from_hist(Fr["v_hist"], Fr["n_pixels"], C.SLOW_MAIN), slow_ratio_smoothdt=X.ratio_from_hist(Fs["v_hist"], Fs["n_pixels"], C.SLOW_MAIN),
             otsu_thr=otsu, fb_otsu_thr=otsu_fb)
    return d


def kp_speeds(kc, kr, tx_l, ty_l, slow_disp, dt_used, dt_raw, dt_smooth):
    """Tốc độ keypoint sau bù. kc, kr N×2×2 đơn vị L_leg; tx_l, ty_l, slow_disp N đơn vị L_leg."""
    def speed(k, dt):
        with np.errstate(all="ignore"):
            return (np.hypot(k[:, :, 0] - tx_l[:, None] - slow_disp[:, None], k[:, :, 1] - ty_l[:, None]) / dt[:, None]).astype(np.float32)
    return dict(kp_speed=speed(kc, dt_used), kp_speed_rawdt=speed(kc, dt_raw), kp_speed_smoothdt=speed(kc, dt_smooth), kp_raw_speed_rawdt=speed(kr, dt_raw), kp_raw_speed_smoothdt=speed(kr, dt_smooth))


def process(vid, meta, state):
    t0 = time.time()
    if not T.done(vid):
        dtf, nf = T.build(vid, meta, state)
        print(f"  {vid}: GMA {nf} cặp, {dtf:.1f}s, kho {(T.STORE / f'{vid}.npz').stat().st_size / 1e6:.1f} MB", flush=True)
    t1 = time.time()
    v = C.load_video(vid, meta)
    c, z = v.c, v.z
    n = v.n
    st = T.Store(vid)
    has_flow = np.zeros(n, bool)
    has_flow[st.a] = True
    dt_raw = np.full(n, np.nan)
    dt_raw[st.a] = c.t[st.a + 1] - c.t[st.a]
    scale = np.full(n, np.nan, np.float32)
    scale[st.a] = st.scale
    crop = np.zeros((n, 4), np.int32)
    crop[st.a] = st.crop
    l_leg = np.where(np.isfinite(v.l_leg), v.l_leg, np.nanmedian(v.l_leg))
    kind, tx0, ty0 = np.full(n, C.GROUND_NONE, np.int8), np.full(n, np.nan), np.full(n, np.nan)
    inl, n_tex, n_in, bg_res = (np.full(n, np.nan, np.float32) for _ in range(4))
    source = np.full(n, "", "U9")
    kc, kr = np.full((n, 2, 2), np.nan), np.full((n, 2, 2), np.nan)
    Px, lf_all, held = {}, [], []
    rng = np.random.default_rng(0)
    pick = set(st.a[np.unique(np.linspace(0, len(st.a) - 1, 8).astype(int))].tolist()) if len(st.a) else set()
    pos = {int(f): i for i, f in enumerate(c.frame)}
    fb_all = np.full(n, np.nan, np.float32)
    gray_store = {}
    prev = None
    for fr, img in T.frames_stream(c.path, c.frame):
        a = pos[fr] - 1
        if a in st.pos and prev is not None:
            F, fb, sb, box, s, shape = st.get(a)
            A, _ = T.grid_pair(prev, img, box, s)
            x0, y0, x1, y1 = sb
            Reg, allm = X.region_masks(v, a, s, box, shape)
            Reg = {o: [m[y0:y1, x0:x1] for m in Reg[o]] for o in (0, 1)}
            strip = G.strip_mask(v, a, s, box, sb, shape, allm)
            gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            Px[a] = X.collect(F, fb, Reg, v.s_final, s, l_leg[a])
            kc[a], kr[a] = X.keypoint_disp(v, a, v.s_final)
            fb_all[a] = fb.mean()
            lf_all.append(float(np.nanmin(z["l_foot"][a])) * s)
            if a in pick:
                held.append((a, A[y0:y1, x0:x1], Reg, fb, sb, lf_all[-1]))
            gray_store[a] = (strip, gray)
        prev = img

    # chữ gắn trên ảnh: dựng mặt nạ cho cả video, rồi mới ước lượng từng cặp
    overlay, cam_static = G.overlay_map(((g, st.get(int(a))[0], s_, *st.get(int(a))[2:5]) for a, (s_, g) in gray_store.items()), c.H, c.W)
    for a, (s_, g) in gray_store.items():
        F, _, sb, box, s, _ = st.get(a)
        s2 = s_ & ~G.overlay_on_box(overlay, sb, box, s, s_.shape)
        gray_store[a] = (s2, g)
        e = G.estimate_pair(g, F, s2, rng)
        kind[a], tx0[a], ty0[a], inl[a], n_tex[a], n_in[a], source[a] = KIND[e["kind"]], e["tx"], e["ty"], e["inlier_frac"], e["n_tex"], e["n_inlier"], e["source"]

    # A: chuỗi tịnh tiến theo thời gian
    t_valid = np.where(has_flow, c.t, np.nan)
    group = c.cont * 100000 + v.shot
    tx, ty, filled = G.temporal(dict(kind=kind, tx=tx0, ty=ty0), t_valid, group)
    no_src = has_flow & ~np.isfinite(tx)
    tx, ty = np.where(no_src, 0.0, tx), np.where(no_src, 0.0, ty)
    comp_state = np.zeros(n, np.uint8)      # 0 đo được, 1 nền đứng yên, 2 nội suy, 3 không có nguồn
    comp_state[has_flow & (kind == C.GROUND_STATIC_KIND)] = 1
    comp_state[has_flow & (filled == 1)] = 2
    comp_state[no_src] = 3
    for a in st.a:
        s_, g_ = gray_store[int(a)]
        F, *_ = st.get(int(a))
        bg_res[a] = G.background_residual(F, g_, s_, tx[a], ty[a], kind[a] == C.GROUND_MOVING or comp_state[a] == 2)
    del gray_store
    with np.errstate(all="ignore"):
        tx_l, ty_l = tx / scale / l_leg * v.s_final, ty / scale / l_leg
    tx_l, ty_l = np.nan_to_num(tx_l), np.nan_to_num(ty_l)
    cam_grid = np.hypot(tx, ty)
    cam_px = cam_grid / scale

    # Δt: giữ nguyên quy tắc
    dt_mode, dt_why, jitter, rho_cam, rho_kp = X.decide_dt(v, dt_raw, cam_px)
    dt_smooth = X.smooth_dt(v, dt_raw)
    dt_used = dt_smooth if dt_mode == "smooth" else dt_raw

    # bản chỉ bù trực tiếp
    PxA = X.shifted(Px, tx_l, ty_l)
    base_ok = z["frame_valid_mask"] & has_flow[:, None]
    FA, _ = X.stats(PxA, n, dt_used, base_ok, False)
    vx_direct = FA["vx_med"][:, :, 3]
    x = vx_direct[np.isfinite(vx_direct)]
    p10_direct = float(np.percentile(x, 10)) if len(x) else np.nan

    otsu_ok = ~z["exclude_time_features"] & z["view_lateral"]
    everything = np.ones((n, 2), bool)
    fr0 = pd.read_csv(C.DATA / "step0_audit" / "frames" / f"{vid}_frames.csv").set_index("frame")
    cum = np.full(n, np.nan)
    for a in st.a:
        w = (fr0.index > c.frame[a]) & (fr0.index <= c.frame[a + 1])
        cum[a] = fr0.cam_dx[w].sum() if fr0.cam_dx[w].notna().any() else np.nan
    g = has_flow & np.isfinite(cum) & (comp_state == 0)
    measured = tx / scale
    cam_dx_rho = K._rho(measured[g], cum[g])[0]
    with np.errstate(all="ignore"):
        cam_dx_ratio = float(np.median(np.abs(cum[g])) / np.median(np.abs(measured[g]))) if g.any() and np.median(np.abs(measured[g])) > 0 else np.nan
    with np.errstate(all="ignore"):
        direct_v = (tx_l / dt_used).astype(np.float32)
        bg_v = np.nanmedian(bg_res / scale / l_leg / dt_used)
    first, second = K.slots_by_label(z["label"])
    i = np.arange(n)
    hf = has_flow
    all_pairs = C.pairs(v)

    def build_variant(name, trigger):
        comp_method = "direct"
        slow = np.zeros(n)
        if np.isfinite(p10_direct) and p10_direct < trigger:
            comp_method = "direct+slow"
            slow = np.nan_to_num(G.slow_component(c.t, vx_direct, st.a, v.stride_T, group))
        slow_disp = np.nan_to_num(slow * dt_used)
        PxF = X.shifted(PxA, slow_disp, np.zeros(n))
        anyF, _ = X.stats(PxF, n, dt_used, everything, False)
        foot_any = anyF["v_med"][:, :, 3]
        cam_bad, cam_reason, bg_p95 = X.camera_flags(v, has_flow, kind, inl, bg_res, cam_grid, foot_any, slow)
        flow_valid = has_flow & ~cam_bad
        ok = z["frame_valid_mask"] & flow_valid[:, None]
        d = features_of(PxF, n, dt_used, dt_raw, dt_smooth, ok, otsu_ok, None)
        d.update(kp_speeds(kc, kr, tx_l, ty_l, slow_disp, dt_used, dt_raw, dt_smooth))
        common = dict(frame=c.frame, t_real_s_v2=c.t, continuity_id=c.cont, shot_id=v.shot, speed_part_id=v.part, label=z["label"], state=z["state"], frame_valid_mask=z["frame_valid_mask"], invalid_reason=z["invalid_reason"],
                      view_lateral=z["view_lateral"], exclude_time_features=z["exclude_time_features"], L_leg=v.l_leg, has_flow=has_flow, flow_valid=flow_valid, cam_fit_bad=cam_bad, cam_fit_reason=cam_reason,
                      dt_real=dt_raw, dt_smooth=dt_smooth, dt_used=dt_used, dt_mode=dt_mode, crop=crop, scale=scale, s_final=v.s_final)
        d.update(common)
        d.update(v_foot_any=foot_any, comp_variant=name, comp_trigger=trigger, comp_method=comp_method, comp_tx=tx.astype(np.float32), comp_ty=ty.astype(np.float32), comp_direct_v=direct_v, comp_slow=slow.astype(np.float32),
                 comp_state=comp_state, comp_source=source, comp_inlier_frac=inl, comp_n_inlier=n_in, comp_n_tex=n_tex, bg_residual=bg_res, cam_flow=cam_px.astype(np.float32), cam_dx_sum=cum.astype(np.float32), frac_fb_all=fb_all,
                 slow_thr=C.SLOW_MAIN, hist_edges=C.HIST_EDGES, cadence_ref=v.cadence, cadence_ref_source=v.cadence_source, config=C.config_hash(), flow_hash=T.flow_hash(), model=C.MODEL, post_config=str(z["post_config"]))
        # bản chỉ A, cùng cờ, để báo bước B đóng góp bao nhiêu
        dA = {**common, **features_of(PxA, n, dt_used, dt_raw, dt_smooth, ok, otsu_ok, None), **kp_speeds(kc, kr, tx_l, ty_l, np.zeros(n), dt_used, dt_raw, dt_smooth)}
        q = K.evaluate(d, v.cadence, part_T=v.part_T)
        qa = K.evaluate(dA, v.cadence, part_T=v.part_T)
        qf = K.evaluate(d, v.cadence, "fb_", part_T=v.part_T)
        qr = K.evaluate({**d, "v_med": d["v_rawdt"], "slow_ratio": d["slow_ratio_rawdt"]}, v.cadence, "", "kp_speed_rawdt", v.part_T)
        qs = K.evaluate({**d, "v_med": d["v_smoothdt"], "slow_ratio": d["slow_ratio_smoothdt"]}, v.cadence, "", "kp_speed_smoothdt", v.part_T)
        parts = q.pop("cadence_parts")
        fv, nall = d["frac_fb_valid"][:, :, 3], d["n_pixels"][:, :, 3]
        v_foot = d["v_med"][:, :, 3][np.isfinite(d["v_med"][:, :, 3])]
        u = ok & (nall >= C.MIN_PIXELS)
        both = np.isfinite(foot_any).all(1)
        q.update(video_id=vid, n_frames=n, n_pairs=int(len(all_pairs)), n_pairs_no_crop=int(len(all_pairs) - hf.sum()), comp_method=comp_method,
                 comp_slow_median=float(np.median(np.abs(slow[hf]))) if hf.any() else 0.0, comp_slow_max=float(np.max(np.abs(slow[hf]))) if hf.any() else 0.0, comp_direct_v_median=float(np.nanmedian(np.abs(direct_v[hf]))),
                 frac_comp_measured=float((comp_state[hf] == 0).mean()), frac_comp_static=float((comp_state[hf] == 1).mean()), frac_comp_filled=float((comp_state[hf] == 2).mean()), frac_comp_none=float((comp_state[hf] == 3).mean()),
                 n_cam_fit_bad=int(cam_bad.sum()), n_cam_bad_inlier=int(((cam_reason & C.CF_INLIER) != 0).sum()), n_cam_bad_bg=int(((cam_reason & C.CF_BG) != 0).sum()), n_cam_bad_tail=int(((cam_reason & C.CF_TAIL) != 0).sum()), n_cam_bad_slow=int(((cam_reason & C.CF_SLOW) != 0).sum()),
                 n_cam_bg_below_floor=int(((cam_reason & C.CF_BG_BELOW_FLOOR) != 0).sum()), frac_flow_valid=float(flow_valid[hf].mean()),
                 scale_min=float(np.nanmin(scale)), scale_max=float(np.nanmax(scale)), l_foot_grid=float(np.median(lf_all)) if lf_all else np.nan,
                 crop_edge_max=float(np.nanmax(np.maximum(crop[:, 2] - crop[:, 0], crop[:, 3] - crop[:, 1]) * np.nan_to_num(scale))),
                 frac_fb_valid=float(np.nansum(fv * nall) / max(nall[np.isfinite(fv)].sum(), 1)),
                 bg_residual_median=float(np.nanmedian(bg_res)), bg_residual_p95=bg_p95, cam_grid_median=float(np.nanmedian(cam_grid[hf])), overlay_frac=float(overlay.mean()), cam_static=cam_static, bg_residual_v=float(bg_v), bg_inlier_frac=float(np.nanmedian(inl)),
                 cam_flow_median=float(np.nanmedian(cam_px)), cam_dx_rho=cam_dx_rho, cam_dx_ratio=cam_dx_ratio,
                 dt_mode=dt_mode, dt_mode_reason=dt_why, dt_jitter=jitter, dt_rho_cam=rho_cam, dt_rho_kp=rho_kp, dt_cv=float(np.nanstd(dt_raw) / np.nanmean(dt_raw)),
                 spearman_fb=qf["spearman_flow_kp"], cadence_fb=qf["cadence_flow"], spearman_rawdt=qr["spearman_flow_kp"], spearman_smoothdt=qs["spearman_flow_kp"], cadence_ref_source=("part_T" if any(p["ref_source"] == "part_T" for p in parts) else v.cadence_source),
                 n_cadence_parts=len([p for p in parts if np.isfinite(p["cadence_flow"])]),
                 spearman_direct=qa["spearman_flow_kp"], cadence_flow_direct=qa["cadence_flow"], cadence_dev_direct=qa["cadence_dev"], vx_p10_direct=p10_direct, minima_per_stride_direct=qa["minima_per_stride"], power_ratio_2f_direct=qa["power_ratio_2f"],
                 alternation_rho_heel_direct=qa["alternation_rho_heel"], slow_ratio_median_direct=float(np.nanmedian(dA["slow_ratio"][:, :, 3])),
                 frac_both_still=float((foot_any[both] < C.TAIL_LO).all(1).mean()) if both.any() else np.nan, otsu_thr=d["otsu_thr"], fb_otsu_thr=d["fb_otsu_thr"],
                 v_foot_p10=float(np.percentile(v_foot, 10)), v_foot_p50=float(np.percentile(v_foot, 50)), v_foot_p90=float(np.percentile(v_foot, 90)),
                 slow_ratio_median=float(np.nanmedian(d["slow_ratio"][:, :, 3])), frac_usable_first=float(u[i, first].mean()), frac_usable_second=float(u[i, second].mean()), abs_time_ok=v.abs_time_ok,
                 device=state["device"], model=C.MODEL, config=C.config_hash())
        q.update(K.stance_metrics(d, v.cadence, v.part_T))
        K.extra_qc(d, q)
        q["gate"], q["gate_reasons"] = K.gate(q)
        return d, q, parts, ok, slow_disp

    results = {}
    for name, trigger in C.VARIANTS:
        results[name] = build_variant(name, trigger)
    default = C.VARIANTS[0][0]
    d, q, parts, ok, slow_disp = results[default]
    labels = sorted(set(z["label"].ravel().tolist()))
    q["foot_labels"] = "/".join(labels)
    q["default_variant"] = default
    for name, (dv, qv, pv, _, _) in results.items():
        suffix = "" if name == default else f"__{name}"
        C.atomic_npz(C.DIRS["features"] / f"{vid}{suffix}.npz", **dv)
        pd.DataFrame(pv).to_csv(C.DIRS["qc"] / f"{vid}{suffix}_cadence_parts.csv", index=False)
        if name != default:
            q.update({f"{name}_{k}": qv[k] for k in VARIANT_COLS})
    for name, (dv, qv, _, _, _) in results.items():
        if name == default:
            R.plot_video(vid, dv, qv, C.DIRS["plots"] / f"{vid}.png")
        elif qv["comp_method"] != q["comp_method"]:
            R.plot_video(vid, dv, qv, C.DIRS["plots"] / f"{vid}__{name}.png")
    cells = []
    for a, A, Reg, fb, sb, lf in held:
        F, *_ = st.get(a)
        res = F - np.array([tx[a], ty[a]], np.float32)
        res[..., 0] -= float(slow_disp[a] * scale[a] * l_leg[a] * v.s_final)
        cells.append(R.render_cell(A, res, Reg, fb, z["label"][a], ok[a], d["v_med"][a, :, 3], int(c.frame[a]), lf))
    R.save_grid(vid, cells, C.DIRS["review"] / f"{vid}.jpg")
    if state["device"] == "mps":
        torch.mps.empty_cache()
    dtv = time.time() - t1
    q.update(time_s=round(dtv, 1), fps=round(len(st.a) / max(dtv, 1e-6), 2))
    (C.DIRS["qc"] / f"{vid}_qc.json").write_text(json.dumps({k: (round(w, 4) if isinstance(w, float) else w) for k, w in q.items()}, ensure_ascii=False, indent=1, default=float))
    print(f"  {vid}: {len(st.a)} cặp, feature {dtv:.1f}s gate={q['gate']} {q['gate_reasons']} {q['comp_method']} P10 {q['vx_p10_direct']:.2f}->{q['vx_p10']:.2f} rho={q['spearman_flow_kp']:.2f} "
          f"cad={q['cadence_flow']:.0f}/{q['cadence_ref_used']:.0f} duty={q['duty_factor_first']:.2f}/{q['duty_factor_second']:.2f} slowBad={q['n_cam_bad_slow']}", flush=True)
    return q


def done(vid):
    p = C.DIRS["qc"] / f"{vid}_qc.json"
    need = [C.DIRS["features"] / f"{vid}.npz", C.DIRS["review"] / f"{vid}.jpg", C.DIRS["plots"] / f"{vid}.png"] + [C.DIRS["features"] / f"{vid}__{name}.npz" for name, _ in C.VARIANTS[1:]]
    if not (p.exists() and all(x.exists() for x in need)):
        return False
    return json.loads(p.read_text()).get("config") == C.config_hash() and T.done(vid)


def write_table(meta):
    rows = [json.loads((C.DIRS["qc"] / f"{v}_qc.json").read_text()) for v in meta.video_id if (C.DIRS["qc"] / f"{v}_qc.json").exists()]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df[QC_COLS].to_csv(C.OUT / "qc_flow.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--store-only", action="store_true", help="chỉ chạy GMA vào flow_store, không tính feature")
    a = ap.parse_args()
    C.ensure_dirs()
    T.STORE.mkdir(parents=True, exist_ok=True)
    meta = C.S4.load_meta()
    vids = [v for x in a.videos for v in x.split(",") if v] if a.videos else meta.video_id.tolist()
    bad = sorted(set(vids) - set(meta.video_id))
    if bad:
        raise SystemExit(f"video không có trong metadata: {bad}")
    if a.store_only:
        need = [v for v in vids if not T.done(v)]
        print(f"{len(vids)} video, {len(need)} cần chạy GMA", flush=True)
        if need:
            device = a.device if a.device != "mps" or torch.backends.mps.is_available() else "cpu"
            state = dict(device=device, model=W.build(device))
            t0 = time.time()
            for k, v in enumerate(need, 1):
                dtf, nf = T.build(v, meta, state)
                print(f"[{k}/{len(need)}] {v}: {nf} cặp, {dtf:.1f}s, kho {(T.STORE / f'{v}.npz').stat().st_size / 1e6:.1f} MB", flush=True)
                if state["device"] == "mps":
                    torch.mps.empty_cache()
            print(f"Xong GMA {len(need)} video trong {time.time() - t0:.1f}s", flush=True)
        return
    todo = [v for v in vids if a.force or not done(v)]
    print(f"{len(vids)} video, {len(vids) - len(todo)} đã xong, {len(todo)} cần chạy; model={C.MODEL}, iters={C.ITERS}, device={a.device}", flush=True)
    t0 = time.time()
    if todo:
        device = a.device if a.device != "mps" or torch.backends.mps.is_available() else "cpu"
        if device != a.device:
            print("  CẢNH BÁO: không có MPS -> cpu", flush=True)
        state = dict(device=device, model=W.build(device) if not all(T.done(v) for v in todo) else None)
        for k, v in enumerate(todo, 1):
            print(f"[{k}/{len(todo)}]", end="", flush=True)
            if state["model"] is None and not T.done(v):
                state["model"] = W.build(device)
            process(v, meta, state)
        print(f"Xong {len(todo)} video trong {time.time() - t0:.1f}s", flush=True)
    df = write_table(meta)
    if df is not None:
        print("\nGate:", df[df.video_id.isin(vids)].gate.value_counts().to_dict())


if __name__ == "__main__":
    main()
