#!/usr/bin/env python3
"""Bước 4: foot mask bằng SAM2 hai object. Chạy từ ROOT/data:

  ../.venv-track/bin/python step4_footmask/code/run_step4.py                     # cả 66 video, resume
  ../.venv-track/bin/python step4_footmask/code/run_step4.py --videos C30,F26
  ... --force  ... --device cpu  ... --checkpoint base_plus
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

import s4_common as C
import s4_regions as G
import s4_report as R
import s4_sam as Q

QC_COLS = ["video_id", "gate", "gate_reasons", "n_frames", "n_blocks", "frac_visible_near", "frac_partial_near", "frac_occluded_near",
           "frac_visible_far", "frac_partial_far", "frac_occluded_far", "frac_visible_A", "frac_partial_A", "frac_occluded_A",
           "frac_visible_B", "frac_partial_B", "frac_occluded_B", "mean_coverage", "mean_coverage_tol", "n_reprompt", "n_reprompt_skipped", "max_reprompt_block",
           "n_mask_trim_frames", "frac_mask_leak", "n_id_conflict_frames", "n_id_conflict_runs", "near_far_occlusion_agreement", "near_far_agreement_area", "n_occlusion_frames", "occlusion_agreement_source",
           "median_mask_iou", "frac_labelled_near_far", "frac_no_axis", "frac_axis_short", "frac_outside_regions", "perp_extent_median",
           "region_area_frac_heel", "region_area_frac_mid", "region_area_frac_toe", "device", "time_s", "fps", "checkpoint", "config"]


def config_hash():
    import hashlib
    d = {k: getattr(C, k) for k in dir(C) if k.isupper() and isinstance(getattr(C, k), (int, float, str, tuple, dict)) and k not in ("DIRS",)}
    return hashlib.sha1(repr(sorted((k, repr(v)) for k, v in d.items())).encode()).hexdigest()[:10]


def analyse_block(c, idx, win, M, ov, log, init, Lf, trimmed):
    off = np.array(win[:2], float)
    n, (H, W) = len(idx), M.shape[2:]
    met = Q.frame_metrics(c, idx, M, Lf, off)
    inter = (M[:, 0] & M[:, 1]).sum((1, 2)).astype(float)
    union = (M[:, 0] | M[:, 1]).sum((1, 2)).astype(float)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    st = G.states(met["area"], met["ratio"], iou, met["cov_tol"])
    conflict = G.conflicts(c, idx, met, Lf)
    origin, axis = np.full((n, 2, 2), np.nan), np.full((n, 2, 2), np.nan)
    borrowed = np.zeros((n, 2), bool)
    for o in (1, 2):
        origin[:, o - 1], axis[:, o - 1], borrowed[:, o - 1] = G.foot_axes(c, idx, o, M[:, o - 1], off)
    region_area = np.full((n, 2, 3), np.nan)
    outside, perp = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    rle, ptr = [], np.zeros((n, 2, 4, 2), np.int64)
    pos = 0
    full = np.zeros((c.H, c.W), bool)
    x0, y0, x1, y1 = win

    def put(a, o, k, m):
        nonlocal pos
        full[y0:y1, x0:x1] = m
        r = C.rle_encode(full)
        rle.append(r)
        ptr[a, o, k] = (pos, pos + len(r))
        pos += len(r)

    for a in range(n):
        for o in (0, 1):
            put(a, o, 0, M[a, o])
            regs = [np.zeros((H, W), bool)] * 3
            if st[a, o] != C.ST_OCCLUDED:
                regs, outside[a, o], perp[a, o] = G.region_split(M[a, o], origin[a, o], axis[a, o], Lf[o + 1], off)
                if np.isfinite(axis[a, o]).all():
                    region_area[a, o] = [r.sum() for r in regs]
            for k, r in enumerate(regs):
                put(a, o, k + 1, r)
    rp = np.zeros((n, 2), bool)
    real = [l for l in log if l["action"] == "prompt_lai"]
    for l in real:
        rp[np.flatnonzero(c.frame[idx] == l["frame"]), l["obj"] - 1] = True
    return dict(idx=idx, area=met["area"], ratio=met["ratio"], cov=met["cov"], cov_tol=met["cov_tol"], other=met["other"], iou=iou, state=st, conflict=conflict,
                origin=origin, axis=axis, borrowed=borrowed, region_area=region_area, outside=outside, perp=perp, rle=rle, ptr=ptr,
                reprompt=rp, trimmed=trimmed, label=G.labels(c, idx), ov=ov, win=win, init=init, Lf=Lf, nrep=len(real), nskip=len(log) - len(real))


def merge(c, res):
    """Ghép các khối theo thứ tự frame."""
    n = c.n
    out = {k: np.full((n, 2) + res[0][k].shape[2:], np.nan) for k in ("area", "ratio", "cov", "cov_tol", "origin", "axis", "region_area", "outside", "perp")}
    state, conflict, rp, borrowed = np.zeros((n, 2), np.uint8), np.zeros((n, 2), bool), np.zeros((n, 2), bool), np.zeros((n, 2), bool)
    trimmed = np.zeros((n, 2), bool)
    iou, ov, lf = np.zeros(n), np.full(n, np.nan), np.full((n, 2), np.nan)
    label = np.empty((n, 2), "U4")
    win, init_f = np.zeros((n, 4), np.int32), np.zeros(n, np.int64)
    ptr, rle, shift = np.zeros((n, 2, 4, 2), np.int64), [], 0
    for r in res:
        i = r["idx"]
        for k in out:
            out[k][i] = r[k]
        state[i], conflict[i], rp[i], borrowed[i], trimmed[i] = r["state"], r["conflict"], r["reprompt"], r["borrowed"], r["trimmed"]
        iou[i], ov[i], label[i] = r["iou"], r["ov"], r["label"]
        win[i] = r["win"]
        init_f[i] = c.frame[i[r["init"]]] if r["init"] is not None else -1
        lf[i] = (r["Lf"][1], r["Lf"][2])
        ptr[i] = r["ptr"] + shift
        rle += r["rle"]
        shift += sum(len(x) for x in r["rle"])
    return dict(**out, state=state, conflict=conflict, reprompt=rp, borrowed=borrowed, trimmed=trimmed, iou=iou, ov=ov, label=label, win=win, init=init_f, l_foot=lf,
                ptr=ptr, counts=np.concatenate(rle).astype(np.uint32) if rle else np.zeros(0, np.uint32))


def qc_row(c, m, res, dt, device, ckpt):
    n = c.n
    q = dict(video_id=c.vid, n_frames=n, n_blocks=len(res))
    st, lab = m["state"], m["label"]
    for name in ("near", "far", "A", "B"):
        w = lab == name
        for s, sn in enumerate(C.STATE_NAMES):
            q[f"frac_{sn}_{name}"] = float((st[w] == s).mean()) if w.any() else np.nan
    allv = [(st[:, o] == C.ST_VISIBLE).mean() for o in (0, 1)]
    # video/khối không có nhãn near/far: gate dùng chân thấy rõ hơn thay cho near, chân còn lại thay cho far
    q["frac_visible_best_AB"], q["frac_visible_worst_AB"] = float(max(allv)), float(min(allv))
    q["mean_coverage"] = float(np.nanmean(m["cov"]))
    q["mean_coverage_tol"] = float(np.nanmean(m["cov_tol"]))
    q["n_reprompt"] = int(sum(r["nrep"] for r in res))
    q["n_reprompt_skipped"] = int(sum(r["nskip"] for r in res))
    q["max_reprompt_block"] = int(max(r["nrep"] for r in res))
    q["n_mask_trim_frames"] = int(m["trimmed"].any(1).sum())
    q["frac_mask_leak"] = float(np.nanmean(m["outside"] > C.LEAK_OUTSIDE)) if np.isfinite(m["outside"]).any() else np.nan
    any_c = m["conflict"].any(1)
    q["n_id_conflict_frames"] = int(any_c.sum())
    # đoạn >= 3 frame unique liên tiếp, không vắt qua khối
    rr = []
    for r in res:
        for a, b in G.runs(r["conflict"].any(1), C.CONFLICT_RUN):
            rr.append(dict(a=int(r["idx"][a]), b=int(r["idx"][b]), frame_from=int(c.frame[r["idx"][a]]), frame_to=int(c.frame[r["idx"][b]]),
                           continuity_id=int(c.cont[r["idx"][a]]), n=int(b - a + 1)))
    q["n_id_conflict_runs"] = len(rr)
    # chân near che chân far: vùng giao thuộc về mask near (logit cao hơn)
    lab_ok = lab[:, 0] != "A"
    near_is_1 = lab[:, 0] == "near"
    # chỉ xét chồng lấn thật: IoU rất cao là mask đã dính vào nhau, không nói gì về chân nào che chân nào
    has = lab_ok & np.isfinite(m["ov"]) & (m["iou"] < C.COLLAPSE_IOU)
    agree = (m["ov"][has] > 0) == near_is_1[has]
    q["near_far_occlusion_agreement"] = float(agree.mean()) if has.sum() >= 3 else np.nan
    q["n_occlusion_frames"] = int(has.sum())
    q["occlusion_agreement_source"] = "logit_vung_giao" if has.sum() >= 3 else "khong_du_frame"
    rn = np.where(near_is_1, m["ratio"][:, 0], m["ratio"][:, 1])
    rf = np.where(near_is_1, m["ratio"][:, 1], m["ratio"][:, 0])
    q["near_far_agreement_area"] = float((rn[has] > rf[has]).mean()) if has.sum() >= 3 else np.nan
    pos = m["iou"][m["iou"] > 0]
    q["median_mask_iou"] = float(np.median(pos)) if len(pos) else 0.0
    q["frac_labelled_near_far"] = float(lab_ok.mean())
    vis = st != C.ST_OCCLUDED
    q["frac_no_axis"] = float((~np.isfinite(m["axis"][:, :, 0]))[vis].mean()) if vis.any() else np.nan
    alen = np.linalg.norm(m["axis"], axis=2) / m["l_foot"]
    q["frac_axis_short"] = float(np.nanmean(alen[vis] < 0.7)) if vis.any() else np.nan
    q["frac_outside_regions"] = float(np.nanmedian(m["outside"]))
    q["perp_extent_median"] = float(np.nanmedian(m["perp"]))
    tot = np.nansum(m["region_area"], axis=(0, 1))
    for k, nm in enumerate(C.REGIONS):
        q[f"region_area_frac_{nm}"] = float(tot[k] / max(tot.sum(), 1))
    q.update(device=device, time_s=round(dt, 1), fps=round(n / max(dt, 1e-6), 2), checkpoint=ckpt, config=config_hash())
    q["gate"], q["gate_reasons"] = R.gate(q)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}, rr


def process(vid, meta, state, ckpt):
    c = C.load_clip(vid, meta)
    t0 = time.time()
    blocks = {int(b): (np.flatnonzero(c.cont == b), C.crop_window(c, np.flatnonzero(c.cont == b))) for b in np.unique(c.cont)}
    dirs = Q.extract_frames(c, blocks)
    t1 = time.time()
    res, logs = [], []
    try:
        for b, (idx, win) in blocks.items():
            while True:
                try:
                    M, ov, log, init, Lf, trimmed = Q.run_block(state["pred"], c, idx, win, dirs[b], state["device"])
                    break
                except (RuntimeError, NotImplementedError) as e:
                    if state["device"] == "cpu":
                        raise
                    print(f"  CẢNH BÁO: SAM2 {state['device']} lỗi ở {vid} khối {b} ({type(e).__name__}: {str(e)[:200]}) -> chuyển cpu", flush=True)
                    state["device"], state["pred"] = "cpu", Q.build(ckpt, "cpu")
            if init is None:
                print(f"  CẢNH BÁO: {vid} khối {b}: không có frame khởi tạo", flush=True)
            res.append(analyse_block(c, idx, win, M, ov, log, init if init is not None else 0, Lf, trimmed))
            logs += log
            del M
            if state["device"] == "mps":
                torch.mps.empty_cache()
    finally:
        Q.cleanup(c)
    t2 = time.time()
    m = merge(c, res)
    q, rr = qc_row(c, m, res, t2 - t0, state["device"], ckpt)
    C.atomic_npz(C.DIRS["masks"] / f"{vid}.npz", frame=c.frame, t=c.t, continuity_id=c.cont, W=c.W, H=c.H,
                 rle_counts=m["counts"], rle_ptr=m["ptr"], state=m["state"], area=m["area"], area_ratio=m["ratio"], coverage=m["cov"], coverage_tol=m["cov_tol"],
                 origin=m["origin"], axis=m["axis"], axis_borrowed=m["borrowed"], l_foot=m["l_foot"], region_area=m["region_area"],
                 iou=m["iou"], label=m["label"], reprompt_flag=m["reprompt"], mask_trimmed=m["trimmed"], mask_id_conflict=m["conflict"], crop_window=m["win"],
                 crop_scale=1.0, init_frame=m["init"], config=config_hash(), checkpoint=ckpt)
    pd.DataFrame(logs, columns=["video_id", "continuity_id", "obj", "frame", "bad_from", "bad_to", "action", "round"]).to_csv(C.DIRS["qc"] / f"{vid}_reprompt.csv", index=False)
    pd.DataFrame(rr, columns=["frame_from", "frame_to", "continuity_id", "n"]).to_csv(C.DIRS["qc"] / f"{vid}_conflicts.csv", index=False)
    z = np.load(C.DIRS["masks"] / f"{vid}.npz")
    R.review(c, z, C.DIRS["review"] / f"{vid}.jpg")
    R.conflict_images(c, z, rr)
    t3 = time.time()
    q["time_extract_s"], q["time_sam_s"], q["time_images_s"] = round(t1 - t0, 1), round(t2 - t1, 1), round(t3 - t2, 1)
    (C.DIRS["qc"] / f"{vid}_qc.json").write_text(json.dumps(q, ensure_ascii=False, indent=1, default=float))
    print(f"  {vid}: {c.n} frame, {t3 - t0:.1f}s (SAM {t2 - t1:.1f}s, {c.n / max(t2 - t1, 1e-6):.1f} fps) gate={q['gate']} {q['gate_reasons']} "
          f"reprompt={q['n_reprompt']} conflict={q['n_id_conflict_frames']}", flush=True)


def done(vid):
    p = C.DIRS["qc"] / f"{vid}_qc.json"
    if not (p.exists() and (C.DIRS["masks"] / f"{vid}.npz").exists() and (C.DIRS["review"] / f"{vid}.jpg").exists()):
        return False
    return json.loads(p.read_text()).get("config") == config_hash()


def write_table(meta):
    rows = [json.loads((C.DIRS["qc"] / f"{v}_qc.json").read_text()) for v in meta.video_id if (C.DIRS["qc"] / f"{v}_qc.json").exists()]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df[QC_COLS].to_csv(C.OUT / "qc_footmask.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--checkpoint", default=C.CHECKPOINT, choices=list(C.CKPT))
    a = ap.parse_args()
    C.ensure_dirs()
    meta = C.load_meta()
    vids = meta.video_id.tolist()
    if a.videos:
        vids = [v for x in a.videos for v in x.split(",") if v]
        bad = sorted(set(vids) - set(meta.video_id))
        if bad:
            raise SystemExit(f"video không có trong metadata: {bad}")
    todo = [v for v in vids if a.force or not done(v)]
    print(f"{len(vids)} video, {len(vids) - len(todo)} đã xong, {len(todo)} cần chạy; checkpoint={a.checkpoint}, device={a.device}", flush=True)
    t0 = time.time()
    if todo:
        device = a.device if a.device != "mps" or torch.backends.mps.is_available() else "cpu"
        if device != a.device:
            print("  CẢNH BÁO: không có MPS -> cpu", flush=True)
        state = dict(device=device, pred=Q.build(a.checkpoint, device))
        for i, v in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}]", end="", flush=True)
            process(v, meta, state, a.checkpoint)
        print(f"Xong {len(todo)} video trong {time.time() - t0:.1f}s", flush=True)
    R.write_review_csv(meta)
    df = write_table(meta)
    if df is not None:
        d = df[df.video_id.isin(vids)]
        print("\nGate:", d.gate.value_counts().to_dict())
        pd.set_option("display.width", 220, "display.max_rows", 100, "display.max_colwidth", 60)
        print(d[["video_id", "gate", "gate_reasons", "frac_visible_near", "frac_visible_far", "frac_visible_A", "frac_visible_B", "mean_coverage", "n_reprompt", "n_id_conflict_frames", "near_far_occlusion_agreement", "median_mask_iou", "fps"]].to_string(index=False))


if __name__ == "__main__":
    main()
