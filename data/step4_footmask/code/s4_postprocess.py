#!/usr/bin/env python3
"""Hậu kỳ Bước 4 trên mask đã lưu, không chạy SAM2. Đọc masks/, ghi masks_post/, qc_post/, qc_footmask.csv, review/. Chạy từ ROOT/data:

  ../.venv-track/bin/python step4_footmask/code/s4_postprocess.py               # cả 66 video, vẽ lại ảnh, kiểm tra
  ../.venv-track/bin/python step4_footmask/code/s4_postprocess.py --no-review   # không vẽ ảnh
  ../.venv-track/bin/python step4_footmask/code/s4_postprocess.py --check       # chỉ kiểm tra output hiện có

Hằng số hậu kỳ nằm ở file này, không nằm trong s4_common.py, để config hash của SAM2 giữ nguyên.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import run_step4 as S
import s4_common as C
import s4_regions as G
import s4_report as R

# 1. Ankle clip
ANKLE_MARGIN = 0.15         # × L_foot, lề phía trên cổ chân
MIN_AXIS_FOR_CLIP = 0.4     # × L_foot: trục ngắn hơn (bàn chân nhìn thẳng từ mũi/gót) thì hướng trục không đáng tin, không clip
MIN_KNEE_OFFSET = 0.2       # × L_foot: gối lệch trục ít hơn mức này thì không dùng gối để xác định phía ống chân
MIN_ANKLE_OFFSET = 0.05     # × L_foot

# 2. Gate
FAIL_CONFLICT = 0.10
FAIL_NEAR = 0.6
WARN_FAR = 0.4
WARN_REPROMPT_PER100 = 20.0
WARN_LEAK = 0.1
# Xung đột ID lấy từ mask trước clip (hợp với sau clip). Clip chỉ cắt bớt phần mask phủ keypoint chân kia nên cờ có thể tắt trong khi mask vẫn nằm sai chân
# (C04_3 f14, f23, f87); clip không sửa danh tính.
CONFLICT_SOURCE = "before_clip_or_after_clip"

# 3. Kiểm tra theo frame, theo từng chân. Ngưỡng theo quyết định review; 2 chuẩn hoá thêm vì luật nguyên văn bắt nhầm mask đúng:
#    F30 26% và C30 48% do độ dày (L_foot đo từ gót tới ngón nên ngắn hơn giày thật), F10_1 26% và F10_2 19% do diện tích (người lại gần/đi xa trong khối).
AREA_HI, AREA_LO = 1.6, 0.4 # × trung vị của chân đó trong khối, sau khi chuẩn hoá area / L_leg(t)²
KP_DIST = 0.2               # × L_foot: cả gót, ngón cái, ngón út đều xa mask hơn mức này
THICK = 0.7                 # × L_foot, P95 − P5 theo hướng vuông góc với trục
THICK_REL = 1.3             # và lớn hơn mức này × trung vị độ dày của chân đó trong khối
R_OCCLUDED, R_CONFLICT, R_AREA_HIGH, R_AREA_LOW, R_NO_KP, R_THICK, R_LITERAL_ONLY = 1, 2, 4, 8, 16, 32, 64
INVALID_BITS = R_OCCLUDED | R_CONFLICT | R_AREA_HIGH | R_AREA_LOW | R_NO_KP | R_THICK   # thêm R_LITERAL_ONLY để dùng luật nguyên văn
REASON_NAMES = {R_OCCLUDED: "occluded", R_CONFLICT: "id_conflict", R_AREA_HIGH: "area_high", R_AREA_LOW: "area_low", R_NO_KP: "no_keypoint", R_THICK: "perp_thick"}

# 4. F10_2. Khối 1: near_track=B của Bước 3 trái với ảnh (f545, f550: chân A đè lên chân B), trái metadata (A), và near_conf=1.0 chỉ từ 1 chunk.
# Điều kiện "nếu Bước 3 tin cậy" không đạt nên khối này giữ A/B. Khối 2: đoạn cuối quay lưng, xác định theo độ rộng hông.
F10_2 = "F10_2"
F10_2_RELIABLE_BLOCKS = {2}
BACK_BLOCK = 2
BACK_HIP_THR = 0.30         # |x hông trái − x hông phải| / chiều dài thân; f1072–1078 quay lưng rõ có khoảng 0,33–0,38, f985–1017 khoảng 0,13
BACK_WIN = 5

# 5. Bước 3
TIME_FLAGS = C.DATA / "step3_footkp" / "time_feature_flags.csv"
TIME_EXCLUDE = {"C20": {1: "Part 1: tỉ số d_max/d_min = 3,78 nên gần như chắc là ramp, nhưng chỉ có 2 bước nên không dựng được k(t); k = 1/5 độ tin cậy thấp"}}

OUT_MASKS = C.OUT / "masks_post"
OUT_QC = C.OUT / "qc_post"
QC_COLS = ["video_id", "gate", "gate_reasons", "n_frames", "n_blocks", "frac_visible_near", "frac_partial_near", "frac_occluded_near",
           "frac_visible_far", "frac_partial_far", "frac_occluded_far", "frac_visible_A", "frac_partial_A", "frac_occluded_A",
           "frac_visible_B", "frac_partial_B", "frac_occluded_B", "frac_frame_valid", "frac_valid_near", "frac_valid_far", "frac_valid_A", "frac_valid_B",
           "frac_removed_new", "frac_removed_new_literal", "n_flag_area", "n_flag_no_keypoint", "n_flag_thick", "n_frames_conflict_invalid",
           "mean_coverage", "mean_coverage_tol", "mean_coverage_tol_before", "n_reprompt", "n_reprompt_skipped", "max_reprompt_block",
           "reprompt_per100", "reprompt_per100_max_block", "n_mask_trim_frames", "frac_mask_leak", "frac_mask_leak_before",
           "n_clip_frames", "n_clip_skipped", "frac_area_clipped", "n_id_conflict_frames", "n_id_conflict_runs",
           "median_mask_iou", "frac_labelled_near_far", "frac_view_lateral", "frac_exclude_time_features", "frac_no_axis", "frac_axis_short",
           "frac_outside_regions", "perp_extent_median", "region_area_frac_heel", "region_area_frac_mid", "region_area_frac_toe",
           "region_area_frac_heel_before", "device", "time_s", "fps", "checkpoint", "config", "post_config"]


def post_config():
    d = {k: globals()[k] for k in ("ANKLE_MARGIN", "MIN_AXIS_FOR_CLIP", "MIN_KNEE_OFFSET", "MIN_ANKLE_OFFSET", "FAIL_CONFLICT", "FAIL_NEAR", "WARN_FAR",
                                   "WARN_REPROMPT_PER100", "WARN_LEAK", "CONFLICT_SOURCE", "AREA_HI", "AREA_LO", "KP_DIST", "THICK", "THICK_REL", "INVALID_BITS", "F10_2_RELIABLE_BLOCKS", "BACK_BLOCK", "BACK_HIP_THR", "BACK_WIN", "TIME_EXCLUDE")}
    return hashlib.sha1(repr(sorted((k, repr(v)) for k, v in d.items())).encode()).hexdigest()[:10]


def f10_2_back_view(c):
    """Đoạn quay lưng: hậu tố dài nhất của khối 2 có trung vị trượt độ rộng hông >= ngưỡng. Trả về mảng bool N."""
    K = np.load(C.DATA / "step2_tracking" / "keypoints_v2" / f"{c.vid}_kp.npz", allow_pickle=True)["kp"].astype(float)
    idx = np.flatnonzero(c.cont == BACK_BLOCK)
    torso = np.linalg.norm((K[idx, 5] + K[idx, 6]) / 2 - (K[idx, 11] + K[idx, 12]) / 2, axis=1)
    hip = pd.Series(np.abs(K[idx, 11, 0] - K[idx, 12, 0]) / torso).rolling(BACK_WIN, center=True, min_periods=3).median().values
    low = np.flatnonzero(hip < BACK_HIP_THR)
    start = low[-1] + 1 if len(low) else 0
    out = np.zeros(c.n, bool)
    out[idx[start:]] = True
    return out


def apply_overrides(c):
    """Sửa c.reliable cho F10_2, trả về view_lateral N."""
    view = np.ones(c.n, bool)
    if c.vid == F10_2:
        back = f10_2_back_view(c)
        view[back] = False
        c.reliable = np.isin(c.cont, [0] + sorted(F10_2_RELIABLE_BLOCKS)) & ~back
    return view


def time_exclude_mask(c):
    z = np.load(C.DATA / "step3_footkp" / "footkp" / f"{c.vid}_footkp.npz", allow_pickle=True)
    out = np.zeros(c.n, bool)
    for part in TIME_EXCLUDE.get(c.vid, {}):
        out |= z["speed_part_id"].astype(int) == part
    return out


def write_time_flags():
    rows = []
    for vid, parts in TIME_EXCLUDE.items():
        z = np.load(C.DATA / "step3_footkp" / "footkp" / f"{vid}_footkp.npz", allow_pickle=True)
        for part, why in parts.items():
            w = z["speed_part_id"].astype(int) == part
            rows.append(dict(video_id=vid, speed_part_id=part, frame_from=int(z["frame"][w][0]), frame_to=int(z["frame"][w][-1]), n_frames_unique=int(w.sum()),
                             exclude_time_features=True, reason=why))
    pd.DataFrame(rows).to_csv(TIME_FLAGS, index=False)


def clip_block(c, idx, M, off, Lf):
    """Bỏ pixel mask ở phía ống chân, quá đường song song với trục bàn chân đi qua cổ chân cộng lề. M sửa tại chỗ.
    Đường vuông góc với trục đi qua cổ chân sẽ cắt ngang bàn chân, nên thước đo là toạ độ q vuông góc với trục. Trả về (clipped N×2, skipped N×2, pixel bỏ)."""
    n = len(idx)
    clipped, skipped = np.zeros((n, 2), bool), np.zeros((n, 2), bool)
    removed = 0
    for o in (1, 2):
        origin, axis, _ = G.foot_axes(c, idx, o, M[:, o - 1], off)
        ank_i, knee_i, L = C.FOOT[o][C.ANK], C.KNEE[o], Lf[o]
        alen = np.linalg.norm(axis, axis=1)
        with np.errstate(all="ignore"):
            u = axis / alen[:, None]
        nrm = np.c_[-u[:, 1], u[:, 0]]
        qa = ((c.kp[idx, ank_i] - origin) * nrm).sum(1)
        qk = ((c.kp[idx, knee_i] - origin) * nrm).sum(1)
        ank_ok = c.valid[idx, ank_i] & np.isfinite(qa)
        knee_ok = c.valid[idx, knee_i] & np.isfinite(qk) & ank_ok
        # độ cao cổ chân so với trục, theo L_foot; frame thiếu cổ chân dùng trung vị khối
        side = np.where(knee_ok & (np.abs(qk - qa) >= MIN_KNEE_OFFSET * L), np.sign(qk - qa),
                        np.where(ank_ok & (np.abs(qa) >= MIN_ANKLE_OFFSET * L), np.sign(qa), 0.0))
        h = np.where(ank_ok, qa * side, np.nan)
        hm = np.nanmedian(h[side != 0]) if np.isfinite(h[side != 0]).any() else 0.0
        for a in range(n):
            if not M[a, o - 1].any():
                continue
            if not np.isfinite(axis[a]).all() or not np.isfinite(origin[a]).all() or alen[a] < MIN_AXIS_FOR_CLIP * L:
                skipped[a, o - 1] = True
                continue
            s = side[a]
            if s == 0:      # không biết phía ống chân
                skipped[a, o - 1] = True
                continue
            lim = max(h[a] if np.isfinite(h[a]) else hm, 0.0) + ANKLE_MARGIN * L   # không bao giờ thấp hơn trục, kể cả khi keypoint cổ chân lệch
            ys, xs = np.nonzero(M[a, o - 1])
            q = ((xs + off[0] - origin[a, 0]) * nrm[a, 0] + (ys + off[1] - origin[a, 1]) * nrm[a, 1]) * s
            drop = q > lim
            if drop.any():
                M[a, o - 1, ys[drop], xs[drop]] = False
                clipped[a, o - 1] = True
                removed += int(drop.sum())
    return clipped, skipped, removed


def mask_geometry(c, idx, M, off, origin, axis, Lf):
    """kp_dist N×2: khoảng cách nhỏ nhất từ gót, ngón cái, ngón út tới mask, theo L_foot; NaN nếu mask rỗng hoặc không có điểm.
    thick N×2: P95 − P5 của toạ độ vuông góc với trục, theo L_foot."""
    n = len(idx)
    kp_dist, thick = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    for o in (1, 2):
        pts = [C.FOOT[o][j] for j in (C.HEEL, C.BIG, C.SMALL)]
        L = Lf[o]
        for a, i in enumerate(idx):
            ys, xs = np.nonzero(M[a, o - 1])
            if not len(xs):
                continue
            xy = np.c_[xs + off[0], ys + off[1]]
            P = c.kp[i, pts]
            P = P[np.isfinite(P).all(1)]
            if len(P):
                kp_dist[a, o - 1] = np.sqrt(((xy[None] - P[:, None]) ** 2).sum(2).min()) / L
            ax, og = axis[a, o - 1], origin[a, o - 1]
            alen = np.linalg.norm(ax)
            if np.isfinite(ax).all() and np.isfinite(og).all() and alen > 1e-6:
                q = (-(xy[:, 0] - og[0]) * ax[1] + (xy[:, 1] - og[1]) * ax[0]) / alen / L
                thick[a, o - 1] = np.percentile(q, 95) - np.percentile(q, 5)
    return kp_dist, thick


def invalid_reasons(c, m, kp_dist, thick, conflict_frame):
    """Trả về (reason N×2 uint8, area_norm N×2 tỉ lệ diện tích đã chuẩn hoá)."""
    Lleg = np.load(C.DATA / "step3_footkp" / "footkp" / f"{c.vid}_footkp.npz", allow_pickle=True)["L_leg"].astype(float)
    area, ratio, st = m["area"], m["ratio"], m["state"]
    an = area / (Lleg[:, None] ** 2)
    norm, trel = np.full(area.shape, np.nan), np.full(area.shape, np.nan)
    for b in np.unique(c.cont):
        w = np.flatnonzero(c.cont == b)
        for o in (0, 1):
            g = area[w, o] > 0
            if np.isfinite(an[w, o][g]).any():
                norm[w, o] = an[w, o] / np.nanmedian(an[w, o][g])
            t = thick[w, o][st[w, o] != C.ST_OCCLUDED]
            if np.isfinite(t).any():
                trel[w, o] = thick[w, o] / np.nanmedian(t)
    norm = np.where(np.isfinite(norm), norm, ratio)
    r = np.zeros(area.shape, np.uint8)
    r[st == C.ST_OCCLUDED] |= R_OCCLUDED
    r[conflict_frame] |= R_CONFLICT
    r[norm > AREA_HI] |= R_AREA_HIGH
    r[(norm < AREA_LO) & (area > 0)] |= R_AREA_LOW
    r[kp_dist > KP_DIST] |= R_NO_KP
    r[(thick > THICK) & (trel > THICK_REL)] |= R_THICK
    literal = (ratio > AREA_HI) | ((ratio < AREA_LO) & (area > 0)) | (thick > THICK)
    r[literal & ((r & (R_AREA_HIGH | R_AREA_LOW | R_THICK)) == 0)] |= R_LITERAL_ONLY
    return r, norm


def process(vid, meta):
    c = C.load_clip(vid, meta)
    view = apply_overrides(c)
    z = np.load(C.DIRS["masks"] / f"{vid}.npz", allow_pickle=True)
    raw_q = json.loads((C.DIRS["qc"] / f"{vid}_qc.json").read_text())
    if str(z["config"]) != S.config_hash() or raw_q.get("config") != S.config_hash():
        raise SystemExit(f"{vid}: masks/ có config {z['config']}, cấu hình SAM2 hiện tại {S.config_hash()}")
    if not np.array_equal(z["frame"], c.frame):
        raise SystemExit(f"{vid}: frame trong masks/ khác Bước 3")
    logs = pd.read_csv(C.DIRS["qc"] / f"{vid}_reprompt.csv")
    shape = (c.H, c.W)
    res, clipped, skipped = [], np.zeros((c.n, 2), bool), np.zeros((c.n, 2), bool)
    removed, total, per_block = 0, 0, []
    kp_dist, thick = np.full((c.n, 2), np.nan), np.full((c.n, 2), np.nan)
    for b in np.unique(c.cont):
        idx = np.flatnonzero(c.cont == b)
        win = tuple(int(v) for v in z["crop_window"][idx[0]])
        x0, y0, x1, y1 = win
        off = np.array(win[:2], float)
        M = np.zeros((len(idx), 2, y1 - y0, x1 - x0), bool)
        for n, a in enumerate(idx):
            for o in (0, 1):
                M[n, o] = R.decode(z, a, o, 0, shape)[y0:y1, x0:x1]
        total += int(M.sum())
        Lf = {o: float(z["l_foot"][idx[0], o - 1]) for o in (1, 2)}
        clipped[idx], skipped[idx], rem = clip_block(c, idx, M, off, Lf)
        removed += rem
        log = logs[logs.continuity_id == b].to_dict("records")
        hit = np.flatnonzero(c.frame[idx] == z["init_frame"][idx[0]])
        init = int(hit[0]) if len(hit) else 0
        res.append(S.analyse_block(c, idx, win, M, np.full(len(idx), np.nan), log, init, Lf, z["mask_trimmed"][idx]))
        res[-1]["conflict"] = res[-1]["conflict"] | z["mask_id_conflict"][idx]
        per_block.append(100.0 * res[-1]["nrep"] / len(idx))
        kp_dist[idx], thick[idx] = mask_geometry(c, idx, M, off, res[-1]["origin"], res[-1]["axis"], Lf)
        del M
    m = S.merge(c, res)
    q, rr = S.qc_row(c, m, res, raw_q["time_s"], raw_q["device"], str(z["checkpoint"]))
    for k in ("near_far_occlusion_agreement", "near_far_agreement_area", "n_occlusion_frames", "occlusion_agreement_source"):
        q.pop(k, None)
    conflict_frame = m["conflict"].any(1)
    reason, area_norm = invalid_reasons(c, m, kp_dist, thick, conflict_frame)
    valid = (reason & INVALID_BITS) == 0
    old_ok = (reason & (R_OCCLUDED | R_CONFLICT)) == 0
    newbits = R_AREA_HIGH | R_AREA_LOW | R_NO_KP | R_THICK
    lab = m["label"]
    fv = {f"frac_valid_{k}": (round(float(valid[lab == k].mean()), 4) if (lab == k).any() else np.nan) for k in ("near", "far", "A", "B")}
    excl = time_exclude_mask(c)
    q.update(**fv, frac_removed_new=round(float((old_ok & ~valid).mean()), 4),
             frac_removed_new_literal=round(float((old_ok & ((reason & (newbits | R_LITERAL_ONLY)) != 0)).mean()), 4),
             n_flag_area=int(((reason & (R_AREA_HIGH | R_AREA_LOW)) != 0).sum()), n_flag_no_keypoint=int(((reason & R_NO_KP) != 0).sum()), n_flag_thick=int(((reason & R_THICK) != 0).sum()))
    q.update(time_s=raw_q["time_s"], fps=raw_q["fps"], post_config=post_config(),
             frac_frame_valid=round(float(valid.mean()), 4), n_frames_conflict_invalid=int(conflict_frame.sum()),
             mean_coverage_tol_before=raw_q["mean_coverage_tol"], frac_mask_leak_before=raw_q["frac_mask_leak"],
             region_area_frac_heel_before=raw_q["region_area_frac_heel"],
             reprompt_per100=round(100.0 * q["n_reprompt"] / c.n, 2), reprompt_per100_max_block=round(max(per_block), 2),
             n_clip_frames=int(clipped.any(1).sum()), n_clip_skipped=int(skipped.sum()), frac_area_clipped=round(removed / max(total, 1), 4),
             frac_view_lateral=round(float(view.mean()), 4), frac_exclude_time_features=round(float(excl.mean()), 4))
    q["gate"], q["gate_reasons"] = gate(q)
    C.atomic_npz(OUT_MASKS / f"{vid}.npz", frame=c.frame, t=c.t, continuity_id=c.cont, W=c.W, H=c.H, rle_counts=m["counts"], rle_ptr=m["ptr"],
                 state=m["state"], area=m["area"], area_ratio=m["ratio"], coverage=m["cov"], coverage_tol=m["cov_tol"], origin=m["origin"], axis=m["axis"],
                 axis_borrowed=m["borrowed"], l_foot=m["l_foot"], region_area=m["region_area"], iou=m["iou"], label=m["label"], reprompt_flag=m["reprompt"],
                 mask_trimmed=m["trimmed"], mask_id_conflict=m["conflict"], crop_window=m["win"], crop_scale=1.0, init_frame=m["init"],
                 frame_valid_mask=valid, invalid_reason=reason, area_ratio_norm=area_norm, kp_mask_dist=kp_dist, perp_thickness=thick, view_lateral=view, exclude_time_features=excl, ankle_clipped=clipped, ankle_clip_skipped=skipped,
                 config=str(z["config"]), checkpoint=str(z["checkpoint"]), post_config=post_config())
    pd.DataFrame(rr, columns=["frame_from", "frame_to", "continuity_id", "n"]).to_csv(OUT_QC / f"{vid}_conflicts.csv", index=False)
    (OUT_QC / f"{vid}_qc.json").write_text(json.dumps(q, ensure_ascii=False, indent=1, default=float))
    if vid == F10_2:
        fb = c.frame[~view]
        (OUT_QC / "F10_2_view.json").write_text(json.dumps(dict(block=BACK_BLOCK, frame_from=int(fb[0]), frame_to=int(fb[-1]), n_frames=int(len(fb)), hip_threshold=BACK_HIP_THR), indent=1))
    return q, rr


def gate(q):
    fail, warn = [], []
    near = q["frac_visible_near"] if np.isfinite(q["frac_visible_near"]) else q["frac_visible_best_AB"]
    far = q["frac_visible_far"] if np.isfinite(q["frac_visible_far"]) else q["frac_visible_worst_AB"]
    if q["n_id_conflict_frames"] > FAIL_CONFLICT * q["n_frames"]:
        fail.append("id_conflict>10%")
    if near < FAIL_NEAR:
        fail.append(f"frac_visible_near<{FAIL_NEAR}")
    if far < WARN_FAR:
        warn.append(f"frac_visible_far<{WARN_FAR}")
    if q["reprompt_per100"] >= WARN_REPROMPT_PER100:
        warn.append(f"reprompt_per100>={WARN_REPROMPT_PER100:g}")
    if q["frac_mask_leak"] >= WARN_LEAK:
        warn.append(f"frac_mask_leak>={WARN_LEAK:g}")
    return ("FAIL" if fail else "WARN" if warn else "PASS"), "; ".join(fail + warn)


def write_table(meta):
    df = pd.DataFrame([json.loads((OUT_QC / f"{v}_qc.json").read_text()) for v in meta.video_id])
    df[QC_COLS].to_csv(C.OUT / "qc_footmask.csv", index=False)
    return df


def draw(meta):
    for vid in meta.video_id:
        c = C.load_clip(vid, meta)
        z = np.load(OUT_MASKS / f"{vid}.npz", allow_pickle=True)
        R.review(c, z, C.DIRS["review"] / f"{vid}.jpg")
        rr = pd.read_csv(OUT_QC / f"{vid}_conflicts.csv")
        pos = {int(f): n for n, f in enumerate(c.frame)}
        R.conflict_images(c, z, [dict(a=pos[int(r.frame_from)], b=pos[int(r.frame_to)]) for r in rr.itertuples()])


def check(meta):
    """Mọi frame: mask mới nằm trong mask gốc, RLE khớp diện tích, vùng rời nhau và nằm trong mask, cờ đúng định nghĩa. Thoát mã 1 nếu có lỗi."""
    bad = {}
    q = pd.read_csv(C.OUT / "qc_footmask.csv").set_index("video_id")
    if set(q.index) != set(meta.video_id):
        bad["qc_footmask.csv"] = [f"{len(q)} dòng"]
    if {"near_far_occlusion_agreement", "near_far_agreement_area"} & set(q.columns):
        bad["qc_footmask.csv cột"] = ["còn cột near_far_*"]
    for vid in meta.video_id:
        err = []
        p = OUT_MASKS / f"{vid}.npz"
        if not p.exists():
            bad[vid] = ["thiếu masks_post"]
            continue
        c = C.load_clip(vid, meta)
        view = apply_overrides(c)
        z, r = np.load(p, allow_pickle=True), np.load(C.DIRS["masks"] / f"{vid}.npz", allow_pickle=True)
        shape = (c.H, c.W)
        if str(z["config"]) != str(r["config"]) or str(z["config"]) != S.config_hash():
            err.append("config SAM2 đổi")
        if str(z["post_config"]) != post_config():
            err.append("post_config cũ")
        if not np.array_equal(z["frame"], c.frame) or not np.array_equal(z["view_lateral"], view):
            err.append("frame hoặc view_lateral không khớp")
        st, conflict = z["state"], z["mask_id_conflict"].any(1)
        if (r["mask_id_conflict"] & ~z["mask_id_conflict"]).any():
            err.append("mất cờ xung đột của mask trước clip")
        reason = z["invalid_reason"]
        if not np.array_equal(z["frame_valid_mask"], (reason & INVALID_BITS) == 0):
            err.append("frame_valid_mask khác invalid_reason")
        if not np.array_equal((reason & R_OCCLUDED) != 0, st == C.ST_OCCLUDED) or not np.array_equal((reason & R_CONFLICT) != 0, np.repeat(conflict[:, None], 2, 1)):
            err.append("invalid_reason sai bit occluded/conflict")
        if not np.array_equal((reason & R_NO_KP) != 0, z["kp_mask_dist"] > KP_DIST) or not ((reason[z["area_ratio_norm"] > AREA_HI] & R_AREA_HIGH) != 0).all():
            err.append("invalid_reason sai bit diện tích/keypoint")
        if (z["frame_valid_mask"][conflict]).any() or (z["frame_valid_mask"][st == C.ST_OCCLUDED]).any():
            err.append("frame conflict/occluded vẫn hợp lệ")
        if (st[np.isfinite(z["region_area"][:, :, 0])] == C.ST_OCCLUDED).any():
            err.append("frame occluded có đặc trưng vùng")
        if not np.array_equal(z["exclude_time_features"], time_exclude_mask(c)):
            err.append("exclude_time_features sai")
        for a in range(len(c.frame)):
            want = ("A", "B") if not c.reliable[a] else (("near", "far") if c.near[a] == "A" else ("far", "near"))
            if tuple(z["label"][a].tolist()) != want:
                err.append(f"nhãn sai f{c.frame[a]}")
                break
            for o in (0, 1):
                m = R.decode(z, a, o, 0, shape)
                if m.sum() != z["area"][a, o]:
                    err.append(f"area lệch f{c.frame[a]}")
                if (m & ~R.decode(r, a, o, 0, shape)).any():
                    err.append(f"mask mới ngoài mask gốc f{c.frame[a]}")
                regs = [R.decode(z, a, o, k, shape) for k in (1, 2, 3)]
                u = np.logical_or.reduce(regs)
                if sum(x.sum() for x in regs) != u.sum() or (u & ~m).any():
                    err.append(f"vùng sai f{c.frame[a]}")
        if err:
            bad[vid] = err[:4]
    tmp = [x.name for x in C.DIRS["_tmp"].glob("*")]
    if tmp:
        bad["_tmp"] = tmp
    print(f"Kiểm tra hậu kỳ: {len(meta) - sum(k in set(meta.video_id) for k in bad)}/{len(meta)} video đạt")
    for k, e in bad.items():
        print(f"  LỖI {k}: {e}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    meta = C.load_meta()
    if a.check:
        sys.exit(check(meta))
    OUT_MASKS.mkdir(exist_ok=True)
    OUT_QC.mkdir(exist_ok=True)
    write_time_flags()
    for i, vid in enumerate(meta.video_id, 1):
        q, _ = process(vid, meta)
        print(f"[{i}/{len(meta)}] {vid}: {q['gate']} {q['gate_reasons']} leak {q['frac_mask_leak_before']}->{q['frac_mask_leak']} "
              f"cov_tol {q['mean_coverage_tol_before']}->{q['mean_coverage_tol']} bỏ {q['frac_area_clipped']}", flush=True)
    df = write_table(meta)
    print("Gate:", df.gate.value_counts().to_dict())
    if not a.no_review:
        draw(meta)
        print("Đã vẽ lại ảnh review")
    sys.exit(check(meta))


if __name__ == "__main__":
    main()
