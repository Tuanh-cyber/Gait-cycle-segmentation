"""Read one fixed Step5 foot slot. No dependency on step5_single_foot."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FOOT_POINTS = [[15, 20, 22, 24], [16, 21, 23, 25]]
PHASE_NAMES = {-1: "UNKNOWN", 0: "SWING", 1: "STANCE"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_npz(path, **values):
    path = Path(path)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **values)
    temp.replace(path)


def runs(valid, groups):
    ids = np.flatnonzero(valid)
    if not len(ids):
        return []
    split = np.diff(ids) != 1
    for group in groups:
        split |= group[ids[1:]] != group[ids[:-1]]
    return np.split(ids, np.flatnonzero(split) + 1)


def two_means(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.array([0., 1.])
    centers = np.quantile(a, [.15, .85])
    for _ in range(40):
        labels = abs(a[:, None] - centers).argmin(1)
        new = np.array([a[labels == k].mean() if (labels == k).any() else centers[k] for k in (0, 1)])
        if np.max(abs(new - centers)) < 1e-7:
            break
        centers = new
    return np.sort(centers)


def semantic_phase(value, centers, margin=.1):
    if not np.isfinite(value) or centers[1] - centers[0] < .15:
        return -1
    threshold = centers.mean()
    if abs(value - threshold) < margin * (centers[1] - centers[0]):
        return -1
    return int(value >= threshold)


def load_video(meta, requested_slot=None):
    vid = meta.video_id
    path = DATA / "step5_flow/features" / f"{vid}.npz"
    fields = ("frame", "t_real_s_v2", "continuity_id", "shot_id", "speed_part_id", "label", "frame_valid_mask",
              "flow_valid", "cam_fit_bad", "has_flow", "view_lateral", "exclude_time_features", "slow_ratio", "v_med", "vx_med", "vy_med")
    with np.load(path, allow_pickle=False) as z:
        f = {key: z[key] for key in fields}
    n = len(f["frame"])
    if np.any(np.diff(f["frame"]) <= 0) or not np.isfinite(f["t_real_s_v2"]).all():
        raise ValueError(f"{vid}: invalid feature frame/time axis")
    if requested_slot is None:
        # Only inspect initial observations for selection, never switch per frame.
        initial = next((row for row in f["label"][:10] if "near" in row), None)
        slot = int(np.flatnonzero(initial == "near")[0]) if initial is not None else 0
        selection = "initial_near_slot_fixed" if initial is not None else "fixed_A_fallback_near_unknown"
    else:
        slot, selection = int(requested_slot), "explicit_fixed_slot"
    if slot not in (0, 1):
        raise ValueError("Fixed foot slot must be 0 or 1")
    groups = [f[key] for key in ("continuity_id", "shot_id", "speed_part_id")]
    valid = f["frame_valid_mask"][:, slot].astype(bool) & f["flow_valid"] & ~f["cam_fit_bad"] & f["has_flow"] & f["view_lateral"]
    valid[-1] = False
    for i in range(n - 1):
        valid[i] &= f["frame_valid_mask"][i + 1, slot] and f["view_lateral"][i + 1] and all(g[i] == g[i + 1] for g in groups)
        valid[i] &= f["t_real_s_v2"][i + 1] > f["t_real_s_v2"][i]
    channels = [f["slow_ratio"][:, slot], np.log1p(np.maximum(f["v_med"][:, slot], 0))]
    for key in ("vx_med", "vy_med"):
        a = f[key][:, slot]
        channels.append(np.sign(a) * np.log1p(abs(a)))
    raw = np.stack(channels, axis=-1).reshape(n, 16).astype(float)
    valid &= np.isfinite(raw[:, 12:]).all(1)
    center, scale = np.zeros(16), np.tile([.1, .25, .25, .25], 4)
    for d in range(16):
        a = raw[valid, d]
        a = a[np.isfinite(a)]
        if len(a):
            center[d], scale[d] = np.median(a), max(np.ptp(np.quantile(a, [.25, .75])), scale[d])
    x = np.clip((raw - center) / (2 * scale), -10, 10)
    slow = f["slow_ratio"][:, slot, :3]
    available = np.isfinite(slow).any(1)
    contact = np.full(n, np.nan)
    contact[available] = np.nanmax(slow[available], axis=1)
    r = runs(valid, groups)
    source = DATA / ("Correct" if meta.label == "CORRECT" else "False") / meta.video
    audit_path = DATA / f"step0_audit/frames/{vid}_frames.csv"
    audit = pd.read_csv(audit_path)
    kp_path = DATA / f"step3_footkp/footkp/{vid}_footkp.npz"
    with np.load(kp_path, allow_pickle=False) as z:
        if not np.array_equal(z["frame"], f["frame"]):
            raise ValueError("Display pose and features have different frame axes")
        kp = z["kp_clean"][:, FOOT_POINTS[slot]]
    mask_path = DATA / f"step4_footmask/masks_post/{vid}.npz"
    with np.load(mask_path, allow_pickle=False) as z:
        if not np.array_equal(z["frame"], f["frame"]):
            raise ValueError("Display masks and features have different frame axes")
        display_mask = {key: z[key] for key in ("rle_ptr", "rle_counts")}
    feature_uid = audit.set_index("frame").loc[f["frame"], "unique_id"].to_numpy(int)
    return dict(video_id=vid, f=f, slot=slot, selection=selection, x=x, raw=raw, valid=valid, groups=groups,
                runs=r, contact=contact, centers=two_means(contact[valid]), source=source, audit=audit, kp=kp,
                display_mask=display_mask, feature_unique_id=feature_uid,
                abs_time_ok=str(meta.abs_time_ok).lower() == "true", scale_center=center, scale=scale,
                inputs={name: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for name, p in
                        (("flow", path), ("source", source), ("audit", audit_path), ("display_pose", kp_path), ("display_mask", mask_path))})


def make_events(v, method, phase, boundaries, quality):
    events = []
    n = len(phase)
    for i in sorted(set(int(x) for x in boundaries)):
        if i <= 0 or i >= n or not (v["valid"][i - 1] and v["valid"][i]):
            continue
        if not all(g[i] == g[i - 1] for g in v["groups"]):
            continue
        before, after = int(phase[i - 1]), int(phase[i])
        kind = "IC" if (before, after) == (0, 1) else "TO" if (before, after) == (1, 0) else "UNKNOWN"
        reason = "phase_transition" if kind != "UNKNOWN" else "ambiguous_phase" if -1 in (before, after) else "within_phase"
        a = max(0, i - 2)
        b = min(n, i + 2)
        left = v["contact"][a:i][v["valid"][a:i]]
        right = v["contact"][i:b][v["valid"][i:b]]
        s0 = float(np.nanmedian(left)) if len(left) else np.nan
        s1 = float(np.nanmedian(right)) if len(right) else np.nan
        strong = np.isfinite(s0 + s1) and ((kind == "IC" and s1 - s0 >= .1) or (kind == "TO" and s0 - s1 >= .1))
        frame = int(v["f"]["frame"][i])
        pts = float(v["audit"].set_index("frame").loc[frame, "pts_s"])
        events.append(dict(video_id=v["video_id"], method=method, foot_slot=v["slot"], source_index=i, frame=frame,
                           t_real_s=float(v["f"]["t_real_s_v2"][i]), source_pts_s=pts,
                           type=kind, phase_before=PHASE_NAMES[before], phase_after=PHASE_NAMES[after], reason=reason,
                           slow_before=s0, slow_after=s1, strong_contact_evidence=bool(strong),
                           boundary_evidence=float(quality.get(i, np.nan)), abs_time_ok=v["abs_time_ok"], candidate=True))
    return events
