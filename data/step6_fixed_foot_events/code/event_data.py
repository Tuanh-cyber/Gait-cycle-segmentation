"""Read one fixed Step5 foot slot. No dependency on step5_single_foot."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


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


def load_video(meta, requested_slot=None, data=DATA):
    vid = meta.video_id
    path = data / "step5_flow/features" / f"{vid}.npz"
    fields = ("frame", "t_real_s_v2", "continuity_id", "shot_id", "speed_part_id", "label", "frame_valid_mask",
              "flow_valid", "cam_fit_bad", "has_flow", "view_lateral", "exclude_time_features", "slow_ratio", "v_med", "vx_med", "vy_med")
    with np.load(path, allow_pickle=False) as z:
        f = {key: z[key] for key in fields}
    n = len(f["frame"])
    if not n or np.any(np.diff(f["frame"]) <= 0) or not np.isfinite(f["t_real_s_v2"]).all():
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
    # Preserve the upstream whole-foot validity gate. New model inputs are built
    # from dense regional pixels in hsmm_inputs, not these saved summaries.
    valid &= np.isfinite(np.stack([f[k][:, slot, 3] for k in ("slow_ratio", "v_med", "vx_med", "vy_med")], axis=1)).all(1)
    audit = pd.read_csv(data / f"step0_audit/frames/{vid}_frames.csv", usecols=["frame"])
    if not np.isin(f["frame"], audit.frame).all():
        raise ValueError(f"{vid}: feature frames are missing from the source audit")
    kp_path = data / f"step3_footkp/footkp/{vid}_footkp.npz"
    with np.load(kp_path, allow_pickle=False) as z:
        if not np.array_equal(z["frame"], f["frame"]):
            raise ValueError("Display pose and features have different frame axes")
    mask_path = data / f"step4_footmask/masks_post/{vid}.npz"
    with np.load(mask_path, allow_pickle=False) as z:
        if not np.array_equal(z["frame"], f["frame"]):
            raise ValueError("Display masks and features have different frame axes")
    return dict(video_id=vid, f=f, slot=slot, selection=selection, valid=valid)
