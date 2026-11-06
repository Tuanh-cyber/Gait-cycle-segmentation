#!/usr/bin/env python3
"""Kiểm tra tiêu chí hoàn thành Bước 3. Thoát với mã 1 nếu có lỗi.

  python step3_footkp/code/s3_validate.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import s3_clean as K
import s3_common as C
import s3_swap as W


def check_video(vid, meta, cfg):
    err = []
    paths = dict(npz=C.DIRS["footkp"] / f"{vid}_footkp.npz", review=C.DIRS["review"] / f"{vid}.jpg",
                 plot=C.DIRS["plots"] / f"{vid}_ankle.png", qc=C.DIRS["qc"] / f"{vid}_qc.json")
    miss = [k for k, p in paths.items() if not p.exists()]
    if miss:
        return [f"thiếu {miss}"]
    c = C.load_clip(vid, meta)
    z = np.load(paths["npz"])
    need = {"frame", "t_real_s", "t_real_s_v2", "time_model", "near_track_reliable", "continuity_id", "speed_part_id", "kp_raw", "kp_clean", "valid_mask", "swap_flag", "near_track", "near_conf"}
    if need - set(z.files):
        err.append(f"npz thiếu {need - set(z.files)}")
    if str(z["config"]) != cfg:
        err.append(f"output cũ: config {z['config']} != {cfg}")
    if not np.array_equal(z["frame"], c.frame) or not np.array_equal(z["continuity_id"], c.cont) or not np.array_equal(z["speed_part_id"], c.part):
        err.append("frame/khối/part không khớp đầu vào")
    if not np.allclose(z["t_real_s_v2"], c.t):
        err.append("t_real_s_v2 không khớp frames CSV")
    if ((z["outlier_reason"] & 8) > 0).any():
        err.append("còn luật (d)")
    if vid in C.NEAR_UNRELIABLE_MANUAL and bool(z["near_track_reliable"]):
        err.append("near_track_reliable phải False")
    if not np.allclose(z["kp_raw"], c.kp.astype(np.float32), equal_nan=True):
        err.append("kp_raw khác keypoints_v2")
    kpc, val, res, inter = z["kp_clean"], z["valid_mask"], z["outlier_reason"], z["interp_mask"]
    # outlier không bao giờ hợp lệ; hợp lệ thì phải có giá trị; ngoài hợp lệ + nội suy ngắn là NaN
    if (val & (res > 0)).any():
        err.append("outlier vẫn valid")
    if (val & ~np.isfinite(kpc[:, :, 0])).any():
        err.append("valid nhưng NaN")
    if (np.isfinite(kpc[:, :, 0]) != (val | inter)).any():
        err.append("kp_clean có giá trị ngoài valid/nội suy")
    # nội suy chỉ trong khoảng trống <= 3 frame unique, không qua ranh giới
    for j in range(26):
        for seg in C.segments(c, val[:, j]):
            if len(seg) > 1 and (np.diff(c.unique_id[seg]) - 1 > C.MAX_GAP_FRAMES).any():
                err.append(f"đoạn điểm {j} vượt khoảng trống")
                break
        ii = np.flatnonzero(inter[:, j])
        for g in np.split(ii, np.flatnonzero(np.diff(ii) != 1) + 1) if len(ii) else []:
            if len(g) > C.MAX_GAP_FRAMES or c.cont[g[0] - 1] != c.cont[g[-1] + 1] or c.part[g[0] - 1] != c.part[g[-1] + 1]:
                err.append(f"nội suy sai ở điểm {j}, frame {c.frame[g[0]]}")
                break
    # kp_clean phải là phiên bản đã hoán đổi của raw: cùng chân, không dính sang chân kia
    fx = W.apply_swap(c.kp, z["swap_flag"])
    d = np.linalg.norm(kpc - fx, axis=2)[val]
    L = np.nanmedian(z["L_leg"])
    if len(d) and np.nanpercentile(d, 99.9) > 0.5 * L:
        err.append("kp_clean lệch xa bản đã sửa swap")
    if not set(np.unique(z["near_track"])) <= {"A", "B"}:
        err.append("near_track lạ")
    # near_track không đổi trong 1 khối
    for b in np.unique(c.cont):
        if len(np.unique(z["near_track"][c.cont == b])) != 1:
            err.append("near_track đổi trong khối")
    return err


def boundary_test(meta):
    """Đổi dữ liệu ở 1 phía ranh giới không được làm đổi kết quả lọc ở phía kia."""
    err = []
    for vid, col in (("C13_1", "part"), ("F11", "cont")):
        c = C.load_clip(vid, meta)
        valid = np.isfinite(c.kp[:, :, 0])
        lab = getattr(c, col)
        b = np.flatnonzero(np.diff(lab) != 0)[0]
        a, _, _ = K.clean_and_filter(c, c.kp, valid)
        k2 = c.kp.copy()
        k2[b + 1:b + 6] += 500.0
        x, _, _ = K.clean_and_filter(c, k2, valid)
        if not np.allclose(a[:b + 1], x[:b + 1], equal_nan=True):
            err.append(f"lọc rò qua ranh giới {col} ở {vid}")
        # Viterbi: chỉ vị trí qua ranh giới part, không dùng vận tốc
    t = np.cumsum(np.r_[0, np.random.default_rng(0).uniform(.01, .1, 40)])
    y = (1 + 2 * t - 3 * t ** 2 + .5 * t ** 3)[:, None]
    if not np.allclose(K.smooth_segment(t, y), y, atol=1e-8):
        err.append("bộ lọc không tái tạo đúng đa thức bậc 3 trên trục không đều")
    return err


def main():
    meta = C.load_meta()
    cfg = C.config_hash()
    bad = {}
    for vid in meta.video_id:
        e = check_video(vid, meta, cfg)
        if e:
            bad[vid] = e
    e = boundary_test(meta)
    if e:
        bad["ranh giới"] = e
    q = pd.read_csv(C.OUT / "qc_footkp.csv")
    if set(q.video_id) != set(meta.video_id) or len(q) != len(meta):
        bad["qc_footkp.csv"] = [f"{len(q)} dòng, cần {len(meta)}"]
    if (q.config.astype(str) != cfg).any():
        bad["qc_footkp.csv cấu hình"] = ["có dòng từ cấu hình cũ"]
    if set(q.gate) - {"PASS", "WARN", "FAIL"}:
        bad["gate"] = [str(set(q.gate))]
    rv = pd.read_csv(C.OUT / "footkp_review.csv", dtype=str, keep_default_na=False)
    if set(rv.video_id) != set(meta.video_id) or list(rv.columns) != ["video_id", "verdict", "notes"]:
        bad["footkp_review.csv"] = ["sai dòng/cột"]
    df = pd.read_csv(C.OUT / "footkp_review_v2_diff.csv", dtype=str, keep_default_na=False)
    if not df[["video_id", "verdict", "notes"]].equals(rv[["video_id", "verdict", "notes"]]) or (df.gate_v2.values != q.set_index("video_id").gate.reindex(df.video_id).values).any():
        bad["footkp_review_v2_diff.csv"] = ["không khớp review hoặc qc"]
    if (q[q.video_id.isin(C.NEAR_UNRELIABLE_MANUAL)].gate == "PASS").any():
        bad["gate tay"] = ["video đánh dấu tay vẫn PASS"]
    v = json.loads(C.VEL_JSON.read_text())
    if v.get("config") != cfg:
        bad["velocity_threshold.json"] = ["ngưỡng từ cấu hình cũ"]
    print(f"{len(meta) - sum(k in set(meta.video_id) for k in bad)}/{len(meta)} video đạt. Gate: {q.gate.value_counts().to_dict()}")
    for k, e in bad.items():
        print(f"  LỖI {k}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
