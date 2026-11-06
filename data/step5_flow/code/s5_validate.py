#!/usr/bin/env python3
"""Kiểm tra output Bước 5 trên mọi video. Thoát với mã 1 nếu có lỗi. Chạy từ ROOT/data:

  ../.venv-track/bin/python step5_flow/code/s5_validate.py
  ../.venv-track/bin/python step5_flow/code/s5_validate.py --videos F30,C04_2
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import s5_checks as K
import s5_common as C
import s5_features as X
import s5_store as T

STAT = ("v_med", "v_p25", "v_p75", "vx_med", "vy_med")
EXTRA = ("slow_ratio", "slow_ratio_1p0", "slow_ratio_1p5", "slow_ratio_3p0", "slow_ratio_otsu", "fb_slow_ratio", "fb_slow_ratio_otsu")


def check_file(vid, v, path, name, trigger, q):
    err = []
    d = np.load(path, allow_pickle=True)
    if str(d["config"]) != C.config_hash():
        err.append(f"output cũ: config {d['config']}")
    if str(d["flow_hash"]) != T.flow_hash():
        err.append("flow_hash khác kho hiện tại")
    if str(d["post_config"]) != str(v.z["post_config"]):
        err.append("masks_post đã đổi sau khi tính feature")
    if str(d["comp_variant"]) != name or float(d["comp_trigger"]) != trigger:
        err.append("sai biến thể")
    if not np.array_equal(d["frame"], v.c.frame) or not np.allclose(d["t_real_s_v2"], v.c.t):
        err.append("frame/thời gian không khớp")

    # cặp frame chỉ trong cùng khối và shot
    a = np.flatnonzero(d["has_flow"])
    if not set(a.tolist()) <= set(C.pairs(v).tolist()):
        err.append("có flow vắt qua ranh giới khối/shot")

    # cờ
    if not np.array_equal(d["flow_valid"], d["has_flow"] & ~d["cam_fit_bad"]):
        err.append("flow_valid sai định nghĩa")
    r = d["cam_fit_reason"]
    if not np.array_equal(d["cam_fit_bad"], (r & C.CF_BAD) != 0):
        err.append("cam_fit_bad khác cam_fit_reason")
    # hiệu chỉnh chậm ngoại lai phải bị loại
    blocks = d["continuity_id"] * 100000 + d["shot_id"]
    slow_abs = np.abs(d["comp_slow"])
    for b in np.unique(blocks[d["has_flow"]]):
        i = np.flatnonzero((blocks == b) & d["has_flow"])
        over = i[(slow_abs[i] > C.COMP_SLOW_BAD) & (slow_abs[i] > C.COMP_SLOW_BAD_REL * np.median(slow_abs[i]))]
        if d["flow_valid"][over].any():
            err.append("frame comp_slow ngoại lai vẫn hợp lệ")
            break

    # feature chỉ có ở frame hợp lệ
    ok = d["frame_valid_mask"] & d["flow_valid"][:, None]
    for k in STAT + EXTRA + ("fb_v_med", "v_rawdt", "v_smoothdt"):
        if np.isfinite(d[k][~ok]).any():
            err.append(f"{k} có giá trị ở frame không hợp lệ")
            break
    if (d["n_pixels"][~ok] != 0).any():
        err.append("n_pixels khác 0 ở frame không hợp lệ")
    if (d["fb_n_pixels"] > d["n_pixels"]).any():
        err.append("bản FB có nhiều pixel hơn bản chính")
    for k in EXTRA + ("frac_fb_valid",):
        x = d[k]
        if np.isfinite(x).any() and (np.nanmin(x) < -1e-6 or np.nanmax(x) > 1 + 1e-6):
            err.append(f"{k} ngoài [0, 1]")
    w = np.isfinite(d["v_p25"]) & np.isfinite(d["v_p75"])
    if (d["v_p25"][w] > d["v_med"][w] + 1e-4).any() or (d["v_med"][w] > d["v_p75"][w] + 1e-4).any():
        err.append("p25 <= median <= p75 không đúng")
    again = X.ratio_from_hist(d["v_hist"], d["n_pixels"], C.SLOW_MAIN)
    if not np.allclose(np.nan_to_num(again), np.nan_to_num(d["slow_ratio"]), atol=1e-4):
        err.append("slow_ratio khác histogram")

    # Δt
    mode = str(d["dt_mode"])
    if mode not in ("raw", "smooth"):
        err.append(f"dt_mode lạ {mode}")
    use = d["dt_smooth"] if mode == "smooth" else d["dt_real"]
    if not np.allclose(np.nan_to_num(use), np.nan_to_num(d["dt_used"])):
        err.append("dt_used khác dt_mode")
    if (d["dt_used"][a] <= 0).any():
        err.append("Δt không dương")
    group = (d["continuity_id"] * 100000 + d["shot_id"]) * 1000 + d["speed_part_id"]
    for g in np.unique(group[a]):
        i = a[group[a] == g]
        if not np.isclose(d["dt_smooth"][i].sum(), d["dt_real"][i].sum(), rtol=1e-6):
            err.append("làm mượt Δt đổi tổng thời gian")
            break
    main_v = d["v_smoothdt"] if mode == "smooth" else d["v_rawdt"]
    if not np.allclose(np.nan_to_num(main_v), np.nan_to_num(d["v_med"]), atol=1e-4):
        err.append("v_med khác bản Δt đã chọn")

    # bù camera: bước B đúng ngưỡng
    method = str(d["comp_method"])
    p10 = q["vx_p10_direct"]
    want = "direct+slow" if np.isfinite(p10) and p10 < trigger else "direct"
    if method != want:
        err.append(f"comp_method {method}, cần {want}")
    slow = d["comp_slow"]
    if method == "direct" and np.abs(slow).max() != 0:
        err.append("direct mà có hiệu chỉnh chậm")
    if not np.isfinite(slow[a]).all():
        err.append("comp_slow có NaN ở frame có flow")
    if not np.isfinite(d["comp_tx"][a]).all():
        err.append("comp_tx có NaN ở frame có flow")

    # nhãn, cờ kế thừa từ Bước 4
    if not np.array_equal(d["label"], v.z["label"]) or not np.array_equal(d["frame_valid_mask"], v.z["frame_valid_mask"]) or not np.array_equal(d["exclude_time_features"], v.z["exclude_time_features"]):
        err.append("nhãn/cờ khác masks_post")
    return err, d


def check_video(vid, meta):
    err = []
    default = C.VARIANTS[0][0]
    paths = [C.DIRS["plots"] / f"{vid}.png", C.DIRS["review"] / f"{vid}.jpg", C.DIRS["qc"] / f"{vid}_qc.json", T.STORE / f"{vid}.npz"]
    files = {name: C.DIRS["features"] / (f"{vid}.npz" if name == default else f"{vid}__{name}.npz") for name, _ in C.VARIANTS}
    if not all(p.exists() for p in paths + list(files.values())):
        return ["thiếu output"]
    if not T.done(vid):
        err.append("kho flow cũ")
    v = C.load_video(vid, meta)
    q = json.loads(paths[2].read_text())
    if q.get("config") != C.config_hash():
        err.append("QC cũ")
    data = {}
    for name, trig in C.VARIANTS:
        e, d = check_file(vid, v, files[name], name, trig, q)
        err += [f"{name}: {x}" for x in e]
        data[name] = d
    # 2 biến thể cùng cách bù thì feature phải trùng
    other = C.VARIANTS[1][0]
    if str(data[default]["comp_method"]) == str(data[other]["comp_method"]) and not np.array_equal(np.nan_to_num(data[default]["v_med"]), np.nan_to_num(data[other]["v_med"])):
        err.append("2 biến thể cùng comp_method mà feature khác nhau")
    if str(data[default]["comp_method"]) != str(data[other]["comp_method"]) and not (C.DIRS["plots"] / f"{vid}__{other}.png").exists():
        err.append(f"thiếu plot {other}")
    if q["gate"] != K.gate(q)[0]:
        err.append("gate trong QC không khớp")
    # QC không đưa chỉ số pha chống vào gate
    if any(k in str(q["gate_reasons"]) for k in ("duty", "stance", "kp_speed_min")):
        err.append("chỉ số pha chống lọt vào gate")
    if C.VARIANTS[0][0] != "v1":
        err.append("mặc định không phải V1")
    if q.get("default_variant") != C.VARIANTS[0][0]:
        err.append("default_variant trong QC sai")
    if list(C.DIRS["features"].glob(f"{vid}__{default}.npz")):
        err.append("còn file hậu tố của biến thể mặc định")
    for m in ("duty_factor_first", "duty_factor_second", "heel_duty_factor_first", "old_duty_factor_first", f"{other}_duty_factor_first"):
        if m not in q:
            err.append(f"thiếu cột {m}")
        elif q[m] is not None and np.isfinite(q[m]) and not 0 <= q[m] <= 1:
            err.append(f"{m} ngoài [0, 1]")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    a = ap.parse_args()
    meta = C.S4.load_meta()
    vids = [x for s in a.videos for x in s.split(",") if x] if a.videos else meta.video_id.tolist()
    bad = {}
    for vid in vids:
        e = check_video(vid, meta)
        if e:
            bad[vid] = e[:5]
    if not a.videos:
        q = pd.read_csv(C.OUT / "qc_flow.csv")
        if set(q.video_id) != set(meta.video_id):
            bad["qc_flow.csv"] = [f"{len(q)} dòng, cần {len(meta)}"]
        elif (q.config.astype(str) != C.config_hash()).any():
            bad["qc_flow.csv"] = ["có dòng config cũ"]
        else:
            print("Gate:", q.gate.value_counts().to_dict())
    left = [p.name for p in C.DIRS["features"].glob("*.tmp*")]
    if left:
        bad["tmp"] = left
    print(f"{len(vids) - sum(k in vids for k in bad)}/{len(vids)} video đạt")
    for k, e in bad.items():
        print(f"  LỖI {k}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
