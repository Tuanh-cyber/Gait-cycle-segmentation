#!/usr/bin/env python3
"""Kiểm tra tiêu chí hoàn thành Bước 2. Thoát với mã 1 nếu có lỗi.

  python step2_tracking/code/s2_validate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import s2_common as C


def check_video(vid, meta):
    err = []
    paths = dict(tracks=C.DIRS["all_tracks"] / f"{vid}_tracks.csv", runner=C.DIRS["runner"] / f"{vid}_runner.csv",
                 kp=C.DIRS["keypoints_v2"] / f"{vid}_kp.npz", review=C.DIRS["review"] / f"{vid}.jpg",
                 scores=C.DIRS["selection"] / f"{vid}_scores.csv", joins=C.DIRS["selection"] / f"{vid}_joins.csv")
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        return [f"thiếu {missing}"]
    v = C.load_video(vid, meta)
    fr = v.fr
    use = set(fr.frame[fr.use].astype(int))
    tr = pd.read_csv(paths["tracks"])
    run = pd.read_csv(paths["runner"])
    z = np.load(paths["kp"])

    # track không vắt qua 2 continuity_id, và chỉ có frame hợp lệ
    if (tr.groupby("track_id").continuity_id.nunique() > 1).any():
        err.append("track vắt qua 2 continuity_id")
    if not set(tr.frame.astype(int)) <= use:
        err.append("all_tracks có frame không hợp lệ")
    if (tr.conf < C.CONF - 1e-6).any():
        err.append("all_tracks có conf < ngưỡng")
    cont = fr.set_index("frame").continuity_id
    if not (tr.continuity_id.values == cont.loc[tr.frame].values).all():
        err.append("all_tracks sai continuity_id")

    # runner: đủ mọi frame giải mã, trạng thái đúng
    if not (run.frame.values == fr.frame.values).all():
        err.append("runner không đủ 1 dòng/frame")
    u = run[run.frame.isin(use)]
    if not set(u.track_status) <= {"tracked", "interpolated", "missing"}:
        err.append(f"trạng thái frame hợp lệ lạ {set(u.track_status)}")
    if run[~run.frame.isin(use) & (run.inherited_from < 0)].track_status.ne("excluded").any():
        err.append("frame loại không có trạng thái excluded")
    if (np.isfinite(u.x0.values) != (u.track_status != "missing").values).any():
        err.append("bbox không khớp trạng thái missing")

    # runner: một track_id chỉ thuộc 1 khối; khối của track khớp all_tracks
    rt = u[u.track_id >= 0]
    if (rt.groupby("track_id").continuity_id.nunique() > 1).any():
        err.append("runner: track vắt qua 2 continuity_id")

    # kế thừa: nguồn là frame unique hợp lệ ngay trước, cùng khối, cùng bbox
    inh = run[run.inherited_from >= 0]
    if len(inh):
        by = run.set_index("frame")
        src = by.loc[inh.inherited_from]
        if not fr.is_dup.iloc[inh.frame.values].all():
            err.append("frame kế thừa không phải frame trùng")
        if not set(inh.inherited_from.astype(int)) <= use:
            err.append("nguồn kế thừa không hợp lệ")
        if (inh.continuity_id.values != src.continuity_id.values).any():
            err.append("kế thừa qua ranh giới khối")
        if (inh.track_status.values != src.track_status.values).any():
            err.append("kế thừa sai trạng thái")
        a, b = inh.x0.values, src.x0.values
        if not (np.isfinite(a) == np.isfinite(b)).all() or not np.allclose(a[np.isfinite(a)], b[np.isfinite(b)]):
            err.append("kế thừa sai bbox")

    # keypoints_v2: khớp frame hợp lệ, có đúng khoá schema, keypoint có <=> bbox có
    need = {"frame", "pts_s", "shot_id", "kp", "score", "bbox", "det", "W", "H", "track_status", "continuity_id", "t_real_s"}
    if need - set(z.files):
        err.append(f"npz thiếu {need - set(z.files)}")
    if set(z["frame"].astype(int)) != use:
        err.append("npz không khớp frame hợp lệ")
    if (np.isfinite(z["kp"][:, 0, 0]) != np.isfinite(z["bbox"][:, 0])).any():
        err.append("npz: kp/bbox không khớp")
    if not np.isfinite(z["t_real_s"]).all() or (z["continuity_id"] < 0).any():
        err.append("npz có frame không có thời gian/khối")
    if int(z["n_read"]) != int(z["n_meta"]):
        err.append(f"số frame đọc {int(z['n_read'])} != audit {int(z['n_meta'])}")
    return err


def main():
    meta = C.load_meta()
    bad, tot = {}, dict(tracked=0, interpolated=0, missing=0, inherited=0, excluded=0)
    for vid in meta.video_id:
        e = check_video(vid, meta)
        if e:
            bad[vid] = e
            continue
        run = pd.read_csv(C.DIRS["runner"] / f"{vid}_runner.csv")
        own = run[run.inherited_from < 0].track_status.value_counts()
        for k, n in own.items():
            tot[k] += int(n)
        tot["inherited"] += int((run.inherited_from >= 0).sum())
    qc = pd.read_csv(C.OUT / "qc_tracking.csv")
    rv = pd.read_csv(C.OUT / "tracking_review.csv", dtype=str, keep_default_na=False)
    sizes = {json.loads((C.DIRS["logs"] / f"timing_{v}.json").read_text()).get("imgsz") for v in meta.video_id}
    if sizes != {C.IMGSZ}:
        bad["imgsz"] = [f"output tạo với imgsz {sizes}, cấu hình hiện tại {C.IMGSZ}: output cũ hoặc trộn lẫn"]
    if "frac_kp_outside_bbox" not in qc.columns:
        bad["qc_tracking.csv cột"] = ["thiếu frac_kp_outside_bbox: output cũ"]
    if set(qc.video_id) != set(meta.video_id) or len(qc) != len(meta):
        bad["qc_tracking.csv"] = [f"{len(qc)} dòng, cần {len(meta)}"]
    if set(rv.video_id) != set(meta.video_id) or len(rv) != len(meta):
        bad["tracking_review.csv"] = [f"{len(rv)} dòng, cần {len(meta)}"]
    print(f"{len(meta) - sum(k in set(meta.video_id) for k in bad)}/{len(meta)} video đạt. Frame: {tot}")
    for k, e in bad.items():
        print(f"  LỖI {k}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
