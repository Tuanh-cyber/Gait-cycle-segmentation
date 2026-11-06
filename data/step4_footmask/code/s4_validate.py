#!/usr/bin/env python3
"""Kiểm tra output Bước 4. Thoát với mã 1 nếu có lỗi.

  ../.venv-track/bin/python step4_footmask/code/s4_validate.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import s4_common as C
import s4_report as R


def config_hash():
    import hashlib
    d = {k: getattr(C, k) for k in dir(C) if k.isupper() and isinstance(getattr(C, k), (int, float, str, tuple, dict)) and k not in ("DIRS",)}
    return hashlib.sha1(repr(sorted((k, repr(v)) for k, v in d.items())).encode()).hexdigest()[:10]


def check(vid, meta, cfg):
    err = []
    p = C.DIRS["masks"] / f"{vid}.npz"
    if not p.exists() or not (C.DIRS["review"] / f"{vid}.jpg").exists():
        return ["thiếu output"]
    c = C.load_clip(vid, meta)
    z = np.load(p)
    if str(z["config"]) != cfg:
        err.append(f"output cũ {z['config']} != {cfg}")
    if not np.array_equal(z["frame"], c.frame) or not np.array_equal(z["continuity_id"], c.cont) or not np.allclose(z["t"], c.t):
        err.append("frame/khối/thời gian không khớp")
    shape = (c.H, c.W)
    st, lab = z["state"], z["label"]
    n = len(z["frame"])
    if (st[np.isfinite(z["region_area"][:, :, 0])] == C.ST_OCCLUDED).any():
        err.append("frame occluded có đặc trưng vùng")
    for i in range(n):
        if c.reliable[i]:
            want = ("near", "far") if c.near[i] == "A" else ("far", "near")
        else:
            want = ("A", "B")
        if tuple(lab[i].tolist()) != want:
            err.append(f"nhãn sai ở frame {c.frame[i]}")
            break
    # mẫu vài frame: giải mã RLE, area khớp, vùng nằm trong mask và rời nhau
    for a in np.linspace(0, n - 1, min(n, 12)).astype(int):
        for o in (0, 1):
            m = R.decode(z, a, o, 0, shape)
            if m.sum() != z["area"][a, o]:
                err.append(f"area lệch RLE ở f{z['frame'][a]}")
            x0, y0, x1, y1 = z["crop_window"][a]
            if m.sum() != m[y0:y1, x0:x1].sum():
                err.append("mask ngoài cửa sổ crop")
            regs = [R.decode(z, a, o, k, shape) for k in (1, 2, 3)]
            if sum(r.sum() for r in regs) != np.logical_or.reduce(regs).sum() or (np.logical_or.reduce(regs) & ~m).any():
                err.append(f"vùng không rời nhau hoặc ngoài mask ở f{z['frame'][a]}")
    # mỗi khối 1 cửa sổ cố định
    for b in np.unique(c.cont):
        if len(np.unique(z["crop_window"][c.cont == b], axis=0)) != 1:
            err.append("cửa sổ crop đổi trong khối")
    return err


def main():
    meta = C.load_meta()
    cfg = config_hash()
    bad = {}
    for vid in meta.video_id:
        e = check(vid, meta, cfg)
        if e:
            bad[vid] = e[:4]
    q = pd.read_csv(C.OUT / "qc_footmask.csv")
    if set(q.video_id) != set(meta.video_id):
        bad["qc_footmask.csv"] = [f"{len(q)} dòng"]
    rv = pd.read_csv(C.OUT / "footmask_review.csv", dtype=str, keep_default_na=False)
    if (rv.verdict != "").any() or (rv.notes != "").any():
        print("  Ghi chú: footmask_review.csv đã có nội dung người kiểm tra")
    left = [p.name for p in C.DIRS["_tmp"].glob("*")]
    if left:
        bad["_tmp"] = [f"còn thư mục tạm {left}"]
    print(f"{len(meta) - sum(k in set(meta.video_id) for k in bad)}/{len(meta)} video đạt. Gate: {q.gate.value_counts().to_dict()}")
    for k, e in bad.items():
        print(f"  LỖI {k}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
