#!/usr/bin/env python3
"""Vẽ lại ảnh review/ từ masks_post/ (bản sau hậu kỳ), không chạy SAM2. Mặc định chỉ video WARN/FAIL. Chạy từ ROOT/data:

  ../.venv-track/bin/python step4_footmask/code/regen_review.py
  ../.venv-track/bin/python step4_footmask/code/regen_review.py --videos C30,F26
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import s4_common as C
import s4_report as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    a = ap.parse_args()
    meta = C.load_meta()
    q = pd.read_csv(C.OUT / "qc_footmask.csv")
    vids = [v for x in a.videos for v in x.split(",") if v] if a.videos else q[q.gate != "PASS"].video_id.tolist()
    for vid in vids:
        c = C.load_clip(vid, meta)
        z = np.load(C.OUT / "masks_post" / f"{vid}.npz", allow_pickle=True)
        R.review(c, z, C.DIRS["review"] / f"{vid}.jpg")
        rr = pd.read_csv(C.OUT / "qc_post" / f"{vid}_conflicts.csv")
        pos = {int(f): n for n, f in enumerate(c.frame)}
        R.conflict_images(c, z, [dict(a=pos[int(r.frame_from)], b=pos[int(r.frame_to)]) for r in rr.itertuples()])
    print(f"Đã vẽ lại {len(vids)} video")


if __name__ == "__main__":
    main()
