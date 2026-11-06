#!/usr/bin/env python3
"""Bước 2 — Subject tracking. Xem step2_tracking/README.md.

  python step2_tracking/code/run_step2.py                                  # toàn bộ, bỏ qua video đã xong
  python step2_tracking/code/run_step2.py --videos C13_1 F10_2             # chạy lại danh sách này
  python step2_tracking/code/run_step2.py --videos F10_2 --from-stage select  # dùng lại kết quả YOLO
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import s2_common as C

STAGES = ["track", "select", "pose", "report"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", help="chạy lại các video này (cách nhau bằng dấu cách hoặc phẩy)")
    ap.add_argument("--from-stage", choices=STAGES, default="track",
                    help="với --videos: chạy lại từ stage này (mặc định track)")
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--det-device", default="mps")
    ap.add_argument("--pose-device", default="mps")
    a = ap.parse_args()
    C.ensure_dirs()
    meta = C.load_meta()
    force = bool(a.videos)
    vids = [x for s in a.videos for x in s.split(",") if x] if a.videos else meta.video_id.tolist()
    bad = [v for v in vids if v not in meta.index]
    if bad:
        ap.error(f"không có trong metadata: {bad}")

    import s2_pose, s2_report, s2_select, s2_track
    mods = dict(track=s2_track, select=s2_select, pose=s2_pose, report=s2_report)
    first = STAGES.index(a.from_stage) if force else 0
    yolo, pose = None, None
    failed = []
    t_all = time.time()
    for n, vid in enumerate(vids, 1):
        t0 = time.time()
        # stage trước chạy lại thì mọi stage sau cũng phải chạy lại
        todo, redo = [], force
        for i, st in enumerate(STAGES):
            if (redo and i >= first) or not mods[st].done(vid):
                todo.append(st)
                redo = True
        print(f"[{n}/{len(vids)}] {vid}: " + ("đã xong, bỏ qua" if not todo else "+".join(todo)), flush=True)
        if not todo:
            continue
        try:
            v = C.load_video(vid, meta)
            if "track" in todo:
                if yolo is None:
                    yolo = s2_track.load_model()
                s2_track.track_video(v, yolo, imgsz=a.imgsz, device=a.det_device)
                if a.det_device == "mps":
                    import torch
                    torch.mps.empty_cache()
            if "select" in todo:
                s2_select.run_video(v)
            if "pose" in todo:
                if pose is None:
                    pose = s2_pose.PoseModel(a.pose_device)
                s2_pose.pose_video(v, pose)
            if "report" in todo:
                s2_report.report_video(v)
            C.update_timing(vid, total_s=round(time.time() - t0, 2))
        except Exception:
            failed.append(vid)
            print(f"  LỖI {vid}:\n{traceback.format_exc()}", flush=True)
    qc = s2_report.write_tables(meta)
    s2_report.print_summary(qc[qc.video_id.isin(vids)] if force else qc)
    print(f"\nTổng thời gian: {time.time()-t_all:.0f}s")
    if failed:
        print("VIDEO LỖI:", " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
