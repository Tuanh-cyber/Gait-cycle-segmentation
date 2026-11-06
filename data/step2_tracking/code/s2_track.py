"""2.1 YOLO + BoT-SORT trên frame hợp lệ, reset tracker tại mỗi thay đổi continuity_id."""
import os
import time

import cv2
import numpy as np
import pandas as pd

from s2_common import (CONF, DIRS, IMGSZ, OUT, TRACKER_YAML, WEIGHTS, atomic_csv, atomic_npz, update_timing)

COLS = ["frame", "pts_s", "continuity_id", "track_id", "x0", "y0", "x1", "y1", "conf"]


def load_model(weights=None):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "models" / ".ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"sync": False})
    DetCapture.install()
    return YOLO(str(weights or DIRS["models"] / WEIGHTS))


class DetCapture:
    """Lấy box detection thô cho từng track. ultralytics thay box bằng trạng thái Kalman (lệch tới ~14% chiều cao,
    có thể cắt đầu/chân) nhưng vẫn trả chỉ số detection ở cột cuối của tracker.update()."""
    dets, idx = None, None

    @classmethod
    def install(cls):
        from ultralytics.trackers.bot_sort import BOTSORT
        if getattr(BOTSORT.update, "_captures", False):
            return
        orig = BOTSORT.update

        def update(self, results, *a, **kw):
            out = orig(self, results, *a, **kw)
            cls.dets = results.xyxy.copy()
            cls.idx = out[:, -1].astype(int).copy() if len(out) else None
            return out
        update._captures = True
        BOTSORT.update = update


def torso_hist(img, box):
    """Histogram H-S của vùng thân (giữa 50% rộng, 20–60% cao); tổng = 1."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    H, W = img.shape[:2]
    a, b = int(max(0, x0 + 0.25 * w)), int(min(W, x1 - 0.25 * w))
    c, d = int(max(0, y0 + 0.2 * h)), int(min(H, y0 + 0.6 * h))
    if b - a < 2 or d - c < 2:
        return np.zeros(16 * 8, np.float32)
    hsv = cv2.cvtColor(img[c:d, a:b], cv2.COLOR_BGR2HSV)
    q = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel().astype(np.float32)
    return q / max(q.sum(), 1.0)


def reset_tracker(model):
    """Xoá hẳn trạng thái tracker (track, lost, Kalman, GMC). Lần gọi đầu chưa có predictor thì không cần."""
    pred = getattr(model, "predictor", None)
    for t in getattr(pred, "trackers", []) if pred is not None else []:
        t.reset()


def track_video(v, model, imgsz=IMGSZ, device="mps"):
    t0 = time.time()
    use = v.fr[v.fr.use]
    want = {int(f): i for i, f in enumerate(use.frame)}
    rows, hists = [], []
    offset, max_id, cur_block, shape, n_blocks = 0, 0, None, None, 0
    cap = cv2.VideoCapture(str(v.path))
    i = 0
    while cap.grab():
        if i in want:
            r = use.iloc[want[i]]
            ok, img = cap.retrieve()
            if not ok:
                raise RuntimeError(f"{v.vid}: không retrieve được frame {i}")
            shape = img.shape
            if r.continuity_id != cur_block:
                reset_tracker(model)         # tracker mới, đếm ID lại từ 1
                offset, cur_block = max_id, r.continuity_id   # ID không bao giờ trùng giữa 2 khối
                n_blocks += 1
            res = model.track(img, persist=True, tracker=str(TRACKER_YAML), classes=[0], conf=CONF,
                              imgsz=imgsz, device=device, verbose=False)[0]
            b = res.boxes
            if b is not None and b.id is not None:
                ids, cf = b.id.cpu().numpy().astype(int), b.conf.cpu().numpy()
                xyxy = DetCapture.dets[DetCapture.idx]     # cùng thứ tự với b (result[idx])
                if len(xyxy) != len(ids):
                    raise RuntimeError(f"{v.vid} f{i}: không khớp detection/track")
                for box, tid, c in zip(xyxy, ids, cf):
                    gid = int(tid) + offset
                    max_id = max(max_id, gid)
                    rows.append((int(r.frame), float(r.pts_s), int(r.continuity_id), gid,
                                 *np.round(box.astype(float), 2).tolist(), round(float(c), 4)))
                    hists.append(torso_hist(img, box))
        i += 1
    cap.release()
    if i != v.n_frames:
        print(f"  !! {v.vid}: OpenCV đọc {i} frame, audit có {v.n_frames}", flush=True)
    df = pd.DataFrame(rows, columns=COLS)
    H = np.stack(hists) if hists else np.zeros((0, 128), np.float32)
    atomic_npz(DIRS["appearance"] / f"{v.vid}_hsv.npz", hist=H, n_read=i, W=shape[1], H=shape[0])
    atomic_csv(df, DIRS["all_tracks"] / f"{v.vid}_tracks.csv")   # ghi sau cùng: dấu hiệu hoàn thành stage
    dt = time.time() - t0
    update_timing(v.vid, track_s=round(dt, 2), n_use=len(use), n_read=i, imgsz=imgsz)
    print(f"  track  {v.vid}: {len(use)} frame, {n_blocks} khối, {df.track_id.nunique()} track, "
          f"{dt:.1f}s ({len(use)/max(dt,1e-6):.1f} fps)", flush=True)


def done(vid):
    return (DIRS["all_tracks"] / f"{vid}_tracks.csv").exists() and (DIRS["appearance"] / f"{vid}_hsv.npz").exists()
