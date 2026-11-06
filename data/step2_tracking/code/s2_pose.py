"""2.4 RTMPose-m Halpe26 theo bbox người chạy. Schema Bước 1 + track_status, continuity_id, t_real_s."""
import os
import time

import cv2
import numpy as np
import pandas as pd

from s2_common import DIRS, atomic_npz, update_timing


class PoseModel:
    """RTMPose với fallback: CoreML (device="mps") lỗi thì in cảnh báo và chuyển cpu, không dừng chương trình."""

    def __init__(self, device="mps"):
        from rtmlib import RTMPose
        from rtmlib.tools.solution.body_with_feet import BodyWithFeet
        self._cls, self._cfg = RTMPose, BodyWithFeet.MODE["balanced"]
        self.device = device
        self.model = None
        try:
            self._load(device)
            self.model(np.zeros((256, 192, 3), np.uint8), bboxes=[[0, 0, 192, 256]])
        except Exception as e:
            self._fallback(e)

    def _load(self, device):
        c = self._cfg
        self.model = self._cls(c["pose"], model_input_size=c["pose_input_size"], backend="onnxruntime", device=device)
        self.device = device
        print(f"  RTMPose device={device}, providers={self.model.session.get_providers()}", flush=True)

    def _fallback(self, e):
        if self.device == "cpu":
            raise e
        print(f"  CẢNH BÁO: RTMPose {self.device} lỗi ({type(e).__name__}: {str(e)[:160]}) -> chuyển sang cpu", flush=True)
        self._load("cpu")

    def __call__(self, img, box):
        try:
            k, s = self.model(img, bboxes=[box])
        except Exception as e:
            self._fallback(e)
            k, s = self.model(img, bboxes=[box])
        return k[0], s[0]


def pose_video(v, pose):
    t0 = time.time()
    run = pd.read_csv(DIRS["runner"] / f"{v.vid}_runner.csv")
    use = v.fr[v.fr.use].reset_index(drop=True)
    rr = run.set_index("frame").loc[use.frame.values]
    N = len(use)
    K, S = np.full((N, 26, 2), np.nan, np.float32), np.zeros((N, 26), np.float32)
    B = rr[["x0", "y0", "x1", "y1"]].values.astype(np.float32)
    has_box = np.isfinite(B).all(1)
    want = {int(f): j for j, f in enumerate(use.frame) if has_box[j]}
    z = np.load(DIRS["appearance"] / f"{v.vid}_hsv.npz")
    W, H = int(z["W"]), int(z["H"])
    cap = cv2.VideoCapture(str(v.path))
    i = 0
    while cap.grab():
        if i in want:
            ok, img = cap.retrieve()
            if not ok:
                raise RuntimeError(f"{v.vid}: không retrieve được frame {i}")
            j = want[i]
            box = np.array([max(0, B[j, 0]), max(0, B[j, 1]), min(W, B[j, 2]), min(H, B[j, 3])], np.float32)
            B[j] = box
            K[j], S[j] = pose(img, box)    # không bao giờ gọi với bbox rỗng: rtmlib sẽ tự lấy cả ảnh
        i += 1
    cap.release()
    atomic_npz(DIRS["keypoints_v2"] / f"{v.vid}_kp.npz",
               frame=use.frame.values, pts_s=use.pts_s.values, shot_id=use.shot_id.values, kp=K, score=S, bbox=B,
               det=(rr.track_status.values == "tracked"), W=W, H=H, n_read=i, n_meta=v.n_frames,
               track_status=rr.track_status.values.astype("U12"),
               continuity_id=use.continuity_id.values.astype(np.int64), t_real_s=use.t_real_s.values)
    dt = time.time() - t0
    update_timing(v.vid, pose_s=round(dt, 2), pose_device=pose.device, n_pose=len(want))
    print(f"  pose   {v.vid}: {len(want)}/{N} frame, {dt:.1f}s", flush=True)


def done(vid):
    return (DIRS["keypoints_v2"] / f"{vid}_kp.npz").exists()
