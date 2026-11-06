"""3.4 scale_perturb_instability: chạy lại RTMPose với bbox ×0.9 và ×1.1, đo độ dịch của heel/toe."""
import cv2
import numpy as np
import pandas as pd

import s3_common as C


class PoseModel:
    """RTMPose: CoreML (device="mps"), lỗi thì in cảnh báo và chuyển cpu, không dừng chương trình."""

    def __init__(self, device="mps"):
        from rtmlib import RTMPose
        from rtmlib.tools.solution.body_with_feet import BodyWithFeet
        self._cls, self._cfg = RTMPose, BodyWithFeet.MODE["balanced"]
        self.device = device
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


def scale_box(b, f, W, H):
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return np.array([max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)], np.float32)


def perturb_video(c, pose, L):
    """Trả về DataFrame (frame, scale, point, disp_L) theo nhãn gốc của model, so với keypoints_v2."""
    ok = np.flatnonzero(np.isfinite(c.bbox).all(1) & np.isfinite(c.kp[:, 15, 0]))
    pick = ok[np.unique(np.linspace(0, len(ok) - 1, min(C.N_PERTURB, len(ok))).astype(int))]
    want = {int(c.frame[i]): i for i in pick}
    last = max(want)
    cap = cv2.VideoCapture(str(C.video_path(c.meta)))
    rows, f = [], 0
    while f <= last and cap.grab():
        if f in want:
            good, img = cap.retrieve()
            if not good:
                raise RuntimeError(f"{c.vid}: không đọc được frame {f}")
            i = want[f]
            for sc in C.PERTURB_SCALES:
                k, s = pose(img, scale_box(c.bbox[i], sc, c.W, c.H))
                d = np.linalg.norm(k.astype(float) - c.kp[i], axis=1) / L[i]
                for j in C.FOOT_IDX:
                    rows.append((f, sc, j, float(d[j]), float(s[j])))
                # có đổi nhãn không: bàn chân kia gần hơn hẳn so với chính nó
                st = sum(np.linalg.norm(k[a] - c.kp[i, a]) for a in C.L_FOOT + C.R_FOOT)
                cr = sum(np.linalg.norm(k[a] - c.kp[i, b]) for a, b in zip(C.L_FOOT + C.R_FOOT, C.R_FOOT + C.L_FOOT))
                rows.append((f, sc, -1, float(cr < st), np.nan))
        f += 1
    cap.release()
    return pd.DataFrame(rows, columns=["frame", "scale", "point", "disp_L", "score"])


def summarize(df):
    p = df[df.point >= 0]
    grp = dict(ankle=[15, 16], heel=[24, 25], toe=[20, 21, 22, 23])
    out = {f"scale_perturb_instability_{k}": round(float(p[p.point.isin(v)].disp_L.median()), 4) for k, v in grp.items()}
    out["scale_perturb_instability"] = round(float(p[p.point.isin(grp["heel"] + grp["toe"])].disp_L.median()), 4)
    out["scale_perturb_p90"] = round(float(p[p.point.isin(grp["heel"] + grp["toe"])].disp_L.quantile(.9)), 4)
    out["scale_perturb_flip_frac"] = round(float(df[df.point == -1].disp_L.mean()), 4)
    return out
