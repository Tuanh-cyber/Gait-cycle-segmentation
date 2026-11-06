"""RTMPose Halpe26 cho mọi frame unique, không thuộc freeze biên tập.
Detector YOLOX chạy mỗi DET_EVERY frame; giữa các lần detect, bbox = bbox keypoint frame trước (nới 25%).
Chọn người: lần đầu = detection gần tâm vùng chuyển động (run_cx/run_cy từ Bước 0) nhất; sau đó = IoU lớn nhất với bbox trước.
Resumable: bỏ qua video đã có keypoints/<vid>_kp.npz. Dừng an toàn sau TIME_BUDGET giây.
Chạy từ data/:  python step1_wu/code/pose_all_frames.py [budget_s]"""
import os, sys, time, cv2, numpy as np, pandas as pd
os.environ.setdefault("TORCH_HOME", os.path.abspath("_rtm"))
from rtmlib import YOLOX, RTMPose
from rtmlib.tools.solution.body_with_feet import BodyWithFeet
DEVICE = os.environ.get("RTM_DEVICE", "cpu")          # trên Mac: RTM_DEVICE=mps cho pose (detector luôn cpu)
DET_EVERY, CONF = 6, 0.3
TIME_BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
c = BodyWithFeet.MODE["balanced"]
det = YOLOX(c["det"], model_input_size=c["det_input_size"], backend="onnxruntime", device="cpu")
pose = RTMPose(c["pose"], model_input_size=c["pose_input_size"], backend="onnxruntime", device=DEVICE)

def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1-x0)*max(0, y1-y0); ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0

def kp_box(k, s, W, H, pad=0.25):
    v = s > CONF
    if v.sum() < 5: return None
    x0, y0 = k[v].min(0); x1, y1 = k[v].max(0); w, h = x1-x0, y1-y0
    return np.array([max(0, x0-pad*w), max(0, y0-pad*h), min(W, x1+pad*w), min(H, y1+pad*h)])

t_start = time.time()
meta = pd.read_csv("step0_audit/step0_metadata.csv")
for m in meta.itertuples():
    out = f"keypoints/{m.video_id}_kp.npz"
    if os.path.exists(out): continue
    if time.time() - t_start > TIME_BUDGET: print("budget reached"); break
    fr = pd.read_csv(f"step0_audit/frames/{m.video_id}_frames.csv")
    excl = set(int(s.replace("shot", "")) for s in str(m.excluded_shots).split(",") if s.startswith("shot"))
    use = fr[(~fr.is_dup) & (~fr.is_edit_freeze) & (~fr.shot_id.isin(excl))]
    want = dict(zip(use.frame.astype(int), range(len(use))))
    cap = cv2.VideoCapture(f"{'Correct' if m.label=='CORRECT' else 'False'}/{m.video}")
    W, H = int(cap.get(3)), int(cap.get(4)); sc = W/480
    K = np.full((len(use), 26, 2), np.nan, np.float32); S = np.zeros((len(use), 26), np.float32)
    B = np.full((len(use), 4), np.nan, np.float32); DET = np.zeros(len(use), bool)
    prev, last_shot, i, n_det = None, None, 0, 0
    while True:
        ok, img = cap.read()
        if not ok: break
        if i in want:
            j = want[i]; r = use.iloc[j]
            if r.shot_id != last_shot: prev = None; last_shot = r.shot_id      # cắt cảnh -> chọn lại người
            box = None
            if prev is None or n_det % DET_EVERY == 0:
                bbs = det(img)
                if len(bbs):
                    if prev is None:
                        if np.isfinite(r.run_cx):
                            cx, cy = r.run_cx*sc, r.run_cy*sc
                            d = [np.hypot((b[0]+b[2])/2-cx, (b[1]+b[3])/2-cy)/max(1, b[3]-b[1]) for b in bbs]
                            box = bbs[int(np.argmin(d))]
                        else:
                            box = bbs[int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in bbs]))]
                    else:
                        ious = [iou(prev, b) for b in bbs]
                        if max(ious) > 0.2: box = bbs[int(np.argmax(ious))]
                        else: box = prev
                    DET[j] = True
            if box is None: box = prev
            n_det += 1
            if box is not None:
                k, s = pose(img, bboxes=[box])
                K[j], S[j], B[j] = k[0], s[0], box
                nb = kp_box(k[0], s[0], W, H); prev = nb if nb is not None else box
        i += 1
    cap.release()
    np.savez_compressed(out, frame=use.frame.values, pts_s=use.pts_s.values, shot_id=use.shot_id.values,
                        kp=K, score=S, bbox=B, det=DET, W=W, H=H, n_read=i, n_meta=m.n_frames)
    print(f"{m.video_id}: {len(use)} frames, read {i}/{m.n_frames}, {time.time()-t_start:.0f}s", flush=True)
print("DONE" if all(os.path.exists(f"keypoints/{v}_kp.npz") for v in meta.video_id) else "PARTIAL")
