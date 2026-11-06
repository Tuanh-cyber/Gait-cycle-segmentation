import cv2, sys, os, pandas as pd
from rtmlib import YOLOX, RTMPose, draw_skeleton
from rtmlib.tools.solution.body_with_feet import BodyWithFeet

cfg = BodyWithFeet.MODE["balanced"]   # cùng model như trước
det  = YOLOX(cfg["det"],  model_input_size=cfg["det_input_size"],  backend="onnxruntime", device="cpu")
pose = RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"], backend="onnxruntime", device="mps")

def model(img):
    bboxes = det(img)
    return pose(img, bboxes=bboxes)
os.makedirs("data/step0_audit/review", exist_ok=True)
pf = pd.read_csv("data/step0_audit/s_keypoint_frames.csv")
meta = pd.read_csv("data/step0_audit/step0_metadata.csv").set_index("video_id")

for vid in sys.argv[1:]:
    m = meta.loc[vid]
    folder = "Correct" if m.label == "CORRECT" else "False"
    frames = pf[pf.video_id == vid].frame.values[::5][:4]   # 4 frame
    cap, i, k = cv2.VideoCapture(f"data/{folder}/{m.video}"), 0, 0
    while True:
        ok, img = cap.read()
        if not ok: break
        if i in frames:
            kp, sc = model(img)
            img = draw_skeleton(img, kp, sc, openpose_skeleton=False, kpt_thr=0.5)
            cv2.imwrite(f"data/step0_audit/review/{vid}_{k}.jpg", img); k += 1
        i += 1
    print(vid, "->", k, "ảnh")