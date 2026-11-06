import cv2, numpy as np, pandas as pd
from rtmlib import BodyWithFeet

DATA = "data"
AUD = "data/step0_audit"
model = BodyWithFeet(mode="balanced", backend="onnxruntime", device="cpu")

# Chỉ số keypoint Halpe26
NOSE, LEAR, REAR = 0, 3, 4
LBT, RBT, LST, RST, LHE, RHE = 20, 21, 22, 23, 24, 25   # big toe, small toe, heel
CONF, N_PER_SHOT = 0.5, 25

def pick_person(kp):
    # người có khung keypoint lớn nhất = thường là người chạy
    areas = [np.ptp(k[:, 0]) * np.ptp(k[:, 1]) for k in kp]
    return int(np.argmax(areas))

def foot_dx(k, s):
    vals = []
    for he, bt, st in [(LHE, LBT, LST), (RHE, RBT, RST)]:
        if s[he] > CONF and s[bt] > CONF:
            toe = (k[bt, 0] + k[st, 0]) / 2 if s[st] > CONF else k[bt, 0]
            vals.append(toe - k[he, 0])          # >0: mũi chân chỉ sang phải
    return np.mean(vals) if vals else np.nan

def face_dx(k, s):
    ears = [k[e, 0] for e in (LEAR, REAR) if s[e] > CONF]
    return k[NOSE, 0] - np.mean(ears) if (s[NOSE] > CONF and ears) else np.nan

meta = pd.read_csv(f"{AUD}/step0_metadata.csv")
rows, per_frame = [], []

for _, m in meta.iterrows():
    vid = m.video_id
    folder = "Correct" if m.label == "CORRECT" else "False"
    fr = pd.read_csv(f"{AUD}/frames/{vid}_frames.csv")
    ok = fr[(~fr.is_dup) & (~fr.is_edit_freeze)]

    # chọn ~25 frame rải đều trong mỗi shot
    want = {}
    for shot, g in ok.groupby("shot_id"):
        pick = np.linspace(0, len(g) - 1, min(N_PER_SHOT, len(g))).astype(int)
        for i in g.frame.values[pick]:
            want[int(i)] = int(shot)

    # đọc video tuần tự (an toàn với video VFR)
    cap, i = cv2.VideoCapture(f"{DATA}/{folder}/{m.video}"), 0
    this = []
    while True:
        okf, img = cap.read()
        if not okf:
            break
        if i in want:
            kp, sc = model(img)
            if len(kp):
                p = pick_person(kp)
                this.append(dict(video_id=vid, shot_id=want[i], frame=i,
                                 foot_dx=foot_dx(kp[p], sc[p]),
                                 face_dx=face_dx(kp[p], sc[p])))
        i += 1
    cap.release()
    if i != m.n_frames:
        print(f"  !! {vid}: OpenCV đọc {i} frame, audit có {m.n_frames} -> cần xem tay")
    per_frame += this

    r = pd.DataFrame(this)
    if r.empty:
        rows.append(dict(video_id=vid, shot_id=0, n_sampled=0, n_foot=0, s_foot=0,
                         foot_agree=np.nan, s_face=0, s_prelim=m.s_prelim))
        print(vid, "KHÔNG phát hiện được người")
        continue

    for shot, g in r.groupby("shot_id"):
        f = np.sign(g.foot_dx.dropna())
        h = np.sign(g.face_dx.dropna())
        s_foot = int(np.sign(f.median())) if len(f) else 0
        s_face = int(np.sign(h.median())) if len(h) else 0
        agree = round(float((f == s_foot).mean()), 2) if len(f) else np.nan
        rows.append(dict(video_id=vid, shot_id=shot, n_sampled=len(g), n_foot=len(f),
                         s_foot=s_foot, foot_agree=agree, s_face=s_face, s_prelim=m.s_prelim))
        print(f"{vid:7s} shot{shot}  s_foot={s_foot:+d}  agree={agree}  s_face={s_face:+d}  s_prelim={m.s_prelim:+d}")

pd.DataFrame(rows).to_csv(f"{AUD}/s_keypoint.csv", index=False)
pd.DataFrame(per_frame).to_csv(f"{AUD}/s_keypoint_frames.csv", index=False)
print("XONG")