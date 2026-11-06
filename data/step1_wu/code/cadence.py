"""Cadence (bước/phút) trong THỜI GIAN VIDEO từ keypoint, không phụ thuộc nhãn L/R:
  s1 = |x_ankleL - x_ankleR| / chiều dài chân  (cực đại 2 lần/stride -> tần số bước)
  s2 = y trung điểm hông (khử trend bậc 2) / chiều dài chân (cực tiểu mỗi bước)
Lomb-Scargle trên timestamp thật của frame unique (không nội suy), tích 2 periodogram chuẩn hóa.
Cửa sổ trượt 2.5 s để phát hiện tốc độ đổi giữa clip."""
import numpy as np, pandas as pd, glob, os, json
from scipy.signal import lombscargle
LA, RA, LH, RH = 15, 16, 11, 12
F = np.linspace(0.12, 4.5, 2000)          # Hz (thời gian video)

def signals(z, t0=None, t1=None):
    t, k, s = z["pts_s"], z["kp"], z["score"]
    ok = (s[:, LA] > .3) & (s[:, RA] > .3) & (s[:, LH] > .3) & (s[:, RH] > .3)
    if t0 is not None: ok &= (t >= t0) & (t < t1)
    t, k = t[ok], k[ok]
    if len(t) < 8: return None
    hip = (k[:, LH] + k[:, RH]) / 2; ank = (k[:, LA] + k[:, RA]) / 2
    L = np.median(np.abs(ank[:, 1] - hip[:, 1])) + 1e-6
    s1 = np.abs(k[:, LA, 0] - k[:, RA, 0]) / L
    s2 = hip[:, 1] / L
    out = []
    for sig in (s1, s2):
        p = np.polyfit(t - t.mean(), sig, 2); r = sig - np.polyval(p, t - t.mean())
        out.append(r / (r.std() + 1e-9))
    return t, out

def periodogram(t, sigs):
    P = np.ones_like(F)
    for r in sigs:
        p = lombscargle(t, r, 2*np.pi*F, precenter=True); P *= p / p.max()
    return P

def peak(t, sigs):
    P = periodogram(t, sigs); i = int(np.argmax(P))
    q = P[i] / (np.median(P) + 1e-12)
    return F[i] * 60, q

rows = []
for f in sorted(glob.glob("keypoints/*_kp.npz")):
    v = os.path.basename(f).replace("_kp.npz", ""); z = np.load(f)
    t = z["pts_s"]
    # đoạn liên tục = cùng shot, không có khoảng trống > 0.3 s (freeze đã bị bỏ)
    gap = np.r_[True, (np.diff(t) > 0.3) | (np.diff(z["shot_id"]) != 0)]
    segid = np.cumsum(gap) - 1
    for g in np.unique(segid):
        m = segid == g; t0, t1 = t[m].min(), t[m].max() + 1/30
        res = signals(z, t0, t1)
        if res is None: continue
        cad, q = peak(*res)
        win = []
        if t1 - t0 >= 5:
            for a in np.arange(t0, t1 - 2.5 + 1e-6, 1.25):
                r = signals(z, a, a + 2.5)
                if r is not None: win.append(round(peak(*r)[0], 1))
        rows.append(dict(video_id=v, seg=int(g), shot_id=int(z["shot_id"][m][0]), t0=round(t0, 3), t1=round(t1, 3),
                         dur_video_s=round(t1 - t0, 2), n_frames=int(m.sum()), cadence_video_spm=round(cad, 1),
                         peak_quality=round(q, 1), window_cadence=json.dumps(win)))
pd.DataFrame(rows).to_csv("step1_wu/cadence_segments.csv", index=False)
print(pd.DataFrame(rows)[["video_id","seg","dur_video_s","n_frames","cadence_video_spm","peak_quality","window_cadence"]].to_string())
