"""4.6: gate, QC, ảnh review, ảnh xung đột."""
import cv2
import numpy as np
import pandas as pd

import s4_common as C

COLOR = {"near": (60, 122, 22), "far": (36, 138, 240), "A": (181, 109, 47), "B": (181, 79, 122)}   # BGR: xanh lá, cam, xanh dương, tím
SHORT = ["vis", "par", "occ"]


def gate(q):
    fail, warn = [], []
    near = q["frac_visible_near"] if np.isfinite(q["frac_visible_near"]) else q["frac_visible_best_AB"]
    if near < C.GATE_NEAR_FAIL:
        fail.append(f"frac_visible_near<{C.GATE_NEAR_FAIL}")
    if q["n_id_conflict_frames"] > C.GATE_CONFLICT_FAIL * q["n_frames"]:
        fail.append("id_conflict>10%")
    far = q["frac_visible_far"] if np.isfinite(q["frac_visible_far"]) else q["frac_visible_worst_AB"]
    if far < C.GATE_FAR_WARN:
        warn.append(f"frac_visible_far<{C.GATE_FAR_WARN}")
    if q["max_reprompt_block"] >= C.GATE_REPROMPT_WARN:
        warn.append(f"reprompt>={C.GATE_REPROMPT_WARN}/khối")
    return ("FAIL" if fail else "WARN" if warn else "PASS"), "; ".join(fail + warn)


def decode(z, a, o, kind, shape):
    """kind 0 = mask, 1/2/3 = heel/mid/toe."""
    s, e = z["rle_ptr"][a, o, kind]
    return C.rle_decode(z["rle_counts"][s:e], shape)


def read_frames(c, frames):
    want, out = set(int(f) for f in frames), {}
    cap = cv2.VideoCapture(str(c.path))
    f, last = 0, max(want)
    while f <= last and cap.grab():
        if f in want:
            ok, img = cap.retrieve()
            if ok:
                out[f] = img
        f += 1
    cap.release()
    return out


def overlay(im, m, color, alpha=0.45):
    if m.any():
        im[m] = (im[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnt, -1, color, 1, cv2.LINE_AA)


def draw_cuts(im, origin, axis, Lf, color):
    if not (np.isfinite(origin).all() and np.isfinite(axis).all()) or np.linalg.norm(axis) < 1e-6:
        return
    u = axis / np.linalg.norm(axis)
    n = np.array([-u[1], u[0]])
    for e in C.REGION_EDGES:
        c0 = origin + u * e * Lf
        a, b = c0 - n * 0.5 * Lf, c0 + n * 0.5 * Lf
        cv2.line(im, tuple(np.round(a).astype(int).tolist()), tuple(np.round(b).astype(int).tolist()), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(im, tuple(np.round(c0 - n * 0.2 * Lf).astype(int).tolist()), tuple(np.round(c0 + n * 0.2 * Lf).astype(int).tolist()), color, 2 if Lf >= 40 else 1, cv2.LINE_AA)


def cell(im, box, size):
    x0, y0, x1, y1 = box
    v = im[y0:y1, x0:x1]
    f = min(size / max(1, v.shape[0]), size / max(1, v.shape[1]))
    v = cv2.resize(v, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 30, np.uint8)
    yy, xx = (size - v.shape[0]) // 2, (size - v.shape[1]) // 2
    out[yy:yy + v.shape[0], xx:xx + v.shape[1]] = v
    return out


def caption(v, text, y):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(v, (0, y - h - 5), (w + 12, y + 5), (30, 30, 30), -1)
    cv2.putText(v, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render(c, z, a, img, size, extra="", show_kp=False):
    """1 ô: vùng bàn chân, tô 2 mask, vẽ đường chia vùng, ghi state."""
    shape = (c.H, c.W)
    im = img.copy()
    masks = [decode(z, a, o, 0, shape) for o in (0, 1)]
    labs = z["label"][a]
    for o in (1, 0):
        overlay(im, masks[o], COLOR[labs[o]])
    i = int(a)
    Lf = z["l_foot"][a]
    for o in (0, 1):
        draw_cuts(im, z["origin"][a, o], z["axis"][a, o], Lf[o], COLOR[labs[o]])
    pts = []
    for o in (0, 1):
        ys, xs = np.nonzero(masks[o])
        if len(xs):
            pts.append(np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]]))
        kp = C.foot_pts(c, i, o + 1)
        if len(kp):
            pts.append(kp)
            if show_kp:
                for p in kp:
                    cv2.circle(im, tuple(np.round(p).astype(int).tolist()), 4, COLOR[labs[o]], -1, cv2.LINE_AA)
                    cv2.circle(im, tuple(np.round(p).astype(int).tolist()), 4, (255, 255, 255), 1, cv2.LINE_AA)
    if pts:
        p = np.vstack(pts)
        x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
        pad = 0.25 * max(x1 - x0, y1 - y0, 40)
        box = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(c.W, x1 + pad)), int(min(c.H, y1 + pad)))
    else:
        box = (0, 0, c.W, c.H)
    v = cell(im, box, size)
    st = z["state"][a]
    if "frame_valid_mask" in z.files:
        extra += "".join(f" X{labs[o]}" for o in (0, 1) if not z["frame_valid_mask"][a, o])
    caption(v, f"f{z['frame'][a]} {labs[0]}:{SHORT[st[0]]} {labs[1]}:{SHORT[st[1]]} iou{z['iou'][a]:.2f}" + extra, 18)
    return v


def review(c, z, path, size=400):
    n = len(z["frame"])
    even = np.unique(np.linspace(0, n - 1, 4).astype(int))
    # ưu tiên frame chồng lấn: IoU cao nhất, không sát nhau
    order = np.argsort(-np.nan_to_num(z["iou"]))
    pick = list(even)
    for a in order:
        if len(pick) >= 8:
            break
        if z["iou"][a] > 0 and all(abs(a - p) >= 3 for p in pick):
            pick.append(int(a))
    for a in np.linspace(0, n - 1, 16).astype(int):
        if len(pick) >= 8:
            break
        if a not in pick:
            pick.append(int(a))
    pick = sorted(pick)
    imgs = read_frames(c, z["frame"][pick])
    cells = [render(c, z, a, imgs[int(z["frame"][a])], size) for a in pick]
    while len(cells) < 8:
        cells.append(np.full((size, size, 3), 30, np.uint8))
    head = np.full((28, size * 4, 3), 245, np.uint8)
    labs = "/".join(sorted(set(z["label"].ravel().tolist())))
    cv2.putText(head, f"{c.vid}  xanh la = near, cam = far, xanh duong/tim = A/B (khong gan near/far)  nhan: {labs}  vach trang = chia vung", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([head, np.hstack(cells[:4]), np.hstack(cells[4:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def conflict_images(c, z, intervals, size=400, max_n=5):
    for old in C.DIRS["review"].glob(f"_conflict_{c.vid}_*.jpg"):
        old.unlink()
    use = intervals[:max_n]
    if not use:
        return
    frames = []
    for it in use:
        frames += [it["a"], (it["a"] + it["b"]) // 2, it["b"]]
    imgs = read_frames(c, z["frame"][frames])
    for k, it in enumerate(use):
        cells = [render(c, z, a, imgs[int(z["frame"][a])], size, show_kp=True) for a in (it["a"], (it["a"] + it["b"]) // 2, it["b"])]
        cv2.imwrite(str(C.DIRS["review"] / f"_conflict_{c.vid}_{k}.jpg"), np.hstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_review_csv(meta):
    p = C.OUT / "footmask_review.csv"
    if not p.exists():
        pd.DataFrame(dict(video_id=meta.video_id, verdict="", notes="")).to_csv(p, index=False)
