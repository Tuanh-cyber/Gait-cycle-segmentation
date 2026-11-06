"""Kho flow: chạy GMA một lần cho mỗi video, lưu flow forward thô và mặt nạ FB trên hộp bao mask chân + dải đất.
Đổi cách bù camera chỉ cần đọc kho này. Lưu flow thô, không lưu residual: residual = flow − mô hình camera, còn đổi mô hình thì cần flow thô."""
import hashlib
import json
import time

import cv2
import numpy as np

import s5_common as C
import s5_features as X
import s5_flow as W

STORE = C.OUT.parent / "flow_store" if C.OUT.name == "_trial" else C.OUT / "flow_store"
STORE_PAD_X, STORE_ABOVE, STORE_BELOW = 2.0, 1.0, 1.5     # × L_foot trên lưới flow, quanh chân
ROUND = 16              # làm tròn flow về 1/16 px trước khi nén


def flow_hash():
    keys = ("MODEL", "ITERS", "BOX_PAD", "BOX_PAD_BOTTOM", "MIN_L_FOOT", "MAX_EDGE", "MAX_EDGE_HARD", "FB_ABS", "FB_REL")
    d = [(k, repr(getattr(C, k))) for k in keys] + [("store", repr((STORE_PAD_X, STORE_ABOVE, STORE_BELOW, ROUND)))]
    return hashlib.sha1(repr(d).encode()).hexdigest()[:10]


def store_box(v, a, s, box, shape, R):
    """Hộp trên lưới flow: bao mask 2 chân và dải đất quanh chân."""
    h, w = shape
    x0, y0 = box[:2]
    lf = float(np.nanmin(v.z["l_foot"][a])) * s
    pts = [C.S4.foot_pts(v.c, a, o) for o in (1, 2)]
    pts = [p for p in pts if len(p)]
    xs, ys = [], []
    if pts:
        p = (np.vstack(pts) - [x0, y0]) * s
        xs += [p[:, 0].min(), p[:, 0].max()]
        ys += [p[:, 1].min(), p[:, 1].max()]
    u = R[0][3] | R[1][3]
    if u.any():
        yy, xx = np.nonzero(u)
        xs += [xx.min(), xx.max()]
        ys += [yy.min(), yy.max()]
    if not xs:
        return 0, 0, w, h
    pad = (STORE_PAD_X * lf, STORE_ABOVE * lf, STORE_BELOW * lf)
    return (int(max(0, min(xs) - pad[0])), int(max(0, min(ys) - pad[1])), int(min(w, max(xs) + pad[0] + 1)), int(min(h, max(ys) + pad[2] + 1)))


def frames_stream(path, want):
    cap = cv2.VideoCapture(str(path))
    want = set(int(f) for f in want)
    f, last = 0, max(want)
    while f <= last and cap.grab():
        if f in want:
            ok, img = cap.retrieve()
            if not ok:
                raise RuntimeError(f"{path}: không đọc được frame {f}")
            yield f, img
        f += 1
    cap.release()


def grid_pair(prev, img, box, s):
    x0, y0, x1, y1 = box
    w, h = int(round((x1 - x0) * s)), int(round((y1 - y0) * s))
    it = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(prev[y0:y1, x0:x1], (w, h), interpolation=it), cv2.resize(img[y0:y1, x0:x1], (w, h), interpolation=it)


def plan(v):
    idx = C.pairs(v)
    boxes = {int(a): C.crop_box(v, int(a)) for a in idx}
    scales, blocks = C.block_scales(v, idx, boxes)
    return idx, boxes, scales, dict(zip(idx.tolist(), blocks.tolist()))


def done(vid):
    p = STORE / f"{vid}.npz"
    if not p.exists():
        return False
    z = np.load(p, allow_pickle=True)
    return str(z["flow_hash"]) == flow_hash()


def build(vid, meta, state):
    """Chạy GMA cho cả video, ghi kho. state: model, device."""
    t0 = time.time()
    v = C.load_video(vid, meta)
    c = v.c
    idx, boxes, scales, block_of = plan(v)
    want = set(idx.tolist())
    pos = {int(f): i for i, f in enumerate(c.frame)}
    rec = {k: [] for k in ("a", "crop", "scale", "sbox", "shape")}
    flows, fbs = [], []
    prev = None
    for fr, img in frames_stream(c.path, c.frame):
        a = pos[fr] - 1
        if a in want and prev is not None and boxes[a] is not None:
            s = scales[block_of[a]]
            A, B = grid_pair(prev, img, boxes[a], s)
            while True:
                try:
                    fw, bw = W.flow_pair(state["model"], A, B, state["device"])
                    break
                except (RuntimeError, NotImplementedError) as e:
                    if state["device"] == "cpu":
                        raise
                    print(f"  CẢNH BÁO: GMA {state['device']} lỗi ở {vid} f{fr} ({type(e).__name__}: {str(e)[:200]}) -> chuyển cpu", flush=True)
                    state["device"], state["model"] = "cpu", W.build("cpu")
            fb = W.fb_valid(fw, bw)
            R, _ = X.region_masks(v, a, s, boxes[a], A.shape[:2])
            b = store_box(v, a, s, boxes[a], A.shape[:2], R)
            x0, y0, x1, y1 = b
            flows.append((np.round(fw[y0:y1, x0:x1] * ROUND) / ROUND).astype(np.float16).ravel())
            fbs.append(np.packbits(fb[y0:y1, x0:x1]))
            rec["a"].append(a)
            rec["crop"].append(boxes[a])
            rec["scale"].append(s)
            rec["sbox"].append(b)
            rec["shape"].append(A.shape[:2])
        prev = img
    C.atomic_npz(STORE / f"{vid}.npz", flow=np.concatenate(flows) if flows else np.zeros(0, np.float16), fb=np.concatenate(fbs) if fbs else np.zeros(0, np.uint8),
                 a=np.array(rec["a"], np.int32), crop=np.array(rec["crop"], np.int32).reshape(-1, 4), scale=np.array(rec["scale"], np.float32), sbox=np.array(rec["sbox"], np.int32).reshape(-1, 4),
                 shape=np.array(rec["shape"], np.int32).reshape(-1, 2), flow_hash=flow_hash(), model=C.MODEL, device=state["device"])
    return time.time() - t0, len(flows)


class Store:
    """Đọc kho theo từng cặp."""

    def __init__(self, vid):
        z = np.load(STORE / f"{vid}.npz", allow_pickle=True)
        self.a, self.crop, self.scale, self.sbox, self.shape = z["a"], z["crop"], z["scale"], z["sbox"], z["shape"]
        self.flow, self.fb = z["flow"], z["fb"]
        sizes = (self.sbox[:, 3] - self.sbox[:, 1]) * (self.sbox[:, 2] - self.sbox[:, 0])
        self.off = np.r_[0, np.cumsum(sizes * 2)]
        self.foff = np.r_[0, np.cumsum(np.ceil(sizes / 8).astype(np.int64))]
        self.pos = {int(a): i for i, a in enumerate(self.a)}

    def get(self, a):
        """Trả về (flow H×W×2 trong hộp, fb H×W, sbox, crop, scale, shape)."""
        i = self.pos[a]
        x0, y0, x1, y1 = self.sbox[i]
        h, w = y1 - y0, x1 - x0
        f = self.flow[self.off[i]:self.off[i + 1]].reshape(h, w, 2).astype(np.float32)
        fb = np.unpackbits(self.fb[self.foff[i]:self.foff[i + 1]])[:h * w].reshape(h, w).astype(bool)
        return f, fb, tuple(int(x) for x in self.sbox[i]), tuple(int(x) for x in self.crop[i]), float(self.scale[i]), tuple(int(x) for x in self.shape[i])
