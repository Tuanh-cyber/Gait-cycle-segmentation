"""4.1–4.2: chạy SAM2 hai object trên 1 khối, re-prompt khi trôi."""
import shutil
import time

import cv2
import numpy as np
import torch

import s4_common as C


def build(name, device):
    from sam2.build_sam import build_sam2_video_predictor
    fn, cfg = C.CKPT[name]
    extra = [f"++model.image_size={C.IMAGE_SIZE}"] if C.IMAGE_SIZE != 1024 else []
    return build_sam2_video_predictor(cfg, str(C.MODELS / fn), device=device, hydra_overrides_extra=extra)


def extract_frames(c, blocks):
    """Trích frame unique từng khối ra JPEG đã crop. blocks: {cont: (idx, window)}. Trả về {cont: thư mục}."""
    tmp = C.DIRS["_tmp"] / c.vid
    shutil.rmtree(tmp, ignore_errors=True)
    want = {}
    dirs = {}
    for b, (idx, win) in blocks.items():
        d = tmp / f"b{b}"
        d.mkdir(parents=True)
        dirs[b] = d
        for n, i in enumerate(idx):
            want[int(c.frame[i])] = (d / f"{n:05d}.jpg", win)
    cap = cv2.VideoCapture(str(c.path))
    f, last, written = 0, max(want), 0
    while f <= last and cap.grab():
        if f in want:
            ok, img = cap.retrieve()
            if not ok:
                raise RuntimeError(f"{c.vid}: không đọc được frame {f}")
            if img.shape[0] != c.H or img.shape[1] != c.W:
                raise RuntimeError(f"{c.vid}: kích thước frame {img.shape} khác Bước 2 ({c.H}x{c.W})")
            p, (x0, y0, x1, y1) = want[f]
            cv2.imwrite(str(p), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, C.JPEG_Q])
            written += 1
        f += 1
    cap.release()
    if written != len(want):
        raise RuntimeError(f"{c.vid}: mới trích {written}/{len(want)} frame")
    return dirs


def prompt_for(c, i, obj, Lf, off):
    """Box + point dương (heel, big toe) + point âm (bàn chân kia), toạ độ crop."""
    other = 3 - obj
    pts = C.foot_pts(c, i, obj)
    x0, y0 = pts.min(0) - C.BOX_PAD * Lf
    x1, y1 = pts.max(0) + C.BOX_PAD * Lf
    f, g = C.FOOT[obj], C.FOOT[other]
    pos = [c.kp[i, f[j]] for j in (C.HEEL, C.BIG) if c.valid[i, f[j]]]
    if not pos:
        pos = [c.kp[i, f[C.SMALL]]]
    neg = [c.kp[i, g[j]] for j in (C.HEEL, C.BIG) if c.valid[i, g[j]]]
    # giữa ống chân của chính nó: không có point này, chân trần/da liền màu làm mask lan lên cẳng chân (C13_1)
    if c.valid[i, C.KNEE[obj]] and c.valid[i, f[C.ANK]]:
        neg.append((c.kp[i, C.KNEE[obj]] + c.kp[i, f[C.ANK]]) / 2)
    P = np.array(pos + neg, np.float32) - off
    L = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    return np.array([x0 - off[0], y0 - off[1], x1 - off[0], y1 - off[1]], np.float32), P, L


def foot_box_iou(c, i):
    a, b = C.foot_pts(c, i, 1), C.foot_pts(c, i, 2)
    if not len(a) or not len(b):
        return 1.0
    a0, a1, b0, b1 = a.min(0), a.max(0), b.min(0), b.max(0)
    iw = max(0.0, min(a1[0], b1[0]) - max(a0[0], b0[0]))
    ih = max(0.0, min(a1[1], b1[1]) - max(a0[1], b0[1]))
    u = np.prod(a1 - a0) + np.prod(b1 - b0) - iw * ih
    return float(iw * ih / u) if u > 0 else 1.0


def choose_init(c, idx):
    """Frame có cả 2 chân hợp lệ và |x_A − x_B| lớn nhất."""
    ok = C.foot_ok(c, idx, 1) & C.foot_ok(c, idx, 2)
    if not ok.any():
        return None
    best, arg = -1.0, None
    for n in np.flatnonzero(ok):
        i = idx[n]
        s = abs(C.foot_pts(c, i, 1)[:, 0].mean() - C.foot_pts(c, i, 2)[:, 0].mean())
        if s > best:
            best, arg = s, int(n)
    return arg


def frame_metrics(c, idx, M, Lf, off):
    """cov: tỉ lệ điểm trúng mask đúng nghĩa đen. cov_tol, other: có dung sai COV_TOL × L_foot. Mọi mảng N×2."""
    n = len(idx)
    area = M.sum((2, 3)).astype(float)
    cov, cov_tol, other = np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    H, W = M.shape[2:]

    def frac(a, pts, o, r):
        if not len(pts):
            return np.nan
        q = np.round(pts - off).astype(int)
        hit = 0
        for x, y in q:
            x0, x1, y0, y1 = max(0, x - r), min(W, x + r + 1), max(0, y - r), min(H, y + r + 1)
            hit += x0 < x1 and y0 < y1 and M[a, o, y0:y1, x0:x1].any()
        return hit / len(q)

    pt3 = {o: [C.FOOT[o][j] for j in (C.HEEL, C.BIG, C.SMALL)] for o in (1, 2)}
    for a, i in enumerate(idx):
        for o in (1, 2):
            r = int(np.ceil(C.COV_TOL * Lf[o]))
            own = c.kp[i, [j for j in pt3[o] if c.valid[i, j]]]
            oth = c.kp[i, [j for j in pt3[3 - o] if c.valid[i, j]]]
            cov[a, o - 1] = frac(a, own, o - 1, 0)
            cov_tol[a, o - 1] = frac(a, own, o - 1, r)
            other[a, o - 1] = frac(a, oth, o - 1, r)
    med = np.array([np.median(area[area[:, o] > 0, o]) if (area[:, o] > 0).any() else np.nan for o in (0, 1)])
    return dict(area=area, cov=cov, cov_tol=cov_tol, other=other, ratio=area / med, med=med)


def find_bad_runs(c, idx, met, min_len):
    """Mọi đoạn >= min_len frame liên tiếp bị trôi mà keypoint hợp lệ. Trả về list (obj, a, b)."""
    out = []
    for o in (1, 2):
        bad = ((met["cov_tol"][:, o - 1] < C.COV_MIN) | (met["ratio"][:, o - 1] < C.AREA_LO) | (met["ratio"][:, o - 1] > C.AREA_HI)
               | (met["area"][:, o - 1] == 0)) & C.foot_ok(c, idx, o)
        a = None
        for i, v in enumerate(np.r_[bad, False]):
            if v and a is None:
                a = i
            if not v and a is not None:
                if i - a >= min_len:
                    out.append((o, a, i - 1))
                a = None
    return out


def pick_prompt_frame(c, idx, a, b):
    """Trong đoạn, frame 2 bàn chân tách xa nhất và không chồng nhau. None nếu cả đoạn đang chồng."""
    best, arg = -1.0, None
    for p in range(a, b + 1):
        i = idx[p]
        if foot_box_iou(c, i) > C.OVERLAP_NO_PROMPT:
            continue
        s = np.linalg.norm(C.foot_pts(c, i, 1).mean(0) - C.foot_pts(c, i, 2).mean(0))
        if s > best:
            best, arg = s, p
    return arg


def keep_own_foot(c, idx, Lg, Lf, off, trimmed):
    """SAM2 có thể cho 1 object phủ cả 2 bàn chân thành 2 mảnh rời (giày giống nhau). Mỗi mảnh rời thuộc về bàn chân gần tâm nó hơn;
    bỏ mảnh của chân kia hoặc nằm xa. Không xét khi 2 bàn chân đang chồng nhau, vì lúc đó khoảng cách không phân biệt được."""
    for a, i in enumerate(idx):
        if foot_box_iou(c, i) > C.OVERLAP_NO_PROMPT:
            continue
        for o in (1, 2):
            own, oth = C.foot_pts(c, i, o), C.foot_pts(c, i, 3 - o)
            if len(own) < C.MIN_PTS or len(oth) < C.MIN_PTS:
                continue
            m = (Lg[a, o - 1] > 0).astype(np.uint8)
            k, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
            big = [j for j in range(1, k) if st[j, cv2.CC_STAT_AREA] >= max(C.COMP_MIN_PX, C.COMP_MIN_FRAC * m.sum())]
            if len(big) < 2:
                continue
            keep = []
            for j in big:
                p = cen[j] + off
                d_own, d_oth = np.linalg.norm(own - p, axis=1).min(), np.linalg.norm(oth - p, axis=1).min()
                if d_own <= d_oth and d_own <= C.COMP_MAX_DIST * Lf[o]:
                    keep.append(j)
            drop = (m > 0) & ~np.isin(lab, keep)
            if drop.any():
                Lg[a, o - 1][drop] = -1
                trimmed[a, o - 1] = True


def run_block(predictor, c, idx, win, frames_dir, device):
    """Trả về M (N×2×H×W bool), ov (N, trung bình logit obj1 − obj2 trên vùng giao), log re-prompt, frame khởi tạo, L_foot, trimmed (N×2)."""
    off = np.array([win[0], win[1]], np.float64)
    H, W = win[3] - win[1], win[2] - win[0]
    n = len(idx)
    Lf = {o: C.l_foot(c, idx, o) for o in (1, 2)}
    fall = np.nanmedian([Lf[1], Lf[2]])
    Lf = {o: (Lf[o] if np.isfinite(Lf[o]) else fall) for o in (1, 2)}
    Lg = np.full((n, 2, H, W), -1, np.int8)    # logit × 4, cắt ±127; dấu giữ đúng ngưỡng > 0 của mask
    ov = np.full(n, np.nan)
    trimmed = np.zeros((n, 2), bool)
    init = choose_init(c, idx)
    log = []
    if init is None or not np.isfinite(fall):
        return Lg > 0, ov, log, None, Lf, trimmed

    def propagate(state, start, reverse, limit=None):
        for fi, ids, logits in predictor.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=limit, reverse=reverse):
            lg = logits.squeeze(1).cpu().numpy()
            for k, oid in enumerate(ids):
                v = np.clip(lg[k] * 4, -127, 127)
                v = np.where(lg[k] > 0, np.maximum(v, 1), np.minimum(v, -1))
                Lg[fi, oid - 1] = v.astype(np.int8)

    with torch.inference_mode():
        state = predictor.init_state(str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=False)
        for o in (1, 2):
            box, P, L = prompt_for(c, idx[init], o, Lf[o], off)
            predictor.add_new_points_or_box(state, frame_idx=init, obj_id=o, points=P, labels=L, box=box)
        propagate(state, init, False)
        if init > 0:
            propagate(state, init, True)
        # sửa từng đoạn lỗi bằng memory mới: nối vào memory cũ sẽ mang theo các frame đã nhảy sang chân kia
        done = set()
        for ps, min_len in enumerate(C.PASS_MIN_RUN):
            keep_own_foot(c, idx, Lg, Lf, off, trimmed)
            met = frame_metrics(c, idx, Lg > 0, Lf, off)
            todo = [r for r in find_bad_runs(c, idx, met, min_len) if r not in done]
            for o, a, b in todo:
                done.add((o, a, b))
                p = pick_prompt_frame(c, idx, a, b)
                if p is None:
                    log.append(dict(video_id=c.vid, continuity_id=int(c.cont[idx[0]]), obj=o, frame=-1, bad_from=int(c.frame[idx[a]]), bad_to=int(c.frame[idx[b]]), action="bo_qua_dang_chong", round=ps))
                    continue
                predictor.reset_state(state)
                box, P, L = prompt_for(c, idx[p], o, Lf[o], off)
                predictor.add_new_points_or_box(state, frame_idx=p, obj_id=o, points=P, labels=L, box=box)
                if b > p or p == a:
                    propagate(state, p, False, limit=b - p)
                if p > a:
                    propagate(state, p, True, limit=p - a)
                log.append(dict(video_id=c.vid, continuity_id=int(c.cont[idx[0]]), obj=o, frame=int(c.frame[idx[p]]), bad_from=int(c.frame[idx[a]]), bad_to=int(c.frame[idx[b]]), action="prompt_lai", round=ps))
        predictor.reset_state(state)
    keep_own_foot(c, idx, Lg, Lf, off, trimmed)
    both = (Lg[:, 0] > 0) & (Lg[:, 1] > 0)
    for fi in np.flatnonzero(both.any((1, 2))):
        w = both[fi]
        ov[fi] = float((Lg[fi, 0][w].astype(float) - Lg[fi, 1][w].astype(float)).mean())
    return Lg > 0, ov, log, init, Lf, trimmed


def cleanup(c):
    shutil.rmtree(C.DIRS["_tmp"] / c.vid, ignore_errors=True)
