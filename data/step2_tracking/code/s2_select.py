"""2.2 chấm điểm + chọn người chạy, 2.3 nối tracklet và nội suy. Không cần giải mã video.

Chọn trên CHUỖI tracklet (sau khi nối) thay vì track đơn: nếu không, người chạy bị đứt ID sẽ thua
một khán giả đứng yên có track dài, và các mảnh của chính người chạy sẽ làm selection_margin thấp giả.
Điểm từng track đơn vẫn được ghi để giải thích lựa chọn."""
import time

import numpy as np
import pandas as pd

from s2_common import (AR_MAX, AR_MIN, CAM_OUTLIER, DIRS, INTERP_MAX_FRAMES, JOIN_MAX_APPEAR, JOIN_MAX_CENTER,
                       JOIN_MAX_CENTER_OVERLAP, JOIN_MAX_DH, JOIN_MAX_GAP_S, JOIN_MAX_OVERLAP_S, JOIN_TOUCH_FRAMES, JOIN_WINDOW, MOTION_FULL_S, SIZE_REF, V_REF, W_LEN, W_MOTION, W_PROX, W_SIZE,
                       atomic_csv, update_timing)


def bhattacharyya(p, q):
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return float(np.sqrt(max(0.0, 1.0 - np.sqrt(p * q).sum())))


class Tracklet:
    def __init__(self, tid, df, hist):
        order = np.argsort(df.frame.values)
        self.ids = [tid]
        self.tid = tid
        d = df.iloc[order]
        self.frame = d.frame.values.astype(int)
        self.pts = d.pts_s.values.astype(float)
        self.box = d[["x0", "y0", "x1", "y1"]].values.astype(float)
        self.conf = d.conf.values.astype(float)
        self.hist = hist[order]
        self.tids = np.full(len(d), tid)      # track gốc của từng frame

    @property
    def n(self): return len(self.frame)
    @property
    def cx(self): return (self.box[:, 0] + self.box[:, 2]) / 2
    @property
    def cy(self): return (self.box[:, 1] + self.box[:, 3]) / 2
    @property
    def h(self): return self.box[:, 3] - self.box[:, 1]
    @property
    def w(self): return self.box[:, 2] - self.box[:, 0]

    def merged(self, other):
        """Ghép theo thời gian; đoạn chồng nhau giữ track đến trước."""
        a, b = (self, other) if self.frame[0] < other.frame[0] else (other, self)
        keep = b.frame > a.frame[-1]
        t = Tracklet.__new__(Tracklet)
        t.tid, t.ids = self.tid, a.ids + b.ids
        for k in ("frame", "pts", "box", "conf", "hist", "tids"):
            setattr(t, k, np.concatenate([getattr(a, k), getattr(b, k)[keep]]))
        return t

    def absorbed(self, other, max_err=JOIN_MAX_CENTER_OVERLAP):
        """Thêm frame của track trùng người. Trong khoảng của chuỗi phải khớp vị trí nội suy; ngoài khoảng thì nhận vì
        cùng một track. Trả về (chuỗi, mảng frame được thêm)."""
        new = ~np.isin(other.frame, self.frame)
        h = np.interp(other.frame, self.frame, self.h)
        err = np.hypot(other.cx - np.interp(other.frame, self.frame, self.cx),
                       other.cy - np.interp(other.frame, self.frame, self.cy)) / np.maximum(h, 1.0)
        outside = (other.frame < self.frame[0]) | (other.frame > self.frame[-1])
        add = new & (outside | (err < max_err))
        order = np.argsort(np.r_[self.frame, other.frame[add]])
        t = Tracklet.__new__(Tracklet)
        t.tid, t.ids = self.tid, self.ids + [i for i in other.ids if i not in self.ids]
        for k in ("frame", "pts", "box", "conf", "hist", "tids"):
            setattr(t, k, np.concatenate([getattr(self, k), getattr(other, k)[add]])[order])
        return t, other.frame[add]


def score_track(t, block, H, s_final):
    """block: DataFrame frame hợp lệ của khối, index = frame, có rcx, rcy (toạ độ gốc), cum_cam, t."""
    b = block.loc[t.frame]
    h_med = float(np.median(t.h))
    # gần vùng chuyển động
    d = np.hypot(t.cx - b.rcx.values, t.cy - b.rcy.values) / np.maximum(t.h, 1.0)
    d = d[np.isfinite(d)]
    prox_d = float(np.median(d)) if len(d) else np.nan
    s_prox = float(np.exp(-prox_d)) if len(d) else 0.5     # không có dữ liệu vùng chuyển động -> trung tính
    s_len = t.n / max(len(block), 1)
    # chuyển động so với nền: cx - cumsum(cam_dx). cam_dx là dịch chuyển NỘI DUNG ảnh giữa 2 frame unique,
    # nên camera bám theo người chạy cũng cho vận tốc nền cùng dấu s_final.
    xw = t.cx - b.cum_cam.values
    tt = b.t.values
    v, dir_frac = 0.0, 0.0
    if t.n >= 3 and tt[-1] > tt[0]:
        v = float(s_final * np.polyfit(tt, xw, 1)[0] / max(h_med, 1.0))
        stride = max(1, t.n // 10)            # bước ~1/10 track để giảm nhiễu dao động bbox theo nhịp chân
        dx = s_final * np.diff(xw[::stride])
        if np.abs(dx).sum() > 0:
            dir_frac = float(np.clip(dx, 0, None).sum() / np.abs(dx).sum())
    block_dur = float(block.t.max() - block.t.min())
    reliability = float(np.clip((tt[-1] - tt[0]) / max(min(MOTION_FULL_S, block_dur), 1e-6), 0, 1))
    s_motion = float(np.clip(v / V_REF, 0, 1) * dir_frac * reliability)
    # kích thước + tỉ lệ khung
    ar = float(np.median(t.h / np.maximum(t.w, 1.0)))
    s_ar = 1.0 if AR_MIN <= ar <= AR_MAX else float(np.exp(-abs(ar - np.clip(ar, AR_MIN, AR_MAX))))
    s_size = float(min(1.0, h_med / (SIZE_REF * H)) * s_ar)
    score = W_PROX * s_prox + W_LEN * s_len + W_MOTION * s_motion + W_SIZE * s_size
    return dict(n_frames=t.n, prox_dist=prox_d, s_proximity=s_prox, s_length=s_len, v_world_bh_per_s=v,
                dir_frac=dir_frac, motion_reliability=reliability, s_motion=s_motion, h_median=h_med, aspect=ar, s_size=s_size, score=score)


def same_person(a, b, min_iou=0.5, min_interleaved=3):
    """YOLO/tracker cho 2 track trên 1 người. Hai dạng: chồng nhau với IoU cao, hoặc xen kẽ gần như không có frame chung
    (C38_5: 36 + 20 frame trên 54 frame khối) — dạng này so với vị trí nội suy, kèm chiều cao và appearance."""
    _, ia, ib = np.intersect1d(a.frame, b.frame, return_indices=True)
    if len(ia) and len(ia) >= 0.5 * min(a.n, b.n):
        x, y = a.box[ia], b.box[ib]
        w = np.clip(np.minimum(x[:, 2], y[:, 2]) - np.maximum(x[:, 0], y[:, 0]), 0, None)
        h = np.clip(np.minimum(x[:, 3], y[:, 3]) - np.maximum(x[:, 1], y[:, 1]), 0, None)
        inter = w * h
        union = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1]) + (y[:, 2] - y[:, 0]) * (y[:, 3] - y[:, 1]) - inter
        return float(np.mean(inter / np.maximum(union, 1.0))) >= min_iou
    lo, hi = max(a.frame[0], b.frame[0]), min(a.frame[-1], b.frame[-1])
    m = (b.frame >= lo) & (b.frame <= hi)
    if m.sum() < min_interleaved:
        return False
    h = np.interp(b.frame[m], a.frame, a.h)
    err = np.hypot(b.cx[m] - np.interp(b.frame[m], a.frame, a.cx), b.cy[m] - np.interp(b.frame[m], a.frame, a.cy))
    app = bhattacharyya(a.hist.mean(0), b.hist.mean(0))
    return (float(np.median(err / np.maximum(h, 1.0))) < JOIN_MAX_CENTER_OVERLAP
            and float(np.median(np.abs(b.h[m] - h) / np.maximum(h, 1.0))) < JOIN_MAX_DH
            and (not np.isfinite(app) or app < JOIN_MAX_APPEAR))


def _pred_err(a, b, forward):
    """Sai số tâm dự đoán bằng vận tốc tuyến tính, chiều cao lệch, appearance — giữa đầu mút a và b."""
    k = JOIN_WINDOW
    if forward:
        ta, ca, cya, ha, qa = a.pts[-k:], a.cx[-k:], a.cy[-k:], a.h[-k:], a.hist[-k:]
        tb, xb, yb, hb, qb = b.pts[0], b.cx[0], b.cy[0], b.h[:3], b.hist[:k]
        t_ref = ta[-1]
    else:
        ta, ca, cya, ha, qa = a.pts[:k], a.cx[:k], a.cy[:k], a.h[:3], a.hist[:k]
        tb, xb, yb, hb, qb = b.pts[-1], b.cx[-1], b.cy[-1], b.h[-3:], b.hist[-k:]
        t_ref = ta[0]
    x_ref, y_ref = (ca[-1], cya[-1]) if forward else (ca[0], cya[0])
    vx = np.polyfit(ta, ca, 1)[0] if len(ta) >= 2 and ta[-1] > ta[0] else 0.0
    vy = np.polyfit(ta, cya, 1)[0] if len(ta) >= 2 and ta[-1] > ta[0] else 0.0
    dt = tb - t_ref
    h_ref = float(np.median(ha))
    err = float(np.hypot(x_ref + vx * dt - xb, y_ref + vy * dt - yb) / max(h_ref, 1.0))
    dh = float(abs(np.median(hb) - h_ref) / max(h_ref, 1.0))
    app = bhattacharyya(qa.mean(0), qb.mean(0))
    return err, dh, app, abs(float(dt))


def _overlap_err(a, b, forward):
    """Hai track chồng thời gian: so b với vị trí nội suy của a trên đoạn chung."""
    lo, hi = max(a.frame[0], b.frame[0]), min(a.frame[-1], b.frame[-1])
    m = (b.frame >= lo) & (b.frame <= hi)
    h = np.interp(b.frame[m], a.frame, a.h)
    d = np.hypot(b.cx[m] - np.interp(b.frame[m], a.frame, a.cx), b.cy[m] - np.interp(b.frame[m], a.frame, a.cy))
    err = float(np.median(d / np.maximum(h, 1.0)))
    dh = float(np.median(np.abs(b.h[m] - h) / np.maximum(h, 1.0)))
    k = JOIN_WINDOW
    qa, qb = (a.hist[-k:], b.hist[:k]) if forward else (a.hist[:k], b.hist[-k:])
    return err, dh, bhattacharyya(qa.mean(0), qb.mean(0))


def build_chain(seed, pool):
    """Hút track trùng người + nối hai chiều, lặp đến khi không đổi. Trả về chuỗi, id thành viên, log các lần xét."""
    chain, used, log = seed, {seed.tid}, []
    changed = True
    while changed:
        changed = False
        for t in pool:
            if t.tid in used or not same_person(chain, t):
                continue
            chain, added = chain.absorbed(t)
            used.add(t.tid)
            changed = True
            if len(added):
                prev = chain.frame[chain.frame < added[0]]
                log.append(dict(direction="absorb", from_track=chain.ids[0], to_track=t.tid,
                                frame_before=int(prev[-1]) if len(prev) else int(added[0]), frame_after=int(added[0]),
                                accepted=True, chosen=True, reject_reason=f"same_person_{len(added)}_frames"))
        for forward in (True, False):
            while True:
                best = None
                for t in pool:
                    if t.tid in used:
                        continue
                    if forward and (t.frame[-1] <= chain.frame[-1] or t.frame[0] <= chain.frame[0]):
                        continue
                    if not forward and (t.frame[0] >= chain.frame[0] or t.frame[-1] >= chain.frame[-1]):
                        continue
                    gap = t.pts[0] - chain.pts[-1] if forward else chain.pts[0] - t.pts[-1]
                    if gap > JOIN_MAX_GAP_S or gap < -JOIN_MAX_OVERLAP_S:
                        continue
                    n_overlap = int((t.frame <= chain.frame[-1]).sum() if forward else (t.frame >= chain.frame[0]).sum())
                    strict = n_overlap > JOIN_TOUCH_FRAMES
                    if strict:
                        err, dh, app = _overlap_err(chain, t, forward)
                    else:
                        err, dh, app, _ = _pred_err(chain, t, forward)
                    reason = []
                    if err >= (JOIN_MAX_CENTER_OVERLAP if strict else JOIN_MAX_CENTER): reason.append("center")
                    if dh >= JOIN_MAX_DH: reason.append("height")
                    if np.isfinite(app) and app >= JOIN_MAX_APPEAR: reason.append("appearance")
                    rec = dict(direction="forward" if forward else "backward", from_track=chain.ids[-1 if forward else 0],
                               to_track=t.tid, frame_before=int(chain.frame[-1] if forward else t.frame[-1]),
                               frame_after=int(t.frame[0] if forward else chain.frame[0]), gap_s=round(float(gap), 4),
                               center_err_bh=round(err, 3), dh_frac=round(dh, 3),
                               appearance_dist=round(app, 3) if np.isfinite(app) else np.nan,
                               accepted=not reason, reject_reason="+".join(reason))
                    log.append(rec)
                    if not reason and (best is None or err < best[0]):
                        best = (err, t, rec)
                if best is None:
                    break
                best[2]["chosen"] = True
                chain = chain.merged(best[1])
                used.add(best[1].tid)
                changed = True
    return chain, used, log


def select_video(v, W, H, tracks, hist):
    """Trả về (bảng điểm track, log nối, bảng runner 1 dòng/frame giải mã, tóm tắt theo khối)."""
    sx, sy = W / v.aw, H / v.ah       # video có tag xoay: Bước 0 giãn ảnh không đều, sx != sy (xem README)
    fr = v.fr
    score_rows, join_rows, block_rows = [], [], []
    runner = {}
    for cid, g in fr[fr.use].groupby("continuity_id", sort=True):
        block = pd.DataFrame(dict(rcx=g.run_cx.values * sx, rcy=g.run_cy.values * sy, t=g.t_real_s.values,
                                  pts=g.pts_s.values), index=g.frame.values)
        cam = g.cam_dx.where(g.cam_dx.abs() < CAM_OUTLIER, 0.0).fillna(0.0).values * sx
        cam[0] = 0.0                  # cam_dx frame đầu khối đo so với frame ngoài khối
        block["cum_cam"] = np.cumsum(cam)
        m = (tracks.continuity_id == cid).values
        sub, shist = tracks[m], hist[m]
        tl = [Tracklet(tid, sub[(sub.track_id == tid).values], shist[(sub.track_id == tid).values])
              for tid in sub.track_id.unique()]
        scores = {t.tid: score_track(t, block, H, v.s_final) for t in tl}
        order = sorted(tl, key=lambda t: -scores[t.tid]["score"])
        remaining, chains = list(order), []
        while remaining:
            seed = remaining[0]
            chain, used, log = build_chain(seed, remaining)
            chains.append((chain, used, log, score_track(chain, block, H, v.s_final), seed.tid))
            remaining = [t for t in remaining if t.tid not in used]
        chains.sort(key=lambda c: (-c[3]["score"], -c[0].n))
        win = chains[0] if chains else None
        margin = np.nan
        if win is not None:
            dup = [same_person(win[0], c[0]) for c in chains]
            rivals = [c[3]["score"] for c, d in zip(chains[1:], dup[1:]) if not d]
            margin = win[3]["score"] - (rivals[0] if rivals else 0.0)
            for cidx, (chain, used, log, cs, seed_id) in enumerate(chains):
                for tid in sorted(used):
                    score_rows.append(dict(video_id=v.vid, continuity_id=int(cid), track_id=int(tid), chain_rank=cidx,
                                           is_seed=tid == seed_id, selected=cidx == 0,
                                           same_person_as_runner=bool(cidx and dup[cidx]),
                                           **{k: round(x, 4) for k, x in scores[tid].items()},
                                           chain_n_frames=cs["n_frames"], chain_s_proximity=round(cs["s_proximity"], 4),
                                           chain_s_length=round(cs["s_length"], 4), chain_s_motion=round(cs["s_motion"], 4),
                                           chain_s_size=round(cs["s_size"], 4), chain_score=round(cs["score"], 4)))
            for rec in win[2]:
                join_rows.append(dict(video_id=v.vid, continuity_id=int(cid), **rec))
            ch = win[0]
            for f, b, c, tid in zip(ch.frame, ch.box, ch.conf, ch.tids):
                runner[int(f)] = dict(track_status="tracked", track_id=int(tid),
                                      x0=b[0], y0=b[1], x1=b[2], y1=b[3], conf=c)
        # nội suy khoảng trống <= INTERP_MAX_FRAMES frame unique (chỉ giữa 2 quan sát cùng khối)
        frames = g.frame.values.astype(int)
        pts = g.pts_s.values
        have = np.array([int(f) in runner for f in frames])
        idx = np.flatnonzero(have)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= INTERP_MAX_FRAMES + 1:
                ba, bb = runner[int(frames[a])], runner[int(frames[b])]
                for j in range(a + 1, b):
                    w = (pts[j] - pts[a]) / max(pts[b] - pts[a], 1e-9)
                    box = {k: ba[k] + w * (bb[k] - ba[k]) for k in ("x0", "y0", "x1", "y1")}
                    runner[int(frames[j])] = dict(track_status="interpolated", track_id=-1, conf=np.nan, **box)
        block_rows.append(dict(continuity_id=int(cid), n_frames=len(g), n_tracks=len(tl), n_chains=len(chains),
                               margin=margin, winner_score=win[3]["score"] if win else np.nan,
                               runner_ids=sorted({int(t) for t in ch.tids}) if win else [],
                               n_joins=len(win[1]) - 1 if win else 0))
    rows = []
    for r in fr.itertuples():
        src = int(r.inherited_from)
        base = dict(frame=int(r.frame), pts_s=r.pts_s, t_real_s=r.t_real_s, continuity_id=int(r.continuity_id),
                    shot_id=int(r.shot_id), is_dup=bool(r.is_dup), inherited_from=src)
        f = int(r.frame) if r.use else src
        if r.use or src >= 0:
            d = runner.get(f, dict(track_status="missing", track_id=-1, x0=np.nan, y0=np.nan, x1=np.nan, y1=np.nan, conf=np.nan))
        else:
            d = dict(track_status="excluded", track_id=-1, x0=np.nan, y0=np.nan, x1=np.nan, y1=np.nan, conf=np.nan)
        rows.append({**base, **d})
    out = pd.DataFrame(rows)
    out[["x0", "y0", "x1", "y1"]] = out[["x0", "y0", "x1", "y1"]].round(2)
    return pd.DataFrame(score_rows), pd.DataFrame(join_rows), out, block_rows


def run_video(v):
    t0 = time.time()
    tracks = pd.read_csv(DIRS["all_tracks"] / f"{v.vid}_tracks.csv")
    z = np.load(DIRS["appearance"] / f"{v.vid}_hsv.npz")
    W, H = int(z["W"]), int(z["H"])
    scores, joins, runner, blocks = select_video(v, W, H, tracks, z["hist"])
    join_cols = ["video_id", "continuity_id", "direction", "from_track", "to_track", "frame_before", "frame_after",
                 "gap_s", "center_err_bh", "dh_frac", "appearance_dist", "accepted", "chosen", "reject_reason"]
    joins = joins.reindex(columns=join_cols)
    joins["chosen"] = joins.chosen.astype("boolean").fillna(False).astype(bool)
    atomic_csv(scores, DIRS["selection"] / f"{v.vid}_scores.csv")
    atomic_csv(joins, DIRS["selection"] / f"{v.vid}_joins.csv")
    atomic_csv(pd.DataFrame(blocks), DIRS["selection"] / f"{v.vid}_blocks.csv")
    atomic_csv(runner, DIRS["runner"] / f"{v.vid}_runner.csv")
    update_timing(v.vid, select_s=round(time.time() - t0, 3))
    st = runner[runner.inherited_from < 0].track_status.value_counts().to_dict()
    print(f"  select {v.vid}: {st}, nối {int(joins.chosen.sum())}", flush=True)


def done(vid):
    return (DIRS["runner"] / f"{vid}_runner.csv").exists()
