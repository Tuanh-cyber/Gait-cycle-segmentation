"""2.5 QC theo video + 2.6 ảnh kiểm tra bằng mắt."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from s2_common import (CENTER_JUMP, COVERAGE_MIN, DIRS, FOOT_IDX, FOOT_SCORE_MIN, FOOT_VIS_MIN, FOOT_VIS_THR, HEAD_IDX,
                       KP_OUT_MAX, KP_OUT_SCORE, KP_OUT_TOL, LR_PAIRS, MARGIN_LOW, MARGIN_TIE, OUT, SKELETON, atomic_csv, atomic_json, update_timing)


def center_jumps(run_use):
    """Bước nhảy tâm giữa 2 frame unique liên tiếp trong cùng khối, cả hai đều có bbox."""
    r = run_use
    cx, cy, h = (r.x0.values + r.x1.values) / 2, (r.y0.values + r.y1.values) / 2, r.y1.values - r.y0.values
    out = []
    for i in range(1, len(r)):
        if r.continuity_id.iat[i] != r.continuity_id.iat[i - 1] or not np.isfinite(cx[i] + cx[i - 1]):
            continue
        d = np.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1])
        if d > CENTER_JUMP * h[i - 1]:
            out.append((int(r.frame.iat[i - 1]), int(r.frame.iat[i]), round(float(d / h[i - 1]), 3)))
    return out


def max_gap_s(run_use):
    """Khoảng pts_s lớn nhất giữa 2 frame có bbox trong cùng khối, kể cả đoạn thiếu ở đầu/cuối khối."""
    best = 0.0
    for _, g in run_use.groupby("continuity_id"):
        pts = g.pts_s.values
        have = pts[np.isfinite(g.x0.values)]
        if not len(have):
            best = max(best, float(pts[-1] - pts[0]))
            continue
        edges = np.r_[pts[0], have, pts[-1]]
        best = max(best, float(np.diff(edges).max()))
    return best


def lr_swap_rate(z):
    """Tỉ lệ cặp frame liền kề mà đổi L<->R ở chi dưới (gối, cổ chân, bàn chân) giảm tổng khoảng cách. Chỉ báo."""
    k, s, c = z["kp"], z["score"], z["continuity_id"]
    n, swaps = 0, 0
    for i in range(1, len(k)):
        if c[i] != c[i - 1] or not np.isfinite(k[i, 0, 0]) or not np.isfinite(k[i - 1, 0, 0]):
            continue
        keep, swap = 0.0, 0.0
        used = 0
        for l, r in LR_PAIRS:
            if min(s[i, l], s[i, r], s[i - 1, l], s[i - 1, r]) <= FOOT_VIS_THR:
                continue
            used += 1
            keep += np.linalg.norm(k[i, l] - k[i - 1, l]) + np.linalg.norm(k[i, r] - k[i - 1, r])
            swap += np.linalg.norm(k[i, l] - k[i - 1, r]) + np.linalg.norm(k[i, r] - k[i - 1, l])
        if used:
            n += 1
            swaps += swap < keep
    return swaps / n if n else np.nan


def frac_kp_outside_bbox(z):
    """Tỉ lệ frame có đầu/bàn chân tin cậy nằm ngoài bbox: dấu hiệu bbox bị cắt cụt."""
    idx = HEAD_IDX + FOOT_IDX
    n, bad = 0, 0
    for k, s, b in zip(z["kp"], z["score"], z["bbox"]):
        if not np.isfinite(k[0, 0]):
            continue
        n += 1
        p = k[idx][s[idx] > KP_OUT_SCORE]
        tol = KP_OUT_TOL * (b[3] - b[1])
        bad += bool(((p[:, 0] < b[0] - tol) | (p[:, 0] > b[2] + tol) | (p[:, 1] < b[1] - tol) | (p[:, 1] > b[3] + tol)).any())
    return bad / n if n else np.nan


def qc_video(v):
    run = pd.read_csv(DIRS["runner"] / f"{v.vid}_runner.csv")
    tracks = pd.read_csv(DIRS["all_tracks"] / f"{v.vid}_tracks.csv")
    joins = pd.read_csv(DIRS["selection"] / f"{v.vid}_joins.csv")
    blocks = pd.read_csv(DIRS["selection"] / f"{v.vid}_blocks.csv")
    z = np.load(DIRS["keypoints_v2"] / f"{v.vid}_kp.npz")
    ru = run[run.frame.isin(z["frame"])].sort_values("frame").reset_index(drop=True)
    has = np.isfinite(ru.x0.values)
    jumps = center_jumps(ru)
    h = (ru.y1 - ru.y0).values[has]
    posed = np.isfinite(z["kp"][:, 0, 0])
    S = z["score"][posed]
    foot = S[:, FOOT_IDX] if len(S) else np.zeros((0, 8))
    margin = float(blocks.margin.min()) if blocks.margin.notna().any() else np.nan
    q = dict(
        video_id=v.vid,
        n_valid_frames=len(ru),
        coverage=round(float(has.mean()), 4) if len(ru) else 0.0,
        n_tracks_total=int(tracks.track_id.nunique()),
        n_tracks_runner_joined=int(blocks.runner_ids.map(lambda s: len(json.loads(s))).sum()),
        n_id_joins=int(joins.chosen.astype(bool).sum()) if len(joins) else 0,
        max_gap_s=round(max_gap_s(ru), 4),
        n_center_jumps=len(jumps),
        bbox_h_cv=round(float(h.std() / h.mean()), 4) if len(h) else np.nan,
        kp_score_median=round(float(np.median(S)), 4) if len(S) else np.nan,
        foot_kp_score_median=round(float(np.median(foot)), 4) if len(S) else np.nan,
        frac_frames_foot_visible=round(float(((foot > FOOT_VIS_THR).sum(1) >= FOOT_VIS_MIN).mean()), 4) if len(S) else np.nan,
        lr_swap_rate=round(float(lr_swap_rate(z)), 4),
        selection_margin=round(margin, 4) if np.isfinite(margin) else np.nan,
        frac_kp_outside_bbox=round(float(frac_kp_outside_bbox(z)), 4),
    )
    flags = []
    if q["coverage"] < COVERAGE_MIN: flags.append("low_coverage")
    if q["n_center_jumps"] > 0: flags.append("center_jump")
    if not np.isfinite(margin) or margin < MARGIN_LOW: flags.append("low_margin")
    if np.isfinite(margin) and margin < MARGIN_TIE: flags.append("near_tie")
    if not np.isfinite(q["foot_kp_score_median"]) or q["foot_kp_score_median"] < FOOT_SCORE_MIN: flags.append("low_foot_score")
    if q["frac_kp_outside_bbox"] > KP_OUT_MAX: flags.append("box_truncated")
    if blocks.winner_score.isna().any(): flags.append("block_without_runner")
    if int(z["n_read"]) != int(z["n_meta"]): flags.append("frame_count_mismatch")
    q["flags"] = ";".join(flags)
    atomic_json(DIRS["qc"] / f"{v.vid}.json", dict(q, center_jumps=jumps))
    return q, jumps


# ---------------------------------------------------------------- vẽ
def draw_frame(img, f, tracks_at, run_row, kp, sc, label, cell_h=400):
    s = cell_h / img.shape[0]
    im = cv2.resize(img, (max(1, int(img.shape[1] * s)), cell_h), interpolation=cv2.INTER_AREA)
    for t in tracks_at.get(f, []):
        p = [int(c * s) for c in t[1:5]]
        cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(im, str(t[0]), (p[0] + 2, max(10, p[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
    if run_row is not None and np.isfinite(run_row.x0):
        p = [int(c * s) for c in (run_row.x0, run_row.y0, run_row.x1, run_row.y1)]
        color = (0, 220, 0) if run_row.track_status == "tracked" else (0, 220, 220)
        cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), color, 3, cv2.LINE_AA)
    if kp is not None and np.isfinite(kp[0, 0]):
        pt = (kp * s).astype(int)
        for a, b in SKELETON:
            if sc[a] > 0.3 and sc[b] > 0.3:
                cv2.line(im, tuple(pt[a]), tuple(pt[b]), (255, 170, 60), 1, cv2.LINE_AA)
        for j in range(26):
            if sc[j] <= 0.3:
                continue
            if j in (24, 25):            # gót: đỏ tươi
                cv2.circle(im, tuple(pt[j]), 4, (40, 40, 255), -1, cv2.LINE_AA)
            elif j in (20, 21, 22, 23):  # mũi chân: vàng
                cv2.circle(im, tuple(pt[j]), 4, (0, 240, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(im, tuple(pt[j]), 2, (255, 200, 120), -1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(im, (0, 0), (tw + 8, th + 10), (0, 0, 0), -1)
    cv2.putText(im, label, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def read_frames(path, wanted):
    wanted, out = set(wanted), {}
    cap = cv2.VideoCapture(str(path))
    i, last = 0, max(wanted) if wanted else -1
    while i <= last and cap.grab():
        if i in wanted:
            out[i] = cap.retrieve()[1]
        i += 1
    cap.release()
    return out


def review_video(v, jumps):
    run = pd.read_csv(DIRS["runner"] / f"{v.vid}_runner.csv").set_index("frame")
    tracks = pd.read_csv(DIRS["all_tracks"] / f"{v.vid}_tracks.csv")
    joins = pd.read_csv(DIRS["selection"] / f"{v.vid}_joins.csv")
    z = np.load(DIRS["keypoints_v2"] / f"{v.vid}_kp.npz")
    frames = z["frame"]
    pos = {int(f): i for i, f in enumerate(frames)}
    tracks_at = {}
    for t in tracks.itertuples():
        tracks_at.setdefault(int(t.frame), []).append((t.track_id, t.x0, t.y0, t.x1, t.y1))
    grid_f = [int(frames[i]) for i in np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)]
    events = [("jump", a, b, f"d={d}") for a, b, d in jumps]
    if len(joins):
        for r in joins[joins.chosen.astype(bool)].itertuples():
            events.append(("join", int(r.frame_before), int(r.frame_after), f"{r.from_track}->{r.to_track}"))
    need = set(grid_f) | {f for e in events for f in e[1:3]}
    imgs = read_frames(v.path, need)

    def cell(f, tag=""):
        rr = run.loc[f]
        p = pos.get(f)
        kp, sc = (z["kp"][p], z["score"][p]) if p is not None else (None, None)
        return draw_frame(imgs[f], f, tracks_at, rr, kp, sc, f"{v.vid} f{f} {rr.track_status} {tag}".rstrip())

    cells = [cell(f) for f in grid_f]
    while len(cells) < 8:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[:4]), np.hstack(cells[4:8])]
    cv2.imwrite(str(DIRS["review"] / f"{v.vid}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for old in DIRS["review"].glob(f"_flagged_{v.vid}_*.jpg"):
        old.unlink()
    for k, (kind, a, b, note) in enumerate(events):
        cv2.imwrite(str(DIRS["review"] / f"_flagged_{v.vid}_{k}.jpg"),
                    np.hstack([cell(a, f"{kind} before"), cell(b, f"{kind} after {note}")]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def report_video(v):
    t0 = time.time()
    _, jumps = qc_video(v)
    review_video(v, jumps)
    update_timing(v.vid, report_s=round(time.time() - t0, 2))


def done(vid):
    return (DIRS["qc"] / f"{vid}.json").exists() and (DIRS["review"] / f"{vid}.jpg").exists()


QC_COLS = ["video_id", "n_valid_frames", "coverage", "n_tracks_total", "n_tracks_runner_joined", "n_id_joins", "max_gap_s",
           "n_center_jumps", "bbox_h_cv", "kp_score_median", "foot_kp_score_median", "frac_frames_foot_visible",
           "lr_swap_rate", "selection_margin", "frac_kp_outside_bbox", "flags"]


def write_tables(meta):
    """Gộp qc/*.json; tracking_review.csv không bao giờ ghi đè nội dung người kiểm tra đã điền."""
    rows = []
    for vid in meta.video_id:
        p = DIRS["qc"] / f"{vid}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    qc = pd.DataFrame(rows).reindex(columns=QC_COLS)
    atomic_csv(qc, OUT / "qc_tracking.csv")
    p = OUT / "tracking_review.csv"
    cols = ["video_id", "verdict", "wrong_person_frames", "notes"]
    old = pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame(columns=cols)
    add = [v for v in meta.video_id if v not in set(old.video_id)]
    new = pd.concat([old, pd.DataFrame(dict(video_id=add, verdict="", wrong_person_frames="", notes=""))], ignore_index=True)
    order = {v: i for i, v in enumerate(meta.video_id)}
    new = new.sort_values("video_id", key=lambda s: s.map(order))
    atomic_csv(new[cols], p)
    return qc


SEVERITY = {"block_without_runner": 4, "near_tie": 3, "box_truncated": 3, "center_jump": 3, "low_margin": 2, "low_coverage": 2,
            "frame_count_mismatch": 2, "low_foot_score": 1}


def print_summary(qc):
    fl = qc[qc["flags"].fillna("") != ""].copy()
    print(f"\n=== TÓM TẮT: {len(qc)} video có QC, {len(fl)} video có cờ ===")
    for name in SEVERITY:
        hit = fl[fl["flags"].str.contains(name)].video_id.tolist()
        if hit:
            print(f"  {name:22s} {len(hit):3d}: {' '.join(hit)}")
    fl["sev"] = fl["flags"].map(lambda s: sum(SEVERITY[x] for x in s.split(";")))
    fl = fl.sort_values(["sev", "coverage"], ascending=[False, True])
    print("Danh sách xem trước (nặng → nhẹ):")
    for r in fl.itertuples():
        print(f"  {r.video_id:7s} cov={r.coverage:.2f} jumps={r.n_center_jumps} margin={r.selection_margin} foot={r.foot_kp_score_median} [{getattr(r, 'flags')}]")
