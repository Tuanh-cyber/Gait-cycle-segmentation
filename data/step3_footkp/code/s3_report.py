"""3.4–3.5 chỉ số QC, gate, ảnh review, ảnh swap, biểu đồ cổ chân."""
import json

import cv2
import numpy as np
import pandas as pd

import s3_common as C

NEAR_BGR, FAR_BGR, X_BGR = (60, 122, 22), (36, 138, 240), (40, 40, 230)     # xanh lá / cam / đỏ
NEAR_HEX, FAR_HEX = "#167A3C", "#F08A24"     # đã qua validator màu: CVD ΔE 14
LEG_LINES = [(0, 1), (1, 2), (2, 5), (2, 3), (3, 4), (5, 3)]                 # hip-knee-ankle-heel, ankle-bigtoe, bigtoe-smalltoe, heel-bigtoe


# ---------------------------------------------------------------- chỉ số
def e_flip(c, kp, pts_a, pts_b):
    """Tỉ lệ frame mà gán chéo L<->R giảm sai lệch so với frame trước (trên toạ độ trừ tâm hông)."""
    hip = (kp[:, 11] + kp[:, 12]) / 2
    a, b = kp[:, pts_a] - hip[:, None], kp[:, pts_b] - hip[:, None]
    n = k = 0
    for seg in C.segments(c):
        for p, i in zip(seg[:-1], seg[1:]):
            if c.unique_id[i] - c.unique_id[p] != 1:
                continue
            ok = np.isfinite(a[i, :, 0]) & np.isfinite(a[p, :, 0]) & np.isfinite(b[i, :, 0]) & np.isfinite(b[p, :, 0])
            if not ok.any():
                continue
            st = np.linalg.norm(a[i][ok] - a[p][ok], axis=1).sum() + np.linalg.norm(b[i][ok] - b[p][ok], axis=1).sum()
            cr = np.linalg.norm(a[i][ok] - b[p][ok], axis=1).sum() + np.linalg.norm(b[i][ok] - a[p][ok], axis=1).sum()
            n += 1
            k += cr < st
    return k / n if n else np.nan


def residual_swap(c, kp, dref, L, margin=0.10):
    """Frame còn nghi hoán đổi SAU sửa. Hai bằng chứng độc lập:
    (a) dấu chênh x cổ chân trái tham chiếu pha (chỉ phần tuần hoàn rõ, |d_ref| > 0.3);
    (b) gán chéo giảm sai lệch so với frame trước quá margin × L_leg mỗi điểm: bước nhảy còn sót."""
    s = int(c.meta.s_final)
    d = s * (kp[:, 15, 0] - kp[:, 16, 0]) / L
    A = np.isfinite(dref) & (np.abs(dref) > 0.3) & np.isfinite(d) & (np.sign(d) != np.sign(dref))
    hip = (kp[:, 11] + kp[:, 12]) / 2
    a, b = kp[:, C.L_FOOT] - hip[:, None], kp[:, C.R_FOOT] - hip[:, None]
    B = np.zeros(c.n, bool)
    ev = np.isfinite(d)
    for seg in C.segments(c):
        for p, i in zip(seg[:-1], seg[1:]):
            ok = np.isfinite(a[i, :, 0]) & np.isfinite(a[p, :, 0]) & np.isfinite(b[i, :, 0]) & np.isfinite(b[p, :, 0])
            if ok.sum() < 2:
                continue
            st = np.linalg.norm(a[i][ok] - a[p][ok], axis=1).sum() + np.linalg.norm(b[i][ok] - b[p][ok], axis=1).sum()
            cr = np.linalg.norm(a[i][ok] - b[p][ok], axis=1).sum() + np.linalg.norm(b[i][ok] - a[p][ok], axis=1).sum()
            B[i] = (st - cr) / L[i] / ok.sum() > margin
    return (A | B), A, B, ev


def gate(q):
    """Ngưỡng ghi trong README. Trả về (PASS/WARN/FAIL, lý do)."""
    fail, warn = [], []
    if q["foot_valid_frac_near"] < C.GATE_NEAR_FAIL:
        fail.append(f"foot_valid_frac_near<{C.GATE_NEAR_FAIL}")
    if q["swap_rate_resid"] > C.GATE_SWAP_FAIL:
        fail.append(f"swap_rate_resid>{C.GATE_SWAP_FAIL}")
    if q["foot_valid_frac_far"] < C.GATE_FAR_WARN:
        warn.append(f"foot_valid_frac_far<{C.GATE_FAR_WARN}")
    if q["video_id"] in C.NEAR_UNRELIABLE_MANUAL:
        warn.append("near/far không tin cậy (review tay)")
    elif q["near_far_conf"] < C.GATE_NEAR_CONF:
        warn.append(f"near_far_conf<{C.GATE_NEAR_CONF}")
    if q["near_match_metadata"] == 0:
        warn.append("near_lệch_metadata")
    elif 0 < q["near_match_metadata"] < 1:
        warn.append("near_lệch_metadata_một_phần")
    if q["phase_periodic_frac"] < C.GATE_PHASE_MIN:
        warn.append("chu_kỳ_không_ổn_định")
    if q["step_time_ratio"] > C.GATE_RAMP:
        warn.append("nghi_tốc_độ_đổi_trong_part")
    if q["scale_perturb_instability"] > C.GATE_PERTURB:
        warn.append(f"scale_perturb>{C.GATE_PERTURB}")
    g = "FAIL" if fail else "WARN" if warn else "PASS"
    return g, "; ".join(fail + warn)


# ---------------------------------------------------------------- vẽ
def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def draw_leg(img, kp, slots, color, th):
    for a, b in LEG_LINES:
        pa, pb = kp[slots[a]], kp[slots[b]]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(img, _pt(pa), _pt(pb), color, th, cv2.LINE_AA)
    for j in slots[2:]:
        if np.isfinite(kp[j]).all():
            cv2.circle(img, _pt(kp[j]), th + 1, color, -1, cv2.LINE_AA)


def draw_x(img, p, th, r):
    x, y = _pt(p)
    cv2.line(img, (x - r, y - r), (x + r, y + r), X_BGR, th, cv2.LINE_AA)
    cv2.line(img, (x - r, y + r), (x + r, y - r), X_BGR, th, cv2.LINE_AA)


def crop_box(c, i, kp, pad=0.15):
    """Khung cắt: bbox người + mọi điểm bàn chân, để thấy cả điểm ngoài bbox."""
    pts = kp[i, C.FOOT_IDX]
    pts = pts[np.isfinite(pts).all(1)]
    b = c.bbox[i]
    if np.isfinite(b).all():
        pts = np.vstack([pts, b[:2], b[2:]])
    if not len(pts):
        return 0, 0, c.W, c.H
    x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    h = max(y1 - y0, x1 - x0, 40)
    return (int(max(0, x0 - pad * h)), int(max(0, y0 - pad * h)), int(min(c.W, x1 + pad * h)), int(min(c.H, y1 + pad * h)))


def cell(img, box, size):
    x0, y0, x1, y1 = box
    im = img[y0:y1, x0:x1]
    f = min(size / max(1, im.shape[0]), size / max(1, im.shape[1]))
    im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 30, np.uint8)
    yy, xx = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
    out[yy:yy + im.shape[0], xx:xx + im.shape[1]] = im
    return out


def caption(im, text, y=20, color=(255, 255, 255)):
    cv2.putText(im, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(im, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def read_frames(c, frames):
    want, out = set(int(f) for f in frames), {}
    if not want:
        return out
    cap = cv2.VideoCapture(str(C.video_path(c.meta)))
    f, last = 0, max(want)
    while f <= last and cap.grab():
        if f in want:
            ok, img = cap.retrieve()
            if ok:
                out[f] = img
        f += 1
    cap.release()
    return out


def slots_near(near):
    """near='A' -> (slot chân gần, slot chân xa)."""
    return (C.L_LEG, C.R_LEG) if near == "A" else (C.R_LEG, C.L_LEG)


def review_grid(c, res, path, size=400):
    ok = np.flatnonzero(np.isfinite(res["kp_clean"][:, 15:17, 0]).any(1))
    if not len(ok):
        ok = np.arange(c.n)
    pick = ok[np.unique(np.linspace(0, len(ok) - 1, 8).astype(int))]
    imgs = read_frames(c, c.frame[pick])
    cells = []
    for i in pick:
        im = imgs[int(c.frame[i])].copy()
        H = max(c.bbox[i, 3] - c.bbox[i, 1], 40) if np.isfinite(c.bbox[i]).all() else 200
        th = max(1, int(round(H / 220)))
        n_s, f_s = slots_near(res["near_track"][i])
        draw_leg(im, res["kp_clean"][i], f_s, FAR_BGR, th)
        draw_leg(im, res["kp_clean"][i], n_s, NEAR_BGR, th)
        for j in np.flatnonzero(res["reason"][i]):
            if np.isfinite(res["kp_fixed"][i, j]).all():
                draw_x(im, res["kp_fixed"][i, j], max(1, th), max(4, int(H / 45)))
        v = cell(im, crop_box(c, i, res["kp_fixed"]), size)
        caption(v, f"f{c.frame[i]} blk{c.cont[i]} conf{res['near_conf'][i]:.2f}" + (" SWAP" if res["swap_flag"][i] else ""))
        cells.append(v)
    while len(cells) < 8:
        cells.append(np.full((size, size, 3), 30, np.uint8))
    head = np.full((28, size * 4, 3), 245, np.uint8)
    warn = "" if res["near_track_reliable"] else "   NEAR/FAR KHONG TIN CAY"
    cv2.putText(head, f"{c.vid}   xanh la = chan gan   cam = chan xa   X do = outlier{warn}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([head, np.hstack(cells[:4]), np.hstack(cells[4:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def swap_events(c, flag):
    return np.flatnonzero((np.diff(flag.astype(int)) != 0) & (np.diff(c.cont) == 0)) + 1


def review_swaps(c, res, size=400, max_n=5):
    """Mỗi lần sửa: frame bị hoán đổi trước và sau sửa, cạnh frame liền kề không bị hoán đổi (nhãn ở đó không đổi)."""
    ev = swap_events(c, res["swap_flag"])[:max_n]
    for old in C.DIRS["review"].glob(f"_swap_{c.vid}_*.jpg"):
        old.unlink()
    if not len(ev):
        return 0
    imgs = read_frames(c, np.r_[c.frame[ev], c.frame[ev - 1]])
    for k, i in enumerate(ev):
        sw, other = (i, i - 1) if res["swap_flag"][i] else (i - 1, i)
        box = crop_box(c, sw, res["kp_fixed"])
        n_s, f_s = slots_near(res["near_track"][i])
        H = max(c.bbox[i, 3] - c.bbox[i, 1], 40) if np.isfinite(c.bbox[i]).all() else 200
        th = max(1, int(round(H / 220)))
        seq = [(f"f{c.frame[sw]} TRUOC sua", sw, c.kp), (f"f{c.frame[sw]} SAU sua", sw, res["kp_fixed"])]
        ctx = (f"f{c.frame[other]} lien ke, khong doi", other, res["kp_fixed"])
        seq = [ctx] + seq if other < sw else seq + [ctx]
        panels = []
        for name, fi, kp in seq:
            im = imgs[int(c.frame[fi])].copy()
            draw_leg(im, kp[fi], f_s, FAR_BGR, th)
            draw_leg(im, kp[fi], n_s, NEAR_BGR, th)
            v = cell(im, box, size)
            caption(v, name)
            panels.append(v)
        cv2.imwrite(str(C.DIRS["review"] / f"_swap_{c.vid}_{k}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return len(ev)


def ankle_plot(c, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = int(c.meta.s_final)
    hip = (c.kp[:, 11] + c.kp[:, 12]) / 2
    L = res["L"]
    x_raw = lambda kp, j: s * (kp[:, j, 0] - hip[:, 0]) / L
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, sharey=True, constrained_layout=True)
    ink, grid = "#3A3A38", "#E4E2DC"

    def series(a, tx, y, color, label, lw=1.6):
        first = True
        for seg in C.segments(c):
            a.plot(tx[seg], y[seg], color=color, lw=lw, solid_capstyle="round", label=label if first else None)
            first = False

    # nhãn gốc: slot của chân gần theo block đầu vẫn tô cùng màu để thấy đoạn bị đảo
    is_near_L = res["near_track"] == "A"
    near_raw = np.where(is_near_L, x_raw(c.kp, 15), x_raw(c.kp, 16))
    far_raw = np.where(is_near_L, x_raw(c.kp, 16), x_raw(c.kp, 15))
    series(ax[0], c.t, near_raw, NEAR_HEX, "slot của chân gần")
    series(ax[0], c.t, far_raw, FAR_HEX, "slot của chân xa")
    near = np.where(is_near_L, x_raw(res["kp_clean"], 15), x_raw(res["kp_clean"], 16))
    far = np.where(is_near_L, x_raw(res["kp_clean"], 16), x_raw(res["kp_clean"], 15))
    series(ax[1], c.t, near, NEAR_HEX, "chân gần")
    series(ax[1], c.t, far, FAR_HEX, "chân xa")
    for a in ax:
        a.axhline(0, color=ink, lw=.6)
        a.grid(axis="y", color=grid, lw=.6)
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(grid)
        a.tick_params(colors=ink, labelsize=8)
        for i in np.flatnonzero(res["swap_flag"]):
            a.axvspan(c.t[i] - 1e-3, c.t[i] + 1e-3, color="#B8B5AC", alpha=.35, lw=2)
        a.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, fontsize=8, ncol=2, borderaxespad=0)
    note = "" if res["near_track_reliable"] else " (near/far KHÔNG tin cậy)"
    ax[0].set_title(f"{c.vid}{note}: trước sửa (nhãn gốc). Vạch xám = frame bị hoán đổi", fontsize=10, color=ink, loc="left")
    ax[1].set_title("Sau sửa + lọc", fontsize=10, color=ink, loc="left")
    ax[1].set_xlabel("t_real_s_v2", fontsize=9, color=ink)
    fig.supylabel("x cổ chân theo chiều chạy, trừ tâm hông (× L_leg)", fontsize=9, color=ink)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_review_csv(meta):
    """Không bao giờ ghi đè kết luận của người kiểm tra."""
    p = C.OUT / "footkp_review.csv"
    if not p.exists():
        pd.DataFrame(dict(video_id=meta.video_id, verdict="", notes="")).to_csv(p, index=False)


def write_diff(meta, df):
    """footkp_review_v2_diff.csv: kết luận v1 giữ nguyên, thêm gate v1, gate v2, cờ thay đổi."""
    rv = pd.read_csv(C.OUT / "footkp_review.csv", dtype=str, keep_default_na=False)
    v1 = pd.read_csv(C.OUT / "qc_footkp_v1.csv", dtype=str, keep_default_na=False).set_index("video_id")
    v2 = df.set_index("video_id")
    rv["gate_v1"] = v1.gate.reindex(rv.video_id).values
    rv["gate_v2"] = v2.gate.reindex(rv.video_id).values
    rv["changed_vs_v1"] = rv.gate_v1 != rv.gate_v2
    rv["gate_reasons_v2"] = v2.gate_reasons.reindex(rv.video_id).fillna("").values
    rv.to_csv(C.OUT / "footkp_review_v2_diff.csv", index=False)
