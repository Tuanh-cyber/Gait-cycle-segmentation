"""Plot theo thời gian và ảnh 8 frame có residual flow."""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import s5_common as C

# màu đã kiểm định ở các bước trước; A/B dùng cặp riêng để không nhầm với near/far
COLOR = {"near": "#167A3C", "far": "#F08A24", "A": "#2F6DB5", "B": "#7A4FB5"}
BGR = {"near": (60, 122, 22), "far": (36, 138, 240), "A": (181, 109, 47), "B": (181, 79, 122)}
HEEL, TOE = 0, 2


def _gaps(t, y, block):
    """Ngắt đường qua ranh giới khối."""
    tt, yy = [], []
    for b in np.unique(block):
        w = block == b
        tt += [t[w], [np.nan]]
        yy += [y[w], [np.nan]]
    return np.concatenate(tt), np.concatenate(yy)


def plot_video(vid, d, q, path):
    t, block = d["t_real_s_v2"], d["continuity_id"] * 100000 + d["shot_id"]
    usable = d["frame_valid_mask"] & d["flow_valid"][:, None]
    lab = d["label"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    panels = ((f"Gót: slow_ratio (v < {C.SLOW_MAIN:g} L_leg/s)", d["slow_ratio"][:, :, HEEL]), (f"Mũi: slow_ratio (v < {C.SLOW_MAIN:g} L_leg/s)", d["slow_ratio"][:, :, TOE]), ("Tốc độ keypoint bàn chân (L_leg/s)", d["kp_speed"]))
    names = [n for n in ("near", "far", "A", "B") if (lab == n).any()]
    for a, (title, y) in zip(ax, panels):
        y = np.where(usable, y, np.nan)
        for n in names:
            s = np.where(lab == n, y, np.nan)
            s = np.nanmax(s, axis=1) if np.isfinite(s).any() else np.full(len(t), np.nan)
            tt, yy = _gaps(t, s, block)
            a.plot(tt, yy, color=COLOR[n], lw=2, marker="o", ms=3, label=n)
        a.set_title(title, loc="left", fontsize=10)
        a.grid(axis="y", color="#E5E5E5", lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
    ex = d["exclude_time_features"]
    if ex.any():
        for a in ax:
            a.axvspan(t[ex].min(), t[ex].max(), color="#DDDDDD", alpha=0.5, lw=0)
    ax[0].legend(loc="upper right", frameon=False, ncol=4)
    ax[0].set_ylim(-0.03, 1.03)
    ax[1].set_ylim(-0.03, 1.03)
    ax[2].set_xlabel("t_real_s_v2 (s)")
    cad = "" if not np.isfinite(q["cadence_dev"]) else f", lệch {q['cadence_dev']:.0%}"
    fig.suptitle(f"{vid}  gate {q['gate']}  Spearman {q['spearman_flow_kp']:.2f}  cadence flow {q['cadence_flow']:.0f} / tham chiếu {q['cadence_ref_used']:.0f} spm{cad}  P10 vx {q['vx_p10']:.1f}  {q['comp_method']}  Δt {q['dt_mode']}", x=0.01, ha="left", fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def flow_color(res, vmax):
    """Hướng = màu, độ lớn = độ sáng."""
    ang = (np.arctan2(res[..., 1], res[..., 0]) + np.pi) / (2 * np.pi) * 179
    mag = np.clip(np.linalg.norm(res, axis=2) / vmax, 0, 1)
    hsv = np.dstack([ang, np.full(ang.shape, 255.0), 60 + 195 * mag]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def render_cell(crop, res, R, fb, labels, valid, foot_v, frame, lf, size=400):
    im = crop.copy()
    vmax = max(np.percentile(np.linalg.norm(res[R[0][3] | R[1][3]], axis=1), 95) if (R[0][3] | R[1][3]).any() else 1.0, 2.0)
    col = flow_color(res, vmax)
    for o in (0, 1):
        m = R[o][3]
        if not m.any():
            continue
        show = m & fb
        im[show] = (im[show] * 0.3 + col[show] * 0.7).astype(np.uint8)
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnt, -1, BGR[str(labels[o])] if valid[o] else (150, 150, 150), 1, cv2.LINE_AA)
        step = max(6, int(lf / 3))
        ys, xs = np.nonzero(show)
        keep = (ys % step == 0) & (xs % step == 0)
        for x, y in zip(xs[keep], ys[keep]):
            d = res[y, x]
            cv2.arrowedLine(im, (int(x), int(y)), (int(x + d[0]), int(y + d[1])), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
    f = min(size / im.shape[0], size / im.shape[1])
    v = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 30, np.uint8)
    yy, xx = (size - v.shape[0]) // 2, (size - v.shape[1]) // 2
    out[yy:yy + v.shape[0], xx:xx + v.shape[1]] = v
    txt = f"f{frame} " + " ".join(f"{labels[o]}:{'x' if not valid[o] or not np.isfinite(foot_v[o]) else format(foot_v[o], '.1f')}" for o in (0, 1))
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (0, 0), (w + 12, h + 10), (30, 30, 30), -1)
    cv2.putText(out, txt, (6, h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_grid(vid, cells, path, size=400):
    cells = list(cells)
    while len(cells) < 8:
        cells.append(np.full((size, size, 3), 30, np.uint8))
    head = np.full((28, size * 4, 3), 245, np.uint8)
    cv2.putText(head, f"{vid}  residual flow tren mask: mau = huong, sang = do lon, mui ten = residual; so = median v ca ban chan (L_leg/s), x = khong hop le", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([head, np.hstack(cells[:4]), np.hstack(cells[4:8])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
