"""Two method phase bars with IC/TO ticks, preserving every source frame/PTS."""
from fractions import Fraction
import json
import subprocess

import av
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from event_data import PHASE_NAMES
from mask_overlay import REGION_COLORS, REGION_NAMES, prepare_regions, apply_regions

# OpenCV BGR: stance green, swing blue, unknown gray. These are phase colors, not QC.
COLORS = {-1: (90, 90, 90), 0: (220, 125, 45), 1: (70, 175, 60)}


def text(im, line, x, y, scale=.6, color=(240, 240, 240)):
    cv2.putText(im, str(line), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fit(im, width, height):
    canvas = np.full((height, width, 3), 22, np.uint8)
    ratio = min(width / im.shape[1], height / im.shape[0])
    resized = cv2.resize(im, (max(1, round(im.shape[1] * ratio)), max(1, round(im.shape[0] * ratio))))
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def probe(path):
    with av.open(str(path)) as source:
        pts = [float(frame.pts * frame.time_base) for frame in source.decode(video=0)]
    pts = np.asarray(pts)
    if not len(pts) or np.any(np.diff(pts) <= 0):
        raise ValueError(f"{path}: missing/non-increasing presentation timestamps")
    return pts


def phase_lookup(v, phases):
    """Step5 row i describes original frames [frame_i, frame_{i+1})."""
    audit = v["audit"]
    frames = audit.frame.to_numpy(int)
    indices = np.searchsorted(v["f"]["frame"], frames, side="right") - 1
    result = {}
    for name, phase in phases.items():
        mapped = np.full(len(frames), -1, np.int8)
        good = (indices >= 0) & (indices < len(phase) - 1)
        ids = np.flatnonzero(good)
        good[ids] &= v["valid"][indices[ids]]
        # Edited freeze/cut/card frames carry no inferred phase.
        for key in ("is_edit_freeze", "is_card"):
            if key in audit:
                good &= ~audit[key].fillna(False).to_numpy(bool)
        ids = np.flatnonzero(good)
        mapped[ids] = phase[indices[ids]]
        result[name] = mapped
    return indices, result


def phase_plot(path, v, phases, events):
    t = v["f"]["t_real_s_v2"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.5), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    for r, label in enumerate(("heel", "mid", "toe")):
        values = v["f"]["slow_ratio"][:, v["slot"], r].copy()
        values[~v["valid"]] = np.nan
        axes[0].plot(t, values, ".-", lw=.8, ms=2, label=label)
    axes[0].set_ylabel("slow ratio")
    axes[0].legend(ncol=3)
    rgb = {k: np.array(c[::-1]) / 255 for k, c in COLORS.items()}
    for ax, (name, phase) in zip(axes[1:], phases.items()):
        for i in range(len(t) - 1):
            ax.axvspan(t[i], t[i + 1], color=rgb[int(phase[i])], linewidth=0)
        for event in events:
            if event["method"] == name and event["type"] in ("IC", "TO"):
                ax.axvline(event["t_real_s"], color="#222222", lw=.8)
                ax.text(event["t_real_s"], .55, event["type"], fontsize=7, rotation=90, ha="center", color="white")
        ax.set(yticks=[], ylabel="Fréchet CPD" if name == "frechet" else "RFF-GP-HSMM", ylim=(0, 1))
    axes[-1].set_xlabel("Step5 real-time axis (s); phase labels and IC/TO are candidates")
    fig.suptitle(f"{v['video_id']} | fixed slot {v['slot']} ({'A' if v['slot']==0 else 'B'}) | tracker bypassed")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render(path, poster, v, phases, events):
    pts = v["audit"].pts_s.to_numpy(float)
    if np.any(np.diff(pts) <= 0):
        raise ValueError("Source audit has non-increasing PTS")
    origin = float(pts[0])
    relative = pts - origin
    source_indices, mapped = phase_lookup(v, phases)
    details = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream_tags=rotate:stream_side_data=rotation", "-of", "json", str(v["source"])], capture_output=True, text=True, check=True).stdout)
    display = details["streams"][0]
    rotation = float(next((s["rotation"] for s in display.get("side_data_list", []) if "rotation" in s), display.get("tags", {}).get("rotate", 0)))
    quarter_turns = int(round(rotation / 90)) % 4
    with av.open(str(v["source"])) as input_container:
        source_stream = input_container.streams.video[0]
        rate = source_stream.average_rate or Fraction(30, 1)
        duration = float(source_stream.duration * source_stream.time_base) if source_stream.duration else relative[-1] + 1 / float(rate)
        duration = max(duration, relative[-1] + .001)
        base = np.full((960, 1280, 3), 22, np.uint8)
        a, b, width = 220, 1250, 1030
        text(base, f"{v['video_id']} | fixed Step5 foot: slot {v['slot']} / {'A' if v['slot']==0 else 'B'} | IC/TO candidates", 18, 30, .73)
        text(base, "Same input foot for both methods | single-foot detector BYPASSED", 18, 59, .57)
        row_y = dict(frechet=770, hsmm=855)
        for name, phase in mapped.items():
            y = row_y[name]
            text(base, "Frechet CPD" if name == "frechet" else "RFF-GP-HSMM", 18, y + 23, .64)
            for i in range(len(pts)):
                x0 = a + round(relative[i] / duration * width)
                right = relative[i + 1] if i + 1 < len(pts) else duration
                x1 = a + round(right / duration * width)
                cv2.rectangle(base, (x0, y), (max(x0, x1), y + 30), COLORS[int(phase[i])], -1)
            for event in events:
                if event["method"] != name or event["type"] not in ("IC", "TO"):
                    continue
                x = a + round((event["source_pts_s"] - origin) / duration * width)
                color = (240, 240, 240) if event["type"] == "IC" else (35, 225, 255)
                cv2.line(base, (x, y - 5), (x, y + 35), color, 1)
                text(base, event["type"], max(a, min(x - 7, b - 16)), y - 10 if event["type"] == "IC" else y + 51, .37, color)
        for phase, label, x in ((1, "STANCE", 20), (0, "SWING", 200), (-1, "UNKNOWN", 380)):
            cv2.rectangle(base, (x, 706), (x + 24, 728), COLORS[phase], -1)
            text(base, label, x + 32, 724, .58)
        for color, label, x in zip(REGION_COLORS, REGION_NAMES, (665, 845, 1025)):
            cv2.rectangle(base, (x, 706), (x + 21, 728), color, -1)
            text(base, "mask " + label, x + 29, 724, .5)
        text(base, "IC: swing -> stance | TO: stance -> swing | white line: current source time", 20, 944, .52)
        typed = [e for e in events if e["type"] in ("IC", "TO")]
        poster_frame = typed[len(typed) // 2]["frame"] if typed else int(v["f"]["frame"][len(v["f"]["frame"]) // 2])
        by_method = {name: [e for e in typed if e["method"] == name] for name in phases}
        output = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        stream = output.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 1280, 960, "yuv420p"
        stream.codec_context.time_base = Fraction(1, 90000)
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.thread_count = 2
        stream.options = {"crf": "21", "preset": "fast"}
        seen_pts, poster_saved = [], False
        cached_row, cached_regions = -1, None
        mask_display_frames = 0
        try:
            for frame_id, frame in enumerate(input_container.decode(video=0)):
                timestamp = float(frame.pts * frame.time_base)
                if frame_id >= len(pts) or abs(timestamp - pts[frame_id]) > 3e-5:
                    raise ValueError(f"{v['video_id']}: decoded frame differs from audited PTS at {frame_id}")
                seen_pts.append(timestamp)
                image = frame.to_ndarray(format="bgr24")
                if quarter_turns:
                    image = np.rot90(image, quarter_turns).copy()
                canvas = base.copy()
                sample = int(source_indices[frame_id])
                crop = None
                marked = image.copy()
                mask_shown = False
                if 0 <= sample < len(v["kp"]) and "display_mask" in v:
                    same_observation = int(v["audit"].unique_id.iloc[frame_id]) == int(v["feature_unique_id"][sample])
                    frozen = bool(v["audit"].is_edit_freeze.iloc[frame_id]) if "is_edit_freeze" in v["audit"] else False
                    if same_observation and not frozen:
                        if cached_row != sample:
                            cached_regions = prepare_regions(v["display_mask"], sample, v["slot"], image.shape[1], image.shape[0])
                            cached_row = sample
                        marked, crop = apply_regions(image, cached_regions)
                        mask_shown = cached_regions is not None
                        mask_display_frames += int(mask_shown)
                # A location crop may use pose when no mask exists; no pose polygon
                # is presented as the boundary of a real segmented mask.
                if crop is None and 0 <= sample < len(v["kp"]) and v["valid"][sample]:
                    points = v["kp"][sample]
                    points = points[np.isfinite(points).all(1)]
                    if len(points) >= 2:
                        center = points.mean(0)
                        radius = max(35., np.ptp(points, axis=0).max() * 1.5)
                        x0, y0 = np.maximum(0, np.floor(center - radius)).astype(int)
                        x1, y1 = np.minimum([image.shape[1], image.shape[0]], np.ceil(center + radius)).astype(int)
                        if x1 > x0 and y1 > y0:
                            crop = image[y0:y1, x0:x1]
                canvas[78:685, 10:950] = fit(marked, 940, 607)
                if crop is not None:
                    canvas[82:360, 968:1268] = fit(crop, 300, 278)
                text(canvas, f"source f{frame_id}", 970, 390, .63)
                text(canvas, f"PTS {timestamp-origin:.3f}s", 970, 418, .63)
                text(canvas, "mask: Step4 heel/mid/toe" if mask_shown else "mask: unavailable here", 970, 446, .43)
                text(canvas, "IC/TO not GT-validated", 970, 474, .48)
                for j, name in enumerate(phases):
                    phase = int(mapped[name][frame_id])
                    y = 515 + j * 80
                    cv2.rectangle(canvas, (967, y), (1268, y + 37), COLORS[phase], -1)
                    label = "FRECHET" if name == "frechet" else "HSMM"
                    text(canvas, f"{label}: {PHASE_NAMES[phase]}", 977, y + 25, .6)
                    recent = [e for e in by_method[name] if 0 <= timestamp - e["source_pts_s"] <= .12]
                    if recent:
                        text(canvas, recent[-1]["type"] + " candidate", 977, y + 61, .6, (35, 225, 255))
                x = a + round((timestamp - origin) / duration * width)
                cv2.line(canvas, (x, 741), (x, 911), (255, 255, 255), 2)
                if frame_id == poster_frame:
                    cv2.imwrite(str(poster), canvas)
                    poster_saved = True
                encoded = av.VideoFrame.from_ndarray(canvas, format="bgr24")
                encoded.time_base = Fraction(1, 90000)
                encoded.pts = round((timestamp - origin) * 90000)
                for packet in stream.encode(encoded):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        finally:
            output.close()
        if len(seen_pts) != len(pts):
            raise ValueError("Source decoder did not deliver all audited frames")
        if not poster_saved:
            raise ValueError("Poster frame was not reached")
    output_pts = probe(path)
    if len(output_pts) != len(pts):
        raise ValueError("Output frame count differs from source")
    timing_error = float(np.max(abs(output_pts - relative)))
    if timing_error > 3e-5:
        raise ValueError(f"Output PTS mismatch {timing_error}")
    return dict(source_frames=len(pts), output_frames=len(output_pts), max_pts_error_s=timing_error,
                first_pts_s=float(output_pts[0]), last_pts_s=float(output_pts[-1]), source_speed_preserved=True,
                audio_included=False, source_display_rotation_applied=rotation, phase_bars=["frechet", "hsmm"],
                regional_mask_overlay=True, mask_display_frames=mask_display_frames)
