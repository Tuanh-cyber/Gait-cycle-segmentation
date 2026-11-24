"""Review both feet's CSV labels against saved IC/TO predictions on the source AVI."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/step6_fixed_foot_events/code"))
from mask_overlay import REGION_COLORS, apply_regions, fit_image, prepare_regions

WIDTH, HEIGHT = 1440, 800
ZOOM_W, ZOOM_H = 172, 145
PANEL_Y = 101
PANEL_H = 307
TIMELINE_X0, TIMELINE_X1 = 201, 1404
TRACK_Y = (482, 539, 603, 660)
BRANCHES = ("embedding96", "embedding192")
EVENTS = {"IC": ("strike", (226, 145, 249)), "TO": ("off", (85, 184, 250))}
PHASE_COLORS = {"IC": EVENTS["IC"][1], "Flat": (137, 196, 87),
                "TO": EVENTS["TO"][1], "Swing": (223, 164, 89)}
BG = (27, 24, 23)
INK = (242, 239, 236)
MUTED = (180, 170, 162)
OUTLINE = (157, 233, 168)
WINDOW_S = .1


def label(image, message, x, y, size=.55, color=INK, bold=False):
    cv2.putText(image, message, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 2 if bold else 1, cv2.LINE_AA)


def event_text(item):
    return f"{item[1]} {item[0]:.2f}s"


def review_tracks(trial, side, truth, source_labels, matches):
    other = "right" if side == "left" else "left"
    raw = source_labels.loc[trial]
    reference = {
        foot: sorted((time, name, "") for name, (field, _) in EVENTS.items()
                     for time in json.loads(raw[f"{foot}_foot_{field}"]))
        for foot in (side, other)
    }
    expected = truth.loc[truth.trial.eq(trial)].sort_values("time_s")
    if not expected.side.eq(side).all() or [(e[0], e[1]) for e in reference[side]] != list(
        expected[["time_s", "event"]].itertuples(index=False, name=None)
    ):
        raise ValueError(f"CSV labels differ from scoring labels in passage {trial}")
    valid = matches.loc[matches.trial.eq(trial) & matches.predicted_s.notna()]
    prediction = [
        [(e.predicted_s, e.event, e.status) for e in valid.loc[valid.branch.eq(branch)].itertuples()]
        for branch in BRANCHES
    ]
    return [(f"GT {side.upper()} (tracked)", reference[side]),
            (f"GT {other.upper()} (other)", reference[other]),
            ("Embedding 96", prediction[0]), ("Embedding 192", prediction[1])]


def draw_timeline(canvas, passage, tracks, supported, phase_windows=None):
    first, end, fps = int(passage.first_frame), int(passage.stop_frame), passage.fps
    stop = end - 1

    def pixel(number):
        return round(TIMELINE_X0 + (number - first) * (TIMELINE_X1 - TIMELINE_X0) / (stop - first))

    for row, (y, (name, events)) in enumerate(zip(TRACK_Y, tracks)):
        label(canvas, name, 23, y + 5, .52)
        cv2.line(canvas, (TIMELINE_X0, y), (TIMELINE_X1, y), (103, 94, 86), 1, cv2.LINE_AA)
        if row >= 2 and phase_windows is not None:
            phases, windows = phase_windows[row-2]
            for s in phases:
                cv2.rectangle(canvas, (pixel(s.start_frame), y-15), (pixel(s.end_frame), y-4),
                              PHASE_COLORS[s.phase], -1)
            for item in windows:
                color = EVENTS[item.event][1]
                x0, x1 = pixel(item.event_start_frame), pixel(item.event_end_frame)
                cv2.rectangle(canvas, (x0, y+2), (max(x0+2,x1), y+12), color, -1 if item.matched else 1)
            continue
        if y in TRACK_Y[2:]:
            for time, event, _ in tracks[0][1]:
                lo, hi = max(first, (time - WINDOW_S) * fps), min(stop, (time + WINDOW_S) * fps)
                if lo <= hi:
                    color = tuple(round(.3 * c + .7 * b) for c, b in zip(EVENTS[event][1], BG))
                    cv2.rectangle(canvas, (pixel(lo), y - 11), (pixel(hi), y + 11), color, -1)
        for time, event, status in sorted(events):
            frame = time * fps
            if not first <= frame < end:
                continue
            x = pixel(min(frame, stop))
            color = EVENTS[event][1]
            if status == "TP":
                cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)
            elif status == "FP":
                cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
            else:
                cv2.line(canvas, (x, y - 9), (x, y + 9), color, 2, cv2.LINE_AA)
            label(canvas, event, min(x - 7, TIMELINE_X1 - 18), y - 15, .42, color)

    label(canvas, "Usable flow", 23, 706, .52)
    cv2.rectangle(canvas, (TIMELINE_X0, 697), (TIMELINE_X1, 706), (77, 71, 66), -1)
    groups = np.flatnonzero(supported[first:end])
    for part in np.split(groups, np.flatnonzero(np.diff(groups) != 1) + 1):
        if len(part):
            cv2.rectangle(canvas, (pixel(first + part[0]), 697), (pixel(first + part[-1]), 706), (138, 181, 104), -1)
    label(canvas, f"{first / fps:.2f}s", TIMELINE_X0, 733, .48, MUTED)
    label(canvas, f"{stop / fps:.2f}s", TIMELINE_X1 - 59, 733, .48, MUTED)
    label(canvas, "IC = violet", 23, 774, .52, EVENTS["IC"][1])
    label(canvas, "TO = orange", 152, 774, .52, EVENTS["TO"][1])
    if phase_windows is None:
        label(canvas, "Shaded = tracked foot GT +/-100 ms     Dot = matched     X = unmatched     White line = current frame",
              302, 774, .49, MUTED)
    else:
        label(canvas, "Flat = green", 286, 774, .52, PHASE_COLORS["Flat"])
        label(canvas, "Swing = blue", 421, 774, .52, PHASE_COLORS["Swing"])
        label(canvas, "Upper band = phase   Lower box = event interval   Filled box = contains GT frame",
              567, 774, .49, MUTED)
    return pixel


def masks_for_passage(mask_file, passage, slot, width, height):
    with np.load(mask_file) as saved:
        payload = {k: saved[k] for k in ("frame", "rle_ptr", "rle_counts", "W", "H")}
    if (int(payload["W"]), int(payload["H"])) != (width, height):
        raise ValueError(f"Wrong mask size for {passage.video_id}")
    first = int(passage.first_frame)
    masks = {
        first + int(frame): tuple(prepare_regions(payload, row, s, width, height) for s in (slot, 1-slot))
        for row, frame in enumerate(payload["frame"])
    }
    centers = [(f, (m[0][0][0] + m[0][0][2])/2, m[0][0][3]) for f, m in masks.items() if m[0] is not None]
    if not centers:
        centers = [(first, width/2, height-35)]
    return masks, np.array(centers).T


def outline(image, mask):
    if mask is None:
        return
    (x0, y0, x1, y1), regions = mask
    union = np.logical_or.reduce(regions).astype(np.uint8)
    lines, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image[y0:y1, x0:x1], lines, -1, OUTLINE, 1, cv2.LINE_AA)


def render_frame(background, raw, frame, passage, slot, pair, centers, pixel, tracks, title, phase_windows=None):
    canvas = background.copy()
    time = frame / passage.fps
    label(canvas, f"{title}  /  PASS {passage.trial:02d}",
          22, 36, .83, bold=True)
    label(canvas, f"Source frame {frame:04d}   |   AVI {time:.3f}s   |   Playback 0.5x", 851, 35, .64)
    label(canvas, "Original context", 22, 85, .55, MUTED)
    label(canvas, "Shoe close-up (raw)", 660, 85, .55, MUTED)
    label(canvas, f"Tracked mask (slot {slot}) and other foot outline", 1037, 85, .49, MUTED)
    seen, xs, ys = centers
    height, width = raw.shape[:2]
    x = int(np.clip(round(np.interp(frame, seen, xs)) - ZOOM_W//2, 0, width-ZOOM_W))
    y = int(np.clip(round(np.interp(frame, seen, ys)) - 105, 0, height-ZOOM_H))
    marked, _ = apply_regions(raw, pair[0])
    outline(marked, pair[1])
    cv2.rectangle(marked, (x, y), (x+ZOOM_W-1, y+ZOOM_H-1), MUTED, 1, cv2.LINE_AA)
    canvas[PANEL_Y:PANEL_Y+PANEL_H, 22:642] = fit_image(marked, 620, PANEL_H)
    canvas[PANEL_Y:PANEL_Y+PANEL_H, 660:1022] = fit_image(raw[y:y+ZOOM_H, x:x+ZOOM_W], 362, PANEL_H)
    canvas[PANEL_Y:PANEL_Y+PANEL_H, 1037:1399] = fit_image(marked[y:y+ZOOM_H, x:x+ZOOM_W], 362, PANEL_H)
    if pair[0] is None:
        label(canvas, "Tracked mask unavailable", 1055, 127, .53, EVENTS["TO"][1], True)
    leg_x = 1029
    for color, name in zip((*REGION_COLORS, OUTLINE), ("heel", "mid", "toe", "other outline")):
        cv2.rectangle(canvas, (leg_x+8, 426), (leg_x+20, 437), color, -1)
        label(canvas, name, leg_x+26, 438, .42)
        leg_x += 73

    cursor = pixel(frame)
    cv2.line(canvas, (cursor, 444), (cursor, 716), INK, 1, cv2.LINE_AA)
    for row, (y, (_, events)) in enumerate(zip(TRACK_Y, tracks)):
        if row >= 2 and phase_windows is not None:
            phases, windows = phase_windows[row-2]
            phase = next((s.phase for s in phases if s.start_frame <= frame <= s.end_frame), None)
            active = next((e for e in windows if e.event_start_frame <= frame <= e.event_end_frame), None)
            if active is not None:
                label(canvas, f"{active.event} [{active.event_start_frame}-{active.event_end_frame}]",
                      24, y+24, .46, EVENTS[active.event][1], True)
            elif phase is not None:
                label(canvas, phase, 24, y+24, .46, PHASE_COLORS[phase])
            continue
        nearby = [e for e in events if abs(e[0] - time) <= WINDOW_S + 1e-9]
        if not nearby:
            continue
        nearest = min(nearby, key=lambda item: abs(item[0] - time))
        delta = round(1000 * (time - nearest[0]))
        caption = f"{event_text(nearest)}  |  frame {delta:+d} ms"
        label(canvas, caption, 24, y+24, .42, EVENTS[nearest[1]][1],
              abs(nearest[0] - time) <= .5/passage.fps + 1e-9)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject", default="BD004")
    parser.add_argument("--session", default="Session1")
    parser.add_argument("--intervals", action="store_true", help="Overlay four-phase spans and event intervals")
    args = parser.parse_args()
    result = args.evaluation
    passages = pd.read_csv(result/"passages.csv").sort_values("trial")
    slots = pd.read_csv(result/"foot_selection.csv").set_index("trial")
    matches = pd.read_csv(result/"matched_events_100ms.csv")
    truth = pd.read_csv(result/"ground_truth_selected_foot.csv")
    if args.intervals:
        phase_table = pd.read_csv(result/"phase_intervals.csv")
        window_table = pd.read_csv(result/"event_intervals.csv")
    condition = args.source.stem
    title = f"{args.subject}  /  {condition.replace('_', ' ').upper()}"
    raw = pd.read_csv(args.ground_truth)
    raw = raw.loc[raw.state.eq(condition.lower())].set_index(["sample", "session"])
    raw = pd.DataFrame({p.trial: raw.loc[(f"{args.subject}/{condition}{p.trial}", args.session)]
                        for p in passages.itertuples()}).T
    scores = pd.read_csv(result/"event_confidence_frames.csv")
    np.testing.assert_array_equal(scores.frame, np.arange(len(scores)))
    supported = scores[[f"{b}_flow_supported" for b in BRANCHES]].any(axis=1).to_numpy()
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(args.source))
    fps = float(passages.fps.iloc[0])
    if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), fps):
        raise ValueError("Could not open original AVI at the expected frame rate")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(".tmp.mp4")
    encode = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
              "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps/2), "-i", "-", "-an",
              "-c:v", "libx264", "-threads", "3", "-preset", "veryfast", "-crf", "20",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)]
    rendered = 0
    try:
        with subprocess.Popen(encode, stdin=subprocess.PIPE) as writer:
            for passage in passages.itertuples():
                selection = slots.loc[passage.trial]
                slot = int(selection.slot)
                tracks = review_tracks(passage.trial, selection.side, truth, raw, matches)
                phase_windows = None
                if args.intervals:
                    phase_windows = [
                        tuple(list(table.loc[table.trial.eq(passage.trial) & table.branch.eq(branch)].itertuples())
                              for table in (phase_table, window_table))
                        for branch in BRANCHES
                    ]
                masks, centers = masks_for_passage(args.masks/f"{passage.video_id}.npz", passage, slot, width, height)
                bg = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
                pixel = draw_timeline(bg, passage, tracks, supported, phase_windows)
                first, end = int(passage.first_frame), int(passage.stop_frame)
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
                    raise ValueError(f"Could not seek to source frame {first}")
                print(f"Pass {passage.trial}: clip {rendered / (fps/2):.2f}s, source frames {first}–{end-1}", flush=True)
                for frame in range(first, end):
                    ok, raw_frame = capture.read()
                    if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                        raise ValueError(f"Could not read source frame {frame}")
                    pair = masks.get(frame, (None, None))
                    image = render_frame(bg, raw_frame, frame, passage, slot, pair, centers, pixel, tracks, title,
                                         phase_windows)
                    writer.stdin.write(image.tobytes())
                    rendered += 1
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("FFmpeg failed to encode the review")
        temp.replace(args.out)
    finally:
        capture.release()
    print(f"{rendered} source frames → {args.out}", flush=True)


if __name__ == "__main__":
    main()
