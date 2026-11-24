"""Prepare overground AVI passages and score their gait-event predictions."""
import argparse
import errno
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("step2_tracking", "step3_footkp", "step4_footmask", "step5_flow")
PRIMARY_TOLERANCE_S = .10


def probe_video(source):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(source)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def visible_passages(source, fps, frame_count):
    command = ["ffmpeg", "-hide_banner", "-i", str(source), "-vf", "blackdetect=d=0.25:pix_th=0.05:pic_th=0.99", "-an", "-f", "null", "-"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    black = [(round(float(a)*fps), round(float(b)*fps)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", result.stderr)]
    runs = []
    end = 0
    for start, stop in black + [(frame_count, frame_count)]:
        if start-end >= fps:
            runs.append((end, start))
        end = stop
    return runs


def walking_direction(frames):
    """Find the moving subject against the stationary camera background."""
    small = np.array([cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (322, 183)) for im in frames[::2]])
    background = np.median(small, axis=0).astype(np.uint8)
    centers = []
    for i, image in enumerate(small):
        foreground = (cv2.absdiff(image, background) > 30).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(foreground)
        if count > 1:
            blob = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            if stats[blob, cv2.CC_STAT_AREA] >= 210:
                centers.append((i, centroids[blob, 0]))
    if len(centers) < 10:
        raise ValueError("Could not infer walking direction from video motion")
    t, x = np.array(centers).T
    correlation = np.corrcoef(t, x)[0,1]
    if abs(correlation) < .6:
        raise ValueError(f"Ambiguous motion direction with correlation {correlation:.2f}")
    return (1 if correlation > 0 else -1), float(correlation)


def prepare(source, output, subject="AJ026", max_passages=None):
    from fractions import Fraction

    stream = probe_video(source)
    fps = float(Fraction(stream["avg_frame_rate"]))
    count = int(stream["nb_frames"])
    width, height = stream["width"], stream["height"]
    intervals = visible_passages(source, fps, count)
    if max_passages is not None:
        if len(intervals) < max_passages:
            raise ValueError(f"Only {len(intervals)} video passages found; expected {max_passages}")
        intervals = intervals[:max_passages]
    data = output/"data"
    (data/"Correct").mkdir(parents=True, exist_ok=True)
    (data/"step0_audit/frames").mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        shutil.copytree(ROOT/"data"/stage/"code", data/stage/"code", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))

    cap = cv2.VideoCapture(str(source))
    metadata, passages = [], []
    for number, (first, end) in enumerate(intervals, start=1):
        vid = f"{subject}_pass{number:02}"
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        images = []
        for _ in range(first, end):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Source decode stopped in {vid}")
            images.append(image)
        sign, correlation = walking_direction(images)
        near_side = "right" if sign > 0 else "left"
        dest = data/"Correct"/f"{vid}.avi"
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Cannot create lossless FFV1 clip")
        for image in images:
            writer.write(image)
        writer.release()
        check = cv2.VideoCapture(str(dest))
        if int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != len(images):
            raise ValueError("Lossless clip frame count differs")
        for i in (0, len(images)//2, len(images)-1):
            check.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, decoded = check.read()
            if not ok or not np.array_equal(images[i], decoded):
                raise ValueError("Clip decode does not match original pixels")
        check.release()
        frames = np.arange(end-first)
        time = frames/fps
        pd.DataFrame(dict(frame=frames, pts_s=time, is_dup=False, is_edit_freeze=False, is_card=False,
                          continuity_id=0, unique_id=frames, shot_id=0, speed_part_id=0,
                          t_real_s=time, t_real_s_v2=time, time_model="constant", k_confidence="cao",
                          cam_dx=0., cam_dy=0., cam_scale=1., cam_inlier=1.,
                          run_cx=np.nan, run_cy=np.nan, run_x0=np.nan, run_x1=np.nan,
                          run_y0=np.nan, run_y1=np.nan)).to_csv(data/"step0_audit/frames"/f"{vid}_frames.csv", index=False)
        metadata.append(dict(video_id=vid, video=dest.name, label="CORRECT", excluded_shots="", width=width,
                             height=height, fps=fps, n_frames=len(frames), s_final=sign, near_leg_camera=near_side,
                             abs_time_ok=True, cadence_real_spm_v2=""))
        passages.append(dict(video_id=vid, trial=number, first_frame=first, stop_frame=end,
                             start_s=first/fps, stop_s=end/fps, fps=fps, direction=sign,
                             motion_correlation=correlation, near_side=near_side))
        print(f"{vid}: frames {first}–{end-1}; direction {sign:+d}, near side {near_side}", flush=True)
    cap.release()
    pd.DataFrame(metadata).to_csv(data/"step0_audit/step0_metadata.csv", index=False)
    pd.DataFrame(passages).to_csv(output/"passages.csv", index=False)


def model_assets(output):
    pairs = [
        (ROOT/".third_party/yolo11m.pt", output/"data/step2_tracking/models/yolo11m.pt"),
        (ROOT/".third_party/sam2.1_hiera_base_plus.pt", output/"data/step4_footmask/models/sam2.1_hiera_base_plus.pt"),
    ]
    for original, link in pairs:
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            try:
                os.link(original, link)
            except OSError as error:
                if error.errno != errno.EXDEV and getattr(error, "winerror", None) != 17:
                    raise
                shutil.copy2(original, link)
    if not (output/".third_party/GMA").exists():
        shutil.copytree(ROOT/".third_party/GMA", output/".third_party/GMA",
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "imgs"))


def trim_untracked(output, from_trial):
    """Skip leading/trailing frames without a person box when making masks."""
    data = output/"data"
    for passage in pd.read_csv(output/"passages.csv").itertuples():
        if passage.trial < from_trial:
            continue
        kpath = data/"step2_tracking/keypoints_v2"/f"{passage.video_id}_kp.npz"
        ppath = data/"step3_footkp/footkp"/f"{passage.video_id}_footkp.npz"
        with np.load(kpath) as saved:
            track = {name:saved[name] for name in saved.files}
        with np.load(ppath) as saved:
            pose = {name:saved[name] for name in saved.files}
        np.testing.assert_array_equal(track["frame"], pose["frame"])
        n = len(track["frame"])
        observed = np.flatnonzero(np.isfinite(track["bbox"]).all(axis=1))
        if not len(observed):
            continue
        first, end = max(0, observed[0]-1), min(n, observed[-1]+2)
        for path, data_dict in ((kpath, track), (ppath, pose)):
            subset = {name:(values[first:end] if values.ndim and len(values)==n else values)
                      for name, values in data_dict.items()}
            temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **subset)
            temp.replace(path)
        print(f"{passage.video_id}: {end-first}/{n} frames retained for SAM2, with original frame IDs", flush=True)


def run_stages(output, names, device):
    data = output/"data"
    env = os.environ.copy()
    modules = [ROOT/".third_party"/part for part in ("overground-runtime", "runtime-deps", "sam2")]
    env["PYTHONPATH"] = os.pathsep.join(map(str, modules)) + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCH_HOME"] = str(ROOT/".third_party/overground-model-cache")
    env["PYTHONUTF8"] = "1"
    scripts = {
        "track": ("step2_tracking/code/run_step2.py", ["--det-device", device, "--pose-device", "cpu"]),
        "keypoints": ("step3_footkp/code/run_step3.py", ["--pose-device", "cpu"]),
        "masks": ("step4_footmask/code/run_step4.py", ["--device", device]),
        "mask_qc": ("step4_footmask/code/s4_postprocess.py", ["--no-review"]),
        "flow": ("step5_flow/code/run_step5.py", ["--device", device]),
    }
    # The original postprocess has a hard-coded C20 annotation. This dataset
    # contains a separate subject and cannot load an unrelated C20 keypoint file.
    postprocess = data/scripts["mask_qc"][0]
    original = postprocess.read_text(encoding="utf-8")
    original = re.sub(r"^TIME_EXCLUDE = .*?$", "TIME_EXCLUDE = {}", original, flags=re.MULTILINE)
    postprocess.write_text(original, encoding="utf-8")
    log_dir = output/"logs"
    log_dir.mkdir(exist_ok=True)
    for stage in names:
        path, flags = scripts[stage]
        print(f"Starting {stage}", flush=True)
        with (log_dir/f"{stage}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, str(data/path), *flags], cwd=data, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if proc.wait():
                raise RuntimeError(f"{stage} failed; see {log.name}")


def load_inference_data(source, output, subject, reuse_clouds=False):
    from event_data import load_video
    from hsmm_inputs import build_clouds, common_view

    passages = pd.read_csv(output/"passages.csv")
    data = output.resolve()/"data"
    metadata = pd.read_csv(data/"step0_audit/step0_metadata.csv", keep_default_na=False).set_index("video_id", drop=False)
    raw, points, pointers, eligible, group, source_frame, slots = [], [], [], [], [], [], []
    continuity_fields = ("continuity_id", "shot_id", "speed_part_id")
    continuity = {name: [] for name in continuity_fields}
    offset = 0
    clouds_dir = output/"clouds"
    clouds_dir.mkdir(exist_ok=True)
    for passage in passages.itertuples():
        v = load_video(metadata.loc[passage.video_id], data=data)
        cache = clouds_dir/f"{passage.video_id}.npz"
        if reuse_clouds and cache.exists():
            with np.load(cache) as saved:
                np.testing.assert_array_equal(saved["frame"], v["f"]["frame"])
                payload = {key: saved[key] for key in ("raw42", "points", "ptr", "valid")}
        else:
            payload, _ = build_clouds(v, cache, data=data)
        n = len(payload["valid"])
        raw.append(payload["raw42"])
        points.append(payload["points"])
        pointers.append(payload["ptr"]+offset)
        offset += len(payload["points"])
        eligible.append(payload["valid"])
        group.append(np.full(n, passage.trial))
        source_frame.append(v["f"]["frame"] + passage.first_frame)
        for name in continuity_fields:
            continuity[name].append(v["f"][name])
        with np.load(output/"data/step3_footkp/footkp"/f"{passage.video_id}_footkp.npz") as pose:
            near_conf = float(np.median(pose["near_conf"]))
        side = passage.near_side if v["selection"] == "initial_near_slot_fixed" else ("left", "right")[v["slot"]]
        slots.append(dict(trial=passage.trial, slot=v["slot"], side=side, expected_near_side=passage.near_side,
                          selection=v["selection"], near_confidence=near_conf, fit_observations=int(payload["valid"].sum())))
        print(f"{passage.video_id}: slot {v['slot']} ({side}), {payload['valid'].sum()}/{n} eligible", flush=True)

    fps = passages.fps.iloc[0]
    frames = np.concatenate(source_frame)
    valid = np.concatenate(eligible)
    trial_id = np.concatenate(group)
    full_frames = np.arange(int(probe_video(source)["nb_frames"]))
    payload = dict(raw42=np.concatenate(raw), points=np.concatenate(points), ptr=np.concatenate(pointers), valid=valid)
    # The pooled model has no single slot number. The predictions attach a side
    # from the per-passage slot audit instead of this diagnostic placeholder.
    source_blocks = np.cumsum(np.r_[False, np.diff(frames) != 1])
    groups = [trial_id, source_blocks, *(np.concatenate(continuity[name]) for name in continuity_fields)]
    base = dict(video_id=subject, slot=-1, valid=valid, groups=groups,
                f=dict(frame=frames, t_real_s_v2=frames/fps), abs_time_ok=True,
                audit=pd.DataFrame(dict(frame=full_frames, pts_s=full_frames/fps)))
    # Fit/tune from video only. The ground-truth CSV enters the separate score stage.
    return common_view(base, payload), payload, slots


def score(gt_path, output, condition, swap_side=False, swap_ground_truth=False, subject="AJ026", session="Session1",
          score_out=None, branches=None, plot=True):
    from scipy.optimize import linear_sum_assignment

    passages = pd.read_csv(output/"passages.csv").set_index("trial")
    slots = pd.read_csv(output/"foot_selection.csv").set_index("trial")
    predict = pd.read_csv(output/"predictions.csv")
    confidence = pd.read_csv(output/"event_confidence_frames.csv").set_index("frame")
    if branches is None:
        suffix = "_ic_confidence"
        branches = tuple(column.removesuffix(suffix) for column in confidence if column.endswith(suffix))
    if not branches:
        raise ValueError(f"No event-confidence branches in {output}")
    raw = pd.read_csv(gt_path)
    opposite = {"left": "right", "right": "left"}
    score_root = score_out if score_out is not None else output
    if swap_ground_truth:
        result_dir = score_root/"ground_truth_swap"
        mapping = "giả thuyết hoán đổi hai cột ground truth"
    elif swap_side:
        result_dir = score_root/"side_swap_diagnostic"
        mapping = "giả thuyết đảo trái/phải"
    else:
        result_dir = score_root
        mapping = "bên chân pipeline gán"
    result_dir.mkdir(parents=True, exist_ok=True)
    if swap_side:
        slots["side"] = slots["side"].map(opposite)
        predict["side"] = predict["side"].map(opposite)
    truth = []
    for trial, passage in passages.iterrows():
        identifier = f"{subject}/{condition}{trial}"
        row, = raw[(raw["sample"] == identifier) & (raw["session"] == session) &
                   (raw["state"] == condition.lower())].to_dict("records")
        side = slots.loc[trial, "side"]
        csv_side = opposite[side] if swap_ground_truth else side
        view = confidence.loc[passage.first_frame:passage.stop_frame-1]
        supported_frames = view[[f"{branch}_flow_supported" for branch in branches]].any(axis=1)
        supported = view.loc[supported_frames, "source_pts_s"].to_numpy()
        for event, field in (("IC", "strike"), ("TO", "off")):
            for time_s in json.loads(row[f"{csv_side}_foot_{field}"]):
                frame = int(np.floor(time_s*passage.fps+.5))
                truth.append(dict(trial=trial, side=side, event=event, time_s=time_s, nearest_frame=frame,
                                  passage_visible=passage.start_s <= time_s < passage.stop_s,
                                  flow_supported=bool(frame in supported_frames.index and supported_frames.at[frame]),
                                  flow_within_100ms=bool(np.any(np.abs(supported-time_s) <= PRIMARY_TOLERANCE_S+1e-9))))
    gt = pd.DataFrame(truth)
    if swap_ground_truth:
        gt.insert(2, "csv_side", gt.side.map(opposite))
    gt.to_csv(result_dir/"ground_truth_selected_foot.csv", index=False)
    primary_matches, summary = [], []
    for scope in ("all_reference", "visible_only"):
        references = gt if scope == "all_reference" else gt[gt.passage_visible]
        for tolerance in (.04, PRIMARY_TOLERANCE_S, .16):
            for branch in branches:
                matches = []
                for trial, passage in passages.iterrows():
                    for event in ("IC", "TO"):
                        target = list(references[(references.trial==trial) & (references.event==event)].itertuples())
                        found = list(predict[(predict.trial==trial) & (predict.event==event) & (predict.branch==branch)].itertuples())
                        dist = np.abs(np.subtract.outer([p.time_s for p in found], [g.time_s for g in target]))
                        a, b = linear_sum_assignment(np.where(dist<=tolerance+1e-9, dist, 1000.))
                        paired = [(i,j) for i,j in zip(a,b) if dist[i,j] <= tolerance+1e-9]
                        pairs = dict(paired)
                        for i, p in enumerate(found):
                            ref = target[pairs[i]] if i in pairs else None
                            matches.append(dict(branch=branch, trial=trial, side=p.side, event=event,
                                                status="TP" if ref else "FP", predicted_frame=p.frame,
                                                inside_gt_window=bool(np.any(dist[i] <= tolerance+1e-9)),
                                                predicted_s=p.time_s, confidence=p.confidence, gt_s=ref.time_s if ref else None,
                                                error_ms=1000*(p.time_s-ref.time_s) if ref else None,
                                                gt_visible=ref.passage_visible if ref else None))
                        for j, ref in enumerate(target):
                            if j not in pairs.values():
                                matches.append(dict(branch=branch, trial=trial, side=ref.side, event=event, status="FN",
                                                    predicted_frame=None, predicted_s=None, confidence=None, gt_s=ref.time_s,
                                                    error_ms=None, gt_visible=ref.passage_visible))
                table = pd.DataFrame(matches)
                if scope=="all_reference" and tolerance==PRIMARY_TOLERANCE_S:
                    primary_matches.extend(matches)
                for event in ("IC", "TO", "all"):
                    part = table if event=="all" else table[table.event==event]
                    tp, fp, fn = (int(part.status.eq(k).sum()) for k in ("TP", "FP", "FN"))
                    errors = part.error_ms.dropna()
                    summary.append(dict(scope=scope, tolerance_ms=int(tolerance*1000), branch=branch, event=event,
                                        TP=tp, FP=fp, FN=fn, precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),
                                        f1=2*tp/max(2*tp+fp+fn,1), mae_ms=errors.abs().mean(), bias_ms=errors.mean()))
    matched = pd.DataFrame(primary_matches)
    if swap_ground_truth:
        matched.insert(3, "gt_csv_side", matched.side.map(opposite).where(matched.gt_s.notna()))
    matched.to_csv(result_dir/"matched_events_100ms.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(result_dir/"metrics.csv", index=False)
    print(result[(result.scope=="all_reference") & (result.tolerance_ms==100)].to_string(index=False), flush=True)
    if plot:
        plot_evaluation(result_dir, passages, gt, matched, confidence, subject, condition, mapping, branches)


def plot_evaluation(output, passages, gt, matched, confidence, subject, condition, mapping, branches):
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = Figure(figsize=(7*len(branches), max(6, 1.3*len(passages)+1.6)), layout="constrained")
    axes = fig.subplots(len(passages), len(branches), sharey=True, squeeze=False)
    colors = {"IC": "#8e246b", "TO": "#b36300"}
    bands = {"IC": "#ead0e6", "TO": "#f9dfb1"}
    predictions = matched[matched.predicted_frame.notna()]
    for i, (trial, passage) in enumerate(passages.iterrows()):
        for j, branch in enumerate(branches):
            ax = axes[i, j]
            prediction = predictions[(predictions.trial==trial) & (predictions.branch==branch)]
            ref = gt[gt.trial==trial]
            start = min(passage.start_s, ref.time_s.min()-PRIMARY_TOLERANCE_S)-.03
            stop = max(passage.stop_s,ref.time_s.max()+PRIMARY_TOLERANCE_S)+.03
            view = confidence.loc[int(np.floor(start*passage.fps)):int(np.ceil(stop*passage.fps))]
            missing = ~view[f"{branch}_flow_supported"]
            ax.fill_between(view.source_pts_s, 0, 1, where=missing, step="mid", color="#e9edf0", linewidth=0)
            for event,color in colors.items():
                gt_times = ref.loc[ref.event==event, "time_s"]
                for t in gt_times:
                    ax.axvspan(t-PRIMARY_TOLERANCE_S, t+PRIMARY_TOLERANCE_S, color=bands[event], alpha=.72, linewidth=0, zorder=1.5)
                ax.plot(view.source_pts_s, view[f"{branch}_{event.lower()}_confidence"], color=color, linewidth=1., zorder=2)
                ax.vlines(gt_times, 0, 1, color=color, linestyle="--", alpha=.65, linewidth=.9, zorder=2)
                selected = prediction[prediction.event==event]
                hits = selected[selected.status.eq("TP")]
                extra = selected[selected.inside_gt_window.eq(True) & selected.status.eq("FP")]
                outside = selected[selected.inside_gt_window.eq(False)]
                ax.scatter(hits.predicted_s, hits.confidence, color=color, edgecolor="white", linewidth=.6, s=32, zorder=4)
                ax.scatter(extra.predicted_s, extra.confidence, facecolor="white", edgecolor=color, linewidth=1., s=29, zorder=4)
                ax.scatter(outside.predicted_s, outside.confidence, color=color, marker="x", linewidth=.9, s=21, zorder=4)
            ax.set(xlim=(start,stop), ylim=(0,1.06))
            ax.grid(axis="y",alpha=.2)
            ax.set_ylabel(f"Lượt {trial} · {ref.side.iloc[0]}")
            if i==0:
                ax.set_title(branch)
    for ax in axes[-1]:
        ax.set_xlabel("Thời gian trên AVI gốc (s)")
    fig.suptitle(f"{subject} · {condition.replace('_', ' ')} · IC/TO theo {mapping}")
    fig.legend(handles=[Patch(facecolor=bands["IC"], label="IC · ±100 ms"), Patch(facecolor=bands["TO"], label="TO · ±100 ms"),
                        Line2D([0],[0],color="#444",linestyle="--",label="Mốc ground truth"),
                        Line2D([0],[0],color="#444",marker="o",linestyle="none",label="Dự đoán ghép đúng"),
                        Line2D([0],[0],color="#444",marker="o",markerfacecolor="white",linestyle="none",label="Trong cửa sổ, dư mốc"),
                        Line2D([0],[0],color="#444",marker="x",linestyle="none",label="Ngoài cửa sổ cùng loại"),
                        Patch(facecolor="#e9edf0", label="Thiếu bằng chứng flow")],
               loc="outside lower center", ncols=4, frameon=False)
    fig.savefig(output/"comparison.png",dpi=145)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject", default="AJ026")
    parser.add_argument("--session", default="Session1")
    parser.add_argument("--stages", nargs="+", default=["prepare"],
                        choices=["prepare", "track", "keypoints", "trim", "masks", "mask_qc", "flow", "score"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--score-out", type=Path, help="Write evaluation artifacts here and read predictions from --out")
    side_options = parser.add_mutually_exclusive_group()
    side_options.add_argument("--side-swap-diagnostic", action="store_true", help="Also compare unchanged event times with opposite-side reference labels")
    side_options.add_argument("--swap-ground-truth-sides", action="store_true", help="Score with the CSV sides exchanged and keep the prediction sides")
    parser.add_argument("--trim-from-trial", type=int, default=1)
    parser.add_argument("--max-passages", type=int, help="Prepare only the first N visible passages")
    args = parser.parse_args()
    if args.max_passages is not None and args.max_passages < 1:
        parser.error("--max-passages must be positive")
    out = args.out.resolve()
    if "prepare" in args.stages:
        prepare(args.source.resolve(), out, args.subject, args.max_passages)
    remaining = [stage for stage in args.stages if stage in ("track", "keypoints", "masks", "mask_qc", "flow")]
    if remaining:
        model_assets(out)
    for stage in args.stages:
        if stage == "trim":
            trim_untracked(out, args.trim_from_trial)
        if stage in remaining:
            run_stages(out, [stage], args.device)
    if "score" in args.stages:
        if args.ground_truth is None:
            parser.error("--ground-truth is required to score predictions")
        condition = args.source.stem
        score(args.ground_truth, out, condition, swap_ground_truth=args.swap_ground_truth_sides,
              subject=args.subject, session=args.session, score_out=args.score_out)
        if args.side_swap_diagnostic:
            score(args.ground_truth, out, condition, swap_side=True, subject=args.subject, session=args.session,
                  score_out=args.score_out)
