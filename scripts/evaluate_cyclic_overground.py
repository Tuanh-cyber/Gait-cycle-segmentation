"""Estimate four cyclic gait phases and IC/TO frame intervals on overground AVI."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from evaluate_overground import ROOT, load_inference_data, score

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RANGE_LEVEL = .90
COLORS = {"IC": "#ad3a84", "Flat": "#43956a", "TO": "#b76814", "Swing": "#417aaa"}


def describe_path(branch, trial, side, frames, path, occupancy, boundaries, fps, min_cycle, phases):
    from cyclic_hsmm import boundary_interval

    spans, events = [], []
    for a, b, state in path:
        event = phases[state]
        spans.append(dict(branch=branch, trial=trial, side=side, phase=event,
                          start_frame=int(frames[a]), end_frame=int(frames[b-1]),
                          start_s=frames[a]/fps, end_exclusive_s=(frames[b-1]+1)/fps,
                          duration_s=(frames[b-1]+1-frames[a])/fps,
                          posterior_mean=float(np.mean(occupancy[a:b, state])),
                          left_censored=a == 0, right_censored=b == len(frames)))
        if event == "IC" and a > 0:
            index, which = a, 0
        elif event == "TO" and b < len(frames):
            index, which = b, 1
        else:
            continue
        weights = boundaries[which]
        lo, hi, mass = boundary_interval(weights, frames, index, fps, min_cycle, RANGE_LEVEL)
        events.append(dict(branch=branch, trial=trial, side=side, event=event, frame=int(frames[index]),
                           time_s=frames[index]/fps, confidence=float(weights[index]),
                           phase_start_frame=int(frames[a]), phase_end_frame=int(frames[b-1]),
                           event_start_frame=lo, event_end_frame=hi,
                           event_start_s=lo/fps, event_end_s=hi/fps, interval_frames=hi-lo+1,
                           local_boundary_mass=mass, interval_level=RANGE_LEVEL))
    return spans, events


def infer_branch(base, payload, slots, cues, size, durations, iterations, out, evidence_seconds):
    from cyclic_hsmm import CyclicModel, PHASES, fit_and_decode, make_runs
    from event_data import save_npz
    from hsmm_inputs import InputTransform

    branch = f"embedding{size*3}"
    transform = InputTransform(size).fit(payload, np.flatnonzero(base["valid"]))
    model_input = transform.transform(payload)
    capture = base["audit"]
    source, group = base["f"]["frame"], base["groups"][0]
    fps = round(1 / np.median(np.diff(capture.pts_s)), 6)
    model = CyclicModel(size*3, durations, observation_weight=min(1., 1/(fps*evidence_seconds)))
    table = np.full((len(capture), len(PHASES) + 2), np.nan)
    flow = np.zeros(len(capture), bool)
    spans, events = [], []
    for run, path, occupancy, boundaries in fit_and_decode(model, make_runs(base, model_input, cues), iterations):
        ids = run["ids"]
        frames, trial = source[ids], int(group[ids[0]])
        if (np.diff(frames) != 1).any():
            raise ValueError("A continuous AVI run has a source-frame gap")
        table[frames, :len(PHASES)] = occupancy
        table[frames[1:], len(PHASES):] = np.column_stack(boundaries)[1:]
        flow[frames[1:]] = True
        this_spans, this_events = describe_path(branch, trial, slots[trial], frames, path, occupancy,
                                               boundaries, fps, float(model.minimum.sum()), PHASES)
        spans.extend(this_spans)
        events.extend(this_events)
    branch_out = out / branch
    branch_out.mkdir(exist_ok=True)
    transform.save(branch_out/"input_transform.npz")
    save_npz(branch_out/"model.npz", omega=model.omega, offset=model.offset, cov=model.cov, weight=model.weight,
             phase_names=PHASES, minimum_s=model.minimum, median_s=model.median, maximum_s=model.maximum,
             duration_mean=model.duration_mean, duration_sd=model.duration_sd, log_trans=model.log_trans,
             log_bos=model.log_bos, likelihood_weight=model.likelihood_weight,
             observation_weight=model.observation_weight, noise_variance=model.noise)
    columns = [f"{branch}_p_phase_{p.lower()}" for p in PHASES] + [f"{branch}_{e}_confidence" for e in ("ic","to")]
    confidence = pd.DataFrame(table, columns=columns)
    confidence[f"{branch}_flow_supported"] = flow
    return confidence, spans, events


def evaluate_ranges(events, truth, branches):
    rows = []
    annotated = events.assign(gt_time_s=np.nan, matched=False)
    for branch in branches:
        for kind in ("IC", "TO"):
            refs = truth[truth.event.eq(kind)]
            ranges = annotated[annotated.branch.eq(branch) & annotated.event.eq(kind)]
            for trial, target in refs.groupby("trial"):
                candidates = ranges[ranges.trial.eq(trial)]
                if candidates.empty:
                    continue
                distance = np.abs(np.subtract.outer(candidates.time_s.to_numpy(), target.time_s.to_numpy()))
                frames = target.nearest_frame.to_numpy()
                inside = ((frames >= candidates.event_start_frame.to_numpy()[:, None])
                          & (frames <= candidates.event_end_frame.to_numpy()[:, None]))
                i, j = linear_sum_assignment(np.where(inside, distance, 1000.))
                valid = inside[i, j]
                annotated.loc[candidates.index[i[valid]], "matched"] = True
                annotated.loc[candidates.index[i[valid]], "gt_time_s"] = target.time_s.iloc[j[valid]].to_numpy()
            hits = int(annotated.loc[ranges.index, "matched"].sum())
            rows.append(dict(branch=branch, event=kind, contained=hits, predicted=len(ranges), references=len(refs),
                             precision=hits/max(1, len(ranges)), recall=hits/max(1, len(refs)),
                             median_width_frames=ranges.interval_frames.median()))
    return pd.DataFrame(rows), annotated


def plot_ranges(out, passages, spans, events, truth, branches, fps, condition):
    fig = Figure(figsize=(7*len(branches), max(5, 1.4*len(passages)+1.4)), layout="constrained")
    axes = fig.subplots(len(passages), len(branches), squeeze=False)
    for row, passage in enumerate(passages.itertuples()):
        trial = passage.trial
        ref = truth[truth.trial.eq(trial)]
        for col, branch in enumerate(branches):
            ax = axes[row, col]
            part = spans[spans.branch.eq(branch) & spans.trial.eq(trial)]
            windows = events[events.branch.eq(branch) & events.trial.eq(trial)]
            ax.axhspan(.60, .93, color="#f0f2f4")
            for phase, segments in part.groupby("phase"):
                ax.broken_barh([(s.start_s, s.end_exclusive_s-s.start_s) for s in segments.itertuples()], (.60,.33),
                               facecolors=COLORS[phase], alpha=.80)
            for interval in windows.itertuples():
                color = COLORS[interval.event]
                ax.broken_barh([(interval.event_start_s-.5/fps, interval.interval_frames/fps)], (.18,.24),
                               facecolors=color, edgecolors=color, alpha=.72 if interval.matched else .36)
            for event, group in ref.groupby("event"):
                ax.vlines(group.time_s, .08, .97, colors=COLORS[event], linestyles="--", linewidth=1.05)
            x_min = min(passage.start_s, ref.time_s.min()) - .04
            x_max = max(passage.stop_s, ref.time_s.max()) + .04
            ax.set(xlim=(x_min,x_max), ylim=(0,1), yticks=[], ylabel=f"Pass {trial} / {ref.side.iloc[0]}")
            ax.grid(axis="x", alpha=.22)
            if row == 0:
                ax.set_title(branch)
            if row == len(passages)-1:
                ax.set_xlabel("Source AVI time (s)")
    fig.suptitle(f"{condition.replace('_',' ')}  |  Four-state HSMM"
                 "\nTop: phase spans   ·   Bottom: IC/TO boundary intervals")
    fig.legend(handles=[Patch(facecolor=color,label=phase) for phase,color in COLORS.items()]
               + [Line2D([0],[0], color="#505050", ls="--", label="CSV event")],
               loc="outside lower center", ncols=5, frameon=False)
    fig.savefig(out / "phase_intervals.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Existing overground output with Step 5 caches")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject", default="BD004")
    parser.add_argument("--session", default="Session1")
    parser.add_argument("--embedding-per-region", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--durations", type=Path, help="JSON object mapping phase to [min, median, max] seconds")
    parser.add_argument("--evidence-seconds", type=float, default=.10, help="Seconds per effective flow observation")
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    sizes = tuple(dict.fromkeys(args.embedding_per_region))
    if (any(size < 1 for size in sizes) or args.iterations < 1 or args.input.resolve() == args.out.resolve()
            or not np.isfinite(args.evidence_seconds) or args.evidence_seconds <= 0):
        parser.error("Use positive sizes and iterations, and set aside a separate output directory")
    sys.path.insert(0, str(ROOT/"data/step6_fixed_foot_events/code"))
    from cyclic_hsmm import OVERGROUND_DURATIONS, WALK_DURATIONS, flow_evidence

    base, payload, selected = load_inference_data(args.source, args.input, args.subject, reuse_clouds=True)
    defaults = OVERGROUND_DURATIONS.get(args.source.stem, WALK_DURATIONS)
    durations = json.loads(args.durations.read_text(encoding="utf-8")) if args.durations else defaults
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input/"passages.csv", out/"passages.csv")
    pd.DataFrame(selected).to_csv(out/"foot_selection.csv", index=False)
    slots = {item["trial"]:item["side"] for item in selected}
    cues = flow_evidence(payload["raw42"], base["valid"])
    spans, events = [], []
    confidence = [base["audit"].rename(columns={"pts_s":"source_pts_s"})]
    for size in sizes:
        print(f"Fitting {args.subject} {args.source.stem} with {size*3} embedding dimensions", flush=True)
        scores, intervals, candidate = infer_branch(base, payload, slots, cues, size, durations, args.iterations,
                                                    out, args.evidence_seconds)
        confidence.append(scores)
        spans.extend(intervals)
        events.extend(candidate)
    branches = tuple(f"embedding{size*3}" for size in sizes)
    pd.concat(confidence, axis=1).to_csv(out/"event_confidence_frames.csv", index=False)
    spans = pd.DataFrame(spans, columns=["branch","trial","side","phase","start_frame","end_frame","start_s","end_exclusive_s",
                                        "duration_s","posterior_mean","left_censored","right_censored"])
    spans.to_csv(out/"phase_intervals.csv", index=False)
    point_cols = ["branch","trial","side","event","frame","time_s","confidence"]
    events = pd.DataFrame(events, columns=[*point_cols,"phase_start_frame","phase_end_frame","event_start_frame","event_end_frame",
                                          "event_start_s","event_end_s","interval_frames","local_boundary_mass","interval_level"])
    # MAP boundary instances permit comparison with the existing point benchmark.
    events[point_cols].to_csv(out/"predictions.csv", index=False)
    score(args.ground_truth, out, args.source.stem, subject=args.subject, session=args.session, branches=branches, plot=False)
    truth = pd.read_csv(out/"ground_truth_selected_foot.csv")
    metrics, events = evaluate_ranges(events, truth, branches)
    metrics.to_csv(out/"interval_metrics.csv", index=False)
    events.to_csv(out/"event_intervals.csv", index=False)
    passages = pd.read_csv(out/"passages.csv")
    plot_ranges(out, passages, spans, events, truth, branches, passages.fps.iloc[0], args.source.stem)
    (out/"interval_settings.json").write_text(json.dumps(
        dict(durations_s=durations, duration_order=["min","median","max"], interval_level=RANGE_LEVEL,
             boundary_definition=dict(IC="Swing -> IC", TO="TO -> Swing"), training="flow only; ground truth used for scoring",
             interval_calibrated=False, seconds_per_effective_observation=args.evidence_seconds,
             max_iterations=args.iterations), indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
