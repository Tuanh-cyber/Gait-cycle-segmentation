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


def describe_path(branch, trial, side, frames, observed, path, occupancy, joint, marginal, fps):
    from cyclic_hsmm import EVENT_PAIRS, PHASES, boundary_interval, phase_interval

    spans, events, windows = [], [], []
    anchors = {pair: [] for pair in EVENT_PAIRS}
    for a, b, state in path:
        event = PHASES[state]
        spans.append(dict(branch=branch, trial=trial, side=side, phase=event,
                          start_frame=int(frames[a]), end_frame=int(frames[b-1]),
                          start_s=frames[a]/fps, end_exclusive_s=(frames[b-1]+1)/fps,
                          duration_s=(frames[b-1]+1-frames[a])/fps,
                          posterior_mean=float(np.mean(occupancy[a:b, state])),
                          left_censored=a == 0, right_censored=b == len(frames)))
        if event in ("IC", "TO"):
            lo, hi, prob = phase_interval(marginal, a, b, state, frames)
            missing = frames[(frames >= lo) & (frames <= hi) & ~observed]
            groups = np.split(missing, np.flatnonzero(np.diff(missing) != 1)+1)
            gaps = [(int(gap[0]), int(gap[-1])) for gap in groups if len(gap)]
            windows.append(dict(branch=branch, trial=trial, side=side, phase=event,
                                map_start_frame=int(frames[a]), map_end_frame=int(frames[b-1]),
                                start_p05_frame=lo, end_p95_frame=hi, anchor_probability=prob,
                                left_censored=a == 0, right_censored=b == len(frames), inferred_gaps=gaps))
        if event == "IC" and a > 0:
            anchors[EVENT_PAIRS[0]].append((a, a, b))
        elif event == "TO" and b < len(frames):
            anchors[EVENT_PAIRS[1]].append((b, a, b))
    for event, pair in zip(("IC", "TO"), EVENT_PAIRS):
        candidates = anchors[pair]
        indices = [index for index, _, _ in candidates]
        edges = [1, *((left+right)//2+1 for left, right in zip(indices, indices[1:])), len(frames)]
        for (index, a, b), start, stop in zip(candidates, edges, edges[1:]):
            weights = marginal.first_event_weights(pair, start, stop)
            lo, hi, mass = boundary_interval(weights, frames[start:stop], RANGE_LEVEL)
            events.append(dict(branch=branch, trial=trial, side=side, event=event, frame=int(frames[index]),
                               time_s=frames[index]/fps, confidence=float(joint[index, pair[0], pair[1]]),
                               phase_start_frame=int(frames[a]), phase_end_frame=int(frames[b-1]),
                               event_start_frame=lo, event_end_frame=hi,
                               event_start_s=lo/fps, event_end_s=hi/fps, interval_frames=hi-lo+1,
                               local_boundary_mass=mass, interval_level=RANGE_LEVEL,
                               flow_supported=bool(observed[index-1] and observed[index])))
    events.sort(key=lambda row: row["frame"])
    return spans, events, windows


def infer_pixels(base, payload, slots, cues, iterations, out):
    from cyclic_hsmm import (
        BRANCH, CyclicModel, EVENT_PAIRS, PHASES, fit_and_decode, make_runs,
    )
    from event_data import save_npz
    from hsmm_inputs import REGIONS
    from pixel_emission import pixel_moments

    moments = pixel_moments(payload)
    capture = base["audit"]
    source, group = base["f"]["frame"], base["groups"][0]
    fps = round(1 / np.median(np.diff(capture.pts_s)), 6)
    model = CyclicModel()
    table = np.full((len(capture), len(PHASES) + 2), np.nan)
    flow = np.zeros(len(capture), bool)
    observed_flow = np.zeros(len(capture), bool)
    inferred_phase = np.zeros(len(capture), bool)
    spans, events, windows = [], [], []
    training = make_runs(base, moments, cues)
    timeline = make_runs(base, moments, cues, include_missing=True)
    for run, path, marginal in fit_and_decode(model, training, iterations, inference_runs=timeline):
        ids = run["ids"]
        frames, trial = source[ids], int(group[ids[0]])
        if (np.diff(frames) != 1).any():
            raise ValueError("A continuous AVI run has a source-frame gap")
        occupancy = marginal.occupancy()
        joint = marginal.transitions_by_frame()
        table[frames, :len(PHASES)] = occupancy
        table[frames[1:], len(PHASES):] = np.column_stack([joint[1:, a, b] for a, b in EVENT_PAIRS])
        observed = run["observed"]
        flow[frames[1:]] = observed[:-1] & observed[1:]
        observed_flow[frames] = observed
        inferred_phase[frames] = ~observed
        this_spans, this_events, this_windows = describe_path(
            BRANCH, trial, slots[trial], frames, observed, path, occupancy, joint, marginal, fps
        )
        spans.extend(this_spans)
        events.extend(this_events)
        windows.extend(this_windows)
    branch_out = out / BRANCH
    branch_out.mkdir(exist_ok=True)
    duration_frames = np.arange(1, len(model.duration_probability)+1)
    save_npz(branch_out/"model.npz", coefficients=model.emission.coefficients, covariance=model.emission.covariance,
             phase_names=PHASES, regions=REGIONS, predictors=("intercept", "xi", "eta", "elapsed_s"),
             fps=fps, duration_frames=duration_frames, duration_counts=model.duration_counts,
             duration_probability=model.duration_probability, log_trans=model.log_trans, log_bos=model.log_bos)
    pd.concat([pd.DataFrame(dict(phase=phase, duration_frames=duration_frames, duration_s=duration_frames/fps,
                                 probability=model.duration_probability[:, i],
                                 complete_segment_count=model.duration_counts[:, i]))
               for i, phase in enumerate(PHASES)], ignore_index=True).to_csv(
                   branch_out/"duration_distribution.csv", index=False)
    columns = [f"{BRANCH}_p_phase_{p.lower()}" for p in PHASES] + [f"{BRANCH}_{e}_confidence" for e in ("ic","to")]
    confidence = pd.DataFrame(table, columns=columns)
    confidence[f"{BRANCH}_flow_supported"] = flow
    confidence[f"{BRANCH}_flow_observed"] = observed_flow
    confidence[f"{BRANCH}_phase_inferred_without_flow"] = inferred_phase
    return confidence, spans, events, windows


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


def plot_ranges(out, passages, spans, phase_windows, truth, branches, fps, condition):
    fig = Figure(figsize=(7*len(branches), max(5, 1.4*len(passages)+1.4)), layout="constrained")
    axes = fig.subplots(len(passages), len(branches), squeeze=False)
    for row, passage in enumerate(passages.itertuples()):
        trial = passage.trial
        ref = truth[truth.trial.eq(trial)]
        for col, branch in enumerate(branches):
            ax = axes[row, col]
            part = spans[spans.branch.eq(branch) & spans.trial.eq(trial)]
            windows = phase_windows[phase_windows.branch.eq(branch) & phase_windows.trial.eq(trial)]
            ax.axhspan(.60, .93, color="#f0f2f4")
            for phase, segments in part.groupby("phase"):
                ax.broken_barh([(s.start_s, s.end_exclusive_s-s.start_s) for s in segments.itertuples()], (.60,.33),
                               facecolors=COLORS[phase], alpha=.80)
            for interval in windows.itertuples():
                color = COLORS[interval.phase]
                lane = (.31, .11) if interval.phase == "IC" else (.18, .11)
                start = interval.start_p05_frame
                width = interval.end_p95_frame-start+1
                ax.broken_barh([((start-.5)/fps, width/fps)], lane, facecolors=color, alpha=.72)
                for lo, hi in interval.inferred_gaps:
                    ax.broken_barh([((lo-.5)/fps, (hi-lo+1)/fps)], lane,
                                   facecolors="#f0f2f4", edgecolors=color, hatch="///", linewidth=.6)
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
                 "\nTop: phase spans   ·   Bottom: IC/TO phase intervals (5–95% edges)")
    fig.legend(handles=[Patch(facecolor=color,label=phase) for phase,color in COLORS.items()]
               + [Line2D([0],[0], color="#505050", ls="--", label="CSV event"),
                  Patch(facecolor="#f0f2f4", edgecolor="#505050", hatch="///", label="Inferred without flow")],
               loc="outside lower center", ncols=6, frameon=False)
    fig.savefig(out / "phase_uncertainty.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Existing overground output with Step 5 caches")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject", default="BD004")
    parser.add_argument("--session", default="Session1")
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    if args.iterations < 1 or args.input.resolve() == args.out.resolve():
        parser.error("Use positive iterations and set aside a separate output directory")
    sys.path.insert(0, str(ROOT/"data/step6_fixed_foot_events/code"))
    from cyclic_hsmm import BRANCH, PHASE_QUANTILES, flow_evidence

    base, payload, selected = load_inference_data(args.source, args.input, args.subject, reuse_clouds=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input/"passages.csv", out/"passages.csv")
    pd.DataFrame(selected).to_csv(out/"foot_selection.csv", index=False)
    slots = {item["trial"]:item["side"] for item in selected}
    cues = flow_evidence(payload["raw42"], base["valid"])
    print(f"Fitting {args.subject} {args.source.stem} using regional pixel sets", flush=True)
    confidence, spans, events, windows = infer_pixels(base, payload, slots, cues, args.iterations, out)
    branches = (BRANCH,)
    audit = base["audit"].rename(columns={"pts_s":"source_pts_s"})
    pd.concat([audit, confidence], axis=1).to_csv(out/"event_confidence_frames.csv", index=False)
    spans = pd.DataFrame(spans, columns=["branch","trial","side","phase","start_frame","end_frame","start_s","end_exclusive_s",
                                        "duration_s","posterior_mean","left_censored","right_censored"])
    spans.to_csv(out/"phase_intervals.csv", index=False)
    windows = pd.DataFrame(windows, columns=[
        "branch", "trial", "side", "phase", "map_start_frame", "map_end_frame", "start_p05_frame",
        "end_p95_frame", "anchor_probability", "left_censored", "right_censored", "inferred_gaps",
    ])
    windows.assign(inferred_gaps=windows.inferred_gaps.map(json.dumps)).to_csv(
        out/"phase_uncertainty_intervals.csv", index=False
    )
    point_cols = ["branch","trial","side","event","frame","time_s","confidence"]
    events = pd.DataFrame(events, columns=[*point_cols,"phase_start_frame","phase_end_frame","event_start_frame","event_end_frame",
                                          "event_start_s","event_end_s","interval_frames","local_boundary_mass","interval_level","flow_supported"])
    # MAP boundary instances permit comparison with the existing point benchmark.
    events[point_cols].to_csv(out/"predictions.csv", index=False)
    score(args.ground_truth, out, args.source.stem, subject=args.subject, session=args.session, branches=branches, plot=False)
    truth = pd.read_csv(out/"ground_truth_selected_foot.csv")
    metrics, events = evaluate_ranges(events, truth, branches)
    metrics.to_csv(out/"interval_metrics.csv", index=False)
    events.to_csv(out/"event_intervals.csv", index=False)
    passages = pd.read_csv(out/"passages.csv")
    plot_ranges(out, passages, spans, windows, truth, branches, passages.fps.iloc[0], args.source.stem)
    (out/"interval_settings.json").write_text(json.dumps(
        dict(emission="pixel_set_conditional_gaussian", pixel_scores="mean within each region",
             duration_model="learned_discrete_by_phase", duration_fit="continuous runs pooled per video",
             boundary_window="midpoints between same-type Viterbi anchors within each run",
             interval_event="first same-type transition in window", interval_level=RANGE_LEVEL,
             boundary_definition=dict(IC="Swing -> IC", TO="TO -> Swing"), training="flow only; ground truth used for scoring",
             phase_edge_quantiles=PHASE_QUANTILES, phase_anchor="Viterbi segment midpoint",
             missing_flow="neutral log emission and cues; visualized inside structurally continuous runs",
             training_runs="contiguous observed spans",
             missing_flow_reference="Yu and Kobayashi (2003), equations 6, 10 and 14",
             interval_calibrated=False, max_iterations=args.iterations), indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
