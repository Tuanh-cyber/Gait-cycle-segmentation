"""Two-method event/phase videos using a fixed original Step5 foot slot."""
import os
for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[thread_variable] = "2"

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

from event_data import DATA, PHASE_NAMES, load_video, make_events, save_npz, sha, write_json
from frechet_cpd import detect
from rff_hsmm import Model, infer_video, train
from render_events import phase_plot, render


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(args):
    cv2.setNumThreads(2)
    out = args.out
    for folder in ("videos", "posters", "plots", "events", "phases", "records", "posterior"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA / "step0_audit/step0_metadata.csv", dtype=str, keep_default_na=False)
    requested = args.videos or sorted(metadata.video_id.tolist())
    if not set(requested).issubset(metadata.video_id) or len(set(requested)) != len(requested):
        raise ValueError("Choose distinct metadata video_id values, not __v2 feature filenames")
    selection = {}
    if args.foot_selection:
        table = pd.read_csv(args.foot_selection)
        if table.video_id.duplicated().any():
            raise ValueError("Duplicate video in fixed foot selection")
        selection = dict(zip(table.video_id, table.foot_slot.astype(int)))
    videos = [load_video(metadata.set_index("video_id", drop=False).loc[vid], selection.get(vid, args.foot_slot)) for vid in requested]
    code_hashes = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
    config = dict(seed=args.seed, classes=args.classes, rff=args.rff, sweeps=args.sweeps, bootstrap=args.bootstrap,
                  cpd_alpha=args.alpha, cpd_window_s=args.window, min_duration_s=args.min_duration, max_duration_s=args.max_duration,
                  phase_switch_penalty=args.phase_switch_penalty, cpd_bootstrap_block_length=args.block_length)
    provenance = dict(config=config, code_hashes=code_hashes, inputs={v["video_id"]: v["inputs"] for v in videos},
                      foot_slots={v["video_id"]: v["slot"] for v in videos})
    frozen_info = None
    if args.model_from:
        source_manifest = json.loads((args.model_from / "manifest.json").read_text(encoding="utf-8"))
        expected = source_manifest["provenance"]["config"]
        for key in ("classes", "rff", "min_duration_s", "max_duration_s", "phase_switch_penalty"):
            if expected[key] != config[key]:
                raise ValueError(f"Frozen-model setting differs: {key}; use {expected[key]}")
        source_training = source_manifest.get("frozen_model") or source_manifest["provenance"].get("frozen_model") or {}
        frozen_info = dict(path=str(args.model_from.resolve()), model_sha256=sha(args.model_from / "model.npz"),
                           manifest_sha256=sha(args.model_from / "manifest.json"),
                           training_videos=source_training.get("training_videos", list(source_manifest["provenance"]["inputs"])), refit=False)
        provenance["frozen_model"] = frozen_info
    write_json(out / "run_status.json", dict(status="running", videos=requested, tracker_bypassed=True))
    cached_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").exists() else {}
    model = Model(classes=args.classes, rff=args.rff, seed=args.seed, min_duration=args.min_duration,
                  max_duration=args.max_duration, phase_penalty=args.phase_switch_penalty)
    if args.model_from:
        with np.load(args.model_from / "model.npz", allow_pickle=False) as z:
            for name in ("omega", "offset", "cov", "weight", "duration_mean", "duration_sd", "phase_labels", "slow_means", "log_trans", "log_bos"):
                setattr(model, name, z[name])
        if model.weight.shape != (args.classes, 16, args.rff + 1):
            raise ValueError("Frozen model dimensions differ from selected regional features")
        trace = []
        print(f"Frozen HSMM loaded from {args.model_from}; no refitting on requested videos", flush=True)
        save_npz(out / "model.npz", **{name: getattr(model, name) for name in
                 ("omega", "offset", "cov", "weight", "duration_mean", "duration_sd", "phase_labels", "slow_means", "log_trans", "log_bos")})
    elif args.resume and cached_manifest.get("provenance") == provenance and (out / "model.npz").exists():
        with np.load(out / "model.npz", allow_pickle=False) as z:
            for name in ("omega", "offset", "cov", "weight", "duration_mean", "duration_sd", "phase_labels", "slow_means", "log_trans", "log_bos"):
                setattr(model, name, z[name])
        trace = cached_manifest.get("training_trace", [])
        print("Reusing matching fitted HSMM", flush=True)
    else:
        _, trace, stats, centers = train(videos, model, sweeps=args.sweeps)
        save_npz(out / "model.npz", **{name: getattr(model, name) for name in
                 ("omega", "offset", "cov", "weight", "duration_mean", "duration_sd", "phase_labels", "slow_means", "log_trans", "log_bos")})
        write_json(out / "training_trace.json", trace)
    rows, all_events, foot_rows = [], [], []
    for order, v in enumerate(videos):
        started = time.monotonic()
        vid = v["video_id"]
        record_path = out / f"records/{vid}.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
        matching_record = record.get("provenance") == provenance
        if args.resume and matching_record and (out / f"videos/{vid}.mp4").exists() and not args.no_video:
            rows.append(record["summary"])
            all_events += pd.read_csv(out / f"events/{vid}.csv").to_dict("records")
            foot_rows.append(record["foot_selection"])
            print(f"{order+1}/{len(videos)} {vid}: matching video reused", flush=True)
            continue
        f_phase, cuts = detect(v, seed=args.seed + order, bootstrap=args.bootstrap, alpha=args.alpha,
                               window_s=args.window, min_duration=args.min_duration, block_length=args.block_length)
        h = infer_video(v, model)
        phases = dict(frechet=f_phase, hsmm=h["phase"])
        f_events = make_events(v, "frechet", f_phase, cuts.keys(), {i: cuts[i]["p"] for i in cuts})
        h_events = make_events(v, "hsmm", h["phase"], h["cuts"], {i: h["boundary_probability"][i] for i in h["cuts"]})
        events = f_events + h_events
        for event in h_events:
            i = event["source_index"]
            event.update(posterior_IC=float(h["IC_probability"][i]), posterior_TO=float(h["TO_probability"][i]))
        event_columns = ["video_id", "method", "foot_slot", "source_index", "frame", "t_real_s", "source_pts_s", "type", "phase_before", "phase_after", "reason", "slow_before", "slow_after", "strong_contact_evidence", "boundary_evidence", "abs_time_ok", "candidate", "posterior_IC", "posterior_TO"]
        pd.DataFrame(events, columns=event_columns).to_csv(out / f"events/{vid}.csv", index=False)
        save_npz(out / f"posterior/{vid}.npz", frame=v["f"]["frame"], state_probability=h["state_probability"],
                 boundary_probability=h["boundary_probability"], IC_probability=h["IC_probability"], TO_probability=h["TO_probability"],
                 phase_labels=model.phase_labels)
        save_npz(out / f"phases/{vid}.npz", frame=v["f"]["frame"], t_real_s_v2=v["f"]["t_real_s_v2"], foot_slot=np.array(v["slot"]),
                 valid=v["valid"], frechet=f_phase, hsmm=h["phase"], latent=h["latent"],
                 input_x_raw=v["raw"], scale_center=v["scale_center"], scale=v["scale"])
        pd.DataFrame(dict(frame=v["f"]["frame"], t_real_s_v2=v["f"]["t_real_s_v2"], input_valid=v["valid"],
                          frechet=[PHASE_NAMES[int(p)] for p in f_phase], hsmm=[PHASE_NAMES[int(p)] for p in h["phase"]],
                          hsmm_latent=h["latent"])).to_csv(out / f"phases/{vid}.csv", index=False)
        phase_plot(out / f"plots/{vid}.png", v, phases, events)
        verification = None if args.no_video else render(out / f"videos/{vid}.mp4", out / f"posters/{vid}.jpg", v, phases, events)
        summary = dict(video_id=vid, foot_slot=v["slot"], n_observations=len(v["valid"]), n_input_valid=int(v["valid"].sum()),
                       foot_selection=v["selection"], abs_time_ok=v["abs_time_ok"])
        for method, phase in phases.items():
            for kind in ("IC", "TO", "UNKNOWN"):
                summary[f"{method}_{kind}"] = sum(e["method"] == method and e["type"] == kind for e in events)
            summary[f"{method}_phase_coverage"] = float((phase >= 0).mean())
        foot_row = dict(video_id=vid, foot_slot=v["slot"], slot_name="A" if v["slot"] == 0 else "B", selection=v["selection"],
                        initial_label=str(v["f"]["label"][0, v["slot"]]), tracker_bypassed=True)
        write_json(record_path, clean_json(dict(provenance=provenance, summary=summary, foot_selection=foot_row,
                                                verification=verification, frechet_boundaries=cuts, hsmm_segments=h["segments"])))
        all_events += events
        rows.append(summary)
        foot_rows.append(foot_row)
        print(f"{order+1}/{len(videos)} {vid} slot{v['slot']}: F IC/TO {summary['frechet_IC']}/{summary['frechet_TO']}; H {summary['hsmm_IC']}/{summary['hsmm_TO']}; {time.monotonic()-started:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(foot_rows).to_csv(out / "foot_selection.csv", index=False)
    pd.DataFrame(all_events).to_csv(out / "events_all.csv", index=False)
    write_json(out / "manifest.json", clean_json(dict(provenance=provenance, tracker_bypassed=True, input_source="original Step5 V1, fixed slot per video",
               n_videos=len(rows), videos_rendered=not args.no_video, training_trace=trace, hsmm_class_phase=model.phase_labels.tolist(),
               hsmm_class_slow_means=model.slow_means.tolist(), ground_truth_available=False, frozen_model=frozen_info,
               phase_mapping="max regional slow ratio -> unsupervised low/high contact regimes; heuristic biological semantics")))
    write_json(out / "run_status.json", dict(status="complete", n_videos=len(rows), videos_rendered=not args.no_video, tracker_bypassed=True))
    print(f"Complete: {len(rows)} videos; single-foot detector bypassed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+")
    parser.add_argument("--out", type=Path, default=DATA / "step6_fixed_foot_events")
    parser.add_argument("--foot-slot", type=int, choices=[0, 1])
    parser.add_argument("--foot-selection", type=Path)
    parser.add_argument("--model-from", type=Path, help="Reuse a frozen HSMM from another result folder; never refit on requested videos")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--rff", type=int, default=12)
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=99)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--window", type=float, default=.6)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--min-duration", type=float, default=.08)
    parser.add_argument("--max-duration", type=float, default=.75)
    parser.add_argument("--phase-switch-penalty", type=float, default=4.)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.bootstrap < 19 or arguments.block_length < 1 or not 0 < arguments.alpha < 1 or not 0 < arguments.min_duration < arguments.max_duration or arguments.sweeps < 1 or arguments.classes < 2:
        parser.error("Invalid bootstrap, duration, class or training parameters")
    main(arguments)
