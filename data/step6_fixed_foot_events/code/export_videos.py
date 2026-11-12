"""Export selected videos from saved inference, without fitting either model again."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from event_data import DATA, ROOT, load_video, sha, write_json
from render_events import phase_plot, render


def main(args):
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(DATA / "step0_audit/step0_metadata.csv", dtype=str, keep_default_na=False).set_index("video_id", drop=False)
    args.out.mkdir(parents=True, exist_ok=True)
    for vid in args.videos:
        expected = manifest["provenance"]["inputs"][vid]
        for source in expected.values():
            if sha(ROOT / source["path"]) != source["sha256"]:
                raise ValueError(f"{vid}: source input changed")
        slot = manifest["provenance"]["foot_slots"][vid]
        v = load_video(metadata.loc[vid], slot)
        with np.load(args.results / f"phases/{vid}.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["frame"], v["f"]["frame"])
            np.testing.assert_array_equal(saved["valid"], v["valid"])
            np.testing.assert_allclose(saved["input_x_raw"], v["raw"], equal_nan=True)
            if int(saved["foot_slot"]) != slot:
                raise ValueError("Different foot in stored inference")
            phases = {name: saved[name] for name in ("frechet", "hsmm")}
        events = pd.read_csv(args.results / f"events/{vid}.csv").to_dict("records")
        verification = render(args.out / f"{vid}.mp4", args.out / f"{vid}.jpg", v, phases, events)
        phase_plot(args.out / f"{vid}_plot.png", v, phases, events)
        shutil.copyfile(args.results / f"events/{vid}.csv", args.out / f"{vid}_events.csv")
        shutil.copyfile(args.results / f"phases/{vid}.csv", args.out / f"{vid}_phases.csv")
        counts = {method: {kind: sum(e["method"] == method and e["type"] == kind for e in events) for kind in ("IC", "TO", "UNKNOWN")} for method in phases}
        write_json(args.out / f"{vid}_export.json", dict(video_id=vid, foot_slot=slot, input_hashes=v["inputs"],
                   inference_manifest_sha256=sha(args.results / "manifest.json"), model_config=manifest["provenance"]["config"],
                   model_training_videos=list(manifest["provenance"]["inputs"]), hsmm_class_phase=manifest["hsmm_class_phase"],
                   hsmm_class_slow_means=manifest["hsmm_class_slow_means"], event_counts=counts,
                   verification=verification, renderer_sha256=sha(Path(__file__).parent / "render_events.py"),
                   tracker_bypassed=True, ground_truth_available=False))
        print(f"{vid}: exported {verification['output_frames']} source-timed frames; {counts}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
