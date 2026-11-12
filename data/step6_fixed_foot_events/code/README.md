# Initial stance/swing detection from a fixed Step5 foot

Read original regional GMA features from `data/step5_flow/features/<video_id>.npz`. Select one A/B slot once per video; both Fréchet CPD and RFF-GP-HSMM use that same slot. The experimental single-foot tracker is not imported or executed.

## Setup

Run commands from the repository root with Python 3.11. Install FFmpeg separately so `ffmpeg` and `ffprobe` are on PATH.

```powershell
py -3.11 -m pip install -r data/step6_fixed_foot_events/code/requirements.txt
```

Source videos and the Step0/3/4/5 caches are the upstream inputs already present in this repository. The delivered [three samples](../three_samples/README.md) include a frozen `model.npz`; no trial cache is needed to reproduce them.

## Reproduce the three samples with the supplied model

```powershell
py -3.11 data/step6_fixed_foot_events/code/run_events.py --videos C13_1 C38_8 F10_2 --classes 2 --phase-switch-penalty 0 --model-from data/step6_fixed_foot_events/three_samples --out data/step6_fixed_foot_events/reproduced_samples
```

This keeps the delivered model frozen and writes to a separate output folder. It does not fit on the three requested videos. The frozen model was trained without event labels on C30, F29, F02 and C38_9. Use `--foot-slot 0` / `--foot-slot 1`, or `--foot-selection` with columns `video_id,foot_slot`, to explicitly select a different fixed upstream slot.

To fit a new model instead, omit `--model-from`. `--no-video` saves inference without rendering; `--resume` reuses matching results. Defaults for a new fit are configurable: explicitly use `--classes 2 --phase-switch-penalty 0` to match the delivered sample configuration.

## Modules

| File | Purpose |
|---|---|
| `event_data.py` | Fixed-slot adapter, validity gates, slow-regime phase mapping and candidate events |
| `frechet_cpd.py` | Local regional Fréchet mean/variance scans, bootstrap and segment phase interpretation |
| `rff_hsmm.py` | RFF Bayesian GP segment emissions, explicit durations, blocked FFBS, Viterbi and conditional posterior |
| `mask_overlay.py` | Decode actual Step4 heel/mid/toe RLE masks for display |
| `render_events.py` | Source-timed MP4, two phase bars, IC/TO markers and mask overlay |
| `run_events.py` | Fit or load a frozen model, infer events and export results |
| `export_videos.py` | Render selected videos from already saved inference |

## Interpretation

STANCE/SWING are heuristic interpretations of motion regimes, not verified contact labels. UNKNOWN is an abstention/missing-data state, not an additional learned HSMM class. Data gaps, cuts and invalid flow interrupt inference; candidate events are not joined across those gaps. A statistical or latent boundary within one phase is recorded as event type UNKNOWN.

The display uses phase colors green/blue/gray and distinct mask region colors. It preserves source frame presentation timestamps and applies source display rotation. It does not include audio.

This code adapts the [Fréchet change-point statistic](https://anson.ucdavis.edu/~mueller/Frechet20.pdf) to Euclidean regional flow features and the [RFF-GP-HSMM idea](https://github.com/naka-lab/RFF-GP-HSMM) to missing regional coordinates and durations in seconds. Local bootstrap p-values are not biological event probabilities; posterior probabilities are conditional on the fitted model and phase mapping. There is no ground-truth accuracy evaluation in these samples.
