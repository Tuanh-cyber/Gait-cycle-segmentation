"""Run BD004 preprocessing on one Modal L4 and evaluate the exported cache locally."""
import csv
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import perf_counter

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = modal.Volume.from_name("bd004-overground-l4", create_if_missing=True, version=2)
APP = modal.App("bd004-overground-l4")
REMOTE_CODE = PurePosixPath("/opt/gait-code")
VOLUME_ROOT = "/vol"
CONDITIONS = {"Walk_Comfortable": 6, "Walk_Fast": 8, "Walk_Slow": 8,
              "Run_Comfortable": 14, "Run_Fast": 14}
RUN_CONDITIONS = ("Run_Comfortable", "Run_Fast")
STAGES = ("step2_tracking", "step3_footkp", "step4_footmask", "step5_flow")
POSE_NAME = "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.onnx"
WEIGHTS = {
    "yolo11m.pt": "yolo11m.pt",
    "sam2.1_hiera_base_plus.pt": "sam2.1_hiera_base_plus.pt",
    "GMA/checkpoints/gma-sintel.pth": "gma-sintel.pth",
    f"overground-model-cache/hub/checkpoints/{POSE_NAME}": POSE_NAME,
}

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "pandas", "scipy", "opencv-python", "matplotlib", "einops", "ultralytics",
                 "rtmlib==0.0.16", "onnxruntime", "lap", "hydra-core", "iopath")
    .env({"SAM2_BUILD_CUDA": "0", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    .pip_install("git+https://github.com/facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4")
    .add_local_file(ROOT / "scripts/evaluate_overground.py", str(REMOTE_CODE / "scripts/evaluate_overground.py"))
    .add_local_dir(ROOT / ".third_party/GMA/core", str(REMOTE_CODE / ".third_party/GMA/core"),
                   ignore=["**/__pycache__/**", "*.pyc"])
)
for stage in STAGES:
    IMAGE = IMAGE.add_local_dir(ROOT / "data" / stage / "code", str(REMOTE_CODE / "data" / stage / "code"),
                                ignore=["**/__pycache__/**", "*.pyc"])


def source_video(dataset, condition):
    family = "Overground_Run" if condition in RUN_CONDITIONS else "Overground_Walk"
    return dataset / family / "BD004" / "Session1" / condition / "Video" / f"{condition}.avi"


def export_cache(work, destination):
    paths = ["passages.csv", "data/step0_audit/step0_metadata.csv",
             "data/step4_footmask/code", "data/step5_flow/code"]
    with (work / "passages.csv").open(newline="", encoding="utf-8") as manifest:
        for passage in csv.DictReader(manifest):
            vid = passage["video_id"]
            paths.extend([f"data/step0_audit/frames/{vid}_frames.csv",
                          f"data/step3_footkp/footkp/{vid}_footkp.npz",
                          f"data/step4_footmask/masks_post/{vid}.npz",
                          f"data/step5_flow/features/{vid}.npz",
                          f"data/step5_flow/flow_store/{vid}.npz"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    with tarfile.open(temp, "w:gz") as archive:
        for relative in paths:
            archive.add(work / relative, arcname=relative)
    temp.replace(destination)


@APP.function(image=IMAGE, gpu="L4", cpu=4, memory=16_384, timeout=3_600,
              volumes={VOLUME_ROOT: VOLUME})
def preprocess(condition: str, pass_count: int, key: str):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Modal container")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name:
        raise RuntimeError(f"Unexpected GPU: {device_name}")
    print(f"GPU: {device_name}; PyTorch: {torch.__version__}", flush=True)
    start = perf_counter()
    volume_root = Path(VOLUME_ROOT)
    work = volume_root / "work" / f"{key}_portable"
    source = volume_root / "input" / f"{condition}.avi"
    destination = volume_root / "exports" / f"{key}.tar.gz"
    with TemporaryDirectory() as directory:
        repo = Path(directory) / "repo"
        shutil.copytree(REMOTE_CODE, repo)
        for relative, name in WEIGHTS.items():
            link = repo / ".third_party" / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(volume_root / "weights" / name)
        subprocess.run(
            [sys.executable, "-u", str(repo / "scripts/evaluate_overground.py"),
             "--source", str(source), "--out", str(work), "--subject", "BD004",
             "--max-passages", str(pass_count), "--device", "cuda",
             "--stages", "prepare", "track", "keypoints", "trim", "masks", "mask_qc", "flow"],
            cwd=repo, check=True,
        )
        export_cache(work, destination)
    VOLUME.commit()
    print(f"{key}: finished {pass_count} passage(s) in {perf_counter() - start:.1f}s", flush=True)
    return str(destination.relative_to(volume_root))


def evaluate_local(condition, key, dataset, archive_path):
    target = ROOT / "outputs/bd004_modal" / key
    work, out = target / "work", target / "results"
    work.mkdir(parents=True, exist_ok=True)
    archive_file = target / "cache.tar.gz"
    with archive_file.open("wb") as stream:
        for chunk in VOLUME.read_file(archive_path):
            stream.write(chunk)
    with tarfile.open(archive_file) as archive:
        archive.extractall(work, filter="data")
    avi = source_video(dataset, condition)
    truth = dataset / "overground_ic_to_timestamps.csv"
    env = os.environ.copy()
    modules = [ROOT / ".third_party" / name for name in ("overground-runtime", "runtime-deps", "sam2")]
    env["PYTHONPATH"] = os.pathsep.join([*map(str, modules), env.get("PYTHONPATH", "")])
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-u", "scripts/evaluate_cyclic_overground.py", "--source", str(avi),
                    "--input", str(work), "--out", str(out), "--ground-truth", str(truth)],
                   cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "-u", "scripts/render_overground_review.py", "--source", str(avi),
                    "--evaluation", str(out), "--masks", str(work / "data/step4_footmask/masks_post"),
                    "--ground-truth", str(truth), "--intervals",
                    "--out", str(out / f"BD004_{key}_four_phase_review.mp4")],
                   cwd=ROOT, env=env, check=True)
    print(f"{key}: evaluation and video saved to {out}", flush=True)


@APP.local_entrypoint()
def main(dataset_directory: str = "D:/Datasets/Gait/Overground", benchmark_only: bool = False,
         condition: str = "", run_group: bool = False):
    dataset = Path(dataset_directory)
    if condition and condition not in CONDITIONS:
        raise ValueError(f"Unknown walking condition: {condition}")
    if sum(map(bool, (benchmark_only, condition, run_group))) > 1:
        raise ValueError("Choose a benchmark, a walking/running condition, or the Run group")
    if run_group:
        conditions = RUN_CONDITIONS
    elif benchmark_only:
        conditions = ("Walk_Comfortable",)
    else:
        conditions = (condition,) if condition else ("Walk_Comfortable", "Walk_Fast")
    inputs = {f"weights/{name}": ROOT / ".third_party" / relative for relative, name in WEIGHTS.items()}
    inputs.update({f"input/{condition}.avi": source_video(dataset, condition) for condition in conditions})
    stored = {entry.path.lstrip("/"): entry.size for entry in VOLUME.listdir("/", recursive=True)}
    missing = {remote: local for remote, local in inputs.items() if stored.get(remote) != local.stat().st_size}
    if missing:
        with VOLUME.batch_upload(force=True) as batch:
            for remote, local in missing.items():
                batch.put_file(local, remote)
    jobs = []
    if benchmark_only:
        jobs.append(("Walk_Comfortable", 1, "Walk_Comfortable_trial1"))
    if not benchmark_only:
        jobs.extend((walk, CONDITIONS[walk], walk) for walk in conditions)
    for walk, count, key in jobs:
        print(f"Starting {key} on Modal L4", flush=True)
        archive = preprocess.remote(walk, count, key)
        evaluate_local(walk, key, dataset, archive)
