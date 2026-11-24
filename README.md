# Gait cycle segmentation — IC/TO từ optical flow một chân cố định

Step 6 dùng HSMM bốn pha tuần hoàn **IC → Flat → TO → Swing** với emission nhận tập pixel. Mô hình chấm vận tốc đã bù của từng pixel theo vị trí trong ba vùng bàn chân. Kết quả gồm khoảng pha, khoảng mốc IC/TO, điểm theo frame và video đối chứng với CSV.

## Pipeline

| Bước | Vai trò | Mã nguồn |
|---|---|---|
| 0–1 | Audit, shot/continuity, thời gian slow-motion | `data/step0_audit`, `data/step1_wu` |
| 2–3 | Tracking người, keypoint và chọn slot chân | `data/step2_tracking`, `data/step3_footkp` |
| 4 | Foot mask theo vùng heel/mid/toe | `data/step4_footmask` |
| 5 | Dense optical flow GMA và bù chuyển động camera | `data/step5_flow` |
| 6 | HSMM bốn pha với emission tập pixel và khoảng IC/TO | `data/step6_fixed_foot_events/code` |

## Chạy trên video Overground

Từ thư mục gốc, với Python 3.11 và FFmpeg/ffprobe trên PATH. Chuẩn bị model cho Bước 2–5 theo hướng dẫn bên dưới. Runner Overground đọc checkpoint ở `.third_party/yolo11m.pt`, `.third_party/sam2.1_hiera_base_plus.pt` và `.third_party/GMA`.

```powershell
py -3.11 -m pip install -r requirements-step6.txt
$dataset = 'D:/Datasets/Gait/Overground'
$avi = "$dataset/Overground_Walk/BD004/Session1/Walk_Slow/Video/Walk_Slow.avi"
$gt = "$dataset/overground_ic_to_timestamps.csv"
$work = 'outputs/reproduction/BD004/Walk_Slow_work'
$out = 'outputs/reproduction/BD004/Walk_Slow'
py -3.11 scripts/evaluate_overground.py --source $avi --out $work --subject BD004 --stages prepare track keypoints trim masks mask_qc flow
py -3.11 scripts/evaluate_cyclic_overground.py --source $avi --input $work --out $out --ground-truth $gt
py -3.11 scripts/render_overground_review.py --source $avi --evaluation $out --masks "$work/data/step4_footmask/masks_post" --ground-truth $gt --intervals --out "$out/BD004_Walk_Slow_four_phase_review.mp4"
```

Dùng `--device cpu` ở lệnh tiền xử lý nếu cần. Khi chỉ review các lượt Walk Fast có nhãn BD004, dùng `--max-passages 8` ở bước chuẩn bị. Step 6 dùng cache flow đã tạo ở Bước 5. Ground truth được đọc ở bước đánh giá và dựng video.

Xem [module Step 6](data/step6_fixed_foot_events/code/README.md) và [đặc tả đầu ra](docs/STEP6_SPEC.md).

---
## 4. Chạy lại Bước 0–5: cài model và repo bên ngoài

Repo **không chứa** môi trường Python, trọng số model, repo bên ngoài và các thư mục trung gian lớn (xem `.gitignore`). Step6 dùng các cache Bước 0–5 và cần GMA khi tái tạo flow_store. Mọi lệnh dưới đây chạy từ **thư mục gốc repo**.

### 4.1 Môi trường (Python 3.11)

```bash
python3.11 -m venv .venv-gait  && .venv-gait/bin/pip  install -r requirements-gait.txt    # RTMPose, cadence, hướng chạy
python3.11 -m venv .venv-speed && .venv-speed/bin/pip install -r requirements-speed.txt   # Wu et al.
python3.11 -m venv .venv-track && .venv-track/bin/pip install -r requirements-track.txt   # Bước 1b–5
```

Các file requirements được xuất từ môi trường macOS arm64. Trên Linux/Windows có thể phải đổi phiên bản `torch`/`torchvision` cho hợp với CUDA của máy.

### 4.2 Repo bên ngoài

```bash
mkdir -p .third_party
git clone https://github.com/facebookresearch/sam2 .third_party/sam2 && git -C .third_party/sam2 checkout 2b90b9f
.venv-track/bin/pip install -e .third_party/sam2
git clone https://github.com/zacjiang/GMA .third_party/GMA && git -C .third_party/GMA checkout 2f1fd29      # đã kèm checkpoints/gma-sintel.pth
git clone https://github.com/Seeing-Fast-and-Slow/Seeing-Fast-and-Slow && git -C Seeing-Fast-and-Slow checkout 06d4c3e
```

### 4.3 Trọng số model

| File / thư mục | Bước | Cách lấy |
|---|---|---|
| `ckpt/speed_predictor/` | 1 | `hf download james020619/VideoMAEv2-SpeedPredictor --local-dir ckpt/speed_predictor` |
| `ckpt/speed_change/` | 1 | `hf download james020619/VideoMAEv2-SpeedChangeDetector --local-dir ckpt/speed_change` |
| `data/_rtm/hub/checkpoints/*.onnx` (RTMPose-m Halpe26, YOLOX-m) | 0–3 | rtmlib tự tải lần chạy đầu, với `TORCH_HOME=data/_rtm` |
| `data/step2_tracking/models/yolo11m.pt` | 2 | ultralytics tự tải `yolo11m.pt`; chép vào thư mục này |
| `data/step4_footmask/models/sam2.1_hiera_base_plus.pt` | 4 | `curl -L -o data/step4_footmask/models/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt` |

`hf` là CLI của Hugging Face (`pip install -U huggingface_hub`); bản cũ dùng `huggingface-cli download`.

### 4.4 Dữ liệu trung gian lớn

| Thư mục | Tạo lại bằng | Thời gian (M4 Pro) |
|---|---|---|
| `data/step1_wu/clips/` | `cd data && ../.venv-gait/bin/python step1_wu/code/make_clean_clips.py` | vài phút |
| `data/step5_flow/flow_store/` | `cd data && caffeinate -i ../.venv-track/bin/python step5_flow/code/run_step5.py --store-only` | khoảng 45 phút |

### 4.5 Thứ tự chạy (từ thư mục `data/`)

```bash
cd data
python step1_wu/code/make_clean_clips.py
#   Wu (trong ../Seeing-Fast-and-Slow/speed_perception, .venv-speed): video_speed_estimation.py, speed_change_detection.py --thres 0.3
TORCH_HOME=_rtm python step1_wu/code/pose_all_frames.py
python step1_wu/code/cadence.py && python step1_wu/code/decide_speed.py && python step1_wu/code/write_time.py
../.venv-track/bin/python step1_wu/code/ramp_time.py
../.venv-track/bin/python step2_tracking/code/run_step2.py && ../.venv-track/bin/python step2_tracking/code/s2_validate.py
../.venv-track/bin/python step3_footkp/code/run_step3.py  && ../.venv-track/bin/python step3_footkp/code/s3_validate.py
caffeinate -i ../.venv-track/bin/python step4_footmask/code/run_step4.py
../.venv-track/bin/python step4_footmask/code/s4_postprocess.py
caffeinate -i ../.venv-track/bin/python step5_flow/code/run_step5.py --store-only
../.venv-track/bin/python step5_flow/code/run_step5.py && ../.venv-track/bin/python step5_flow/code/s5_validate.py
```

Lưu ý:
- Pipeline phát triển trên macOS arm64 (MPS). Máy khác đổi `device="mps"` thành `"cuda"` hoặc `"cpu"`; `caffeinate` chỉ có trên macOS, máy khác bỏ đi.
- `step0_audit/code/merge.py` và `shots.py` còn đường dẫn `~/work` của môi trường chạy Bước 0. Bước 0 không cần chạy lại, vì output đã có trong repo; nếu chạy lại thì phải sửa đường dẫn trước.

## 5. Quy ước làm việc

- Mỗi bước ghi output vào thư mục riêng `data/stepN_*/`, gồm `README.md`, `code/`, `qc_*.csv`, `review/`, và có script `*_validate.py` kiểm tra output.
- **Gate tự động luôn đi kèm review bằng mắt.** Kết luận review ghi ở `*_review.csv`.
- Gắn cờ theo frame và theo chân thay vì loại cả video.
- Nhãn trong folder `False/` được ghi là `INCORRECT` trong mọi bảng (pandas đọc chuỗi `FALSE` thành boolean).

## 6. Tham khảo chính

- Ceriola et al. (2026). *Monocular Optical-Flow Method for Gait Event Detection and Foot-Strike Pattern Evaluation.* IEEE Sensors Journal 26(10).
- Wu et al. (2026). *Seeing Fast and Slow.* https://github.com/Seeing-Fast-and-Slow/Seeing-Fast-and-Slow
- Ravi et al. SAM 2. https://github.com/facebookresearch/sam2
- Jiang et al. (2021). GMA: *Learning to Estimate Hidden Motions with Global Motion Aggregation.* https://github.com/zacjiang/GMA
- Zeni et al. (2008). *Two simple methods for determining gait events during treadmill and overground walking using kinematic data.* Gait & Posture.
