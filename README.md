# Running gait – Gait cycle segmentation từ video RGB một camera

Phân loại kỹ thuật chạy **CORRECT / INCORRECT** từ video mạng xã hội (TikTok, YouTube, Instagram). Repo này chứa pipeline tách chu kỳ dáng chạy dựa trên optical flow theo vùng bàn chân, theo phương pháp FOOF của Ceriola et al. (2026).

> **Trạng thái (06/11/2026):** Bước 0–5 đã hoàn tất và đã review. **Bước 6 (mô hình thống kê phát hiện IC/TO và cắt chu kỳ) chưa làm.** Đặc tả ở [`docs/STEP6_SPEC.md`](docs/STEP6_SPEC.md).

**Đọc trước:** [`reports/Bao_cao_day_du_Buoc0-5.md`](reports/Bao_cao_day_du_Buoc0-5.md). Báo cáo ghi toàn bộ phương pháp, quyết định, kết quả và hạn chế của Bước 0–5. Mỗi thư mục `data/stepN_*/` cũng có `README.md` riêng với định nghĩa từng cột và khoá.

---

## 1. Pipeline

| Module | Bước | Làm gì | Output chính | Trạng thái |
|---|---|---|---|---|
| Video speed processing | 0 | Audit: VFR, frame trùng, freeze, cắt cảnh, camera, hướng chạy, mirror | `data/step0_audit/` | ✅ |
| | 1, 1b | Trục thời gian thật: slow-mo (Wu et al. + cadence), speed ramp | `t_real_s_v2` trong `data/step0_audit/frames/` | ✅ |
| Subject tracking | 2 | YOLO11m + BoT-SORT + RTMPose Halpe26 theo người chạy | `data/step2_tracking/` | ✅ |
| Persistent foot region tracking | 3 | Sửa hoán nhãn L/R (Viterbi), chân gần/xa, làm sạch keypoint bàn chân | `data/step3_footkp/` | ✅ |
| | 4 | SAM2.1 mask từng bàn chân, trạng thái che khuất, vùng gót/giữa/mũi, cờ hợp lệ | `data/step4_footmask/masks_post/` | ✅ |
| Optical flow | 5 | GMA + bù camera trên dải đất + feature tốc độ theo vùng | `data/step5_flow/features/` | ✅ |
| **Statistical gait event model** | **6** | **IC/TO bằng FOOF-rule, Zeni, HSMM hai chân; cắt chu kỳ** | `data/step6_events/` | ⏳ **Chưa làm** |
| Validation | 7 | Ground truth (đề xuất: video 240 fps tự quay) | | ⏳ |

Dataset: **66 video**, gồm 39 CORRECT (`data/Correct/`) và 27 INCORRECT (`data/False/`), tổng 8.407 frame unique dùng được.

---

## 2. Cấu trúc repo

```
README.md                    ← file này
docs/STEP6_SPEC.md           ← đặc tả Bước 6 cho người làm tiếp
reports/                     ← báo cáo kỹ thuật (đọc Bao_cao_day_du_Buoc0-5.md)
requirements-step6.txt       ← tối thiểu để làm Bước 6
requirements-{gait,speed,track}.txt  ← môi trường đầy đủ để chạy lại Bước 0–5
scripts/                     ← script xác định hướng chạy (Bước 0, phần A)
data/
  Correct/, False/           ← video gốc
  step0_audit/               ← metadata, trục thời gian từng frame
  step1_wu/                  ← Wu et al., cadence, speed parts, ramp
  keypoints/                 ← RTMPose toàn frame (Bước 1)
  step2_tracking/            ← bbox người chạy, keypoint theo bbox
  step3_footkp/              ← keypoint bàn chân đã sửa và làm sạch
  step4_footmask/            ← mask bàn chân (masks_post/ là bản dùng)
  step5_flow/                ← feature optical flow (features/ là bản dùng)
biomech_signals.py           ← hàm tín hiệu sinh cơ học (từ dự án trước)
```

Mỗi `data/stepN_*/` có `README.md`, thư mục `code/`, file `qc_*.csv` và thư mục `review/` hoặc `plots/`.

---

## 3. Bắt đầu nhanh cho người làm Bước 6

```bash
git clone https://github.com/Tuanh-cyber/Gait-cycle-segmentation.git
cd Gait-cycle-segmentation
python3.11 -m venv .venv-step6 && source .venv-step6/bin/activate
pip install -r requirements-step6.txt
```

Bước 6 **không cần GPU, không cần trọng số model, không cần chạy lại Bước 0–5.** Mọi input đã có sẵn trong repo.

### 3.1 Input cho Bước 6

| File | Dùng để |
|---|---|
| `data/step5_flow/features/<vid>.npz` | **Tín hiệu chính:** tốc độ và `slow_ratio` theo frame × chân × vùng (biến thể V1) |
| `data/step5_flow/qc_flow.csv` | Gate và chất lượng flow theo video |
| `data/step4_footmask/masks_post/<vid>.npz` | Mask, vùng, trạng thái che khuất, `frame_valid_mask` (đã được chép sang features) |
| `data/step3_footkp/footkp/<vid>_footkp.npz` | `kp_clean` (cho phương pháp Zeni), `L_leg`, `near_track` |
| `data/step0_audit/frames/<vid>_frames.csv` | `t_real_s_v2`, `continuity_id`, `speed_part_id` |
| `data/step0_audit/step0_metadata.csv` | `label`, `s_final`, `abs_time_ok`, `cadence_real_spm_v2` |
| `data/step3_footkp/time_feature_flags.csv` | Đoạn loại khỏi feature thời gian (C20 part 1) |

### 3.2 Đọc feature

```python
import numpy as np, pandas as pd

vid = "C30"
F = np.load(f"data/step5_flow/features/{vid}.npz", allow_pickle=True)

t      = F["t_real_s_v2"]          # (N,)  thời gian thật, giây  ← trục thời gian chuẩn
block  = F["continuity_id"]        # (N,)  không cho chu kỳ vắt qua 2 khối
part   = F["speed_part_id"]        # (N,)  không cho chu kỳ vắt qua 2 phần tốc độ
label  = F["label"]                # (N, 2) "near"/"far" hoặc "A"/"B" cho slot 0/1
slow   = F["slow_ratio"]           # (N, 2, 4) tỉ lệ pixel v < 2,0 L_leg/s; vùng 0 gót, 1 giữa, 2 mũi, 3 cả bàn chân
v_med  = F["v_med"]                # (N, 2, 4) tốc độ trung vị, L_leg/s
valid  = F["frame_valid_mask"] & F["flow_valid"][:, None]   # (N, 2) chỉ dùng khi True; NaN ở chỗ không hợp lệ
kp_v   = F["kp_speed"]             # (N, 2) tốc độ keypoint bàn chân sau bù camera (đối chiếu)

meta = pd.read_csv("data/step0_audit/step0_metadata.csv")   # label: CORRECT / INCORRECT
```

Danh sách đầy đủ các khoá nằm ở [`data/step5_flow/README.md`](data/step5_flow/README.md). Hàm giải mã mask (RLE) nằm ở [`data/step4_footmask/README.md`](data/step4_footmask/README.md).

### 3.3 Quy tắc dùng dữ liệu (bắt buộc)

| Quy tắc | Lý do |
|---|---|
| Thời gian luôn lấy theo `t_real_s_v2`, không dùng frame index hay `pts_s` | Video có slow-mo, speed ramp, VFR, freeze |
| Chỉ dùng chỗ có `frame_valid_mask` và `flow_valid` = True; frame không hợp lệ là **không có quan sát**, không phải swing | Chân bị che, mask sai, lỗi bù camera |
| Chu kỳ phải nằm trọn trong **một `continuity_id` và một `speed_part_id`** | Không có liên tục thời gian qua cắt cảnh hoặc đổi tốc độ |
| **Slot 0/1 là chân A/B, không phải trái/phải giải phẫu.** Ngữ nghĩa nằm ở `label` (near/far) | 12 video không xác định được mirror |
| Feature thời gian tuyệt đối (GCT, cadence, flight) chỉ dùng khi `abs_time_ok` = True và `exclude_time_features` = False | 27/66 video không có mốc real-time chắc chắn |
| Bỏ frame có `view_lateral` = False | F10_2 f1052–1078 quay lưng |
| **Không dùng nhãn CORRECT/INCORRECT khi fit hay chọn tham số** | Tránh rò rỉ nhãn vào feature |
| Không sửa output của `data/step0`–`step5`; ghi kết quả mới vào `data/step6_events/` | Các output này đã được review |

### 3.4 Video cần chú ý

- **Bước 5 FAIL** (tương quan flow–keypoint < 0,4): C04_3, C04_2, C09_2.
- **Bước 4 major:** C04_3, C38_8, C27. Mask kém; C04_3 và C27 không có nhãn near/far.
- **Không đo được mặt đất** (toàn bộ bù camera dựa vào spline chậm): C28_2, F28_2, F28_4.
- **Video rất ngắn**, khoảng 2 sải: bộ C28, F28, C38_1, C38_5.
- **C09_1, C09_2:** hệ số slow-mo 1/8 với độ tin cậy thấp, nên thời lượng tuyệt đối không tin được.

Trạng thái từng video qua các bước: Phụ lục A của báo cáo.

---

## 4. Chạy lại Bước 0–5: cài model và repo bên ngoài

Repo **không chứa** môi trường Python, trọng số model, repo bên ngoài và hai thư mục trung gian lớn (xem `.gitignore`). **Làm Bước 6 thì không cần phần này.** Chỉ làm khi muốn chạy lại hoặc sửa Bước 0–5. Mọi lệnh dưới đây chạy từ **thư mục gốc repo**.

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
