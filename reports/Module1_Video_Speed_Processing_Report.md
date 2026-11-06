# Báo cáo Module 1 — Xử lý tốc độ video (Video speed processing)

**Dự án:** Gait cycle segmentation cho bài toán phân loại kỹ thuật chạy (CORRECT / INCORRECT)
**Framework:** Single camera RGB → **Video speed processing** → Subject detection & tracking → Foot regional extraction → Persistent foot region tracking → Optical flow & regional features → Statistical gait event modeling → Validation
**Core paper:** Ceriola et al. (2026), *Monocular Optical-Flow Method for Gait Event Detection and Foot-Strike Pattern Evaluation in Parkinsonian Gait*, IEEE Sensors Journal 26(10)
**Công cụ tốc độ video:** Wu et al. (2026), *Seeing Fast and Slow: Learning the Flow of Time in Videos* (ECCV 2026), github.com/Seeing-Fast-and-Slow/Seeing-Fast-and-Slow
**Ngày:** 04/11/2026

---

## 1. Mục tiêu

Module 1 dựng một **trục thời gian thật, đáng tin** cho từng frame của từng video trước khi phát hiện sự kiện dáng chạy (IC/TO). Lý do: dataset gồm video mạng xã hội đã qua biên tập, nên frame index ≠ thời gian. Các vấn đề phải xử lý:

1. Video **VFR** (frame rate thay đổi) và **frame trùng** (duplication) → độ phân giải thời gian thật thấp hơn fps khai báo.
2. **Freeze-frame do biên tập** (ảnh tĩnh chèn chú thích) → dễ bị nhầm thành pha stance.
3. **Slow-motion** (nhân frame hoặc quay bằng camera tốc độ cao) và **đổi tốc độ giữa clip** → sai mọi đặc trưng thời gian tuyệt đối.
4. **Cắt cảnh** → một chu kỳ dáng chạy không được vắt qua hai shot.
5. Metadata hình học cần cho các module sau: **hướng chạy `s`**, **mirror**, **chân gần camera**, góc quay, chuyển động camera.

Module được chia thành hai phần: **Bước 0 — Audit** (đo đạc, không sửa dữ liệu) và **Bước 1 — Trục thời gian thật** (Wu + cadence).

---

## 2. Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Số video | **66** (39 CORRECT, 27 INCORRECT); `F0x.mov` đã bị loại khỏi dataset theo quyết định của nhóm |
| Hướng khung hình | 41 dọc, 25 ngang |
| Độ dài | 1.33–28.22 s (trung vị 4.71 s) |
| fps khai báo (container) | 60 (25 video), 30 (17), 23.976 (12), các giá trị lẻ khác |
| Nguồn | TikTok / YouTube / Instagram — nhiều video có chú thích, watermark, title card |

Quy ước nhãn: folder `False` được ghi là `INCORRECT` trong mọi bảng (chuỗi `FALSE` bị pandas đọc thành boolean).

---

## 3. Bước 0 — Audit dữ liệu

### 3.1 Phương pháp

| Hạng mục | Cách đo |
|---|---|
| Timestamp thật | `ffprobe` → `best_effort_timestamp_time` của từng frame; giải mã bằng `ffmpeg -vsync passthrough` để không thêm/bớt frame |
| Frame trùng | MAD ảnh xám (480 px) so với frame trước; trùng nếu MAD < min(1.0, 0.12 × median MAD của frame thật). Phân bố hai mode rất rõ: frame trùng < 0.3, frame thật > 3 |
| Effective fps | số frame unique / thời lượng thật; `effective_fps_active` tính sau khi bỏ freeze biên tập |
| Freeze biên tập | chuỗi frame trùng hoặc gần tĩnh (MAD < 0.25 × median frame thật, để bắt cả chú thích động trên ảnh tĩnh) kéo dài ≥ 0.30 s |
| Cắt cảnh | khoảng cách Bhattacharyya histogram HSV giữa hai frame liên tiếp > 0.45 |
| Chuyển động camera | Lucas–Kanade + RANSAC similarity transform trên frame unique → dx, dy, scale |
| Góc quay | đánh giá trực quan trên contact sheet và crop |

### 3.2 Kết quả chính

- **VFR:** 23/66 video. Khai báo 60 fps nhưng khoảng cách frame nhảy 17/33/50 ms → **mọi phép tính thời gian phải dùng `pts_s`, không dùng frame index.**
- **Frame trùng:** 43 video có frame trùng; tỉ lệ trùng tối đa 89%.
- **Effective fps (phần chạy):** 26 video ≥ 28 fps · 27 video 20–28 · 8 video 12–20 · 5 video < 12 (min 6.6, trung vị 24.7).
- **Freeze biên tập:** 6 video — C11 (3.9 s), C12 (4.1 s), F10_2 (20.5 s), F11 (6.7 s), F12 (1.4 s), F30 (0.5 s). F10_2 có 20.5 s đứng hình trên tổng 28 s.
- **Cắt cảnh:** C01 (cắt sang cận cảnh bàn chân tại 2.67 s), C38_5 (title card tại 1.80 s), C38_8 (2.80 s).
- **Góc quay:** 26 video góc chéo (chủ yếu chéo phía trước), 40 video lateral.
- **Chuyển động camera:** 6 tĩnh · 30 pan nhẹ · 19 pan vừa · 11 pan mạnh → bắt buộc bù chuyển động camera ở module optical flow.
- **Clip ngắn:** 10 video có phần chạy < 2 s (≈ 2 sải chân): bộ C28, F28, C38_1, C38_5.

---

## 4. Hướng chạy, mirror và chân gần camera

### 4.1 Hướng chạy `s`

Quy ước: **`s = +1` chạy trái→phải màn hình, `s = −1` phải→trái.**

- *Thử nghiệm ban đầu* (chuyển động người so với nền + quan sát thumbnail) cho độ tin cậy thấp. Khi đối chiếu với keypoint, **23 shot mâu thuẫn**; kiểm tra full-res 5 trường hợp cho thấy 4/5 lỗi nằm ở cách sơ bộ → cột `s_prelim`, `dir_auto`, `dir_visual` chỉ còn để tham khảo.
- *Cách chốt:* RTMPose-m Halpe26 (rtmlib, chế độ `balanced`) trên ~25 frame/shot; tín hiệu chính là dấu thành phần x của vector **heel → big/small toe** (hai chân), tín hiệu phụ là **mũi so với tai**. Hai tín hiệu độc lập khớp nhau 96–100% ở hầu hết video.
- *Ngoại lệ:* F10_2 — detector chọn nhầm khán giả (người lớn nhất khung) → chốt bằng kiểm tra tay (`s = +1`). Bài học này dẫn tới cách chọn người theo vùng chuyển động ở Bước 1.
- Kết quả: **34 video `+1`, 32 video `−1`**; 2 shot bị loại (C38_5 shot 1 = title card; C38_8 shot 1 = 6 frame).

### 4.2 Mirror

Mirror **không suy ra được** từ hình học. Căn cứ là chữ trong khung hình:

| Giá trị | Số video | Ý nghĩa |
|---|---|---|
| `no` | 15 | có chữ thật trong cảnh đọc xuôi (biển "Irma", "fitnessdk", "HOME OF THE AZTECS", chữ "USA" trên áo, bảng "tcs"…) |
| `no (overlay)` | 39 | chỉ có chú thích/watermark đọc xuôi → chắc chắn không bị lật **sau khi tải về**, không loại trừ việc người quay lật trước khi chèn chữ |
| `unknown` | 12 | không có chữ nào: C09_2, C30, C31, C38_1, C38_3, C38_4, C38_6, C38_7, C38_8, C38_9, F02, F30 |

### 4.3 Chân gần camera

`s = −1` & không lật → **trái**; `s = +1` & không lật → **phải**. Kết quả: 30 trái, 24 phải, 12 chỉ dùng nhãn near/far (mirror unknown). Với video bị lật, nhãn L/R của RTMPose cũng bị đảo theo.

---

## 5. Bước 1 — Trục thời gian thật

### 5.1 Clip sạch cho Wu

Đọc mã nguồn Wu (`dataset.py`, `video_speed_estimation.py`) cho thấy 4 ràng buộc: chỉ đọc `*.mp4`; dùng `average_rate` và coi frame cách đều; cửa sổ `num_frames/7.5` s (≈ 2.13 s); clip ngắn hơn cửa sổ gây lỗi chia 0. Do đó mỗi video được:

1. bỏ đoạn freeze biên tập, 2. tách theo shot, 3. chuyển **VFR → CFR 30 fps theo timestamp thật** (`ffmpeg -vf fps=30`, H.264 CRF 18).

Kết quả: **57 clip** (số frame khớp thời lượng × 30, sai lệch ≤ 1 frame), 33 đoạn quá ngắn (< 2.3 s), 2 shot bị loại. 11 video không có clip đủ dài cho Wu: C28_1–4, F28_1–4, C38_1, C38_5, F12.

### 5.2 Chạy Wu et al.

Checkpoint `VideoMAEv2-SpeedPredictor` và `VideoMAEv2-SpeedChangeDetector` (`--thres 0.3`), chạy trên GPU M4 Pro (MPS).
**Kiểm chứng quy ước:** video mẫu `4x_slomo` → 0.225, `20x_slomo` → 0.051 ⇒ *speed < 1 = slow-motion; thời gian thật = thời gian video × speed.*

**Đánh giá trên dataset:**
- Wu **nhận ra slow-motion mạnh rất ổn định**: C09_1/2 ≈ 0.26; cụm quay bằng camera tốc độ cao C17/F17, C19–C21/F19–F21, C24/F24 ≈ 0.28–0.40; F02 = 0.16. Cặp C/F cùng người quay cho giá trị gần như trùng nhau.
- Ở vùng 0.5–1.5, Wu **nhiễu ±40%**: C13_1 = 0.77 và C13_2 = 1.14 dù cùng buổi quay; bộ C38 (cùng người quay) dao động 0.54–1.48; vài clip > 1.4 ("tua nhanh") là không hợp lý với video hướng dẫn.
- Nguyên nhân chính: Wu trả **một giá trị cho cả clip**, trong khi nhiều clip **đổi tốc độ giữa chừng** → giá trị là trung bình nhân của hai tốc độ.
⇒ **Không dùng giá trị Wu liên tục làm hệ số hiệu chỉnh**; Wu chỉ dùng để phân loại và để chọn giữa các ứng viên.

### 5.3 Cadence từ keypoint

- RTMPose Halpe26 trên **toàn bộ 8.424 frame unique** (không thuộc freeze, không thuộc shot bị loại) của 66 video. Detector YOLOX chạy mỗi 6 frame; giữa các lần detect, bbox = bbox keypoint frame trước (nới 25%). **Người chạy được chọn theo tâm vùng chuyển động** (từ Bước 0), sau đó bám theo IoU; chọn lại khi cắt cảnh. OpenCV đọc đủ số frame ở mọi video.
- Hai tín hiệu **không phụ thuộc nhãn L/R** (chống lỗi hoán đổi chân của RTMPose): khoảng cách ngang hai cổ chân và dao động dọc trung điểm hông, chuẩn hóa theo chiều dài chân, khử trend bậc 2.
- Tần số bước bằng **Lomb–Scargle trên timestamp thật** (không nội suy), lấy tích hai periodogram chuẩn hóa; cửa sổ trượt 2 s để phát hiện đổi tốc độ.

### 5.4 Chốt hệ số tốc độ `k`

1. **Phát hiện đổi tốc độ:** cadence các cửa sổ 2 s (bước 0.25 s) → phân 2 cụm trong không gian log; tách nếu tâm cụm lệch ≥ 1.6× và mỗi phần ≥ 0.5 s; phần "nhanh" phải có cadence ≥ 120 và peak rõ.
2. **Clip có cả phần nhanh và chậm:** phần nhanh `k = 1`; phần chậm `k = cadence_chậm / cadence_nhanh` (cùng vận động viên, cùng buổi), snap về {1/2, 1/3, 1/4, 1/5, 1/6, 1/8} nếu lệch < 15%.
3. **Phần đơn tốc độ:** ứng viên `k` sao cho cadence thực ∈ [140, 215] bước/phút; chọn ứng viên gần Wu nhất (log); không có Wu → ứng viên lớn nhất.
4. **Không chuẩn hóa cadence về một giá trị cố định** — "slow cadence" là một lỗi trong nhãn INCORRECT; các ứng viên cách nhau ≥ 1.2× nên cadence chậm thật vẫn được giữ.
5. **Độ tin cậy:** `cao` = real-time, ≥ 4 bước, peak rõ; `trung bình` = ≥ 4 bước hoặc có tỉ lệ cadence đo trực tiếp; `thấp` = phần slow-mo < 4 bước, mâu thuẫn Wu hoặc nhiều ứng viên.

### 5.5 Kết quả

- **32/66 video có slow-motion.** Phát hiện quan trọng: cụm C17, C19–C21, C24 (và F tương ứng) **không có frame trùng** nên ở Bước 0 trông như 30 fps thật, nhưng thực chất là slow-mo ≈ 4× quay bằng camera tốc độ cao.
- **Đổi tốc độ giữa clip** được tách: C13_1 (0.5× trong 11.9 s đầu → real-time), C18, C19 (nhanh → chậm → nhanh), C20, C21, F11, F18, F24. Ví dụ C13_1: cadence thực hai nửa đều ≈ 205, khớp C13_2 (203).
- **85 phần tốc độ:** k = 1 (48), 1/2 (14), 1/3 (8), 1/4 (4), 1/5 (6), 1/6 (1), 1/8 (4). Độ tin cậy: 36 cao · 20 trung bình · 29 thấp.
- **Cadence thực** (phần có độ tin cậy ≥ trung bình, 47 video): 142–205 bước/phút, trung vị 171.
- **`abs_time_ok`:** 40 video True / 26 False; phân bố cân bằng giữa hai nhãn (CORRECT 24/39, INCORRECT 16/27).

---

## 6. Đầu ra

### 6.1 Cấu trúc thư mục

```
data/
├── step0_audit/
│   ├── step0_metadata.xlsx / .csv   ← 1 dòng/video, toàn bộ metadata module 1 (sheet README giải thích cột)
│   ├── frames/<video>_frames.csv    ← 1 dòng/frame: trục thời gian thật
│   ├── s_keypoint.csv               ← hướng chạy theo shot (s_foot, s_face, s_final, ghi_chu)
│   ├── s_keypoint_frames.csv
│   ├── contact_sheets/              ← ảnh kiểm tra trực quan (hướng, mirror, mâu thuẫn)
│   └── code/                        ← script audit Bước 0
├── step1_wu/
│   ├── clips/  manifest.csv         ← 57 clip sạch CFR 30 fps + bảng ánh xạ về timestamp gốc
│   ├── out/speeds.json, changes.json, test_speeds.json   ← output Wu
│   ├── cadence_segments.csv         ← cadence theo đoạn liên tục
│   ├── speed_parts.csv              ← các phần tốc độ + k + độ tin cậy + cờ
│   └── code/                        ← make_clean_clips, pose_all_frames, cadence, decide_speed, write_time
├── keypoints/<video>_kp.npz          ← RTMPose Halpe26 mọi frame unique (kp, score, bbox, pts_s, shot_id)
└── _rtm/hub/checkpoints/             ← file ONNX của YOLOX-m và RTMPose-m (bản sao cache rtmlib)
scripts/  a0_test.py, a_direction.py, a_review.py   ← chốt hướng chạy (việc A)
```

### 6.2 Cột trong `frames/<video>_frames.csv`

| Cột | Ý nghĩa |
|---|---|
| `frame`, `pts_s` | chỉ số frame (giải mã passthrough) và timestamp thật trong video |
| `diff_prev`, `is_dup`, `unique_id` | MAD so với frame trước, cờ frame trùng, chỉ số frame unique |
| `is_edit_freeze` | thuộc đoạn freeze biên tập — **không phải stance** |
| `cam_dx`, `cam_dy`, `cam_scale`, `cam_inlier` | chuyển động camera giữa hai frame unique |
| `run_*` | bbox vùng chuyển động (thô, dùng để chọn người) |
| `hist_dist`, `is_cut`, `shot_id` | cắt cảnh |
| `speed_part_id`, `speed_k`, `k_confidence` | phần tốc độ và hệ số `k` (−1 = freeze / shot bị loại) |
| `continuity_id` | khối liên tục về thời gian — **chu kỳ không được vắt qua hai khối** |
| `t_real_s` | thời gian thật (s) = Σ dt × k; freeze được thu gọn |

### 6.3 Cột chính trong `step0_metadata`

`effective_fps_active`, `n_edit_freeze`, `edit_freeze_segments_s`, `n_shots`, `cut_times_s`, `camera_view`, `camera_motion`, `s_final`, `excluded_shots`, `mirrored`, `mirror_evidence`, `near_leg_camera`, `near_leg_confidence`, `speed_parts`, `has_slowmo`, `realtime_fraction`, `cadence_real_spm`, `abs_time_ok`, `speed_flags`.

---

## 7. Hạn chế

1. **Hệ số k của phần slow-mo ngắn không chắc chắn.** Nhiều phần chỉ có 2–3 bước nên không phân biệt được 1/4 với 1/5 (ví dụ C20 cho tỉ lệ 0.21, C19 cho 0.24, F24 cho 0.30 — cùng một kiểu quay).
2. **Một số clip có ~1 s real-time ở đầu chưa tách được** vì quá ngắn: C17, F17, F19, F20 (Wu change-detection có báo).
3. **Wu không chạy được** cho 11 video ngắn; với các video này `k` chỉ dựa vào cadence.
4. **Mirror `unknown`** ở 12 video; `no (overlay)` không loại trừ việc lật trước khi chèn chữ.
5. **Script Bước 0** (`step0_audit/code/`) được chạy trong môi trường staging; đường dẫn `~/work` trong `merge.py` cần sửa trước khi chạy lại.
6. Cadence dựa trên giả định khoảng sinh lý 140–215 bước/phút; vận động viên chạy nước rút có thể vượt ngưỡng trên.

---

## 8. Khuyến nghị cho các module sau

- **Mọi đạo hàm và mọi đặc trưng thời gian dùng `t_real_s`**, không dùng frame index hay `pts_s`.
- **Đặc trưng thời gian tuyệt đối** (cadence, ground contact time, flight time) **chỉ dùng video `abs_time_ok = True`**. Đặc trưng chuẩn hóa theo pha chu kỳ (%cycle) dùng được cho mọi video, miễn chu kỳ nằm trọn trong một `speed_part_id` và một `continuity_id`.
- **Bỏ frame `is_edit_freeze`** khỏi optical flow (GMA flow = 0 sẽ bị đọc thành stance) và chỉ tính flow giữa **frame unique liên tiếp**.
- **Bù chuyển động camera** cho 30 video pan vừa/mạnh.
- **Chọn người chạy theo vùng chuyển động**, không theo kích thước bbox (bài học F10_2); kiểm tra tận mắt các video đông người (F10_1, F10_2) ở module tracking.
- **Dùng lại `keypoints/*.npz`** cho module foot regional extraction (heel / big toe / small toe đã có).
- Với video `mirrored = unknown`: chỉ dùng nhãn chân **near/far**.
- **Ablation đề xuất cho paper:** chạy Wu trên video gốc vs clip sạch; Wu liên tục vs k chốt bằng cadence — chênh lệch là bằng chứng định lượng cho bước tiền xử lý.

---

## 9. Tái lập

| Môi trường | Dùng cho | Cài đặt |
|---|---|---|
| `.venv-gait` (Python 3.11) | RTMPose, hướng chạy, pose toàn bộ frame, cadence | `pip install rtmlib onnxruntime opencv-python pandas numpy scipy openpyxl` |
| `.venv-speed` (Python 3.11) | Wu et al. | `pip install torch torchvision "av<14" "transformers==4.49" timm "accelerate>=0.26.0" tqdm huggingface_hub pandas` (bỏ `pytorchvideo`) |

Lưu ý phần cứng (MacBook Pro M4 Pro): detector YOLOX **không chạy được trên CoreML** (lỗi shape động `{1,8400}`) → detector chạy `cpu`, pose model có thể chạy `mps`; Wu chạy trên MPS với `PYTORCH_ENABLE_MPS_FALLBACK=1`.

Thứ tự chạy (từ thư mục `data/`):

```bash
# Bước 1 (sau khi đã có step0_audit/)
python step1_wu/code/make_clean_clips.py
# -> chạy Wu (trong Seeing-Fast-and-Slow/speed_perception, .venv-speed):
#    video_speed_estimation.py  --input_path data/step1_wu/clips --output_path data/step1_wu/out/speeds.json  --num_workers 0
#    speed_change_detection.py --thres 0.3 ... --output_path data/step1_wu/out/changes.json --num_workers 0
TORCH_HOME=_rtm python step1_wu/code/pose_all_frames.py
python step1_wu/code/cadence.py
python step1_wu/code/decide_speed.py
python step1_wu/code/write_time.py
```

---

## Phụ lục A — Metadata theo video

| Video | Nhãn | Dài (s) | VFR | Tỉ lệ trùng | eff. fps | Freeze (s) | Shot | Góc quay | Camera | s | Mirror | Chân gần |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | CORRECT | 5.58 | ✓ | 0.43 | 17.3 |  | 2 | lateral + cắt cận cảnh bàn chân | tĩnh | -1 | no (overlay) | left |
| C04_1 | CORRECT | 4.84 | ✓ | 0.07 | 23.7 |  | 1 | lateral | pan vừa | +1 | no (overlay) | right |
| C04_2 | CORRECT | 3.13 | ✓ | 0.51 | 14.8 |  | 1 | oblique-front | pan mạnh | +1 | no (overlay) | right |
| C04_3 | CORRECT | 6.95 | ✓ | 0.24 | 23.6 |  | 1 | lateral-oblique | pan mạnh | -1 | no | left |
| C09_1 | CORRECT | 8.53 |  | 0.89 | 6.6 |  | 1 | lateral (camera tĩnh) | tĩnh | +1 | no (overlay) | right |
| C09_2 | CORRECT | 10.38 |  | 0.84 | 9.5 |  | 1 | lateral (camera tĩnh) | tĩnh | -1 | unknown | chỉ dùng near/far |
| C11 | CORRECT | 11.48 | ✓ | 0.49 | 30.1 | 3.86 | 1 | oblique-front | pan vừa | -1 | no (overlay) | left |
| C12 | CORRECT | 10.68 | ✓ | 0.53 | 29.7 | 4.13 | 1 | lateral → oblique-front cuối clip | pan vừa | +1 | no (overlay) | right |
| C13_1 | CORRECT | 22.10 | ✓ | 0.01 | 30.0 |  | 1 | lateral (có frame chuyển cảnh trắng) | pan vừa | -1 | no (overlay) | left |
| C13_2 | CORRECT | 19.92 | ✓ | 0.02 | 30.0 |  | 1 | lateral | pan mạnh | -1 | no (overlay) | left |
| C17 | CORRECT | 5.67 |  | 0.00 | 30.2 |  | 1 | lateral, khung chặt | pan vừa | -1 | no (overlay) | left |
| C18 | CORRECT | 6.37 |  | 0.43 | 17.1 |  | 1 | lateral | pan vừa | -1 | no (overlay) | left |
| C19 | CORRECT | 7.26 | ✓ | 0.33 | 23.9 |  | 1 | lateral | pan nhẹ | +1 | no (overlay) | right |
| C20 | CORRECT | 6.08 | ✓ | 0.25 | 30.2 |  | 1 | lateral | pan vừa | +1 | no (overlay) | right |
| C21 | CORRECT | 5.94 | ✓ | 0.41 | 24.2 |  | 1 | lateral | pan nhẹ | -1 | no (overlay) | left |
| C23_1 | CORRECT | 5.58 |  | 0.64 | 9.6 |  | 1 | oblique-front | pan nhẹ | +1 | no (overlay) | right |
| C23_2 | CORRECT | 2.52 |  | 0.32 | 18.1 |  | 1 | oblique-front | pan nhẹ | +1 | no (overlay) | right |
| C23_3 | CORRECT | 4.85 |  | 0.00 | 26.4 |  | 1 | oblique-front | pan nhẹ | +1 | no (overlay) | right |
| C24 | CORRECT | 4.62 |  | 0.01 | 29.7 |  | 1 | lateral, khung chặt | pan vừa | -1 | no (overlay) | left |
| C25 | CORRECT | 8.83 |  | 0.63 | 9.0 |  | 1 | oblique-front → lateral | pan mạnh | -1 | no (overlay) | left |
| C26 | CORRECT | 4.80 |  | 0.13 | 26.2 |  | 1 | lateral | tĩnh | +1 | no (overlay) | right |
| C27 | CORRECT | 3.95 |  | 0.03 | 29.1 |  | 1 | lateral | pan nhẹ | +1 | no (overlay) | right |
| C28_1 | CORRECT | 1.42 |  | 0.00 | 24.7 |  | 1 | lateral | pan nhẹ | -1 | no (overlay) | left |
| C28_2 | CORRECT | 1.38 |  | 0.00 | 24.7 |  | 1 | lateral | pan nhẹ | -1 | no (overlay) | left |
| C28_3 | CORRECT | 1.46 |  | 0.03 | 24.0 |  | 1 | lateral | pan nhẹ | -1 | no (overlay) | left |
| C28_4 | CORRECT | 1.42 |  | 0.03 | 24.0 |  | 1 | lateral | pan nhẹ | -1 | no (overlay) | left |
| C29 | CORRECT | 10.13 |  | 0.00 | 30.0 |  | 1 | oblique-front | pan mạnh | -1 | no | left |
| C30 | CORRECT | 10.52 | ✓ | 0.00 | 24.1 |  | 1 | oblique-front | pan vừa | -1 | unknown | chỉ dùng near/far |
| C31 | CORRECT | 6.08 | ✓ | 0.17 | 24.3 |  | 1 | lateral | pan mạnh | +1 | unknown | chỉ dùng near/far |
| C38_1 | CORRECT | 1.93 |  | 0.00 | 30.5 |  | 1 | lateral | pan nhẹ | +1 | unknown | chỉ dùng near/far |
| C38_10 | CORRECT | 5.77 |  | 0.00 | 30.2 |  | 1 | lateral | pan nhẹ | +1 | no | right |
| C38_2 | CORRECT | 3.20 |  | 0.00 | 30.3 |  | 1 | oblique-rear (chạy ra xa) | pan nhẹ | +1 | no | right |
| C38_3 | CORRECT | 3.00 |  | 0.00 | 30.3 |  | 1 | oblique-front | pan nhẹ | +1 | unknown | chỉ dùng near/far |
| C38_4 | CORRECT | 2.37 |  | 0.00 | 30.4 |  | 1 | lateral | pan mạnh | +1 | unknown | chỉ dùng near/far |
| C38_5 | CORRECT | 2.00 |  | 0.08 | 28.0 |  | 2 | lateral + title card cuối | pan nhẹ | +1 | no (overlay) | right |
| C38_6 | CORRECT | 3.00 |  | 0.00 | 30.3 |  | 1 | lateral | pan nhẹ | +1 | unknown | chỉ dùng near/far |
| C38_7 | CORRECT | 2.37 |  | 0.00 | 30.4 |  | 1 | lateral | pan mạnh | +1 | unknown | chỉ dùng near/far |
| C38_8 | CORRECT | 3.00 |  | 0.00 | 30.3 |  | 2 | lateral, người chạy rất nhỏ/xa | pan vừa | +1 | unknown | chỉ dùng near/far |
| C38_9 | CORRECT | 4.07 |  | 0.00 | 30.2 |  | 1 | lateral | pan nhẹ | +1 | unknown | chỉ dùng near/far |
| F02 | INCORRECT | 19.54 | ✓ | 0.32 | 24.8 |  | 1 | oblique-front | pan vừa | +1 | unknown | chỉ dùng near/far |
| F10_1 | INCORRECT | 3.85 | ✓ | 0.02 | 30.4 |  | 1 | oblique-front, đám đông + vật che tiền cảnh | pan mạnh | +1 | no | right |
| F10_2 | INCORRECT | 28.22 | ✓ | 0.89 | 14.7 | 20.53 | 1 | lateral, đám đông | pan mạnh | +1 | no | right |
| F11 | INCORRECT | 15.50 | ✓ | 0.56 | 29.6 | 6.71 | 1 | oblique-front | pan vừa | -1 | no (overlay) | left |
| F12 | INCORRECT | 3.88 | ✓ | 0.49 | 30.8 | 1.43 | 1 | lateral | tĩnh | +1 | no (overlay) | right |
| F17 | INCORRECT | 6.52 | ✓ | 0.25 | 30.1 |  | 1 | lateral | pan vừa | -1 | no (overlay) | left |
| F18 | INCORRECT | 5.57 |  | 0.41 | 17.7 |  | 1 | oblique-front | pan vừa | -1 | no (overlay) | left |
| F19 | INCORRECT | 4.93 | ✓ | 0.34 | 24.1 |  | 1 | lateral | pan nhẹ | +1 | no (overlay) | right |
| F20 | INCORRECT | 5.77 | ✓ | 0.28 | 30.1 |  | 1 | lateral | pan vừa | +1 | no (overlay) | right |
| F21 | INCORRECT | 4.83 | ✓ | 0.40 | 24.3 |  | 1 | lateral, khung chặt | pan nhẹ | -1 | no (overlay) | left |
| F22_1 | INCORRECT | 2.59 |  | 0.00 | 24.4 |  | 1 | oblique-front | pan nhẹ | -1 | no | left |
| F22_2 | INCORRECT | 2.54 |  | 0.00 | 24.4 |  | 1 | oblique-front | pan nhẹ | -1 | no | left |
| F22_3 | INCORRECT | 2.50 |  | 0.00 | 24.4 |  | 1 | oblique-front | pan nhẹ | -1 | no | left |
| F22_4 | INCORRECT | 2.50 |  | 0.00 | 24.4 |  | 1 | oblique-front | pan nhẹ | -1 | no | left |
| F23_1 | INCORRECT | 2.44 |  | 0.00 | 26.6 |  | 1 | oblique-front | pan vừa | +1 | no (overlay) | right |
| F23_2 | INCORRECT | 3.67 |  | 0.54 | 12.1 |  | 1 | oblique-front | pan nhẹ | +1 | no (overlay) | right |
| F23_3 | INCORRECT | 3.86 |  | 0.42 | 15.4 |  | 1 | oblique-front | pan vừa | +1 | no (overlay) | right |
| F24 | INCORRECT | 4.98 |  | 0.01 | 29.7 |  | 1 | oblique-front | pan vừa | -1 | no (overlay) | left |
| F25 | INCORRECT | 10.79 |  | 0.61 | 9.3 |  | 1 | oblique-front | pan mạnh | -1 | no (overlay) | left |
| F26 | INCORRECT | 4.93 |  | 0.10 | 27.1 |  | 1 | lateral | tĩnh | +1 | no | right |
| F27 | INCORRECT | 4.12 |  | 0.01 | 29.9 |  | 1 | lateral | pan nhẹ | +1 | no (overlay) | right |
| F28_1 | INCORRECT | 1.42 |  | 0.00 | 24.7 |  | 1 | lateral | pan vừa | -1 | no | left |
| F28_2 | INCORRECT | 1.33 |  | 0.00 | 24.8 |  | 1 | lateral | pan nhẹ | -1 | no | left |
| F28_3 | INCORRECT | 1.38 |  | 0.00 | 24.7 |  | 1 | lateral | pan nhẹ | -1 | no | left |
| F28_4 | INCORRECT | 1.46 |  | 0.03 | 24.0 |  | 1 | lateral | pan nhẹ | -1 | no | left |
| F29 | INCORRECT | 2.40 |  | 0.00 | 30.4 |  | 1 | oblique-front | pan nhẹ | -1 | no (overlay) | left |
| F30 | INCORRECT | 3.98 | ✓ | 0.14 | 24.4 | 0.5 | 1 | lateral | pan nhẹ | +1 | unknown | chỉ dùng near/far |

## Phụ lục B — Các phần tốc độ

| Video | Thời gian video (s) | Cadence video | Wu | k | Cadence thực | Tin cậy | Cờ |
|---|---|---|---|---|---|---|---|
| C01 | 0.00–2.67 | 114.2 | 0.73 | 1/2 | 228.4 | thấp | cadence ngoài khoảng sinh lý |
| C01 | 2.67–5.58 | 96.6 | 0.65 | 1/2 | 193.2 | trung bình |  |
| C04_1 | 0.00–4.84 | 87.9 | 0.63 | 1/2 | 175.8 | trung bình |  |
| C04_2 | 0.00–3.08 | 94.8 | 0.96 | 1/2 | 189.6 | thấp | mâu thuẫn Wu (0.96) |
| C04_3 | 0.00–6.95 | 198.7 | 1.10 | 1 | 198.7 | cao |  |
| C09_1 | 0.00–8.47 | 22.7 | 0.26 | 1/8 | 181.6 | thấp | mâu thuẫn Wu (0.26); < 4 bước trong phần này |
| C09_2 | 0.00–10.38 | 21.3 | 0.27 | 1/8 | 170.4 | thấp | mâu thuẫn Wu (0.27); < 4 bước trong phần này |
| C11 | 0.00–6.30 | 43.4 | 0.48 | 1/4 | 173.6 | thấp | mâu thuẫn Wu (0.48) |
| C11 | 10.23–11.50 | 80.3 |  | 1/2 | 160.6 | thấp | < 4 bước trong phần này; peak cadence yếu |
| C12 | 0.00–2.59 | 166.4 | 0.87 | 1 | 166.4 | cao |  |
| C12 | 4.25–4.96 | 180.7 |  | 1 | 180.7 | trung bình | < 4 bước trong phần này; peak cadence yếu |
| C12 | 6.16–7.52 | 178.8 |  | 1 | 178.8 | cao |  |
| C12 | 8.69–10.69 | 172.7 |  | 1 | 172.7 | cao |  |
| C13_1 | 0.00–11.88 | 102.9 | 0.77 | 1/2 | 205.8 | trung bình |  |
| C13_1 | 11.88–22.10 | 204.5 | 0.77 | 1 | 204.5 | cao |  |
| C13_2 | 0.00–19.92 | 203.3 | 1.14 | 1 | 203.3 | cao |  |
| C17 | 0.00–5.67 | 32.8 | 0.34 | 1/5 | 164.0 | thấp | nhiều ứng viên k: 1/5,1/6; < 4 bước trong phần này |
| C18 | 0.00–2.88 | 162.5 | 0.87 | 1 | 162.5 | cao |  |
| C18 | 2.88–6.33 | 81.3 | 0.87 | 1/2 | 162.6 | trung bình |  |
| C19 | 0.00–2.88 | 144.2 | 0.30 | 1 | 144.2 | cao |  |
| C19 | 2.88–5.38 | 34.2 | 0.30 | 1/4 | 136.8 | thấp | < 4 bước trong phần này |
| C19 | 5.38–7.21 | 140.0 | 0.30 | 1 | 140.0 | cao |  |
| C20 | 0.00–1.88 | 154.8 | 0.32 | 1 | 154.8 | cao |  |
| C20 | 1.88–6.09 | 33.2 | 0.32 | 1/5 | 166.0 | thấp | < 4 bước trong phần này |
| C21 | 0.00–1.62 | 142.1 | 0.28 | 1 | 142.1 | trung bình | < 4 bước trong phần này |
| C21 | 1.62–5.92 | 30.2 | 0.28 | 1/5 | 151.0 | thấp | < 4 bước trong phần này |
| C23_1 | 0.00–5.57 | 51.5 | 0.58 | 1/3 | 154.5 | thấp | nhiều ứng viên k: 1/3,1/4; mâu thuẫn Wu (0.58) |
| C23_2 | 0.00–2.52 | 105.4 | 1.04 | 1/2 | 210.8 | thấp | mâu thuẫn Wu (1.04) |
| C23_3 | 0.00–4.84 | 170.6 | 1.42 | 1 | 170.6 | cao |  |
| C24 | 0.00–4.62 | 30.1 | 0.37 | 1/5 | 150.5 | thấp | nhiều ứng viên k: 1/5,1/6; < 4 bước trong phần này |
| C25 | 0.00–8.74 | 53.9 | 0.59 | 1/3 | 161.7 | thấp | mâu thuẫn Wu (0.59) |
| C26 | 0.00–4.73 | 42.3 | 0.83 | 1/4 | 169.2 | thấp | nhiều ứng viên k: 1/4,1/5; < 4 bước trong phần này |
| C27 | 0.00–3.95 | 102.4 | 0.81 | 1/2 | 204.8 | thấp | mâu thuẫn Wu (0.81) |
| C28_1 | 0.00–1.41 | 167.5 |  | 1 | 167.5 | trung bình | < 4 bước trong phần này |
| C28_2 | 0.00–1.37 | 170.1 |  | 1 | 170.1 | trung bình | < 4 bước trong phần này |
| C28_3 | 0.00–1.45 | 167.1 |  | 1 | 167.1 | cao |  |
| C28_4 | 0.00–1.41 | 165.4 |  | 1 | 165.4 | trung bình | < 4 bước trong phần này |
| C29 | 0.00–10.13 | 168.0 | 1.21 | 1 | 168.0 | cao |  |
| C30 | 0.00–10.51 | 173.1 | 1.00 | 1 | 173.1 | cao |  |
| C31 | 0.00–6.08 | 172.6 | 0.98 | 1 | 172.6 | cao |  |
| C38_1 | 0.00–1.93 | 191.8 |  | 1 | 191.8 | cao |  |
| C38_10 | 0.00–5.77 | 197.6 | 0.54 | 1 | 197.6 | cao | mâu thuẫn Wu (0.54) |
| C38_2 | 0.00–3.20 | 187.8 | 0.91 | 1 | 187.8 | cao |  |
| C38_3 | 0.00–3.00 | 196.1 | 0.62 | 1 | 196.1 | cao | mâu thuẫn Wu (0.62) |
| C38_4 | 0.00–2.37 | 192.0 | 0.70 | 1 | 192.0 | cao |  |
| C38_5 | 0.00–1.80 | 186.3 |  | 1 | 186.3 | cao |  |
| C38_6 | 0.00–3.00 | 188.6 | 1.43 | 1 | 188.6 | cao |  |
| C38_7 | 0.00–2.37 | 191.3 | 1.37 | 1 | 191.3 | cao |  |
| C38_8 | 0.00–2.80 | 188.4 | 0.89 | 1 | 188.4 | cao |  |
| C38_9 | 0.00–4.07 | 190.9 | 1.48 | 1 | 190.9 | cao |  |
| F02 | 0.00–19.54 | 22.6 | 0.16 | 1/8 | 180.8 | trung bình |  |
| F10_1 | 0.00–3.85 | 184.5 | 1.61 | 1 | 184.5 | cao |  |
| F10_2 | 0.00–3.78 | 92.9 | 0.64 | 1/2 | 185.8 | trung bình |  |
| F10_2 | 13.70–15.20 | 51.2 |  | 1/3 | 153.6 | thấp | nhiều ứng viên k: 1/3,1/4; < 4 bước trong phần này; peak cadence yếu |
| F10_2 | 25.75–28.22 | 179.8 | 1.04 | 1 | 179.8 | trung bình | peak cadence yếu |
| F11 | 0.00–2.38 | 155.0 | 0.52 | 1 | 155.0 | cao |  |
| F11 | 2.38–5.82 | 40.9 | 0.52 | 1/4 | 163.6 | thấp | < 4 bước trong phần này |
| F11 | 9.65–10.62 | 87.5 |  | 1/2 | 175.0 | thấp | < 4 bước trong phần này |
| F11 | 13.62–15.52 | 47.7 |  | 1/3 | 143.1 | thấp | nhiều ứng viên k: 1/3,1/4; < 4 bước trong phần này |
| F12 | 0.00–1.43 | 162.3 |  | 1 | 162.3 | trung bình | < 4 bước trong phần này |
| F12 | 2.83–3.90 | 165.1 |  | 1 | 165.1 | trung bình | < 4 bước trong phần này |
| F17 | 0.00–6.54 | 28.4 | 0.40 | 1/5 | 142.0 | thấp | nhiều ứng viên k: 1/5,1/6; < 4 bước trong phần này |
| F18 | 0.00–2.62 | 191.4 | 0.76 | 1 | 191.4 | cao |  |
| F18 | 2.62–5.57 | 93.8 | 0.76 | 1/2 | 187.6 | trung bình |  |
| F19 | 0.00–4.93 | 27.3 | 0.31 | 1/6 | 163.8 | thấp | < 4 bước trong phần này |
| F20 | 0.00–5.78 | 20.1 | 0.32 | 1/8 | 160.8 | thấp | < 4 bước trong phần này |
| F21 | 0.00–4.85 | 30.2 | 0.33 | 1/5 | 151.0 | thấp | nhiều ứng viên k: 1/5,1/6; < 4 bước trong phần này |
| F22_1 | 0.00–2.58 | 168.4 | 1.09 | 1 | 168.4 | cao |  |
| F22_2 | 0.00–2.54 | 162.9 | 0.88 | 1 | 162.9 | cao |  |
| F22_3 | 0.00–2.49 | 171.4 | 0.78 | 1 | 171.4 | cao |  |
| F22_4 | 0.00–2.49 | 170.2 | 0.79 | 1 | 170.2 | cao |  |
| F23_1 | 0.00–2.44 | 169.6 | 1.57 | 1 | 169.6 | cao |  |
| F23_2 | 0.00–3.66 | 66.9 | 0.74 | 1/3 | 200.7 | thấp | mâu thuẫn Wu (0.74) |
| F23_3 | 0.00–3.81 | 80.0 | 0.81 | 1/2 | 160.0 | thấp | mâu thuẫn Wu (0.81) |
| F24 | 0.00–2.38 | 146.3 | 0.39 | 1 | 146.3 | cao |  |
| F24 | 2.38–4.98 | 43.2 | 0.39 | 1/3 | 129.6 | thấp | < 4 bước trong phần này |
| F25 | 0.00–10.78 | 51.2 | 0.54 | 1/3 | 153.6 | thấp | nhiều ứng viên k: 1/3,1/4; mâu thuẫn Wu (0.54) |
| F26 | 0.00–4.93 | 47.6 | 0.93 | 1/3 | 142.8 | thấp | nhiều ứng viên k: 1/3,1/4; < 4 bước trong phần này |
| F27 | 0.00–4.12 | 76.2 | 0.63 | 1/2 | 152.4 | trung bình |  |
| F28_1 | 0.00–1.41 | 170.1 |  | 1 | 170.1 | trung bình | < 4 bước trong phần này |
| F28_2 | 0.00–1.33 | 172.2 |  | 1 | 172.2 | trung bình | < 4 bước trong phần này |
| F28_3 | 0.00–1.37 | 165.5 |  | 1 | 165.5 | trung bình | < 4 bước trong phần này |
| F28_4 | 0.00–1.45 | 162.6 |  | 1 | 162.6 | trung bình | < 4 bước trong phần này |
| F29 | 0.00–2.40 | 142.5 | 1.20 | 1 | 142.5 | cao |  |
| F30 | 0.00–3.48 | 154.0 | 1.00 | 1 | 154.0 | cao |  |
