# Báo cáo kỹ thuật chi tiết: Bước 0 → Bước 4

**Dự án:** Gait cycle segmentation cho bài toán phân loại kỹ thuật chạy (CORRECT / INCORRECT) từ video RGB một camera trên mạng xã hội
**Core paper:** Ceriola et al. (2026), *Monocular Optical-Flow Method for Gait Event Detection and Foot-Strike Pattern Evaluation*, IEEE Sensors Journal 26(10) (FOOF)
**Công cụ tốc độ video:** Wu et al. (2026), *Seeing Fast and Slow*, github.com/Seeing-Fast-and-Slow/Seeing-Fast-and-Slow
**Phần cứng:** MacBook Pro M4 Pro (MPS)
**Cập nhật:** 05/11/2026
**Phạm vi:** Bước 0 (audit), Bước 1 và 1b (trục thời gian thật), Bước 2 (tracking), Bước 3 (keypoint bàn chân), Bước 4 (mask bàn chân SAM2). Bước 5 trở đi chưa chạy.

Báo cáo này thay thế và mở rộng `Module1_Video_Speed_Processing_Report.md`. Số liệu Module 1 ở đây đã cập nhật theo Bước 1b, ví dụ `abs_time_ok` hiện là 39 True / 27 False, không còn 40 / 26.

---

## 0. Tóm tắt

### 0.1 Ánh xạ framework → bước thực thi

| Module trong framework | Bước | Câu hỏi bước đó trả lời | Trạng thái |
|---|---|---|---|
| 1. Video speed processing | Bước 0, 1, 1b | Mỗi frame ứng với thời điểm thật nào? | Xong |
| 2. Subject detection & tracking | Bước 2 | Người chạy là ai, nằm ở đâu trong từng frame? | Xong |
| 3a. Persistent foot identity | Bước 3 | Chân nào là chân nào (A/B, gần/xa), keypoint bàn chân có sạch không? | Xong |
| 3b. Foot regional extraction & tracking | Bước 4 | Pixel nào thuộc bàn chân, vùng gót/giữa/mũi ở đâu, lúc nào bị che? | Xong |
| 4. Optical flow & regional features | Bước 5 | Vùng nào của bàn chân đứng yên so với mặt đất? | Prompt đã viết, chưa chạy |
| 5. Statistical gait event modeling | Bước 6 | IC/TO xảy ra lúc nào? | Chưa |
| 6. Validation | Bước 7 | Sự kiện có đúng không? | Chưa |

### 0.2 Luồng dữ liệu

```
video gốc (66)
  └─ Bước 0  audit ─────────► step0_metadata.csv, frames/<vid>_frames.csv (pts_s, is_dup, is_edit_freeze, shot_id, cam_dx…)
       └─ Bước 1  Wu + cadence ─► speed_k, continuity_id, t_real_s            (ghi thêm vào frames CSV)
            └─ Bước 1b ramp ────► speed_k_t, t_real_s_v2, time_model         (ghi thêm vào frames CSV)
                 └─ Bước 2  YOLO11m + BoT-SORT + RTMPose ─► runner/<vid>_runner.csv, keypoints_v2/<vid>_kp.npz
                      └─ Bước 3  Viterbi L/R + near/far + làm sạch ─► footkp/<vid>_footkp.npz (kp_clean, near_track…)
                           └─ Bước 4  SAM2.1 + hậu kỳ ─► masks_post/<vid>.npz (mask, vùng gót/giữa/mũi, frame_valid_mask)
                                └─ Bước 5  GMA optical flow (chưa chạy)
```

### 0.3 Kết quả chính

| Bước | Kết quả | Số liệu |
|---|---|---|
| 0 | Phát hiện vấn đề của dữ liệu | 23/66 video VFR, 43 video có frame trùng, 6 video có freeze biên tập, 3 video có cắt cảnh, 30 video camera pan vừa/mạnh |
| 0 | Hướng chạy, mirror, chân gần | `s`: 34 (+1) / 32 (−1); mirror: 15 `no`, 39 `no (overlay)`, 12 `unknown` |
| 1 | Hệ số tốc độ `k` | 32/66 video có slow-mo; 85 phần tốc độ; cadence thực 142–205 spm (trung vị 171) |
| 1b | Slow-mo dạng ramp | 12 part ở 11 video, dựng `k(t)` bằng PCHIP |
| 1b | Thời gian tuyệt đối dùng được | `abs_time_ok`: 39 True / 27 False (CORRECT 24/39, INCORRECT 15/27) |
| 2 | Tracking người chạy | Coverage trung vị 1,00 (thấp nhất 0,969); review tay: 60 ok, 6 minor, **0 frame sai người** |
| 3 | Keypoint bàn chân | Gate 58 PASS / 8 WARN / 0 FAIL; review tay 60 ok / 6 minor; near/far khớp metadata 62/64 khối |
| 4 | Mask bàn chân | Gate 59 PASS / 5 WARN / 2 FAIL; review tay 35 ok / 28 minor / 3 major; 94,1% cặp (frame, chân) hợp lệ |

### 0.4 Nguyên tắc xuyên suốt

1. **Thời gian lấy theo `t_real_s_v2`.** Không dùng frame index hay `pts_s` để tính vận tốc hay thời lượng.
2. **Chỉ xử lý frame unique.** Bỏ frame trùng (`is_dup`) và freeze biên tập (`is_edit_freeze`). Không cho một chu kỳ vắt qua hai khối `continuity_id`.
3. **Không loại cả video vì một vài frame lỗi.** Mỗi bước gắn cờ theo frame và theo chân; bước sau tự lọc.
4. **Gate tự động đi kèm review bằng mắt.** Ở Bước 4, 23 video PASS vẫn được review là `minor`, nên gate không thay được việc xem ảnh.
5. **Không sửa output đã review của bước trước.** Quyết định mới được ghi thành cờ ở bước sau (ví dụ C20 part 1, F10_2 khối 1).

---

## 1. Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Số video | **66**: 39 CORRECT, 27 INCORRECT. `F0x.mov` đã bị loại khỏi dataset |
| Hướng khung hình | 41 dọc, 25 ngang |
| Độ dài | 1,33–28,22 s, trung vị 4,71 s |
| fps khai báo | 60 (25 video), 30 (17), 23,976 (12), các giá trị khác |
| Nguồn | TikTok, YouTube, Instagram; nhiều video có chú thích, watermark, title card |
| Số frame unique dùng được | 8.424 sau Bước 1; 8.407 từ Bước 2 trở đi |

Quy ước: folder `False` được ghi là `INCORRECT` trong mọi bảng, vì pandas đọc chuỗi `FALSE` thành boolean.

Đường dẫn gốc: `/Users/ngotuananh/Documents/Running-gait/Running-Gait-Clustering/` (gọi tắt `ROOT`). Mọi output nằm trong `ROOT/data/`.

---

## 2. Bước 0: Audit dữ liệu

### 2.1 Mục tiêu

Đo mọi đặc tính của video có thể làm sai việc phát hiện sự kiện về sau. Bước này không sửa dữ liệu.

### 2.2 Input

66 file video gốc trong `data/Correct/` và `data/False/`.

### 2.3 Phương pháp

| Hạng mục | Cách đo |
|---|---|
| Timestamp thật | `ffprobe` lấy `best_effort_timestamp_time` của từng frame; giải mã bằng `ffmpeg -vsync passthrough` để không thêm hay bớt frame |
| Frame trùng | MAD ảnh xám (rộng 480 px) so với frame trước. Trùng nếu MAD < min(1,0; 0,12 × trung vị MAD frame thật). Phân bố hai mode rõ: frame trùng < 0,3, frame thật > 3 |
| Effective fps | Số frame unique / thời lượng thật. `effective_fps_active` tính sau khi bỏ freeze |
| Freeze biên tập | Chuỗi frame trùng hoặc gần tĩnh (MAD < 0,25 × trung vị) kéo dài ≥ 0,30 s |
| Cắt cảnh | Khoảng cách Bhattacharyya histogram HSV giữa hai frame liên tiếp > 0,45 |
| Chuyển động camera | Lucas–Kanade + RANSAC similarity transform giữa các frame unique → `cam_dx`, `cam_dy`, `cam_scale` |
| Góc quay | Xem trực quan contact sheet và crop |

### 2.4 Kết quả

- **VFR ở 23/66 video.** Có video khai báo 60 fps nhưng khoảng cách frame nhảy 17, 33, 50 ms, nên mọi phép tính thời gian phải dùng timestamp.
- **Frame trùng ở 43 video**, tỉ lệ trùng cao nhất 89% (C09_1, F10_2).
- **Effective fps của phần chạy:** 26 video ≥ 28 fps, 27 video 20–28 fps, 8 video 12–20 fps, 5 video < 12 fps (thấp nhất 6,6, trung vị 24,7).
- **Freeze biên tập ở 6 video:** C11 (3,9 s), C12 (4,1 s), F10_2 (20,5 s trên tổng 28 s), F11 (6,7 s), F12 (1,4 s), F30 (0,5 s).
- **Cắt cảnh ở 3 video:** C01 (sang cận cảnh bàn chân tại 2,67 s), C38_5 (title card tại 1,80 s), C38_8 (2,80 s).
- **Góc quay:** 40 lateral, 26 chéo (chủ yếu chéo phía trước).
- **Camera:** 6 tĩnh, 30 pan nhẹ, 19 pan vừa, 11 pan mạnh.
- **Clip rất ngắn:** 10 video có phần chạy < 2 s, khoảng 2 sải chân (bộ C28, F28, C38_1, C38_5).

### 2.5 Hướng chạy, mirror, chân gần camera

**Hướng chạy `s`.** Quy ước `s = +1` là chạy từ trái sang phải màn hình.
- Cách sơ bộ (chuyển động người so với nền) mâu thuẫn với keypoint ở 23 shot, và 4/5 ca kiểm tra full-res cho thấy cách sơ bộ sai. Các cột `s_prelim`, `dir_auto`, `dir_visual` chỉ giữ để tham khảo.
- Cách chốt dùng RTMPose-m Halpe26 trên khoảng 25 frame/shot. Tín hiệu chính là dấu thành phần x của vector gót → ngón (hai chân); tín hiệu phụ là mũi so với tai. Hai tín hiệu khớp nhau 96–100% ở hầu hết video.
- Kết quả: 34 video `+1`, 32 video `−1`. F10_2 được sửa tay thành `−1` sau khi kiểm tra full-res (bản đầu ghi `+1` vì detector chọn nhầm khán giả). Loại 2 shot: C38_5 shot 1 (title card) và C38_8 shot 1 (6 frame).

**Mirror** không suy ra được từ hình học, chỉ dựa vào chữ trong khung hình:

| Giá trị | Số video | Ý nghĩa |
|---|---|---|
| `no` | 15 | Có chữ thật trong cảnh đọc xuôi |
| `no (overlay)` | 39 | Chỉ có chú thích hoặc watermark đọc xuôi; chắc chắn không bị lật sau khi chèn chữ, nhưng không loại trừ việc lật trước đó |
| `unknown` | 12 | Không có chữ: C09_2, C30, C31, C38_1, C38_3, C38_4, C38_6, C38_7, C38_8, C38_9, F02, F30 |

**Chân gần camera:** `s = −1` và không lật thì chân **trái** gần camera; `s = +1` và không lật thì chân **phải**. Kết quả: 30 trái, 24 phải, 12 video chỉ dùng nhãn near/far.

### 2.6 Output

| File | Nội dung |
|---|---|
| `step0_audit/step0_metadata.csv` / `.xlsx` | 1 dòng/video, toàn bộ metadata của Module 1. Sheet README trong file xlsx giải thích từng cột |
| `step0_audit/frames/<vid>_frames.csv` | 1 dòng/frame giải mã, là trục thời gian của video (cột ở mục 3.6) |
| `step0_audit/s_keypoint.csv` | Hướng chạy theo shot: `s_foot`, `foot_agree`, `s_face`, `s_prelim`, `s_final`, `ghi_chu` |
| `step0_audit/s_keypoint_frames.csv` | Keypoint dùng để chốt hướng |
| `step0_audit/contact_sheets/`, `review/` | Ảnh kiểm tra trực quan |
| `step0_audit/frames_backup_20261104/`, `step0_metadata_backup_20261104.*` | Bản trước khi Bước 1b ghi thêm cột |
| `step0_audit/code/` | `step0_audit.py`, `shots.py`, `merge.py`, `sheets.py` |
| `scripts/a0_test.py`, `a_direction.py`, `a_review.py` | Script chốt hướng chạy |

Cột chính của `step0_metadata`: `effective_fps_active`, `n_edit_freeze`, `edit_freeze_segments_s`, `n_shots`, `cut_times_s`, `camera_view`, `camera_motion`, `s_final`, `s_final_note`, `excluded_shots`, `mirrored`, `mirror_evidence`, `near_leg_camera`, `near_leg_confidence`; và từ Bước 1/1b: `speed_parts`, `has_slowmo`, `realtime_fraction`, `cadence_real_spm`, `abs_time_ok`, `speed_flags`, `time_model`, `n_ramp_parts`, `cadence_real_spm_v2`.

---

## 3. Bước 1 và 1b: Trục thời gian thật

### 3.1 Mục tiêu

Gán cho mỗi frame unique một thời điểm thật `t_real_s_v2`, đã khử slow-motion, đổi tốc độ, VFR và freeze.

### 3.2 Clip sạch cho Wu

Đọc mã nguồn Wu cho thấy 4 ràng buộc: chỉ đọc `*.mp4`; dùng `average_rate` và coi frame cách đều; cửa sổ `num_frames/7,5` s (khoảng 2,13 s); clip ngắn hơn cửa sổ gây lỗi chia cho 0.

Vì vậy mỗi video được bỏ freeze, tách theo shot, rồi chuyển từ VFR sang CFR 30 fps theo timestamp thật (`ffmpeg -vf fps=30`, H.264 CRF 18). Kết quả là **57 clip**. 11 video không có clip đủ dài cho Wu: C28_1–4, F28_1–4, C38_1, C38_5, F12.

### 3.3 Wu et al.

Dùng checkpoint `VideoMAEv2-SpeedPredictor` và `VideoMAEv2-SpeedChangeDetector` (`--thres 0.3`) trên MPS. Kiểm chứng quy ước trên video mẫu: `4x_slomo` cho 0,225 và `20x_slomo` cho 0,051, tức speed < 1 là slow-motion và thời gian thật = thời gian video × speed.

Đánh giá:
- **Wu nhận ra slow-mo mạnh ổn định.** C09_1/2 khoảng 0,26; cụm quay bằng camera tốc độ cao (C17/F17, C19–21/F19–21, C24/F24) 0,28–0,40; F02 0,16.
- **Ở vùng 0,5–1,5 Wu nhiễu khoảng ±40%.** C13_1 cho 0,77 còn C13_2 cho 1,14 dù cùng buổi quay; bộ C38 dao động 0,54–1,48.
- Nguyên nhân chính: Wu trả một giá trị cho cả clip, trong khi nhiều clip đổi tốc độ giữa chừng.

**Kết luận:** không dùng giá trị liên tục của Wu làm hệ số hiệu chỉnh. Wu chỉ dùng để phân loại và để chọn giữa các ứng viên `k`.

### 3.4 Cadence từ keypoint

- Chạy RTMPose Halpe26 trên toàn bộ 8.424 frame unique. Người chạy được chọn theo tâm vùng chuyển động của Bước 0, rồi bám theo IoU.
- Hai tín hiệu không phụ thuộc nhãn L/R: khoảng cách ngang hai cổ chân, và dao động dọc trung điểm hông. Cả hai chuẩn hoá theo chiều dài chân và khử trend bậc 2.
- Tính tần số bước bằng Lomb–Scargle trên timestamp thật (không nội suy), với cửa sổ trượt 2 s để phát hiện đổi tốc độ.

### 3.5 Chốt hệ số `k`

1. **Tách phần tốc độ:** cadence theo cửa sổ 2 s, phân 2 cụm trong không gian log. Tách nếu tâm hai cụm lệch ≥ 1,6× và mỗi phần dài ≥ 0,5 s.
2. **Clip có cả phần nhanh và phần chậm:** phần nhanh `k = 1`; phần chậm `k` = cadence chậm / cadence nhanh, snap về {1/2, 1/3, 1/4, 1/5, 1/6, 1/8} nếu lệch < 15%.
3. **Phần đơn tốc độ:** chọn ứng viên `k` sao cho cadence thực nằm trong [140, 215] spm; nếu có nhiều ứng viên thì chọn ứng viên gần Wu nhất.
4. **Không chuẩn hoá cadence về một giá trị cố định**, vì "cadence thấp" là một trong các lỗi thuộc nhãn INCORRECT.
5. **Độ tin cậy:** `cao` khi real-time, ≥ 4 bước và peak rõ; `trung bình` khi ≥ 4 bước hoặc có tỉ lệ cadence đo trực tiếp; `thấp` khi < 4 bước, mâu thuẫn Wu, hoặc có nhiều ứng viên.

Kết quả Bước 1:
- **32/66 video có slow-mo.** Cụm C17, C19–21, C24 (và F tương ứng) không có frame trùng nên ở Bước 0 trông như 30 fps thật, nhưng thực chất là slow-mo khoảng 4× quay bằng camera tốc độ cao.
- **85 phần tốc độ:** k = 1 (48), 1/2 (14), 1/3 (8), 1/4 (4), 1/5 (6), 1/6 (1), 1/8 (4). Độ tin cậy: 36 cao, 20 trung bình, 29 thấp.
- **Cadence thực** (47 video có độ tin cậy ≥ trung bình): 142–205 spm, trung vị 171.

### 3.6 Bước 1b: slow-mo dạng ramp

Một số video đổi tốc độ ngay trong một part (speed ramp kiểu TikTok), trong khi Bước 1 gán `k` hằng cho cả part.

Cách làm:
1. Lấy tín hiệu x cổ chân near − far (chuẩn hoá theo L_leg) từ keypoint Bước 3 bản v1.
2. Mỗi bước là khoảng giữa hai lần chéo chân. Dùng bộ kích Schmitt (15% biên độ p95) để bỏ nhiễu quanh 0, và nội suy thời điểm chéo dưới mức frame.
3. Part được coi là ramp nếu `max(d)/min(d)` > 1,8 và có ≥ 4 bước.
4. Lấy `T_step` là trung vị thời lượng bước ở phần real-time tin cậy. Nếu không có thì lấy các bước ngắn nhất, và đặt `abs_time_ok = False`.
5. Tính `k_i = T_step / d_i` (chặn ≤ 1, snap nếu lệch < 15%), rồi nội suy PCHIP theo `pts_s`, mỗi bước 2 mốc (ở 15% và 85% thời lượng).
6. Cộng dồn ra `t_real_s_v2`. Video không có ramp cho `t_real_s_v2` trùng `t_real_s`, và script tự dừng nếu lệch.

Kết quả:
- 12 part ở 11 video: C17, C24, C26, C27, F10_2, F17, F19, F20, F21, F26, F27. Cả 12 đạt CV thời lượng bước < 0,15.
- Kiểm chứng độc lập bằng `corr_k_image_rate` (tương quan giữa `k` và cường độ đổi ảnh): 11/12 part ≥ 0,79. **C27 chỉ đạt 0,05, nên ramp của C27 chưa được xác nhận.**
- F27 chuyển từ `abs_time_ok = True` sang `False`.
- Cảnh báo phương pháp: giả định thời lượng bước thực là hằng trong part, nên biến thiên cadence thật trong part ramp bị xoá. Với video ramp, `t_real_s_v2` chỉ đúng về tỉ lệ tương đối giữa các pha.

### 3.7 Output

| File | Nội dung |
|---|---|
| `step1_wu/clips/`, `manifest.csv` | 57 clip CFR 30 fps và bảng ánh xạ về timestamp gốc, kèm `wu_speed`, `wu_change_windows` |
| `step1_wu/out/speeds.json`, `changes.json`, `test_speeds.json` | Output Wu |
| `step1_wu/cadence_segments.csv` | Cadence theo đoạn liên tục |
| `step1_wu/speed_parts.csv` | 85 phần tốc độ: `k`, độ tin cậy, cờ |
| `step1_wu/ramp_parts.csv`, `ramp_tested_parts.csv` | Part ramp (`cv_old`, `cv_v2`, `corr_k_image_rate`); toàn bộ 85 part đã kiểm |
| `step1_wu/plots_ramp/<vid>.png` | Thời lượng bước và `k` trước/sau |
| `step1_wu/README_ramp.md` | Tài liệu Bước 1b |
| `keypoints/<vid>_kp.npz` | RTMPose Halpe26 mọi frame unique của Bước 1 |
| `step1_wu/code/` | `make_clean_clips.py`, `pose_all_frames.py`, `cadence.py`, `decide_speed.py`, `write_time.py`, `ramp_time.py` |

**Cột trong `step0_audit/frames/<vid>_frames.csv`** (Bước 0, 1 và 1b cùng ghi vào):

| Cột | Ý nghĩa |
|---|---|
| `frame`, `pts_s` | Chỉ số frame và timestamp trong video |
| `diff_prev`, `is_dup`, `unique_id` | MAD so với frame trước, cờ frame trùng, chỉ số frame unique |
| `is_edit_freeze` | Thuộc đoạn freeze biên tập; **không phải stance** |
| `cam_dx`, `cam_dy`, `cam_scale`, `cam_inlier` | Chuyển động camera |
| `run_*` | Bbox vùng chuyển động (toạ độ ảnh phân tích 480 px) |
| `hist_dist`, `is_cut`, `shot_id` | Cắt cảnh |
| `speed_part_id`, `speed_k`, `k_confidence` | Phần tốc độ và hệ số `k` (−1 = freeze hoặc shot bị loại) |
| `continuity_id` | Khối liên tục; chu kỳ không được vắt qua hai khối |
| `t_real_s` | Thời gian thật theo `k` hằng (Bước 1) |
| `speed_k_t`, `t_real_s_v2`, `time_model` | `k(t)` sau Bước 1b, **trục thời gian chuẩn cho mọi bước sau**, `constant` hoặc `ramp` |

### 3.8 Hạn chế

1. 33/85 part có < 4 bước nên không kiểm được ramp. **C20 part 1** có tỉ số 3,78 với chỉ 2 bước, gần như chắc là ramp nhưng vẫn giữ `k` cũ; đoạn này đã được gắn `exclude_time_features` (xem mục 5.6).
2. `k` của phần slow-mo ngắn không chắc chắn; nhiều phần không phân biệt được 1/4 với 1/5.
3. Cả 12 part ramp đều không có mốc real-time thật. F20, C24, F27 cho cadence 127–138 spm, dưới dải 140–215, nên đoạn nhanh nhất có thể vẫn là slow-mo.
4. Wu không chạy được cho 11 video ngắn.
5. `step0_audit/code/merge.py` còn đường dẫn `~/work`, cần sửa trước khi chạy lại Bước 0.
6. Chạy lại `write_time.py` thì phải chạy lại `ramp_time.py` để cột mới khớp.

---

## 4. Bước 2: Phát hiện và bám người chạy

### 4.1 Mục tiêu

Xác định đúng một người chạy ở mỗi frame (kể cả video đông người), rồi chạy pose theo bbox đó.

### 4.2 Input

Video gốc; `frames/<vid>_frames.csv` (frame unique, `continuity_id`, `cam_dx`, `run_*`); `s_final`.

### 4.3 Phương pháp

1. **Detect và track:** YOLO11m (`imgsz = 640`, `conf ≥ 0,25`) với BoT-SORT có ReID. Hai thay đổi so với cấu hình mặc định: `fuse_score: False` và `match_thresh: 0.9`. Lý do: người nhỏ và fps thấp. Với cấu hình mặc định, C38_8 chỉ có track ở 49/84 frame và bị chia thành 8 ID; sau khi sửa còn 1 ID trên 82/84 frame.
2. **Nối tracklet:** nối các track nối tiếp nhau có khoảng trống, chồng ngắn (≤ 2 frame), chồng dài (≤ 0,5 s, ngưỡng tâm chặt hơn), hoặc trùng người (`absorb`), dựa trên tâm, kích thước và histogram HSV vùng thân.
3. **Chọn người chạy theo chuỗi đã nối** (không theo từng track), với điểm `0,25·s_proximity + 0,25·s_length + 0,35·s_motion + 0,15·s_size`:
   - `s_proximity`: gần tâm vùng chuyển động của Bước 0
   - `s_length`: độ dài track so với khối
   - `s_motion`: vận tốc đã bù camera theo hướng `s_final`, nhân với độ tin cậy theo độ dài track
   - `s_size`: kích thước và tỉ lệ khung
4. **Nội suy** bbox qua khoảng trống ≤ 3 frame unique.
5. **Pose:** RTMPose Halpe26 theo bbox người chạy (`keypoints_v2`).

**Quyết định quan trọng nhất là giữ `imgsz = 640`.** Ở 1280, người gần camera vượt receptive field của detector và bị tách thành 2 box:

| | Frame có bàn chân ngoài box | Frame có đầu ngoài box |
|---|---|---|
| `imgsz = 1280` | 25,3% | 33,6% |
| `imgsz = 640` | 2,9% | 0,1% |

Các khác biệt khác so với đặc tả: sửa hệ số toạ độ `sy = H/h_an` cho 10 video có tag xoay; lưu box detection thay vì box Kalman (box Kalman lệch tới 14% chiều cao); thêm điều kiện `continuity_id ≥ 0`, nên 4 video (C11, C12, F11, F12) mất 17 frame lẻ giữa đoạn freeze.

### 4.4 Kết quả

| Chỉ số | Trung vị | Thấp nhất / cao nhất |
|---|---|---|
| `coverage` (tỉ lệ frame có bbox) | 1,000 | 0,969 |
| `max_gap_s` | 0,058 s | 0,250 s |
| `kp_score_median` | 0,80 | 0,65 |
| `foot_kp_score_median` | 0,78 | 0,63 |
| `frac_frames_foot_visible` | 1,000 | 0,988 |
| `selection_margin` | 0,62 | **0,016 (F10_2)** |
| `frac_kp_outside_bbox` | 0,000 | 0,262 |
| `lr_swap_rate` của RTMPose (chỉ báo, chưa sửa) | 0,000 | 0,111 |

- Cờ: `box_truncated` ở 6 video, chủ yếu do bàn chân đang lăng nằm ngoài box trong khi keypoint vẫn đúng. `low_margin`/`near_tie` ở 1 video (F10_2).
- **Review tay: 60 ok, 6 minor, không frame nào sai người.**

| Video minor | Lý do |
|---|---|
| C01 | Shot 1 là cận cảnh, chỉ thấy từ hông xuống bàn chân |
| C30, F30 | Góc chéo phía sau, dễ bị RTMPose đảo L/R |
| C38_8 | Người rất nhỏ (khoảng 15% chiều cao khung), bị thân cây che |
| F10_2 | Đông người, margin 0,016, nhưng cả 8 frame kiểm tra đều đúng người |
| F26 | f81 RTMPose vẽ sai chân sau |

### 4.5 Output (`data/step2_tracking/`)

| File | Nội dung |
|---|---|
| `runner/<vid>_runner.csv` | **Bảng tra chính:** 1 dòng mỗi frame giải mã; `frame, pts_s, t_real_s, continuity_id, shot_id, is_dup, inherited_from, track_status, track_id, x0, y0, x1, y1, conf`. `track_status` ∈ {`tracked`, `interpolated`, `missing`, `excluded`} |
| `keypoints_v2/<vid>_kp.npz` | RTMPose theo bbox người chạy: `frame, pts_s, shot_id, kp (N×26×2), score (N×26), bbox, det, W, H, track_status, continuity_id, t_real_s` |
| `all_tracks/<vid>_tracks.csv` | Mọi người ở mọi frame: `frame, pts_s, continuity_id, track_id, x0…y1, conf` |
| `selection/<vid>_scores.csv`, `_joins.csv`, `_blocks.csv` | Điểm từng track và chuỗi, mọi cặp được xét nối, tóm tắt theo khối |
| `qc_tracking.csv` | QC 1 dòng/video |
| `review/<vid>.jpg`, `_flagged_*.jpg`; `tracking_review.csv` | Ảnh 8 frame và kết luận review tay |
| `appearance/`, `qc/`, `logs/`, `models/` | Histogram HSV, QC chi tiết, log, trọng số YOLO |
| `code/` | `run_step2.py`, `s2_track.py`, `s2_select.py`, `s2_pose.py`, `s2_report.py`, `s2_validate.py`, `botsort_reid.yaml` |

Môi trường `.venv-track`: ultralytics 8.4.202, torch 2.15.0, rtmlib 0.0.16, onnxruntime 1.31.1. YOLO chạy MPS; detector YOLOX của rtmlib chạy CPU vì CoreML lỗi với shape động.

### 4.6 Lưu ý cho bước sau

- Vị trí bàn chân lấy từ keypoint, không lấy từ bbox người.
- Keypoint thân trên ở shot cận cảnh của C01 không đáng tin.
- `s_motion` phụ thuộc chất lượng `cam_dx`. Ở F10_2 khối 1, một người trên ban công đạt `s_motion` 0,96 trong khi người chạy đạt 0; lựa chọn vẫn đúng nhờ `s_proximity` và `s_size`.

---

## 5. Bước 3: Kiểm định và làm sạch keypoint bàn chân

### 5.1 Mục tiêu

- Sửa lỗi RTMPose đảo nhãn chân trái/phải theo thời gian.
- Xác định chân gần/xa camera.
- Làm sạch 8 điểm bàn chân (cổ chân, gót, ngón cái, ngón út × 2 chân).
- Cho mỗi video một gate trước khi tạo mask.

### 5.2 Input

`keypoints_v2/`, `runner/`, `frames/` (`t_real_s_v2`, `speed_part_id`), `step0_metadata` (`near_leg_camera`, `s_final`).

### 5.3 Phương pháp

**Sửa đảo nhãn (Viterbi).** Trạng thái là cặp (S[t−1], S[t]) để dự đoán tuyến tính. Chi phí là sai lệch dự đoán của 12 điểm chân trên toạ độ đã trừ tâm hông, có trọng số theo score.
- Thêm **prior lệch pha:** d² (d là chênh x hai cổ chân) không đổi khi hoán nhãn và có chu kỳ bằng 1 bước. Giải điều chế d² cho một tham chiếu pha không phụ thuộc nhãn, đưa vào lượt Viterbi thứ hai.
- Prior chỉ bật ở part có độ tuần hoàn R ≥ 0,8 (≥ 0,7 với part ramp đã sửa thời gian).
- Độ nhạy, đo bằng swap tiêm vào chuỗi thật:

| Kiểu tiêm | Có prior | Không prior |
|---|---|---|
| Nhảy nhãn giữa pha | 98,8% | 93,4% |
| Đúng nửa chu kỳ giữa hai lần chéo chân (khó nhất) | 96,8% | 84,1% |

**Near/far.** Bốn bằng chứng tổng hợp theo khúc 8 frame, bootstrap ra `near_conf`:
- `foot_size`
- `leg_size`
- `ground_y`: chân gần chạm đất ở vị trí thấp hơn trong ảnh
- `occlusion`: khi chồng lấn, chân xa có score thấp hơn

Không khớp theo metadata; metadata chỉ dùng để kiểm chứng.

**Làm sạch.**
- (a) Loại điểm score thấp.
- (b) Loại khi độ dài gót → ngón bất thường.
- (c) Loại theo vận tốc, chỉ khi điểm nhảy đi rồi quay lại trong ≤ 5 frame. Ngưỡng **28,65 L_leg/s** là p99.5 trên `t_real_s_v2` toàn dataset.
- (collapse) Loại khi hai cổ chân chập một chỗ trong lúc tham chiếu pha cho thấy chúng phải cách xa.
- Bỏ luật (d) "ngoài bbox" ở bản v2, vì nó loại cả bàn chân đúng.

Sau đó lọc Savitzky–Golay bậc 3, 7 điểm, viết lại cho trục thời gian không đều. Lọc ngắt tại ranh giới khối, part và khoảng trống > 3 frame. Khoảng trống ≤ 3 frame được nội suy nhưng `valid_mask` vẫn False. L_leg tính theo thời gian (trung vị ±0,5 s) vì người chạy lại gần camera ở F10_1, F10_2.

### 5.4 Kết quả (bản v2)

| Chỉ số | Trung vị | Thấp nhất / cao nhất |
|---|---|---|
| `foot_valid_frac_near` | 1,000 | 0,816 |
| `foot_valid_frac_far` | 0,991 | 0,791 |
| `swap_rate` (tỉ lệ frame đã sửa) | 0,000 | 0,360 (C27) |
| `swap_rate_resid` (còn nghi hoán sau sửa) | 0,000 | 0,044 |
| `near_far_conf` | 1,000 | 0,508 |
| `phase_periodic_frac` | 1,000 | 0,000 |
| `ankle_amp_retained` (biên độ giữ lại sau lọc) | 0,987 | 0,946 |
| `outlier_any_frac` | 0,006 | 0,102 |
| Thời lượng sải trung vị | 0,70 s | 0,58–1,04 s |

- Near/far khớp metadata ở 62/64 khối; 53 video khớp hoàn toàn, 12 video không có metadata để so (mirror `unknown`), F10_2 khớp một phần.
- **Gate: 58 PASS, 8 WARN, 0 FAIL.**

| Video WARN | Lý do |
|---|---|
| C04_3 | `near_far_conf` < 0,6 (0,58) |
| C27 | Near/far không tin cậy (review tay), chu kỳ không ổn định |
| F10_2 | Near lệch metadata một phần |
| F27 | Nghi đổi tốc độ trong part (đã xử lý ở Bước 1b) |
| C13_2, C20, C29, F25 | Chu kỳ không ổn định (`phase_periodic_frac` thấp) |

- **Review tay: 60 ok, 6 minor:** C01, C04_3, C27, C38_8, F10_2, F27.
- `near_track_reliable = False` ở C04_3 và C27. Các video này chỉ dùng nhãn A/B.

### 5.5 Output (`data/step3_footkp/`)

**`footkp/<vid>_footkp.npz`:**

| Khoá | Kích thước | Ý nghĩa |
|---|---|---|
| `frame`, `pts_s`, `continuity_id`, `speed_part_id` | N | Frame unique đã xử lý ở Bước 2 |
| `t_real_s_v2`, `time_model` | N | Trục thời gian chuẩn; `ramp` hoặc `constant` (đọc với `allow_pickle=True`) |
| `kp_raw`, `kp_clean` | N×26×2 | Nguyên bản / đã sửa swap, bỏ outlier, lọc (NaN nếu không có) |
| `valid_mask`, `interp_mask` | N×26 | Quan sát thật còn lại / điểm được nội suy |
| `swap_flag` | N | Frame bị hoán nhãn so với model |
| `near_track`, `near_conf` | N | `A`/`B` và độ tin cậy, không đổi trong một khối |
| `near_track_reliable` | 1 | False ở C04_3, C27 |
| `outlier_reason` | N×26 | Bitmask: 1 score, 2 heel_toe, 4 velocity, 16 collapse |
| `score_fixed`, `L_leg`, `d_ref`, `config` | | Score theo nhãn đã sửa, chiều dài chân theo thời gian, tham chiếu pha, mã cấu hình |

**Lưu ý: slot trái trong `kp_clean` là chân A, không phải chân trái giải phẫu.** Chân A gồm chỉ số 11, 13, 15, 20, 22, 24; chân B gồm 12, 14, 16, 21, 23, 25. A/B chỉ có nghĩa trong một khối `continuity_id`.

| File khác | Nội dung |
|---|---|
| `qc_footkp.csv`, `footkp_review.csv`, `footkp_review_v2_diff.csv` | QC, kết luận review, so sánh gate v1/v2 |
| `time_feature_flags.csv` | C20 part 1 (f76–243): `exclude_time_features = True` (thêm ngày 05/11) |
| `review/`, `review/_swap_*`, `plots/<vid>_ankle.png` | Ảnh 8 frame, ảnh trước/sau sửa swap, x cổ chân theo thời gian |
| `qc/<vid>_blocks.csv`, `_perturb.csv`, `velocity_threshold.json`, `inject_results.csv` | Near/far theo khối, độ nhạy theo bbox, ngưỡng vận tốc, kết quả tiêm swap |
| `footkp_v1/`, `qc_footkp_v1.csv` | Bản v1 |
| `code/` | `run_step3.py`, `s3_common.py`, `s3_validate.py`, `s3_inject.py`… |

### 5.6 Quyết định sau review (05/11)

- **C20 part 1:** gắn `exclude_time_features`, vì gần như chắc là ramp nhưng chỉ có 2 bước.
- **Ngưỡng vận tốc:** giữ 28,65 L_leg/s. Hạ xuống khoảng 17 sẽ cắt nhầm keypoint thật của chân đang lăng nhanh.
- **F10_2 khối 1 (f523–583, 12 frame):** Bước 3 cho near = B với `near_conf` = 1,0, nhưng ảnh f545/f550 và metadata đều cho thấy A mới là chân gần. `near_conf` chỉ tính từ 1 chunk. File Bước 3 giữ nguyên; Bước 4 dùng nhãn A/B cho khối này.

### 5.7 Hạn chế

- Không có ground truth L/R. Kết quả tiêm swap đo độ nhạy với giả định chuỗi nền đúng.
- Viterbi hoán đổi cả chân. Lỗi chỉ ở bàn chân (ngón nhảy sang chân kia) chỉ bị luật (b)/(c) loại.
- Luật (b) vừa bắt lỗi thật (ngón chân xa bị đặt lên ống chân ở C13_1/2), vừa bắt nhầm bàn chân đang lăng bị co ngắn trong ảnh (F22_4, C04_1).
- C27: đoạn hoán đổi f76–f117 gần trùng đoạn đổi màu (từ f83), có thể do Viterbi sửa nhầm.
- Bằng chứng `occlusion` yếu nhất: khớp metadata 77%, so với 95% của `leg_size`.

---

## 6. Bước 4: Mask bàn chân bằng SAM2

### 6.1 Mục tiêu

- Tách mask từng bàn chân theo thời gian.
- Chia mask thành 3 vùng gót / giữa / mũi theo trục bàn chân (theo FOOF).
- Gán trạng thái che khuất và nhãn near/far.
- Đánh dấu frame nào dùng được cho optical flow.

### 6.2 Input

Video gốc, `footkp/` (kp_clean, near_track, L_leg), `runner/` (bbox), `frames/` (frame unique, `continuity_id`).

### 6.3 Phương pháp

**Cấu hình SAM2:**

| Mục | Giá trị |
|---|---|
| Model | SAM2.1 `hiera_base_plus`, `image_size = 768`, MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`); cả 66 video chạy MPS |
| Crop | Nới 15% mỗi phía, cố định theo khối, giữ độ phân giải gốc |
| Prompt | Box nới 35% L_foot; điểm dương: gót và ngón cái; điểm âm: trên chân kia và giữa ống chân của chính nó |
| Phát hiện trôi | `coverage_tol` < 0,5, hoặc diện tích ra ngoài [0,4; 2,5] × trung vị (dung sai 0,15 L_foot) |
| Re-prompt | Không giới hạn số lần; mỗi đoạn lỗi được prompt lại bằng memory mới, chỉ cho object đó; không re-prompt khi box hai bàn chân có IoU > 0,1 |
| Bỏ mảnh rời | Mảnh ≥ 5% diện tích và ≥ 20 px thuộc về chân có keypoint gần hơn; xa quá 1 L_foot thì bỏ |
| Trạng thái | occluded: diện tích < 0,25, rỗng, hoặc dính sang chân kia; partial: 0,25–0,6 hoặc IoU > 0,3; còn lại visible |
| Vùng | Gót [−0,15; 0,33], giữa (0,33; 0,66], mũi (0,66; 1,15] L_foot dọc trục; giới hạn vuông góc ±0,6 L_foot |

Lý do chọn cấu hình, đo trên video thử:

| Cấu hình | Tốc độ | `coverage_tol` F26 / C38_8 |
|---|---|---|
| `small` @1024 | 0,9 fps | Hơn 2,6 giờ cho cả dataset |
| `base_plus` @1024 | 0,74–0,8 fps | Hơn 2,6 giờ cho cả dataset |
| `small` @768 | 2,3 fps | 0,901 / 0,609 |
| **`base_plus` @768** | 2,0 fps | **0,973 / 0,696** |

So với 1024, mask ở 768 có IoU 0,96–0,97 trên F26 và 0,87–0,89 trên C38_8 (người nhỏ). Thời gian thực tế cả dataset khoảng 1,45 giờ.

**Hậu kỳ** (`s4_postprocess.py`, không chạy lại SAM2):
1. **Ankle clip:** bỏ pixel nằm phía ống chân, vượt quá đường song song với trục bàn chân đi qua cổ chân, cộng lề 0,15 L_foot. Đường cắt không bao giờ thấp hơn trục. Không clip khi trục ngắn hơn 0,4 L_foot (82/16.814 cặp).
2. **Xung đột ID** = hợp cờ trước và sau clip, vì clip chỉ làm cờ tắt chứ không sửa mask nằm sai chân.
3. **`frame_valid_mask` theo từng chân,** với `invalid_reason` dạng bit:

| Bit | Tên | Điều kiện | Làm False |
|---|---|---|---|
| 1 | occluded | Chân đó occluded | Có |
| 2 | id_conflict | Frame có xung đột ID (đặt cho cả hai chân) | Có |
| 4 | area_high | Diện tích / L_leg(t)² > 1,6 × trung vị khối | Có |
| 8 | area_low | < 0,4 × trung vị khối, mask không rỗng | Có |
| 16 | no_keypoint | Cả gót, ngón cái, ngón út cách mask > 0,2 L_foot | Có |
| 32 | perp_thick | Độ dày vuông góc > 0,7 L_foot và > 1,3 × trung vị của chân đó | Có |
| 64 | literal_only | Chỉ luật nguyên văn (chưa chuẩn hoá) bắt | Không |

4. **Gate mới:**
   - FAIL khi xung đột ID > 10% số frame, hoặc `frac_visible_near` < 0,6.
   - WARN khi `reprompt_per100` ≥ 20, `frac_mask_leak` ≥ 0,1, hoặc `frac_visible_far` < 0,4.
   - Bỏ tiêu chí `max_reprompt_block` ≥ 3 (42/42 WARN cũ chỉ vì tiêu chí này).
   - Bỏ `near_far_occlusion_agreement`: chỉ khớp 0,51–0,53, ngang tung đồng xu.
5. **F10_2:** khối 1 dùng A/B; đoạn f1052–1078 (23 frame, quay lưng) có `view_lateral = False` và không có nhãn near/far.
6. **C20:** chép cờ `exclude_time_features` vào `masks_post/C20.npz`.

### 6.4 Kết quả

Tổng: 8.407 frame unique, 784 lần re-prompt.

| Chỉ số | Trung vị | Thấp nhất / cao nhất |
|---|---|---|
| `frac_frame_valid` (cặp frame–chân hợp lệ) | 0,972 | 0,699 (C04_3) |
| `frac_valid_near` | 0,984 | 0,738 (C38_8) |
| `frac_valid_far` | 0,968 | 0,812 (C01) |
| `mean_coverage_tol` | 0,990 | 0,837 |
| `reprompt_per100` | 4,09 | 37,42 (C04_3) |
| `frac_mask_leak` trước → sau clip | 0,007 → 0,004 | 0,350 → 0,239 |
| Tỉ phần diện tích gót / giữa / mũi | 0,40 / 0,32 / 0,29 | |

Hiệu quả ankle clip: F17 giảm từ 0,350 còn 0,008, C13_2 từ 0,306 còn 0,033, C09_1 từ 0,196 còn 0,027; coverage gần như không đổi (giảm tối đa 0,004). Số video có leak ≥ 0,1 giảm từ 9 xuống 3. Toàn dataset: 94,1% cặp (frame, chân) hợp lệ; ba kiểm tra hình học mới loại thêm 3,2%.

**Gate: 59 PASS, 5 WARN, 2 FAIL** (trước hậu kỳ là 22 / 42 / 2).

| Video | Gate | Lý do | Xung đột ID | Re-prompt/100 | Leak |
|---|---|---|---|---|---|
| C04_3 | FAIL | Xung đột ID, re-prompt, leak | 11,0% | 37,4 | 0,24 |
| C38_8 | FAIL | Xung đột ID, re-prompt | 11,9% | 21,4 | 0,08 |
| C01 | WARN | Re-prompt, leak | 4,2% | 27,1 | 0,17 |
| C04_2 | WARN | Leak | 8,7% | 19,6 | 0,11 |
| C13_1 | WARN | Re-prompt | 7,9% | 27,0 | 0,03 |
| C13_2 | WARN | Re-prompt | 6,0% | 26,5 | 0,03 |
| F22_4 | WARN | Re-prompt (đúng ngưỡng) | 10,0% | 20,0 | 0,03 |

**Review tay (`footmask_review.csv`): 35 ok, 28 minor, 3 major.**

| | major | minor | ok |
|---|---|---|---|
| Gate FAIL | 2 | 0 | 0 |
| Gate WARN | 0 | 5 | 0 |
| Gate PASS | 1 (C27) | 23 | 35 |

- **Major (C04_3, C38_8, C27):** chỉ dùng frame hợp lệ; near/far không tin cậy hoặc không có.
- **Ba loại lỗi ở nhóm minor:**
  1. Mask lan lên ống chân ở frame mờ lúc swing (C28_x, F28_x, F20, F22_4, F10_2).
  2. Ở cảnh cận và mờ, hai mask dồn vào một chân (F02, F11, F18, C31).
  3. Mask bám nhầm vật khác trong khi giày thật không có mask (C26 f96 trên cột, F26 f0 trên nền, F25 f0/f15, C11 f55, C30 f204). Loại này nguy hiểm nhất.
- **Đối chiếu với `frame_valid_mask`:** bắt được 11/23 frame lỗi đã liệt kê, gồm toàn bộ lỗi loại 3. 9 frame còn sót đều là lan ống chân lúc swing, có số đo hình học bình thường. Quyết định: **không thử so màu giày.** Lỗi ở pha swing gần như không ảnh hưởng tỉ lệ pixel chậm, còn IC/TO được xác định ở pha tiếp đất.

### 6.5 Output (`data/step4_footmask/`)

**Các bước sau đọc `masks_post/`, không đọc `masks/`, và chỉ dùng frame có `frame_valid_mask = True`.**

**`masks_post/<vid>.npz`** (N = số frame unique; chiều thứ hai: 0 = slot A, 1 = slot B):

| Khoá | Kích thước | Nội dung |
|---|---|---|
| `frame`, `t`, `continuity_id`, `W`, `H` | N | Chỉ số frame, `t_real_s_v2`, khối, kích thước frame gốc |
| `rle_counts`, `rle_ptr` | M, N×2×4×2 | RLE của mask và 3 vùng (0 mask, 1 gót, 2 giữa, 3 mũi), toạ độ frame gốc |
| `state` | N×2 | 0 visible, 1 partial, 2 occluded |
| `label` | N×2 | `near`/`far`, hoặc `A`/`B` khi khối không tin cậy |
| `area`, `area_ratio`, `region_area` | N×2(×3) | Diện tích mask, tỉ lệ so với trung vị khối, diện tích từng vùng |
| `coverage`, `coverage_tol` | N×2 | Tỉ lệ keypoint bàn chân trúng mask, không và có dung sai |
| `origin`, `axis`, `axis_borrowed`, `l_foot` | N×2(×2) | Gốc (gót), vector gót → giữa hai ngón, trục mượn, L_foot |
| `iou`, `reprompt_flag`, `mask_trimmed`, `mask_id_conflict` | N, N×2 | IoU hai mask, các cờ của SAM2 |
| `frame_valid_mask`, `invalid_reason` | N×2 | **Dùng để lọc** |
| `area_ratio_norm`, `kp_mask_dist`, `perp_thickness` | N×2 | Số đo gốc của ba kiểm tra hình học |
| `view_lateral`, `exclude_time_features` | N | False ở đoạn quay lưng F10_2; True ở C20 part 1 |
| `ankle_clipped`, `ankle_clip_skipped`, `crop_window`, `init_frame`, `config`, `post_config` | | Thông tin hậu kỳ, crop, cấu hình |

Hàm giải mã RLE có trong `step4_footmask/README.md`.

| File khác | Nội dung |
|---|---|
| `masks/<vid>.npz` | Output gốc của SAM2 (config `ed4897ca59`), giữ để làm lại hậu kỳ mà không chạy SAM2 |
| `qc_footmask.csv`, `qc_post/<vid>_qc.json`, `qc_post/<vid>_conflicts.csv`, `qc_post/F10_2_view.json` | QC sau hậu kỳ (post_config `7b8f8caa7d`) |
| `qc/<vid>_reprompt.csv`, `qc/<vid>_qc.json` | Mọi lần re-prompt; QC gốc |
| `review/<vid>.jpg`, `review/_conflict_*.jpg` | 8 frame (ưu tiên frame chồng lấn nặng nhất); X đánh dấu chân không hợp lệ |
| `footmask_review.csv` | Kết luận review tay (verdict, notes) |
| `code/` | `run_step4.py`, `s4_common.py`, `s4_sam.py`, `s4_regions.py`, `s4_report.py`, `s4_postprocess.py`, `s4_validate.py`, `regen_review.py` |

**Lưu ý vận hành:** `run_step4.py` ghi đè `qc_footmask.csv` bằng bảng gốc, nên sau khi chạy phải chạy lại `s4_postprocess.py`. Khi chạy cả dataset nên dùng `caffeinate -i`; lần trước máy ngủ từ 03:41 làm F24 và F25 mỗi video mất gần 1 giờ.

### 6.6 Hạn chế

1. PASS và `frame_valid_mask = True` không đảm bảo mọi frame đều đúng; 9 frame lỗi đã biết vẫn có cờ True.
2. Ankle clip chỉ cắt ống chân của chính chân đó. Mask vẫn có thể lan sang chân kia (F17 f240).
3. Tỉ phần vùng gót phình chủ yếu không do ống chân (clip chỉ giảm 0,01–0,03), nên không so diện tích vùng gót trực tiếp giữa các video.
4. Leak còn cao ở C04_3, C01, C04_2 do mờ chuyển động.
5. Khi hai bàn chân chồng nhau, hai mask thường dính lại. Các frame này được đánh dấu partial/occluded chứ không được sửa.
6. Keypoint Bước 3 có frame dồn cả hai chân vào một bàn chân (F22_4 f15/f47, C04_3 f14/f33/f87).
7. Hệ số 1,3 của `perp_thick` chọn trên chính danh sách review, chưa kiểm định độc lập. Vẫn còn bắt nhầm, ví dụ giày dựng đứng lúc swing ở C38_3 f1/f56.

---

## 7. Chuyển giao cho Bước 5 (optical flow)

### 7.1 Điều kiện dùng dữ liệu

| Dùng cho | Điều kiện |
|---|---|
| Mọi tính toán | Frame unique, không freeze, cùng `continuity_id`; thời gian `t_real_s_v2` |
| Feature theo vùng | `masks_post`, `frame_valid_mask = True` |
| Feature theo chân gần/xa | `label ∈ {near, far}`; với C04_3, C27 chỉ dùng đặc trưng không phân biệt chân |
| Feature thời gian tuyệt đối (cadence, GCT, flight time) | `abs_time_ok = True` (39 video) và `exclude_time_features = False` |
| Feature theo %chu kỳ | Mọi video, miễn chu kỳ nằm trọn trong một `speed_part_id` và một `continuity_id` |
| Góc quay | Bỏ đoạn `view_lateral = False` (F10_2 f1052–1078) |

### 7.2 Kế hoạch Bước 5 (prompt đã viết, chưa chạy)

- **Model:** GMA (`gma-sintel`); nếu không chạy được trên MPS thì dùng `torchvision raft_large`.
- **Flow:** forward + backward giữa hai frame unique liên tiếp trong cùng khối, kiểm tra nhất quán fb.
- **Bù camera:** homography RANSAC trên flow nền ở dải đất quanh chân, loại bbox người và mask chân.
- **Chuẩn hoá:** `v = |residual| / L_leg(t) / Δt_real`; thành phần ngang nhân `s_final`.
- **Feature:** cho từng vùng gót/giữa/mũi và từng chân: median/p25/p75 của `v`, `slow_ratio_otsu` (theo FOOF), `slow_ratio_fixed`, `frac_fb_valid`.
- **Kiểm tra hợp lý không cần nhãn:**
  - Spearman giữa tốc độ flow và tốc độ keypoint, kỳ vọng > 0,7.
  - Chu kỳ heel `slow_ratio` so với cadence của Bước 1b.
  - Hai chân phải luân phiên.
- **Chạy thử 6 video:** C17, F30, C09_1, C13_2, C38_8, và video pan mạnh nhất.

---

## 8. Các quyết định đã chốt

| Ngày | Quyết định | Lý do |
|---|---|---|
| 04/11 | Loại `F0x` khỏi dataset | Quyết định của nhóm |
| 04/11 | Không dùng giá trị Wu liên tục làm `k` | Nhiễu ±40% ở vùng 0,5–1,5 |
| 04/11 | `s_final` theo keypoint, không theo chuyển động | Cách sơ bộ sai ở 23 shot |
| 04/11 | F10_2 `s_final = −1`, `near_leg_camera = left` | Kiểm tra full-res và bằng chứng Bước 3 |
| 04/11 | YOLO `imgsz = 640` | 1280 tách người gần camera thành 2 box |
| 04/11 | Bỏ luật (d) ngoài bbox ở Bước 3 | Loại nhầm bàn chân đúng |
| 05/11 | Ngưỡng vận tốc giữ 28,65 L_leg/s | Hạ xuống 17 cắt nhầm keypoint chân lăng nhanh |
| 05/11 | C20 part 1: `exclude_time_features` | Ramp chỉ 2 bước, không dựng được `k(t)` |
| 05/11 | SAM2 `base_plus` @768 | Cân bằng chất lượng và thời gian |
| 05/11 | Bỏ giới hạn 5 re-prompt/khối | Giày giống nhau làm object nhảy chân ở mỗi lần chéo |
| 05/11 | Ankle clip song song trục | Đường vuông góc cắt ngang bàn chân |
| 05/11 | Xung đột ID tính cả trước clip | Clip chỉ tắt cờ, không sửa mask |
| 05/11 | Kiểm tra hình học chuẩn hoá theo L_leg(t) | Luật nguyên văn loại gần nửa số frame đúng ở C30, F10_1, F10_2 |
| 05/11 | F10_2 khối 1 giữ A/B | Chỉ 12 frame; không sửa tay kết quả Bước 3 |
| 05/11 | Không thử so màu giày | Lỗi còn sót ở pha swing, ít ảnh hưởng IC/TO |

---

## 9. Việc còn mở

1. Bước 5–7 chưa chạy.
2. Ramp của C27 chưa được xác nhận (`corr_k_image_rate` = 0,05).
3. F20, C24, F27: đoạn nhanh nhất có thể vẫn là slow-mo (cadence 127–138 spm).
4. 12 video mirror `unknown` chỉ dùng near/far.
5. Chưa có ground truth sự kiện. Đề xuất: tự quay video 240 fps có nhãn IC/TO để kiểm định.
6. Sửa đường dẫn `~/work` trong `step0_audit/code/merge.py`.

---

## 10. Tái lập

| Môi trường | Dùng cho |
|---|---|
| `.venv-gait` (Python 3.11) | RTMPose, hướng chạy, pose Bước 1, cadence |
| `.venv-speed` (Python 3.11) | Wu et al. (torch, `av<14`, `transformers==4.49`, timm, accelerate) |
| `.venv-track` | Bước 1b, 2, 3, 4 (ultralytics, torch, rtmlib, SAM2 tại `ROOT/.third_party/sam2` commit `2b90b9f`) |

Thứ tự chạy, từ `ROOT/data`:

```bash
# Bước 1
python step1_wu/code/make_clean_clips.py
#   Wu (trong Seeing-Fast-and-Slow/speed_perception, .venv-speed): video_speed_estimation.py, speed_change_detection.py --thres 0.3
TORCH_HOME=_rtm python step1_wu/code/pose_all_frames.py
python step1_wu/code/cadence.py && python step1_wu/code/decide_speed.py && python step1_wu/code/write_time.py
# Bước 1b
../.venv-track/bin/python step1_wu/code/ramp_time.py
# Bước 2
../.venv-track/bin/python step2_tracking/code/run_step2.py && ../.venv-track/bin/python step2_tracking/code/s2_validate.py
# Bước 3
../.venv-track/bin/python step3_footkp/code/run_step3.py && ../.venv-track/bin/python step3_footkp/code/s3_validate.py
# Bước 4
caffeinate -i ../.venv-track/bin/python step4_footmask/code/run_step4.py
../.venv-track/bin/python step4_footmask/code/s4_postprocess.py
```

Thời gian chạy (M4 Pro): Bước 3 khoảng 45 s cho cả dataset; Bước 4 khoảng 1,45 giờ cho SAM2 và vài phút cho hậu kỳ.

---

## Phụ lục A: Trạng thái từng video qua các bước

Cột "Hợp lệ near/far" là tỉ lệ frame có `frame_valid_mask = True` ở Bước 4. Video không có nhãn near/far hiện theo A/B.

| Video | Nhãn | s | Slow-mo | Trục thời gian | abs_time_ok | B2 review | B3 gate / review | near/far tin cậy | B4 gate / review | Frame unique | Hợp lệ near | Hợp lệ far |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | CORRECT | −1 | ✓ | constant | ✗ | minor | PASS / minor | ✓ | WARN / minor | 96 | 0.82 | 0.81 |
| C04_1 | CORRECT | +1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 114 | 0.99 | 0.98 |
| C04_2 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | WARN / minor | 46 | 0.87 | 0.89 |
| C04_3 | CORRECT | −1 |  | constant | ✓ | ok | WARN / minor | ✗ | FAIL / major | 163 | A 0.68 | B 0.72 |
| C09_1 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 56 | 0.96 | 0.96 |
| C09_2 | CORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 98 | 0.98 | 0.97 |
| C11 | CORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / minor | 228 | 0.87 | 0.95 |
| C12 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 195 | 0.99 | 0.96 |
| C13_1 | CORRECT | −1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | WARN / minor | 662 | 0.88 | 0.85 |
| C13_2 | CORRECT | −1 |  | constant | ✓ | ok | WARN / ok | ✓ | WARN / minor | 597 | 0.90 | 0.87 |
| C17 | CORRECT | −1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 170 | 0.99 | 0.97 |
| C18 | CORRECT | −1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 108 | 1.00 | 0.97 |
| C19 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 173 | 0.99 | 0.98 |
| C20 | CORRECT | +1 | ✓ | constant | ✗ | ok | WARN / ok | ✓ | PASS / ok | 183 | 1.00 | 0.98 |
| C21 | CORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 143 | 1.00 | 0.97 |
| C23_1 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 53 | 1.00 | 0.96 |
| C23_2 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 45 | 0.98 | 0.84 |
| C23_3 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 127 | 0.97 | 0.97 |
| C24 | CORRECT | −1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 136 | 0.98 | 0.90 |
| C25 | CORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 79 | 1.00 | 0.97 |
| C26 | CORRECT | +1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / minor | 125 | 0.96 | 0.99 |
| C27 | CORRECT | +1 | ✓ | ramp | ✗ | ok | WARN / minor | ✗ | PASS / major | 114 | A 0.93 | B 0.94 |
| C28_1 | CORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 34 | 1.00 | 0.94 |
| C28_2 | CORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 33 | 1.00 | 1.00 |
| C28_3 | CORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 34 | 1.00 | 0.97 |
| C28_4 | CORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 33 | 1.00 | 0.97 |
| C29 | CORRECT | −1 |  | constant | ✓ | ok | WARN / ok | ✓ | PASS / minor | 303 | 0.96 | 0.95 |
| C30 | CORRECT | −1 |  | constant | ✓ | minor | PASS / ok | ✓ | PASS / minor | 253 | 0.98 | 0.96 |
| C31 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 147 | 0.97 | 0.97 |
| C38_1 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 58 | 0.97 | 0.97 |
| C38_10 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 173 | 0.99 | 0.92 |
| C38_2 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 96 | 0.98 | 0.97 |
| C38_3 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 90 | 1.00 | 0.98 |
| C38_4 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 71 | 1.00 | 0.99 |
| C38_5 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 54 | 1.00 | 1.00 |
| C38_6 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 90 | 1.00 | 0.99 |
| C38_7 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 71 | 0.99 | 0.96 |
| C38_8 | CORRECT | +1 |  | constant | ✓ | minor | PASS / minor | ✓ | FAIL / major | 84 | 0.74 | 0.85 |
| C38_9 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 122 | 0.98 | 0.96 |
| F02 | INCORRECT | +1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 484 | 0.98 | 0.98 |
| F10_1 | INCORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 116 | 0.92 | 0.88 |
| F10_2 | INCORRECT | −1 | ✓ | ramp | ✗ | minor | WARN / minor | ✓ | PASS / minor | 113 | 0.85 | 0.85 |
| F11 | INCORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / minor | 258 | 0.93 | 0.93 |
| F12 | INCORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 76 | 1.00 | 0.97 |
| F17 | INCORRECT | −1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 196 | 0.90 | 0.88 |
| F18 | INCORRECT | −1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 98 | 0.99 | 0.98 |
| F19 | INCORRECT | +1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 118 | 1.00 | 0.97 |
| F20 | INCORRECT | +1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / minor | 173 | 0.98 | 0.99 |
| F21 | INCORRECT | −1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 117 | 1.00 | 0.99 |
| F22_1 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 62 | 0.98 | 1.00 |
| F22_2 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 61 | 1.00 | 1.00 |
| F22_3 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 60 | 0.97 | 0.97 |
| F22_4 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | WARN / minor | 60 | 0.83 | 0.87 |
| F23_1 | INCORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 64 | 1.00 | 0.95 |
| F23_2 | INCORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 44 | 1.00 | 0.91 |
| F23_3 | INCORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 59 | 0.90 | 0.95 |
| F24 | INCORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 147 | 0.99 | 0.97 |
| F25 | INCORRECT | −1 | ✓ | constant | ✗ | ok | WARN / ok | ✓ | PASS / minor | 100 | 0.94 | 0.88 |
| F26 | INCORRECT | +1 | ✓ | ramp | ✗ | minor | PASS / ok | ✓ | PASS / minor | 133 | 0.95 | 0.97 |
| F27 | INCORRECT | +1 | ✓ | ramp | ✗ | ok | WARN / minor | ✓ | PASS / minor | 122 | 0.95 | 0.89 |
| F28_1 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 34 | 1.00 | 1.00 |
| F28_2 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 32 | 0.97 | 0.97 |
| F28_3 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 33 | 0.97 | 0.91 |
| F28_4 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 34 | 0.97 | 0.97 |
| F29 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 72 | 1.00 | 0.93 |
| F30 | INCORRECT | +1 |  | constant | ✓ | minor | PASS / ok | ✓ | PASS / ok | 84 | 0.94 | 0.99 |


## Phụ lục B: Các phần tốc độ (Bước 1)

Bảng đầy đủ 85 phần tốc độ (thời gian video, cadence, Wu, `k`, cadence thực, độ tin cậy, cờ) nằm ở `Module1_Video_Speed_Processing_Report.md`, Phụ lục B, và trong `data/step1_wu/speed_parts.csv`. Các part ramp của Bước 1b nằm ở `data/step1_wu/ramp_parts.csv`.
