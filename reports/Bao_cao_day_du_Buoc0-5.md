# Báo cáo kỹ thuật đầy đủ: Pipeline gait cycle segmentation, Bước 0 → Bước 5

**Dự án:** Gait cycle segmentation cho bài toán phân loại kỹ thuật chạy (CORRECT / INCORRECT) từ video RGB một camera trên mạng xã hội
**Core paper:** Ceriola et al. (2026), *Monocular Optical-Flow Method for Gait Event Detection and Foot-Strike Pattern Evaluation*, IEEE Sensors Journal 26(10) (FOOF)
**Công cụ tốc độ video:** Wu et al. (2026), *Seeing Fast and Slow*, github.com/Seeing-Fast-and-Slow/Seeing-Fast-and-Slow
**Phần cứng:** MacBook Pro M4 Pro (MPS)
**Cập nhật:** 05/11/2026
**Trạng thái:** Bước 0–5 đã hoàn tất và đã review. Bước 6 (phát hiện IC/TO, cắt chu kỳ) đã có thiết kế nhưng **đang tạm dừng**, chưa chạy.

Báo cáo này thay thế `Module1_Video_Speed_Processing_Report.md` và `Bao_cao_chi_tiet_Buoc0-4.md`. Nó bổ sung Bước 5, và viết lại phần **persistent foot region tracking** (Bước 3 + Bước 4) ở mức chi tiết để dùng cho phần Methods của paper.

---

## 0. Tóm tắt

### 0.1 Ánh xạ framework → bước thực thi

| Module trong framework | Bước | Câu hỏi bước đó trả lời | Trạng thái |
|---|---|---|---|
| 1. Video speed processing | Bước 0, 1, 1b | Mỗi frame ứng với thời điểm thật nào? | Xong |
| 2. Subject detection & tracking | Bước 2 | Người chạy là ai, nằm ở đâu? | Xong |
| **3. Persistent foot region tracking** | **Bước 3** (tầng định danh) + **Bước 4** (tầng vùng pixel) | Chân nào là chân nào qua toàn bộ clip? Pixel nào thuộc gót/giữa/mũi của chân đó? Lúc nào chân bị che? | Xong |
| 4. Optical flow & regional features | Bước 5 | Vùng nào của bàn chân đứng yên so với mặt đất? | Xong |
| 5. Statistical gait event modeling | Bước 6 | IC/TO xảy ra lúc nào; cắt chu kỳ | Đã thiết kế, tạm dừng |
| 6. Validation | Bước 7 | Sự kiện có đúng không? | Chưa |

### 0.2 Luồng dữ liệu

```
video gốc (66)
 └ Bước 0  audit ─────────────► frames/<vid>_frames.csv: pts_s, is_dup, is_edit_freeze, shot_id, cam_dx …
    └ Bước 1/1b  Wu + cadence + ramp ─► speed_k(t), continuity_id, t_real_s_v2
       └ Bước 2  YOLO11m + BoT-SORT + RTMPose ─► runner/<vid>_runner.csv, keypoints_v2/<vid>_kp.npz
          └ Bước 3  [TẦNG ĐỊNH DANH] Viterbi L/R + near/far + làm sạch ─► footkp/<vid>_footkp.npz
             └ Bước 4  [TẦNG VÙNG] SAM2.1 2 object + re-prompt + trạng thái + vùng + hậu kỳ ─► masks_post/<vid>.npz
                └ Bước 5  GMA + bù camera trên dải đất + feature theo vùng ─► features/<vid>.npz
                   └ Bước 6  FOOF-rule / Zeni / HSMM hai chân ─► events, cycles   (tạm dừng)
```

### 0.3 Kết quả chính

| Bước | Kết quả | Số liệu |
|---|---|---|
| 0 | Vấn đề của dữ liệu | 23/66 VFR; 43 có frame trùng; 6 có freeze biên tập; 3 có cắt cảnh; 30 camera pan vừa/mạnh |
| 0 | Hình học | `s`: 34 (+1) / 32 (−1); mirror: 15 `no`, 39 `no (overlay)`, 12 `unknown` |
| 1 | Hệ số tốc độ | 32/66 video có slow-mo; 85 phần tốc độ; cadence thực 142–205 spm |
| 1b | Ramp | 12 part ở 11 video; `abs_time_ok` 39 True / 27 False |
| 2 | Tracking | Coverage trung vị 1,00; review 60 ok / 6 minor; **0 frame sai người** |
| 3 | Định danh chân | Gate 58 PASS / 8 WARN / 0 FAIL; near/far khớp metadata 62/64 khối; độ nhạy phát hiện swap 96,8–98,8% |
| 4 | Vùng bàn chân | Gate 59 PASS / 5 WARN / 2 FAIL; review 35 ok / 28 minor / 3 major; 94,1% cặp (frame, chân) hợp lệ |
| 5 | Optical flow | Gate 45 PASS / 18 WARN / 3 FAIL; Spearman flow–keypoint trung vị 0,92; duty factor trung vị 0,29 / 0,32 |

### 0.4 Nguyên tắc xuyên suốt

1. **Thời gian lấy theo `t_real_s_v2`**, không dùng frame index hay `pts_s`.
2. **Chỉ dùng frame unique**, bỏ freeze. Một chu kỳ không được vắt qua hai khối `continuity_id`.
3. **Gắn cờ theo frame và theo chân, không loại cả video.** Bước sau tự lọc.
4. **Gate tự động luôn đi kèm review bằng mắt.** Ví dụ ở Bước 4 có 23 video PASS nhưng review là `minor`.
5. **Không sửa output đã review của bước trước.** Quyết định mới được ghi thành cờ ở bước sau.
6. **Không dùng nhãn CORRECT/INCORRECT trong bất kỳ bước xử lý nào.** Khi có hai cách xử lý, ưu tiên cách được áp dụng đồng đều giữa hai nhãn (ví dụ chọn V1 ở Bước 5).

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

1. 33/85 part có < 4 bước nên không kiểm được ramp. **C20 part 1** có tỉ số 3,78 với chỉ 2 bước, gần như chắc là ramp nhưng vẫn giữ `k` cũ; đoạn này đã được gắn `exclude_time_features` (xem mục 5.14).
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

## 5. Module 3: Persistent foot region tracking (Bước 3 + Bước 4)

### 5.1 Bài toán

Phương pháp FOOF (Ceriola) đo tỉ lệ pixel chậm trong vùng gót, giữa và mũi của **từng bàn chân**. Vì vậy mỗi frame phải trả lời được ba câu hỏi:

1. **Định danh:** đây là chân nào, và có phải đúng chân đó ở frame trước không?
2. **Vùng:** pixel nào thuộc bàn chân đó, và chia gót/giữa/mũi theo trục bàn chân ra sao?
3. **Khả dụng:** bàn chân có bị che không, mask có đáng tin không?

Các câu trả lời phải **bền theo thời gian** trong suốt một khối liên tục. Nếu định danh nhảy sang chân kia giữa chừng, chuỗi tốc độ của "chân A" sẽ ghép nửa pha chống của chân này với nửa pha swing của chân kia, và mọi sự kiện IC/TO sau đó đều sai.

**Vì sao khó trên dữ liệu này:**

| Yếu tố | Hệ quả | Video ví dụ |
|---|---|---|
| Hai giày giống hệt nhau | Bộ tách dựa vào ngoại hình (SAM2) nhảy sang chân kia ở mỗi lần hai chân chéo nhau (2 lần/stride) | C13_1 cần 179 lần re-prompt |
| RTMPose đảo nhãn L/R | Nhãn trái/phải của model đổi theo thời gian, nhất là ở góc chéo phía sau | C27 (36% frame bị đảo), C38_8 (26%) |
| Chân gần che chân xa | Chân xa biến mất ở pha chéo nhau; hai mask dính vào nhau | Mọi video lateral |
| Mirror không xác định | Không biết chân trái giải phẫu ở 12 video | C30, C31, F02, F30… |
| Mờ chuyển động lúc swing | Mask lan lên ống chân, keypoint lệch | C28_x, F28_x, F20 |
| Người nhỏ / xa | Bàn chân chỉ khoảng 17 px | C38_8 |
| Slow-mo, VFR, freeze | Khoảng cách frame trong thời gian thật thay đổi rất mạnh | C09_1 (k = 1/8), F10_2 |
| Chân trần / da liền màu | Mask lan lên cẳng chân | C13_1, C13_2 |
| Phụ đề, vật che | Mask bám chữ hoặc vật che tiền cảnh | C26 (cột), C29, F29 |

### 5.2 Kiến trúc hai tầng

```
              keypoints_v2 (RTMPose, nhãn L/R của model)
                              │
     ┌────────── TẦNG 1: ĐỊNH DANH (Bước 3, keypoint) ──────────┐
     │ 3.1 Viterbi hoán nhãn L/R + prior lệch pha  → slot A/B   │
     │ 3.2 Near/far từ 4 bằng chứng hình học       → near_track │
     │ 3.3 Làm sạch keypoint + Savitzky–Golay trên t thật       │
     │ 3.4 Gate + review                                        │
     └─────────────────────────────┬────────────────────────────┘
                                   │ kp_clean, near_track, L_leg(t)
     ┌────────── TẦNG 2: VÙNG PIXEL (Bước 4, SAM2) ─────────────┐
     │ 4.1 Prompt từ keypoint đã định danh (2 object = A, B)    │
     │ 4.2 Lan truyền + phát hiện trôi + re-prompt memory mới   │
     │ 4.3 Bỏ mảnh mask nằm trên chân kia                       │
     │ 4.4 Trạng thái visible / partial / occluded              │
     │ 4.5 Hệ toạ độ bàn chân, chia gót / giữa / mũi            │
     │ 4.6 Kiểm chéo định danh: mask_id_conflict                │
     └─────────────────────────────┬────────────────────────────┘
                                   │ masks/ (SAM2 gốc)
     ┌────────── HẬU KỲ (Bước 4, không chạy SAM2) ──────────────┐
     │ ankle clip · frame_valid_mask theo chân (bitmask)        │
     │ sửa nhãn theo review · cờ view_lateral, exclude_time     │
     └─────────────────────────────┬────────────────────────────┘
                                   ▼
                     masks_post/<vid>.npz → Bước 5
```

**Nguyên tắc thiết kế: định danh trước, vùng sau.** SAM2 tách vùng tốt nhưng không biết chân nào là chân nào khi hai giày giống nhau. Keypoint có cấu trúc khung xương, nên định danh được bằng động học. Vì vậy định danh được giải ở tầng keypoint, rồi dùng để **prompt** và **kiểm chéo** tầng mask. SAM2 không bao giờ tự quyết định định danh.

### 5.3 Biểu diễn định danh

| Khái niệm | Định nghĩa | Phạm vi |
|---|---|---|
| **Slot A / B** | Hai "đường đời" chân sau khi sửa hoán nhãn. A là chân mang nhãn L của model ở đa số frame trong khối | Trong một `continuity_id` |
| **near / far** | Chân gần / xa camera, suy ra từ hình học | Một nhãn cho mỗi khối, gán vào A hoặc B |
| **trái / phải giải phẫu** | Chỉ suy ra được khi biết `s_final` và mirror | Dùng làm metadata kiểm chứng, không dùng làm định danh |

**Vì sao không dùng trái/phải giải phẫu làm định danh chính:** 12 video không xác định được mirror, và nhãn L/R của RTMPose không ổn định. Near/far có nghĩa vật lý trực tiếp (chân gần ít bị che, ảnh to hơn), không phụ thuộc mirror, và là thứ phương pháp optical flow thực sự cần.

**Vì sao định danh chỉ có nghĩa trong một khối:** qua cắt cảnh, freeze hoặc đoạn bị loại, không có liên tục động học để nối A của khối này với A của khối sau. Mỗi khối được giải độc lập; near/far là cầu nối ngữ nghĩa giữa các khối.

### 5.4 Tầng 1a: sửa hoán nhãn L/R (Viterbi)

**Trạng thái.** Mỗi frame t có biến nhị phân S_t ∈ {0 giữ nhãn, 1 hoán nhãn}. Dự đoán tuyến tính cần 2 frame trước, nên Viterbi chạy trên **cặp trạng thái (S_{t−1}, S_t)**. Đây là nghiệm tối ưu toàn cục, không phải xấp xỉ tham lam.

**Toạ độ.** Mọi điểm được trừ tâm hông. Tâm hông không đổi khi hoán nhãn, nên phép trừ loại được chuyển động camera và chuyển động của cả người mà không phụ thuộc vào nhãn.

**Chi phí quan sát** cho 12 điểm chân (hông, gối, cổ chân, ngón cái, ngón út, gót × 2 chân):

```
pred_j(t) = p_j(t−1) + r·(p_j(t−1) − p_j(t−2)),   r = (t − t₋₁) / (t₋₁ − t₋₂)   (theo t thật)
obs(t)    = (1/12) Σ_j w_j · min(‖p_j(t) − pred_j(t)‖ / L_leg(t), 1.0)
```

- `w_j` là score nhỏ nhất của điểm đó qua 3 frame.
- Mỗi điểm bị chặn ở 1 L_leg, để một điểm hỏng không đủ kéo theo đổi nhãn.
- Ở ranh giới `speed_part_id` hoặc qua khoảng trống > 3 frame, bỏ vận tốc và dự đoán bằng vị trí frame trước.
- Mỗi lần đổi trạng thái bị phạt `SWITCH_COST` = 0,30.

**Prior lệch pha.** Gọi d(t) = s·(x_ankle_L − x_ankle_R) / L_leg (s là hướng chạy).
- d(t) đổi dấu 2 lần mỗi stride, còn **d²(t) không đổi khi hoán nhãn** và có chu kỳ bằng 1 bước.
- Giải điều chế d² trong cửa sổ Gauss σ = 1 stride cho pha θ(t) và biên độ A(t). Tham chiếu dấu cho từng frame là d_ref(t) = A(t)·cos(θ(t)/2), dấu tổng thể lấy theo lượt Viterbi đầu.
- Lượt Viterbi thứ hai thêm chi phí đơn `W_PHASE · w_ankle · |d(t)·(1−2S_t) − d_ref(t)|`.
- Prior chỉ bật khi độ tuần hoàn R ≥ 0,80 (part ramp đã sửa thời gian: ≥ 0,70). Hạ ngưỡng chung sẽ gây sửa nhầm ở đoạn tốc độ đổi chưa được sửa (C20 part 1).

**Chuẩn hoá.** Sau Viterbi, nếu quá nửa số frame của khối có S = 1 thì lật toàn bộ, để A luôn là chân mang nhãn L của model ở đa số frame.

**Độ nhạy**, đo bằng swap tiêm vào chuỗi thật (`s3_inject.py`):

| Kiểu tiêm | Có prior | Không prior |
|---|---|---|
| Nhảy nhãn giữa pha | 98,8% | 93,4% |
| Đúng nửa chu kỳ giữa hai lần chéo chân (khó nhất, vì dự đoán động học mơ hồ nhất) | **96,8%** | 84,1% |

**Kết quả trên dataset:** 12/66 video có frame bị hoán, tổng 33 sự kiện hoán. Nhiều nhất là C27 (36% frame, 1 sự kiện kéo đến hết clip), C38_8 (26%, 7 sự kiện), F27 (4,9%), F26 (3,8%), F24 (3,4%). `swap_rate_resid` (tỉ lệ frame sau sửa còn nghi hoán) có trung vị 0, cao nhất 0,044.

### 5.5 Tầng 1b: chân gần / xa camera

Bốn bằng chứng độc lập, mỗi bằng chứng dương nếu A gần hơn:

| Bằng chứng | Cách tính | Thang chuẩn hoá | Khớp metadata |
|---|---|---|---|
| `foot_size` | log(độ dài gót→ngón A / B) | 0,10 | |
| `leg_size` | log((đùi + cẳng chân) A / B); ít nhiễu hơn bàn chân | 0,05 | 95% |
| `ground_y` | So P90 toạ độ y (mức thấp nhất, gần lúc chạm đất) trong cửa sổ ±12 frame; camera cao hơn mặt đất nên chân gần chạm đất thấp hơn trong ảnh | 0,05 | |
| `occlusion` | Khi hai cổ chân cách nhau < 0,25 L_leg: score trung bình chân A − chân B; chân xa bị che nên score thấp | 0,05 | 77% (yếu nhất) |

**Tổng hợp:**
1. Gom frame thành khúc 8 frame, vì các frame liền kề không độc lập.
2. Mỗi khúc lấy trung vị từng bằng chứng, chia cho thang, chặn ±3.
3. Trọng số bằng nhau. **Không khớp tham số theo metadata**, để metadata còn dùng được làm kiểm chứng.
4. Bootstrap 1.000 lần trên các khúc cho p = P(tổng > 0). Chân near = A nếu p ≥ 0,5; `near_conf` = |2p − 1|.

**Kết quả:**
- Khớp metadata ở 62/64 khối. Phép khớp này kiểm tra đồng thời bằng chứng hình học, nhãn đa số của model và `s_final`.
- `near_far_conf` trung vị 1,00, thấp nhất 0,51.
- `near_track_reliable` = False ở C04_3 (conf 0,58) và C27 (review tay thấy màu near/far đổi chân từ f83).
- **F10_2 khối 1 là trường hợp đáng chú ý.** `near_conf` = 1,0 nhưng sai: ảnh f545/f550 và metadata cho thấy A mới là chân gần. Khối chỉ có 12 frame, tức 1 khúc, nên bootstrap không có biến thiên. **Bài học:** `near_conf` của khối rất ngắn không đáng tin; cần ngưỡng số khúc tối thiểu.

### 5.6 Tầng 1c: làm sạch keypoint bàn chân

| Luật | Điều kiện | Ghi chú |
|---|---|---|
| (a) score | score < 0,30 | |
| (b) heel_toe | Độ dài gót→ngón lệch > 40% so với trung vị | Bắt được ngón chân xa bị đặt lên ống chân (C13_1/2); bắt nhầm bàn chân đang lăng bị co ngắn trong ảnh (F22_4, C04_1) |
| (c) velocity | Vận tốc > 28,65 L_leg/s (p99,5 trên `t_real_s_v2`) **và** điểm quay lại quỹ đạo cũ trong ≤ 5 frame | Một bước nhảy đơn lẻ không cho biết phía nào sai |
| collapse | Hai cổ chân cách nhau < 0,15 L_leg trong khi tham chiếu pha cho thấy phải > 0,40 | Hoán nhãn không sửa được; hiếm (6 frame toàn dataset) |
| ~~(d) ngoài bbox~~ | Đã bỏ ở v2 | Loại cả bàn chân đúng đang lăng ra ngoài bbox; bỏ luật này làm `foot_valid_frac` chân gần tăng ở C38_8 (0,83 → 0,93), C01 (0,83 → 0,89) |

- **Lọc Savitzky–Golay** bậc 3, 7 điểm, viết lại cho trục thời gian không đều: mỗi điểm khớp đa thức trên `t_real_s_v2` thật. Ngắt tại ranh giới khối, part và khoảng trống > 3 frame. Biên độ cổ chân giữ lại trung vị 98,8%, thấp nhất 93%.
- **L_leg(t)** là trung vị trong ±0,5 s, vì người chạy lại gần camera làm kích thước gần gấp đôi trong khối (F10_1, F10_2).
- **Khoảng trống ≤ 3 frame** được nội suy (`interp_mask`), nhưng `valid_mask` vẫn False.
- **Ngưỡng vận tốc 28,65** cao hơn mức kỳ vọng khoảng 16, vì đuôi phân bố đến từ các phần slow-mo, nơi một cú nhảy pose trong dt rất nhỏ cho vận tốc lớn. Hạ xuống 17 đã bị bác vì sẽ cắt nhầm chân đang lăng nhanh.

### 5.7 Tầng 1d: gate và review của Bước 3

| Mức | Điều kiện |
|---|---|
| FAIL | `foot_valid_frac_near` < 0,7, hoặc `swap_rate_resid` > 0,05 |
| WARN | `foot_valid_frac_far` < 0,5; `near_far_conf` < 0,6; near/far lệch metadata; `phase_periodic_frac` < 0,5; `step_time_ratio` > 2; `scale_perturb_instability` > 0,05 |

- **Kiểm tra nhiễu loạn bbox:** chạy lại RTMPose 30 lần với bbox co giãn ±10%. Gót/mũi dịch trung vị bao nhiêu phần L_leg (`scale_perturb_instability`) cho biết keypoint có bám chắc hay chỉ là ảo giác của crop.
- **Kết quả:** 58 PASS / 8 WARN / 0 FAIL; review tay 60 ok / 6 minor (C01, C04_3, C27, C38_8, F10_2, F27).
- `foot_valid_frac` trung vị 1,00 (near) và 0,99 (far); thấp nhất 0,82 / 0,79.

### 5.8 Tầng 2a: khởi tạo và lan truyền SAM2

**Model:** SAM2.1 `hiera_base_plus`, `image_size` 768, MPS. Mỗi khối là một "video" SAM2 riêng, gồm các frame unique đã crop (nới 15% mỗi phía, cố định theo khối, giữ độ phân giải gốc). Hai object: 1 = slot A, 2 = slot B.

**Frame khởi tạo:** frame có cả hai chân hợp lệ và |x̄_A − x̄_B| lớn nhất, tức lúc hai chân tách xa nhau nhất, ít mơ hồ nhất.

**Prompt cho mỗi object, từ keypoint đã định danh:**
- **Box:** bao các điểm bàn chân, nới 0,35 L_foot.
- **Điểm dương:** gót và ngón cái của chính chân đó (nếu thiếu thì dùng ngón út).
- **Điểm âm:** gót và ngón cái của **chân kia** (để tách hai giày giống nhau), cộng **điểm giữa gối–cổ chân của chính nó** (để chặn mask lan lên cẳng chân ở chân trần; thiếu điểm này C13_1 bị lan).

**Lan truyền:** xuôi từ frame khởi tạo tới cuối khối, rồi ngược về đầu khối.

### 5.9 Tầng 2b: phát hiện trôi và re-prompt

**Số đo mỗi frame, mỗi object:**
- `coverage`: tỉ lệ điểm gót/ngón cái/ngón út của chính chân đó nằm trong mask.
- `coverage_tol`: như trên nhưng cho dung sai 0,15 L_foot, vì keypoint hay nằm sát mép giày nên `coverage` báo sai.
- `other`: tỉ lệ điểm của **chân kia** nằm trong mask (dung sai 0,15 L_foot).
- `area_ratio`: diện tích / trung vị diện tích của object đó trong khối.

**Frame bị coi là trôi** khi keypoint chân đó hợp lệ và: `coverage_tol` < 0,5, hoặc `area_ratio` ngoài [0,4; 2,5], hoặc mask rỗng.

**Re-prompt** chạy 3 vòng, xử lý đoạn trôi dài tối thiểu 3, 3, rồi 1 frame:
1. Trong đoạn trôi, chọn frame hai bàn chân tách xa nhất và **không chồng nhau** (IoU box hai bàn chân ≤ 0,1). Nếu cả đoạn đang chồng thì bỏ qua và ghi log `bo_qua_dang_chong`, vì prompt lúc chồng sẽ dính vào bàn chân phía trước.
2. **Reset memory** rồi prompt lại chỉ object đó, lan truyền chỉ trong đoạn lỗi (xuôi và ngược từ frame prompt).
3. Lý do dùng memory mới: nối tiếp memory cũ sẽ mang theo các frame đã nhảy sang chân kia, nên lỗi tái phát.

**Không giới hạn số lần re-prompt mỗi khối.** Đặc tả ban đầu là tối đa 5, nhưng giày giống nhau làm object nhảy chân ở **mỗi** lần chéo chân, nên 5 lần hết rất sớm. Toàn dataset có 784 lần re-prompt; trung vị 4,1 lần / 100 frame, cao nhất 37,4 (C04_3).

### 5.10 Tầng 2c: bỏ mảnh mask nằm trên chân kia

SAM2 có thể cho một object phủ **cả hai bàn chân** thành hai mảnh rời, vì giày giống nhau. Với mỗi object, khi hai bàn chân không chồng nhau:
1. Tìm thành phần liên thông có diện tích ≥ 5% mask và ≥ 20 px.
2. Mỗi mảnh thuộc về chân có keypoint gần tâm mảnh hơn. Mảnh thuộc chân kia, hoặc xa quá 1 L_foot, bị bỏ.

Kết quả: 499 frame ở 44 video được sửa. Bước này đưa C04_2 và C23_2 từ FAIL lên WARN.

### 5.11 Tầng 2d: mô hình trạng thái che khuất

| Trạng thái | Điều kiện | Ý nghĩa cho bước sau |
|---|---|---|
| **occluded** | `area_ratio` < 0,25, mask rỗng, hoặc **mask gần trùng mask chân kia (IoU > 0,8) mà `coverage_tol` < 0,5** (mask đang nằm ở chân kia) | Không có quan sát; Bước 6 phải marginalize, **không coi là swing** |
| **partial** | `area_ratio` 0,25–0,6, hoặc IoU hai mask > 0,3 | Dùng được nếu qua các kiểm tra hậu kỳ |
| **visible** | Còn lại | Dùng được |

Mask ở frame occluded **không được nội suy**: không có pixel nào được bịa ra.

Tỉ lệ visible thấp nhất (theo nhãn near/far) là F10_1: near 0,76, far 0,75 (đám đông, vật che tiền cảnh). Ở các khối chỉ có A/B, thấp nhất là F10_2 (A 0,69, B 0,63, gồm đoạn quay lưng) và C04_3 (A 0,77, B 0,74).

### 5.12 Tầng 2e: hệ toạ độ bàn chân và chia vùng

**Gốc và trục:** gốc O = gót; trục **u** = (trung điểm ngón cái–ngón út − gót) / ‖·‖. Nếu frame thiếu điểm, mượn trục của frame hợp lệ gần nhất trong ≤ 3 frame unique (`axis_borrowed`); thiếu gót thì dời gót của frame mượn theo độ dịch trọng tâm mask.

Mỗi pixel x của mask có toạ độ dọc và vuông góc, chuẩn hoá theo L_foot của khối:

```
p = ((x − O) · u) / L_foot          q = ((x − O) × u) / L_foot
```

| Vùng | Khoảng p | Điều kiện q |
|---|---|---|
| Gót | [−0,15; 0,33] | \|q\| ≤ 0,6 |
| Giữa | (0,33; 0,66] | \|q\| ≤ 0,6 |
| Mũi | (0,66; 1,15] | \|q\| ≤ 0,6 |
| Ngoài vùng | còn lại | tính vào `frac_mask_leak` khi > 30% pixel |

- Giới hạn vuông góc ±0,6 được thêm vào vì nếu không, ống chân chiếu lên trục sẽ rơi vào vùng gót.
- Mũi kéo dài tới 1,15 và gót lùi tới −0,15 vì keypoint nằm bên trong mép giày, nên L_foot đo bằng keypoint ngắn hơn giày thật.

**Kết quả:** tỉ phần diện tích gót / giữa / mũi trung vị 0,40 / 0,32 / 0,29 (gót rộng 0,48 L_foot, giữa chỉ 0,33 L_foot).

### 5.13 Kiểm chéo định danh giữa hai tầng: `mask_id_conflict`

Đây là cơ chế để **tầng mask kiểm tra tầng keypoint** và ngược lại. Một mask bị coi là xung đột định danh khi:
- hai bàn chân đủ tách nhau (khoảng cách tâm ≥ 0,5 L_foot), **và**
- `other` ≥ 0,5 và `other` > `coverage_tol`, tức mask chủ yếu nằm trên keypoint của chân kia.

Khi xung đột, cả hai chân ở frame đó bị gắn bit `id_conflict` và `frame_valid_mask` = False.

**Lấy hợp cờ trước và sau ankle clip.** Nếu chỉ tính sau clip, số frame xung đột giảm từ 180 xuống 111 và không còn video FAIL. Nhưng đó không phải sửa thật: clip chỉ cắt bớt phần mask phủ keypoint chân kia nên cờ tắt, trong khi mask vẫn nằm sai chân (ví dụ C04_3 f14, f23, f87).

Toàn dataset: 180 frame xung đột (2,1% số frame); hai video vượt 10% là C04_3 (11,0%) và C38_8 (11,9%).

Xung đột có thể do mask sai **hoặc** do keypoint dồn cả hai chân vào một bàn chân (thấy ở F22_4 f15/f47, C04_3 f14/f33/f87). Cơ chế này không phân biệt được hai nguyên nhân, nên chỉ loại frame chứ không sửa.

### 5.14 Hậu kỳ: ankle clip và cờ hợp lệ theo chân

**Ankle clip.** Bỏ pixel ở phía ống chân, vượt quá **đường song song với trục bàn chân** đi qua cổ chân, cộng lề 0,15 L_foot.
- Quyết định ban đầu ghi "đường vuông góc với trục qua cổ chân". Làm đúng nghĩa đen thì đường đó **cắt ngang bàn chân**: cổ chân nằm ở 0,13–0,18 L_foot dọc trục, nên 56–66% mask nằm phía mũi. Bản đã sửa đo khoảng cách theo hướng vuông góc với trục.
- Phía ống chân xác định theo gối nếu gối lệch trục ≥ 0,2 L_foot, không thì theo cổ chân. Đường cắt không bao giờ thấp hơn trục.
- Không clip khi trục ngắn hơn 0,4 L_foot (82/16.814 cặp).
- Hiệu quả: `frac_mask_leak` của F17 giảm 0,350 → 0,008, C13_2 0,306 → 0,033, C09_1 0,196 → 0,027; coverage giảm tối đa 0,004. Số video có leak ≥ 0,1 giảm từ 9 xuống 3.

**`frame_valid_mask` theo từng chân,** kèm `invalid_reason` dạng bit:

| Bit | Tên | Điều kiện | Làm False |
|---|---|---|---|
| 1 | occluded | Trạng thái occluded | Có |
| 2 | id_conflict | Xung đột định danh (cả hai chân) | Có |
| 4 | area_high | Diện tích / L_leg(t)² > 1,6 × trung vị khối | Có |
| 8 | area_low | < 0,4 × trung vị khối, mask không rỗng | Có |
| 16 | no_keypoint | Cả gót, ngón cái, ngón út cách mask > 0,2 L_foot | Có |
| 32 | perp_thick | Độ dày vuông góc > 0,7 L_foot **và** > 1,3 × trung vị của chân đó | Có |
| 64 | literal_only | Chỉ luật nguyên văn (chưa chuẩn hoá) bắt | Không (để tra cứu) |

Chuẩn hoá theo L_leg(t)² và theo trung vị của chính chân đó là bắt buộc. Luật nguyên văn loại nhầm 26% frame của F10_1 (người chạy lại gần camera), 48% của C30 và 26% của F30 (giày dài hơn L_foot đo bằng keypoint).

Kết quả: 94,1% cặp (frame, chân) hợp lệ. Ba kiểm tra hình học loại thêm 3,2%; theo video trung vị 1,6%, cao nhất 14,1% (C04_3).

**Sửa nhãn theo review:**
- F10_2 khối 1 dùng A/B (xem 5.5).
- F10_2 f1052–1078 (23 frame, người chạy quay lưng; phát hiện bằng độ rộng hông / chiều dài thân ≥ 0,30) có `view_lateral` = False và không có nhãn near/far.
- C20 part 1 có `exclude_time_features` = True.

### 5.15 Gate và review của Bước 4

| Mức | Điều kiện |
|---|---|
| FAIL | Xung đột ID > 10% số frame, hoặc `frac_visible_near` < 0,6 |
| WARN | `reprompt_per100` ≥ 20, `frac_mask_leak` ≥ 0,1, hoặc `frac_visible_far` < 0,4 |

Gate trải qua hai lần sửa:
- **Bỏ `max_reprompt_block` ≥ 3.** Sau khi bỏ giới hạn 5 lần, 42/42 WARN cũ chỉ vì tiêu chí này, kể cả C23_1 có visible 1,00/1,00.
- **Bỏ `near_far_occlusion_agreement`.** Phép kiểm "logit SAM2 cho biết chân nào ở trước" chỉ khớp 0,51–0,53 trên 526 frame, tức tung đồng xu, vì SAM2 suy luận từng object độc lập.

**Kết quả:** gate 59 PASS / 5 WARN / 2 FAIL (trước hậu kỳ: 22 / 42 / 2).

**Review tay toàn bộ 66 video** (`footmask_review.csv`): 35 ok, 28 minor, 3 major (C04_3, C38_8, C27).

| | major | minor | ok |
|---|---|---|---|
| Gate FAIL | 2 | 0 | 0 |
| Gate WARN | 0 | 5 | 0 |
| Gate PASS | 1 (C27) | 23 | 35 |

**Phân loại lỗi tìm thấy khi review:**

| Loại | Mô tả | Video | Mức nguy hiểm | Đã xử lý? |
|---|---|---|---|---|
| L1 | Mask lan lên ống chân ở frame mờ lúc swing | C28_x, F28_x, F20, F22_4, F10_2, F12 | Thấp: ống chân lúc swing cũng nhanh, không tạo pha chống giả | Một phần (bit 4, 32); 9 frame còn sót |
| L2 | Cận cảnh mờ: hai mask dồn vào một chân | F02, F11, F18, C31, C11 | Trung bình | Phần lớn được đánh dấu partial/occluded |
| L3 | Mask bám vật khác trong khi giày thật không có mask | C26 f96 (cột), F26 f0 (nền), F25 f0/f15, C11 f55, C30 f204 | **Cao**: tạo pha chống giả | **Đã loại hết** (bit 2, 16, 4) |
| L4 | Keypoint dồn hai chân vào một bàn chân | F22_4, C04_3 | Cao | Loại qua `id_conflict` |
| L5 | Mask đúng nhưng bị loại nhầm (giày dựng đứng lúc swing, bàn chân bị mép ảnh cắt) | C38_3 f1/f56, C38_4 f13, F30 f5/f54 | Thấp (mất dữ liệu) | Chấp nhận |

Đối chiếu 23 frame lỗi đã liệt kê khi review: `frame_valid_mask` bắt 11, gồm **toàn bộ lỗi loại L3**. 9 frame sót đều thuộc L1. Quyết định: **không thử so màu giày**, vì L1 nằm ở pha swing và gần như không ảnh hưởng IC/TO.

**Kiểm tra thiên lệch theo nhãn** (chỉ để chẩn đoán): `frac_frame_valid` trung vị 0,974 (CORRECT) và 0,967 (INCORRECT); `foot_valid_frac_near` của Bước 3 đều là 1,00. Ba video major đều là CORRECT, còn INCORRECT có tỉ lệ minor cao hơn (16/27 so với 12/39). Lý do chính là các bộ INCORRECT có cận cảnh mờ (F28_x, F22_x). Chưa thấy dấu hiệu chất lượng tracking khác nhau một cách hệ thống theo nhãn, nhưng cần theo dõi khi phân loại.

### 5.16 Bằng chứng ở tầng sau rằng định danh đã bền

Bước 5 là phép thử độc lập cho tầng định danh. Nếu A/B hay near/far nhảy giữa chừng, tốc độ flow của "một chân" sẽ trộn pha chống của chân này với pha swing của chân kia.

- Spearman giữa tốc độ flow (trên mask) và tốc độ keypoint (trên `kp_clean`) cùng chân: trung vị 0,92, 75% video ≥ 0,86.
- Ở các video tốt (F29, C30, F02, F18), bậc pha chống của chân gần và chân xa **luân phiên đều suốt clip**, đúng nhịp cadence.
- Chỉ số pha chống dùng mask theo vùng (max ba vùng): duty factor trung vị 0,29 / 0,32, độ dài một lần chống 0,19 s, khoảng 1,1–1,2 đoạn chống mỗi stride.

### 5.17 Bài học thiết kế

1. **Định danh phải giải ở tầng có cấu trúc (keypoint), không ở tầng ngoại hình (mask).** Với giày giống nhau, SAM2 nhảy chân ở mỗi lần chéo chân.
2. **Dùng d² làm tham chiếu pha không phụ thuộc nhãn** là mấu chốt để bắt kiểu hoán khó nhất (giữa hai lần chéo chân): độ nhạy tăng từ 84% lên 97%.
3. **Re-prompt phải dùng memory mới** và không prompt lúc hai chân đang chồng nhau.
4. **Điểm âm trên cẳng chân của chính nó** ngăn mask lan ở chân trần.
5. **Mọi ngưỡng hình học phải chuẩn hoá theo kích thước người theo thời gian** (L_leg(t)) và theo chính chân đó. Ngưỡng tuyệt đối loại nhầm hàng loạt.
6. **Không để một bước sửa chỉ làm cờ biến mất.** Cờ xung đột được lấy trên cả mask trước clip.
7. **Gate theo video không đủ:** 23 video PASS vẫn có lỗi khi review; cần cờ theo frame và theo chân.
8. **Độ tin cậy bootstrap trên rất ít khúc là ảo** (F10_2 khối 1).

### 5.18 Hạn chế

1. Không có ground truth L/R hay mask. Độ nhạy Viterbi đo bằng tiêm swap trên chuỗi đã sửa, giả định chuỗi nền đúng; chất lượng mask chỉ có review bằng mắt.
2. Viterbi hoán cả chân. Lỗi chỉ ở bàn chân (ngón nhảy sang chân kia) chỉ được luật (b)/(c) loại.
3. Định danh không nối qua các khối; near/far là cầu nối duy nhất và phụ thuộc chất lượng bằng chứng hình học.
4. Khi hai bàn chân chồng nhau, mask không được sửa mà chỉ bị đánh dấu.
5. 9 frame lan ống chân lúc swing vẫn có cờ hợp lệ. Hệ số 1,3 của `perp_thick` được chọn trên chính danh sách review.
6. `view_lateral` chỉ được xét ở F10_2; các video khác mặc định True.
7. Tỉ phần vùng gót còn phụ thuộc hình dạng giày (cổ giày nằm trong vùng gót), nên không so diện tích vùng giữa các video.

### 5.19 Hướng cải thiện

- Gán nhãn tay một tập nhỏ (ví dụ 200 frame, 10 video) để đo IoU mask và độ đúng A/B.
- Dùng optical flow (Bước 5) để kiểm tra nhất quán thời gian của mask: mask(t+1) ≈ warp(mask(t)).
- Thêm số khúc tối thiểu cho `near_conf`.
- Đặc trưng ngoại hình theo khối (màu giày trái/phải nếu khác nhau) như một bằng chứng định danh bổ sung.
- Kiểm tra `view_lateral` tự động cho mọi video.

### 5.20 Đoạn mô tả phương pháp (bản tiếng Anh cho paper)

> **Persistent foot region tracking.** Foot identity is resolved at the keypoint level before any pixel-level segmentation. Within each temporally continuous block, left/right label swaps of the 2D pose estimator (RTMPose, Halpe-26) are corrected by a second-order Viterbi decoder over the swap state, whose observation cost is the score-weighted, leg-length-normalised linear-prediction error of the 12 lower-limb keypoints expressed relative to the pelvis centre, and whose unary prior is a label-invariant phase reference obtained by demodulating the squared inter-ankle distance (period = one step). Swap-injection experiments give a detection sensitivity of 96.8–98.8%. The near/far foot is then inferred per block from four geometric cues (projected foot length, projected leg length, ground-contact height and occlusion-induced confidence drop), aggregated over 8-frame chunks with bootstrap confidence, without fitting to metadata (agreement with metadata-derived expectations: 62/64 blocks). Cleaned keypoints (score, heel–toe length, plateau-velocity and collapse rules; Savitzky–Golay filtering on the real-time axis) prompt a two-object SAM2.1 video predictor with a padded box, positive heel and hallux points, and negative points on the contralateral foot and the ipsilateral shank. Drift (tolerance-based keypoint coverage < 0.5 or area ratio outside [0.4, 2.5]) triggers segment-wise re-prompting with a fresh memory, skipped while the two feet overlap; disconnected mask components closer to the contralateral foot are removed. Each foot mask is assigned a visibility state (visible / partial / occluded), split into heel, midfoot and toe regions in a foot-aligned coordinate frame (origin at the heel, axis to the toe midpoint, bounds normalised by foot length), clipped above the ankle, and cross-checked against keypoint identity (mask–identity conflict). A per-frame, per-foot validity mask combines occlusion, identity conflict and size- and shape-normalised geometric checks; 94.1% of frame–foot pairs are retained.

---
## 6. Module 4: Optical flow và feature theo vùng (Bước 5)

### 6.1 Mục tiêu

Đo tốc độ của từng vùng gót / giữa / mũi của từng bàn chân **so với mặt đất**, theo chiều dài chân trên giây thật. Đây là tín hiệu để Bước 6 tìm pha chống, nơi bàn chân đứng yên so với đất.

### 6.2 Input

`masks_post/` (mask, vùng, `frame_valid_mask`, nhãn), `footkp/` (`kp_clean`, L_leg), `runner/` (bbox), `frames/` (`t_real_s_v2`, `continuity_id`, `shot_id`, `speed_part_id`), video gốc.

### 6.3 Phương pháp

**Optical flow.**
- Dùng GMA (checkpoint `gma-sintel`, 12 vòng lặp) trên MPS; kết quả khớp CPU nên không cần chuyển sang RAFT. Tăng lên 32 vòng lặp gần như không đổi gì; tăng độ phân giải 1,5× còn làm Spearman giảm.
- Cặp frame là 2 frame unique liền nhau trong cùng `continuity_id` và `shot_id`.
- Crop nửa dưới bbox người, resize cố định theo khối sao cho L_foot ≥ 48 px và cạnh dài ≤ 640 px (tới 800 px nếu cần giữ L_foot).

**Kiểm tra forward–backward (FB): chỉ dùng cho QC.** Trên bàn chân mảnh và nhanh, FB loại đúng những pixel có flow đúng và giữ pixel bám theo nền. Ở C13_2, frame có keypoint > 5 L_leg/s: keypoint cho 8,4, flow không lọc 6,4, flow lọc FB chỉ 3,2. Bản lọc FB vẫn được lưu (`fb_*`).

**Bù chuyển động camera** (đã sửa một lần sau khi phát hiện bù thiếu ở 33/66 video):

*A. Bù trực tiếp trên dải đất*, áp dụng cho mọi video:
1. Dải đất từ 0,5 L_foot phía trên tới 1,0 L_foot phía dưới điểm chân thấp nhất, trừ mask chân đã nới và ống chân.
2. Mặt nạ chữ cho cả video: ô có chi tiết cao mà đứng yên tuyệt đối (< 0,5 px) ở ≥ 3 cặp được coi là phụ đề. Bước này sửa lỗi phụ đề kéo phép fit về "camera đứng yên": ở C13_2, P90 nền dư giảm từ 20 px còn 0,3 px.
3. Chỉ dùng pixel có chi tiết theo cả hai hướng; chọn cụm độ dịch lớn nhất trong các cụm đủ lớn.
4. Mô hình tịnh tiến, không dùng affine.
5. Theo thời gian: loại giá trị đổi hướng hoặc tụt quá nửa; cặp không đo được thì nội suy.

*B. Thành phần biến thiên chậm*, chỉ khi A vẫn thiếu (P10 của vx < −1 L_leg/s):
- Spline có nút cách 1 stride, fit theo P10 vx của cả hai chân trong cửa sổ 1 stride, chặn trong khoảng P10 đo được. Hiệu chỉnh này được trừ khỏi cả pixel và keypoint.
- **Không phải ràng buộc "chân chống đứng yên" theo từng frame.** Ràng buộc đó bị bác vì gây lỗi vòng tròn: nó tự tạo pha chống, kể cả ở pha bay, và xoá lỗi trượt chân thật của video INCORRECT. Spline chỉ dịch mức nền chậm hơn 1 stride, nên không dời được thời điểm chạm đất.

*Hai biến thể:*

| | Ngưỡng kích hoạt B | Video `direct+slow` | CORRECT | INCORRECT |
|---|---|---:|---:|---:|
| **V1 (mặc định)** | P10 vx < −1 | 25 | 15/39 (38%) | 10/27 (37%) |
| V2 (ablation) | P10 vx < −2 | 16 | 7/39 (18%) | 9/27 (33%) |

**Chọn V1** vì tỉ lệ dùng bước B cân bằng giữa hai nhãn, tránh tiền xử lý khác nhau một cách có hệ thống theo nhãn.

**Chuẩn hoá:** v = |residual| / L_leg(t) / Δt; vx nhân `s_final` để dương là hướng chạy.

**Δt cho "VFR giả".** Ở nhiều video, pts xen kẽ (ví dụ 33/50 ms) nhưng chuyển động thật không đổi theo. Đây là lệch khi đóng gói file, không phải frame cách nhau không đều. Quy tắc cho mỗi video:
1. Nếu Δt không rung (`dt_jitter` < 0,05): giữ `raw`.
2. Nếu Δt tương quan với độ dịch của `kp_raw` hoặc camera (rho ≥ 0,4): giữ `raw`.
3. Còn lại dùng `smooth` (trung bình trượt 5 cặp, giữ tổng thời gian).

Kết quả: 37 video `smooth`, 29 `raw`. Ở F30, độ răng cưa của flow giảm từ 0,83 còn 0,36.

**Feature** theo từng chân và từng vùng (gót, giữa, mũi, cả bàn chân):
- Phân vị của v: `v_med`, `v_p25`, `v_p75`; thành phần `vx_med`, `vy_med`.
- `slow_ratio` (tỉ lệ pixel có v < 2,0 L_leg/s), cùng các ngưỡng 1,0 / 1,5 / 3,0 và Otsu.
- Histogram của v, để đổi ngưỡng mà không tính lại flow.
- `frac_fb_valid` và tốc độ keypoint.

Ngưỡng 2,0 có cơ sở vật lý: L_leg ≈ 0,9 m nên 2 L_leg/s ≈ 1,8 m/s; bàn chân lúc chống ≈ 0, lúc swing khoảng 8–10 L_leg/s. Otsu bị loại làm ngưỡng chính vì không ổn định (4–14 L_leg/s tuỳ video).

**Cờ fit camera theo frame** (`cam_fit_bad`): tỉ lệ inlier thấp; nền dư cao (sàn 1 px); cả hai chân cùng ở đuôi phân phối khi camera đang pan; hiệu chỉnh chậm lớn bất thường (C38_3 f0–13). Tổng 232/8.331 cặp (2,8%).

### 6.4 Kiểm tra hợp lý (không cần nhãn)

| Kiểm tra | Cách tính | Kết quả |
|---|---|---|
| Spearman flow–keypoint | Giữa `v_med` cả bàn chân và tốc độ `kp_clean` cùng chân | Trung vị **0,92** (trước khi sửa bù camera: 0,85) |
| Cadence | Lomb–Scargle của `slow_ratio` gót, **tính riêng từng speed_part**, so với tham chiếu Bước 1b hoặc Bước 3 | 0 video FAIL vì cadence (trước khi sửa: 6). C19 FAIL trước đó do gộp qua các part có tốc độ khác nhau |
| Luân phiên hai chân | Tương quan near–far của `slow_ratio` gót | Trung vị −0,31 (V1); 58/66 video âm. Chỉ để tham khảo, vì bị kéo về 0 khi phần lớn frame có slow_ratio = 0 ở cả hai chân |
| Pha chống (chỉ QC) | Chân chống khi max(slow_ratio 3 vùng) > 0,5; gộp khoảng hở ≤ max(1 frame, 30 ms) | Duty factor 0,29 / 0,32; một lần chống 0,19 s; 1,11 / 1,19 đoạn/stride; 12 video đạt cả 3 khoảng ở cả hai chân |

Đoạn chống ban đầu trông bị "vụn" (trung vị 1–2 frame) chủ yếu do cách đếm: một frame tụt lẻ giữa pha chống làm một lần chống thành 2–3 đoạn, và việc chỉ xét gót bỏ sót người chạy đáp mũi. **Feature không bị làm mượt**; nhiễu một frame để mô hình thời lượng của Bước 6 xử lý.

### 6.5 Kết quả

**Gate:**
- FAIL khi Spearman < 0,4, hoặc cadence lệch > 15% mà vượt độ phân giải tần số.
- WARN khi Spearman < 0,7, nền dư cao, `frac_fb_valid` < 0,7, hoặc bù camera thiếu.

Kết quả: **45 PASS, 18 WARN, 3 FAIL**.

| Gate | Video |
|---|---|
| FAIL (Spearman < 0,4) | C04_3 (0,34), C04_2 (0,37), C09_2 (0,37) |
| WARN (Spearman < 0,7 và FB < 0,7) | C38_8 |
| WARN (nền dư cao và FB < 0,7) | C01, C28_4, C29, F22_1, F22_2, F22_3, F22_4, F28_1 |
| WARN (chỉ FB < 0,7) | C12, C13_1, C13_2, C28_1, C28_2, C28_3, F28_2, F28_3, F28_4 |

- Cả 3 video FAIL đều là CORRECT; C04_3 cũng là video major của Bước 4.
- Không đo được mặt đất ở C28_2, F28_2, F28_4: toàn bộ phần bù đến từ bước B.

**Ví dụ minh hoạ (plot trước/sau sửa bù camera):**
- **F29:** trước khi sửa, cadence lệch 98% (đo được hai cực tiểu mỗi stride). Sau khi sửa, có bậc pha chống 6–8 frame ở gót và mũi, hai chân luân phiên, mỗi bậc trùng cực tiểu tốc độ keypoint.
- **C38_9:** Spearman 0,98 ở cả hai bản, nhưng V2 chỉ có gai 1–3 frame vì nền còn trôi khoảng 2 L_leg/s. V1 xử lý được trường hợp này.

### 6.6 Output (`data/step5_flow/`)

| File | Nội dung |
|---|---|
| `features/<vid>.npz` | Feature V1 (mặc định), N frame × 2 chân × 4 vùng: `v_med`, `v_p25`, `v_p75`, `vx_med`, `vy_med`, `slow_ratio*`, `v_hist`, `fb_*`, `kp_speed`, `comp_*`, `dt_*`, `flow_valid`, `cam_fit_bad`, cờ của Bước 4 |
| `features/<vid>__v2.npz` | Feature V2, để ablation |
| `flow_store/<vid>.npz` | Flow forward thô (float16) và mặt nạ FB trên hộp bao mask chân + dải đất, tổng 781 MB; đổi cách bù không cần chạy lại GMA |
| `qc_flow.csv`, `qc/<vid>_qc.json`, `qc/<vid>_cadence_parts.csv` | QC; cột `v2_*` cho V2 |
| `plots/<vid>.png`, `review/<vid>.jpg` | slow_ratio gót/mũi của near/far kèm tốc độ keypoint; 8 frame có residual flow |
| `baseline_v1/` | Feature trước khi sửa bù camera (ablation; tên "v1" ở đây không liên quan biến thể V1) |
| `README.md`, `code/`, `s5_validate.py` | `run_step5.py`, `s5_common.py`, `s5_store.py`, `s5_flow.py`, `s5_ground.py`, `s5_features.py`, `s5_checks.py`, `s5_report.py` |

Môi trường: GMA tại `ROOT/.third_party/GMA` (commit `2f1fd29`), thêm `einops` vào `.venv-track`. GMA mất khoảng 45 phút, tính feature khoảng 5 phút.

### 6.7 Hạn chế

1. Tham số bù camera được chỉnh trên 12 video thử rồi áp dụng cho 54 video còn lại.
2. Bước B có thể làm giảm khác biệt thật nếu bàn chân trượt. So V1 với V2 để ablation.
3. Spearman ở video `direct+slow` có thể được nâng một phần, vì keypoint và flow bị trừ cùng hiệu chỉnh.
4. Δt làm mượt vẫn là ước lượng. Với C09_1, C09_2 (k = 1/8, độ tin cậy thấp), thời lượng pha chống theo giây không có ý nghĩa.
5. Chỉ số pha chống chưa đạt ở đa số video (12/66 đạt cả 3 khoảng). Không có mảng đoạn chống trong features; Bước 6 phải tự xác định.

---

## 7. Bước 6: Thiết kế (tạm dừng, chưa chạy)

**Mục tiêu:** tìm IC/TO của từng chân và cắt video thành chu kỳ IC → IC cùng chân. Mỗi chu kỳ là một instance cho phân loại MIL.

**Ba phương pháp đối chiếu:**
- **A. FOOF-rule** theo Ceriola: IC là lúc vùng đầu tiên chậm lại, TO là lúc vùng cuối cùng nhanh lên. Vùng chạm đất đầu tiên cho foot-strike pattern.
- **B. Zeni** trên keypoint: IC ở cực đại x(gót) − x(hông), TO ở cực tiểu x(mũi) − x(hông). Phương pháp này độc lập với flow.
- **C. HSMM hai chân** (phương pháp chính):
  - Các trạng thái đi theo vòng: A chống → bay → B chống → bay, cộng một trạng thái hai chân cùng chạm đất có xác suất thấp.
  - Mô hình thời lượng theo giây thật, trên lưới đều 200 Hz.
  - Frame không hợp lệ được marginalize, không coi là swing.
  - Emission fit bằng EM chung toàn dataset; prior thời lượng rộng.
  - Có posterior, nên mỗi sự kiện có độ tin cậy.

**Nguyên tắc:**
- Không dùng nhãn khi fit.
- Không ép thời lượng về một chuẩn, vì thời gian chạm đất chính là feature để phân loại.
- Chu kỳ phải nằm trọn trong một `continuity_id` và một `speed_part_id`.

**Kiểm tra dự kiến:**
- Độ khớp C–A và C–B (|ΔIC|, |ΔTO| tính bằng ms).
- Cadence tính từ sự kiện so với tham chiếu.
- CV của thời lượng stride, tỉ lệ vi phạm luân phiên hai chân.
- So phân phối lỗi giữa hai nhãn, chỉ để chẩn đoán.
- Ảnh frame quanh IC/TO để review bằng mắt.
- Chạy thử 6 video (C30, F02, F29, C38_9, F22_4, C17) rồi dừng chờ review.

---

## 8. Các quyết định đã chốt

| Ngày | Bước | Quyết định | Lý do |
|---|---|---|---|
| 04/11 | — | Loại `F0x` khỏi dataset | Quyết định của nhóm |
| 04/11 | 1 | Không dùng giá trị Wu liên tục làm `k` | Nhiễu ±40% ở vùng 0,5–1,5 |
| 04/11 | 0 | `s_final` theo keypoint | Cách sơ bộ sai ở 23 shot |
| 04/11 | 0 | F10_2 `s_final = −1`, chân gần = trái | Kiểm tra full-res và bằng chứng Bước 3 |
| 04/11 | 2 | YOLO `imgsz = 640` | 1280 tách người gần camera thành 2 box |
| 04/11 | 3 | Bỏ luật (d) ngoài bbox | Loại nhầm bàn chân đúng |
| 05/11 | 3 | Giữ ngưỡng vận tốc 28,65 L_leg/s | Hạ xuống 17 cắt nhầm chân lăng nhanh |
| 05/11 | 3 | C20 part 1: `exclude_time_features` | Ramp chỉ 2 bước |
| 05/11 | 4 | SAM2 `base_plus` @768 | Cân bằng chất lượng và thời gian |
| 05/11 | 4 | Bỏ giới hạn 5 re-prompt/khối; re-prompt bằng memory mới | Giày giống nhau làm object nhảy chân ở mỗi lần chéo |
| 05/11 | 4 | Ankle clip song song trục | Đường vuông góc cắt ngang bàn chân |
| 05/11 | 4 | Xung đột ID lấy hợp cờ trước và sau clip | Clip chỉ tắt cờ, không sửa mask |
| 05/11 | 4 | Kiểm tra hình học chuẩn hoá theo L_leg(t) và theo chính chân đó | Luật nguyên văn loại gần nửa số frame đúng |
| 05/11 | 4 | F10_2 khối 1 giữ A/B | Chỉ 12 frame; không sửa tay Bước 3 |
| 05/11 | 4 | Không so màu giày | Lỗi còn sót ở pha swing, ít ảnh hưởng IC/TO |
| 05/11 | 5 | GMA trên MPS, 12 vòng lặp | Khớp CPU; 32 vòng lặp và độ phân giải cao hơn không giúp |
| 05/11 | 5 | FB chỉ dùng cho QC | FB loại pixel đúng của bàn chân nhanh |
| 05/11 | 5 | `slow_ratio` chính ở 2,0 L_leg/s | Cơ sở vật lý; Otsu không ổn định |
| 05/11 | 5 | Δt làm mượt cho VFR giả | pts xen kẽ không phản ánh thời điểm chụp |
| 05/11 | 5 | Bù camera trên dải đất + spline chậm; **không** dùng ràng buộc chân chống theo frame | Tránh lỗi vòng tròn |
| 05/11 | 5 | Cadence tính theo từng speed_part | Gộp part làm sai C19 |
| 05/11 | 5 | Mặc định V1 | Bước B cân bằng giữa hai nhãn |
| 05/11 | 5 | Không làm mượt feature; QC pha chống dùng max ba vùng | Nhiễu 1 frame để Bước 6 xử lý; người đáp mũi |
| 05/11 | 5 | Giữ `baseline_v1/`, xoá `flow_debug/` và `_trial/` | Giữ dữ liệu cho ablation |

---

## 9. Việc còn mở

1. **Bước 6–7 chưa chạy** (Bước 6 đang tạm dừng theo yêu cầu).
2. **Chưa có ground truth** sự kiện, mask hay nhãn L/R. Đề xuất: tự quay video 240 fps có nhãn IC/TO, và gán nhãn tay khoảng 200 frame mask.
3. Ramp của C27 chưa được xác nhận; F20, C24, F27 có thể vẫn là slow-mo ở đoạn nhanh nhất.
4. 12 video mirror `unknown` chỉ dùng near/far.
5. Ba video major (C04_3, C38_8, C27) và ba video FAIL ở Bước 5 (C04_3, C04_2, C09_2) cần cân nhắc loại khỏi tập huấn luyện, hoặc chỉ dùng các chu kỳ có độ tin cậy cao.
6. Kiểm tra ảnh hưởng của bước B (V1 so với V2) và của tiền xử lý lên kết quả phân loại.
7. Sửa đường dẫn `~/work` trong `step0_audit/code/merge.py`.

---

## 10. Tái lập

| Môi trường | Dùng cho |
|---|---|
| `.venv-gait` (Python 3.11) | RTMPose, hướng chạy, pose Bước 1, cadence |
| `.venv-speed` (Python 3.11) | Wu et al. |
| `.venv-track` | Bước 1b–5: ultralytics, torch, rtmlib, SAM2 (`ROOT/.third_party/sam2`, commit `2b90b9f`), GMA (`ROOT/.third_party/GMA`, commit `2f1fd29`), einops |

Thứ tự chạy, từ `ROOT/data`:

```bash
# Bước 1
python step1_wu/code/make_clean_clips.py
#   Wu (Seeing-Fast-and-Slow/speed_perception, .venv-speed): video_speed_estimation.py, speed_change_detection.py --thres 0.3
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
# Bước 5
caffeinate -i ../.venv-track/bin/python step5_flow/code/run_step5.py --store-only
../.venv-track/bin/python step5_flow/code/run_step5.py && ../.venv-track/bin/python step5_flow/code/s5_validate.py
```

Thời gian (M4 Pro): Bước 3 khoảng 45 s; Bước 4 khoảng 1,45 giờ (SAM2) và vài phút (hậu kỳ); Bước 5 khoảng 45 phút (GMA) và 5 phút (feature).

---

## Phụ lục A: Trạng thái từng video qua các bước

"B4 hợp lệ" là tỉ lệ frame có `frame_valid_mask` = True; video không có nhãn near/far hiện theo A/B. "B5 duty factor" là duty factor của chân near/A và far/B theo định nghĩa max ba vùng (chỉ QC).

| Video | Nhãn | s | Slow-mo | Trục thời gian | abs_time_ok | B2 review | B3 gate / review | near/far tin cậy | B4 gate / review | B4 hợp lệ (near/far) | B5 gate | B5 Spearman | B5 bù camera | B5 duty factor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | CORRECT | −1 | ✓ | constant | ✗ | minor | PASS / minor | ✓ | WARN / minor | 0.82 / 0.81 | WARN | 0.78 | direct+slow | 0.07 / 0.03 |
| C04_1 | CORRECT | +1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.99 / 0.98 | PASS | 0.91 | direct | 0.26 / 0.35 |
| C04_2 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | WARN / minor | 0.87 / 0.89 | FAIL | 0.37 | direct | 0.52 / 0.66 |
| C04_3 | CORRECT | −1 |  | constant | ✓ | ok | WARN / minor | ✗ | FAIL / major | A 0.68 / B 0.72 | FAIL | 0.34 | direct | 0.55 / 0.55 |
| C09_1 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.96 / 0.96 | PASS | 0.86 | direct | 0.36 / 0.28 |
| C09_2 | CORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.98 / 0.97 | FAIL | 0.37 | direct | 0.52 / 0.41 |
| C11 | CORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / minor | 0.87 / 0.95 | PASS | 0.91 | direct | 0.43 / 0.23 |
| C12 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.99 / 0.96 | WARN | 0.82 | direct | 0.40 / 0.42 |
| C13_1 | CORRECT | −1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | WARN / minor | 0.88 / 0.85 | WARN | 0.75 | direct | 0.45 / 0.59 |
| C13_2 | CORRECT | −1 |  | constant | ✓ | ok | WARN / ok | ✓ | WARN / minor | 0.90 / 0.87 | WARN | 0.72 | direct | 0.49 / 0.56 |
| C17 | CORRECT | −1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.99 / 0.97 | PASS | 0.98 | direct+slow | 0.21 / 0.16 |
| C18 | CORRECT | −1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.97 | PASS | 0.93 | direct+slow | 0.23 / 0.25 |
| C19 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.99 / 0.98 | PASS | 0.93 | direct | 0.38 / 0.33 |
| C20 | CORRECT | +1 | ✓ | constant | ✗ | ok | WARN / ok | ✓ | PASS / ok | 1.00 / 0.98 | PASS | 0.97 | direct | 0.23 / 0.41 |
| C21 | CORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.97 | PASS | 0.94 | direct | 0.35 / 0.43 |
| C23_1 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.96 | PASS | 0.95 | direct | 0.20 / 0.24 |
| C23_2 | CORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.98 / 0.84 | PASS | 0.83 | direct | 0.22 / 0.32 |
| C23_3 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.97 / 0.97 | PASS | 0.93 | direct+slow | 0.15 / 0.16 |
| C24 | CORRECT | −1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.98 / 0.90 | PASS | 0.95 | direct+slow | 0.30 / 0.28 |
| C25 | CORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.97 | PASS | 0.95 | direct | 0.24 / 0.26 |
| C26 | CORRECT | +1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / minor | 0.96 / 0.99 | PASS | 0.91 | direct+slow | 0.24 / 0.20 |
| C27 | CORRECT | +1 | ✓ | ramp | ✗ | ok | WARN / minor | ✗ | PASS / major | A 0.93 / B 0.94 | PASS | 0.79 | direct | 0.34 / 0.34 |
| C28_1 | CORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.94 | WARN | 0.95 | direct+slow | 0.28 / 0.23 |
| C28_2 | CORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 1.00 / 1.00 | WARN | 0.78 | direct+slow | 0.25 / 0.23 |
| C28_3 | CORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 1.00 / 0.97 | WARN | 0.90 | direct+slow | 0.30 / 0.25 |
| C28_4 | CORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 1.00 / 0.97 | WARN | 0.92 | direct+slow | 0.31 / 0.29 |
| C29 | CORRECT | −1 |  | constant | ✓ | ok | WARN / ok | ✓ | PASS / minor | 0.96 / 0.95 | WARN | 0.86 | direct+slow | 0.27 / 0.18 |
| C30 | CORRECT | −1 |  | constant | ✓ | minor | PASS / ok | ✓ | PASS / minor | 0.98 / 0.96 | PASS | 0.91 | direct | 0.32 / 0.32 |
| C31 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 0.97 / 0.97 | PASS | 0.79 | direct | 0.51 / 0.59 |
| C38_1 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.97 / 0.97 | PASS | 0.95 | direct | 0.28 / 0.24 |
| C38_10 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.99 / 0.92 | PASS | 0.95 | direct+slow | 0.27 / 0.30 |
| C38_2 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.98 / 0.97 | PASS | 0.93 | direct | 0.37 / 0.38 |
| C38_3 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.98 | PASS | 0.95 | direct+slow | 0.28 / 0.30 |
| C38_4 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.99 | PASS | 0.98 | direct | 0.30 / 0.28 |
| C38_5 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 1.00 | PASS | 0.98 | direct | 0.26 / 0.41 |
| C38_6 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.99 | PASS | 0.95 | direct+slow | 0.27 / 0.24 |
| C38_7 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.99 / 0.96 | PASS | 0.92 | direct | 0.25 / 0.27 |
| C38_8 | CORRECT | +1 |  | constant | ✓ | minor | PASS / minor | ✓ | FAIL / major | 0.74 / 0.85 | WARN | 0.51 | direct | 0.71 / 0.58 |
| C38_9 | CORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.98 / 0.96 | PASS | 0.98 | direct+slow | 0.24 / 0.25 |
| F02 | INCORRECT | +1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 0.98 / 0.98 | PASS | 0.94 | direct | 0.41 / 0.48 |
| F10_1 | INCORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.92 / 0.88 | PASS | 0.77 | direct | 0.35 / 0.47 |
| F10_2 | INCORRECT | −1 | ✓ | ramp | ✗ | minor | WARN / minor | ✓ | PASS / minor | 0.85 / 0.85 | PASS | 0.92 | direct | 0.28 / 0.31 |
| F11 | INCORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / minor | 0.93 / 0.93 | PASS | 0.83 | direct | 0.49 / 0.47 |
| F12 | INCORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 1.00 / 0.97 | PASS | 0.94 | direct | 0.27 / 0.37 |
| F17 | INCORRECT | −1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.90 / 0.88 | PASS | 0.91 | direct | 0.26 / 0.45 |
| F18 | INCORRECT | −1 | ✓ | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 0.99 / 0.98 | PASS | 0.87 | direct | 0.44 / 0.37 |
| F19 | INCORRECT | +1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.97 | PASS | 0.93 | direct+slow | 0.18 / 0.14 |
| F20 | INCORRECT | +1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / minor | 0.98 / 0.99 | PASS | 0.95 | direct | 0.26 / 0.32 |
| F21 | INCORRECT | −1 | ✓ | ramp | ✗ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.99 | PASS | 0.93 | direct | 0.33 / 0.48 |
| F22_1 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 0.98 / 1.00 | WARN | 0.92 | direct+slow | 0.23 / 0.40 |
| F22_2 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 1.00 | WARN | 0.93 | direct+slow | 0.39 / 0.25 |
| F22_3 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 0.97 / 0.97 | WARN | 0.92 | direct+slow | 0.17 / 0.13 |
| F22_4 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | WARN / minor | 0.83 / 0.87 | WARN | 0.84 | direct+slow | 0.09 / 0.27 |
| F23_1 | INCORRECT | +1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.95 | PASS | 0.95 | direct | 0.30 / 0.38 |
| F23_2 | INCORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 1.00 / 0.91 | PASS | 0.86 | direct | 0.24 / 0.38 |
| F23_3 | INCORRECT | +1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.90 / 0.95 | PASS | 0.86 | direct | 0.40 / 0.31 |
| F24 | INCORRECT | −1 | ✓ | constant | ✗ | ok | PASS / ok | ✓ | PASS / ok | 0.99 / 0.97 | PASS | 0.98 | direct+slow | 0.27 / 0.31 |
| F25 | INCORRECT | −1 | ✓ | constant | ✗ | ok | WARN / ok | ✓ | PASS / minor | 0.94 / 0.88 | PASS | 0.94 | direct | 0.26 / 0.31 |
| F26 | INCORRECT | +1 | ✓ | ramp | ✗ | minor | PASS / ok | ✓ | PASS / minor | 0.95 / 0.97 | PASS | 0.89 | direct | 0.47 / 0.32 |
| F27 | INCORRECT | +1 | ✓ | ramp | ✗ | ok | WARN / minor | ✓ | PASS / minor | 0.95 / 0.89 | PASS | 0.95 | direct | 0.37 / 0.23 |
| F28_1 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 1.00 / 1.00 | WARN | 0.94 | direct+slow | 0.32 / 0.41 |
| F28_2 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 0.97 / 0.97 | WARN | 0.90 | direct+slow | 0.27 / 0.37 |
| F28_3 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 0.97 / 0.91 | WARN | 0.83 | direct+slow | 0.32 / 0.29 |
| F28_4 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 0.97 / 0.97 | WARN | 0.89 | direct+slow | 0.23 / 0.35 |
| F29 | INCORRECT | −1 |  | constant | ✓ | ok | PASS / ok | ✓ | PASS / minor | 1.00 / 0.93 | PASS | 0.91 | direct | 0.47 / 0.48 |
| F30 | INCORRECT | +1 |  | constant | ✓ | minor | PASS / ok | ✓ | PASS / ok | 0.94 / 0.99 | PASS | 0.91 | direct | 0.40 / 0.40 |

## Phụ lục B: Bảng 85 phần tốc độ

Bảng đầy đủ nằm trong `Module1_Video_Speed_Processing_Report.md` (Phụ lục B) và `data/step1_wu/speed_parts.csv`; các part ramp nằm ở `data/step1_wu/ramp_parts.csv`.
