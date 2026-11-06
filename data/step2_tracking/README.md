# Bước 2 — Subject detection & tracking

Bám người chạy trên từng video bằng YOLO11m + BoT-SORT, rồi chạy RTMPose Halpe26 theo bbox đó.
Đầu ra là đầu vào cho bước foot region/mask.

## Cách chạy

Chạy từ `data/`, dùng `.venv-track` (venv riêng; `.venv-gait` không có torch/ultralytics).

```bash
PY=../.venv-track/bin/python
$PY step2_tracking/code/run_step2.py                                     # toàn bộ, bỏ qua video đã xong
$PY step2_tracking/code/run_step2.py --videos C13_1 F10_2                # chạy lại từ đầu các video này
$PY step2_tracking/code/run_step2.py --videos F10_2 --from-stage select  # dùng lại kết quả YOLO
$PY step2_tracking/code/s2_validate.py                                   # kiểm tra tiêu chí hoàn thành
```

Tham số dòng lệnh: `--videos`, `--from-stage {track,select,pose,report}`, `--imgsz` (mặc định 640, xem mục dưới trước khi đổi),
`--det-device` (mặc định `mps`), `--pose-device` (mặc định `mps`; CoreML lỗi thì tự chuyển `cpu` và in cảnh báo).
Các ngưỡng chọn track, nối track và cờ QC nằm trong `code/s2_common.py`.

Đổi ngưỡng ở stage nào thì chạy lại từ stage đó cho **mọi** video bị ảnh hưởng: `--from-stage` chỉ có tác dụng khi đi kèm `--videos`.

Phiên bản: ultralytics 8.4.202, torch 2.15.0, rtmlib 0.0.16, onnxruntime 1.31.1, opencv 5.0.0.93.
`s2_track.DetCapture` bọc `BOTSORT.update` để lấy box detection thô, nên khi nâng cấp ultralytics phải kiểm tra lại.

## Mã nguồn

| File | Việc |
|---|---|
| `code/run_step2.py` | Điều phối, resume, in tóm tắt |
| `code/s2_track.py` | 2.1 YOLO + BoT-SORT, histogram HSV vùng thân |
| `code/s2_select.py` | 2.2 chấm điểm và chọn người chạy, 2.3 nối tracklet, nội suy |
| `code/s2_pose.py` | 2.4 RTMPose |
| `code/s2_report.py` | 2.5 QC, 2.6 ảnh kiểm tra |
| `code/s2_validate.py` | Kiểm tra tiêu chí hoàn thành |
| `code/botsort_reid.yaml` | Cấu hình tracker |

## Khác biệt so với đặc tả ban đầu

Các điểm này do dữ liệu thực tế đòi hỏi. Mỗi điểm đều có bằng chứng đo được.

1. **Toạ độ `run_cy` không nhân `W/480` trên video có tag xoay.** Dùng `sx = W/480`, `sy = H/h_an`, với `h_an` là chiều cao ảnh phân tích của Bước 0.
   Bước 0 tính `h_an` từ `width/height` của ffprobe (chưa xoay) rồi để ffmpeg xoay trước khi scale, nên ảnh bị giãn không đều trên 10 video: C01, C04_1, C09_2, C12, C19, C20, C21, C30, F02, F17.
   Ví dụ C01: H = 1094 nhưng `run_cy` tối đa 842, nhân `W/480 = 4` thành 3368. Ảnh OpenCV resize về `480×h_an` tương quan ≥ 0.9995 với ảnh của Bước 0.
   `cam_dx` và `run_cx` không đổi vì `sx = W/480` vẫn đúng.
2. **Thêm điều kiện `continuity_id >= 0`.** 17 frame (C11, C12, F11, F12) qua được quy tắc 1 nhưng nằm lẻ giữa đoạn freeze, không có `t_real_s` và không thuộc khối nào, nên bị loại. Vì vậy `keypoints_v2` của 4 video này ít hơn Bước 1 vài frame.
3. **Tracker không dùng nguyên `botsort.yaml`.** Ngoài `with_reid: True` còn có:
   - `fuse_score: False`: nhân IoU với conf làm mất người nhỏ conf thấp. C38_8 có detection ở 82/84 frame nhưng chỉ ra track 49/84, với 8 ID.
   - `match_thresh: 0.9`: người gầy và fps thấp nên IoU giữa 2 frame nhỏ. C38_8 còn 1 ID, 82/84 frame.
4. **Lưu box detection, không lưu box Kalman.** Tracker của ultralytics xuất trạng thái Kalman, lệch tới khoảng 14% chiều cao bbox so với detection.
5. **Chọn người trên chuỗi tracklet đã nối, không trên từng track.** Nếu không, người chạy bị đứt ID có thể thua khán giả đứng yên có track dài, và các mảnh của chính người chạy làm `selection_margin` thấp giả. Điểm từng track vẫn được ghi.
6. **`imgsz = 640`, không tăng lên.** Đây là tham số quan trọng nhất của bước này. Với `imgsz = 1280`, người gần camera vượt quá receptive field của detector và vỡ thành 2 box từng phần (đầu→gối, hông→chân). Đối chiếu với keypoint Bước 1 trên cả 66 video:

   | | Frame có bàn chân ngoài box | Frame có đầu ngoài box |
   |---|---|---|
   | `imgsz = 1280` | 25.3% | 33.6% |
   | `imgsz = 640` | 2.9% | 0.1% |

   Ở 640 không có video kiểm tra nào thiếu detection, kể cả video người nhỏ (C38_8) và video đông người (F10_1, F10_2), đồng thời nhanh hơn khoảng 3 lần.
7. **Nối cả các trường hợp ngoài khoảng trống thuần tuý.** Đặc tả chỉ xét track nối tiếp nhau có khoảng trống. Trên dữ liệu còn có:
   - **Chồng rất ngắn** (≤ 2 frame): coi như khoảng trống bằng 0, dùng ngưỡng nối thường (C38_8).
   - **Chồng dài hơn** (≤ 0.5 s): ngưỡng tâm chặt hơn, < 0.25 × chiều cao.
   - **Track trùng người** (`direction = absorb`): IoU trung bình ≥ 0.5 trên frame chung, hoặc xen kẽ gần như không có frame chung (C38_5). Frame của track đó được gộp vào chuỗi nếu khớp vị trí nội suy.
8. **`selection_margin` không so với track trùng người**, vì đó không phải đối thủ. Cột `same_person_as_runner` đánh dấu các track này.
9. **`s_motion` giảm theo độ dài track** (`motion_reliability`): track ngắn hơn 1 s thật có thể đạt vận tốc tối đa chỉ vì nhiễu. C38_9 từng chọn track 3 frame thay vì track 122 frame vì lỗi này.
10. **Thêm cột `frac_kp_outside_bbox` và cờ `box_truncated`.** QC theo đặc tả không phát hiện được lỗi ở mục 6.

## Lưu ý cho bước sau

- **Lấy vùng bàn chân từ keypoint, không từ bbox.** Box YOLO thường không bao bàn chân đang lăng; RTMPose vẫn định vị được nhờ phần đệm 25% của crop. Cờ `box_truncated` trên 6 video hiện tại chủ yếu là trường hợp này.
- **Shot cận cảnh của C01 (khối 1) không có thân trên** nên keypoint thân trên ở đó không đáng tin; bàn chân và cẳng chân vẫn dùng được.
- **`s_motion` phụ thuộc chất lượng `cam_dx`.** Ở F10_2 khối 1 (12 frame), một người trên ban công đạt `s_motion` 0.96 trong khi người chạy đạt 0, nên margin chỉ 0.016. Lựa chọn vẫn đúng nhờ `s_proximity` và `s_size`.

## Output

Tất cả nằm trong `step2_tracking/`. Không file nào ở `step0_audit/`, `step1_wu/`, `keypoints/` bị sửa.

### `all_tracks/<video_id>_tracks.csv`
Mọi người ở mọi frame đã xử lý: `frame, pts_s, continuity_id, track_id, x0, y0, x1, y1, conf`.
Toạ độ ảnh gốc (sau khi xoay), box detection, `conf >= 0.25`. `track_id` là duy nhất trong video và không lặp giữa các khối.
Frame trùng không có dòng ở đây; thông tin kế thừa nằm trong `runner/`.

### `runner/<video_id>_runner.csv`
Một dòng cho mỗi frame giải mã, là bảng tra chính cho bước sau:
`frame, pts_s, t_real_s, continuity_id, shot_id, is_dup, inherited_from, track_status, track_id, x0, y0, x1, y1, conf`.

| `track_status` | Nghĩa |
|---|---|
| `tracked` | Có detection của người chạy |
| `interpolated` | Nội suy tuyến tính theo `pts_s`, khoảng trống ≤ 3 frame unique. `track_id = -1` |
| `missing` | Frame hợp lệ, không có bbox |
| `excluded` | Không xử lý: freeze biên tập, shot bị loại, `continuity_id = -1`, hoặc frame trùng của các frame đó |

`inherited_from` là số frame unique được kế thừa (cùng khối), `-1` nếu tự xử lý. Frame kế thừa mang `track_status` và bbox của frame nguồn.

### `keypoints_v2/<video_id>_kp.npz`
Schema Bước 1 (N = số frame được xử lý): `frame, pts_s, shot_id, kp (N×26×2), score (N×26), bbox (N×4), det, W, H, n_read, n_meta`, thêm `track_status, continuity_id, t_real_s`.
- `det = True` khi bbox từ detection (`tracked`).
- Frame `missing`: `kp` và `bbox` là NaN, `score` là 0.
- `W, H` là kích thước ảnh sau khi xoay.
- `score` là SimCC thô nên có thể > 1.

### `selection/`
- `<video_id>_scores.csv`: điểm từng track và chuỗi. `s_proximity`, `s_length`, `s_motion`, `s_size`, `score` là điểm track; `chain_*` là điểm chuỗi dùng để chọn; `selected` đánh dấu chuỗi được chọn.
- `<video_id>_joins.csv`: mọi cặp được xét nối của chuỗi thắng (`direction` = `forward`, `backward` hoặc `absorb`), kèm `gap_s` (âm khi chồng), `center_err_bh`, `dh_frac`, `appearance_dist`, `accepted`, `chosen`, `reject_reason`.
- `<video_id>_blocks.csv`: tóm tắt theo khối, gồm `margin`, `winner_score`, `runner_ids`.

**Công thức điểm:** `0.25·s_proximity + 0.25·s_length + 0.35·s_motion + 0.15·s_size`.
- `s_proximity = exp(−d)`, với `d` là trung vị khoảng cách tới tâm vùng chuyển động, chia chiều cao bbox.
- `s_length` = số frame của track / số frame của khối.
- `s_motion = clip(v / 1.5, 0, 1) · dir_frac · motion_reliability`. `v` là vận tốc của `cx − cumsum(cam_dx · sx)` theo hướng `s_final`, tính bằng chiều cao bbox trên giây thật. `dir_frac` là tỉ lệ quãng đường đi đúng hướng. `motion_reliability = min(1, thời lượng track / min(1 s, thời lượng khối))`. `cam_dx` là dịch chuyển của nội dung ảnh, nên camera bám theo người chạy vẫn cho vận tốc nền cùng dấu `s_final`.
- `s_size = min(1, h / 0.25H) · s_ar`, với `s_ar` = 1 khi tỉ lệ cao/rộng trong [1.0, 4.5].

### `qc_tracking.csv`
Cột theo đặc tả, thêm `frac_kp_outside_bbox`. Các định nghĩa cần lưu ý:
- `max_gap_s`: khoảng `pts_s` lớn nhất giữa 2 frame có bbox trong một khối, kể cả đoạn thiếu ở đầu và cuối khối.
- `n_center_jumps`: tính trong toạ độ ảnh, nên pan rất nhanh cũng có thể gây nhảy tâm.
- `selection_margin`: giá trị nhỏ nhất qua các khối; khối không có đối thủ thì margin = điểm người thắng.
- `frac_frames_foot_visible`, `kp_score_median`, `foot_kp_score_median`: tính trên các frame có pose.
- `frac_kp_outside_bbox`: tỉ lệ frame có đầu hoặc bàn chân (score > 0.5) nằm ngoài bbox quá 3% chiều cao bbox.
- `lr_swap_rate`: chỉ xét chi dưới (gối, cổ chân, bàn chân), điểm có score > 0.3. Chỉ báo, không sửa dữ liệu.

| Cờ | Điều kiện |
|---|---|
| `low_coverage` | `coverage < 0.9` |
| `center_jump` | `n_center_jumps > 0` |
| `low_margin` | `selection_margin < 0.15` |
| `near_tie` | `selection_margin < 0.05` |
| `low_foot_score` | `foot_kp_score_median < 0.5` |
| `box_truncated` | `frac_kp_outside_bbox > 0.1` |
| `block_without_runner` | Có khối không có track nào |
| `frame_count_mismatch` | Số frame OpenCV đọc khác audit |

### `review/` và `tracking_review.csv`
- `<video_id>.jpg`: lưới 8 frame rải đều. Box xám là mọi track, xanh lá là người chạy, vàng là bbox nội suy; chấm đỏ là gót, chấm vàng là mũi chân.
- `_flagged_<video_id>_<k>.jpg`: frame trước và sau mỗi center jump hoặc lần nối tracklet.
- Lưới 8 frame có thể không chạm tới khối ngắn; muốn kiểm tra từng khối thì xem `selection/<video_id>_blocks.csv`.
- `tracking_review.csv`: để người kiểm tra điền `verdict (ok/minor/fail), wrong_person_frames, notes`. Chạy lại không ghi đè nội dung đã điền.

### Khác
`appearance/`: histogram HSV theo từng dòng của `all_tracks` (để chạy lại stage select). `qc/`: QC từng video kèm danh sách center jump. `logs/`: thời gian từng stage và log chạy. `models/`: trọng số YOLO.
