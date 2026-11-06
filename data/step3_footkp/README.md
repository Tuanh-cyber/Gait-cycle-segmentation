# Bước 3: Kiểm định & làm sạch keypoint bàn chân

Sửa đảo nhãn L/R theo thời gian, xác định chân gần/xa camera, làm sạch 8 điểm bàn chân, và cho mỗi video một gate trước khi tạo foot mask ở Bước 4.

Đây là bản v2: dùng `t_real_s_v2` của Bước 1b (`step1_wu/README_ramp.md`), bỏ luật (d) ngoài bbox, thêm `near_track_reliable`. Bản v1 giữ ở `footkp_v1/`, `qc_footkp_v1.csv`, `qc/velocity_threshold_v1.json`. Kết luận của người kiểm tra nằm ở `footkp_review.csv` (không bị ghi đè); `footkp_review_v2_diff.csv` thêm gate v1, gate v2 và cờ thay đổi.

## Chạy

Chạy từ `ROOT/data`, dùng `.venv-track` (đã có rtmlib, numpy, matplotlib; không cần scipy):

```bash
../.venv-track/bin/python step3_footkp/code/run_step3.py                    # cả 66 video, tự bỏ qua video đã xong
../.venv-track/bin/python step3_footkp/code/run_step3.py --videos C30,F30   # danh sách
../.venv-track/bin/python step3_footkp/code/run_step3.py --force            # làm lại
../.venv-track/bin/python step3_footkp/code/run_step3.py --pose-device cpu
../.venv-track/bin/python step3_footkp/code/s3_validate.py                  # kiểm tra toàn bộ output, mã thoát 1 nếu lỗi
../.venv-track/bin/python step3_footkp/code/s3_inject.py                    # đo độ nhạy Viterbi bằng swap tiêm vào
```

Đổi hằng số trong `code/s3_common.py` thì mã cấu hình đổi theo: output cũ tự chạy lại, validator báo lỗi nếu còn sót. Ngưỡng vận tốc luôn tính trên cả 66 video, kể cả khi dùng `--videos`. Cả 66 video mất khoảng 45 s.

## Dùng output ở Bước 4

`footkp/<vid>_footkp.npz`:

| Khoá | Kích thước | Ý nghĩa |
|---|---|---|
| `frame`, `pts_s`, `continuity_id`, `speed_part_id` | N | Chỉ frame unique đã xử lý ở Bước 2 |
| `t_real_s_v2`, `time_model` | N | Trục thời gian dùng cho mọi tính toán; `ramp` hoặc `constant` |
| `t_real_s` | N | Trục cũ của Bước 1, chỉ để so sánh |
| `near_track_reliable` | 1 | True nếu `near_far_conf` ≥ 0.6 và không bị đánh dấu tay. Hiện False ở C04_3, C27 |
| `kp_raw` | N×26×2 | Nguyên bản từ `keypoints_v2` |
| `kp_clean` | N×26×2 | Đã sửa swap, bỏ outlier, lọc. NaN nếu không có giá trị |
| `valid_mask` | N×26 | True nếu là quan sát thật còn lại sau làm sạch |
| `swap_flag` | N | Frame bị hoán đổi nhãn so với model |
| `near_track`, `near_conf` | N | `A`/`B` và độ tin cậy, không đổi trong 1 khối |
| `outlier_reason` | N×26 | Bitmask: 1 score, 2 heel_toe, 4 velocity, 16 collapse (bit 8 là luật (d), đã bỏ) |
| `interp_mask` | N×26 | Nội suy tuyến tính qua khoảng trống ≤ 3 frame; `valid_mask` ở đó là False |
| `score_fixed`, `L_leg`, `d_ref`, `config` | | Score theo nhãn đã sửa, L_leg theo thời gian, tham chiếu pha, mã cấu hình |

Ba điểm dễ dùng sai:

- **Slot trái trong `kp_clean` là chân A, không phải chân trái giải phẫu.** Chân A gồm các chỉ số 11, 13, 15, 20, 22, 24; chân B gồm 12, 14, 16, 21, 23, 25. A/B chỉ có nghĩa trong 1 khối `continuity_id`. Chân gần lấy từ `near_track`.
- **Dùng `t_real_s_v2`, và chỉ dùng thời gian tuyệt đối khi `abs_time_ok = True`.** Ở video ramp, trục này chỉ đúng về tỉ lệ tương đối.
- **Kiểm tra `near_track_reliable` trước khi dùng near/far.** Ở C27 nên dùng đặc trưng không phân biệt chân.
- **Vị trí bàn chân nên lấy từ keypoint, không từ bbox người.** Bàn chân đang đu thường nằm ngoài bbox dù keypoint đúng.

## Các file khác

- `qc_footkp.csv`: 1 dòng/video.
- `footkp_review.csv`: để trống cho người kiểm tra.
- `review/<vid>.jpg`: 8 frame; xanh lá là chân gần, cam là chân xa, X đỏ là outlier. Ở v2 chỉ tạo lại cho video ramp hoặc có gate đổi; ảnh và plot của video khác vẫn là bản v1, nên có thể còn X đỏ của luật (d) đã bỏ.
- `review/_swap_<vid>_<k>.jpg`: frame bị hoán đổi trước và sau sửa, cạnh frame liền kề; tối đa 5 ảnh/video.
- `plots/<vid>_ankle.png`: x cổ chân theo `t_real_s` trước và sau sửa.
- `qc/<vid>_blocks.csv`: near/far từng khối và giá trị từng bằng chứng.
- `qc/<vid>_perturb.csv`: độ dịch khi đổi bbox.
- `qc/velocity_threshold.json`: ngưỡng vận tốc.
- `qc/inject_results.csv`: kết quả tiêm swap.

## Gate

| Mức | Điều kiện |
|---|---|
| FAIL | `foot_valid_frac_near` < 0.7, hoặc `swap_rate_resid` > 0.05 |
| WARN | `foot_valid_frac_far` < 0.5 |
| WARN | `near_far_conf` < 0.6 |
| WARN | near/far lệch metadata ở một phần hoặc cả video |
| WARN | `phase_periodic_frac` < 0.5: dưới nửa số frame có chu kỳ đủ rõ để kiểm chéo swap (R ≥ 0.8) |
| WARN | `step_time_ratio` > 2: thời gian một bước (p90/p10) đổi mạnh trong cùng part |
| WARN | `scale_perturb_instability` > 0.05 |

Hai ngưỡng FAIL và ngưỡng `foot_valid_frac_far` lấy từ đề bài. Ngưỡng 0.6 của `near_far_conf` đặt sau khi đối chiếu metadata: khối có conf ≤ 0.6 chỉ khớp 2/3.

`swap_rate` là tỉ lệ frame đã sửa, nên nó cao không phải là lỗi (C27 là 0.36 vì model đảo nhãn đến hết clip). Gate dùng `swap_rate_resid`: tỉ lệ frame SAU sửa còn nghi hoán đổi, tức dấu chênh x cổ chân trái tham chiếu pha, hoặc gán chéo giảm sai lệch quá 0.1 L_leg mỗi điểm.

## Cách hoạt động

**3.1 Swap.** Viterbi trên cặp trạng thái (S[t-1], S[t]), vì dự đoán tuyến tính cần 2 frame trước. Chi phí là sai lệch dự đoán của 12 điểm chân trên toạ độ trừ tâm hông, có trọng số score. Lượt đầu chỉ dùng vận tốc.

Prior lệch pha: d² (d là chênh x cổ chân) không đổi khi hoán đổi nhãn và có chu kỳ bằng 1 bước. Giải điều chế d² cho tham chiếu pha mà không dùng nhãn nào; tham chiếu này đi vào lượt Viterbi thứ hai. Prior bật ở part có độ tuần hoàn R ≥ 0.8; riêng part ramp đã sửa thời gian bật từ R ≥ 0.7. Không hạ ngưỡng chung vì tốc độ đổi chưa sửa làm tham chiếu sai và sinh ra sửa nhầm (C20 part 1, chỉ có 2 bước nên Bước 1b không sửa được).

Độ nhạy, đo bằng swap tiêm vào chuỗi thật, lấy chuỗi đã sửa làm nền (bản v2):

| Kiểu tiêm | Có prior | Không prior |
|---|---|---|
| Nhảy nhãn giữa pha | 98.8% | 93.4% |
| Đúng nửa chu kỳ giữa 2 lần chéo chân (khó nhất) | 96.8% | 84.1% |

**3.2 Near/far.** Bốn bằng chứng, tổng hợp theo khúc 8 frame, bootstrap ra `near_conf`: `foot_size`, `leg_size`, `ground_y` (chân gần chạm đất thấp hơn trong ảnh), `occlusion` (lúc chồng lấn, chân xa có score thấp hơn). Trọng số bằng nhau, không khớp theo metadata.

Mức khớp với metadata: 62/64 khối. A được định nghĩa là chân mang nhãn L của model ở đa số frame, nên mức khớp này kiểm tra đồng thời bằng chứng hình học, nhãn đa số của model và `s_final`.

**3.3 Làm sạch.** Các luật (a), (b), (c) theo đề bài, thêm luật collapse. Lọc Savitzky–Golay bậc 3, 7 điểm, viết lại cho trục không đều: mỗi điểm khớp đa thức trên `t_real_s` thật. Đoạn ngắt tại ranh giới khối, part và khoảng trống > 3 frame; đoạn < 7 điểm giữ nguyên không lọc. Biên độ cổ chân giữ lại: trung vị 98.8%, thấp nhất 93%.

## Khác với đề bài

| Thay đổi | Lý do |
|---|---|
| `foot_valid_frac` dùng ≥ 3/4 điểm (ankle, big toe, small toe, heel) mỗi chân. Bản ≥ 4/6 của tập "chân" ở cột `leg_valid_frac_*` | Mỗi bàn chân chỉ có 4 điểm nên "4/6 điểm bàn chân mỗi chân" không áp dụng được. Theo tập 6 điểm thì hip + knee gần như luôn đạt, nên 2 điểm bàn chân đã đủ qua |
| Ngưỡng vận tốc v2 là 28.65 L_leg/s, cao hơn mức kỳ vọng ~16 | p99.5 thẳng trên `t_real_s_v2` toàn dataset. Phần `k` tin cậy cao 16.4, video `abs_time_ok = True` 17.2, phần slow-mo 39–46: ở đó một cú nhảy pose trong dt rất nhỏ cho vận tốc lớn, nên đuôi phân bố là lỗi pose, không phải sinh lý. Không còn chuẩn hoá cục bộ của v1, vì trục thời gian mới đã đồng nhất (trung vị 4–7 L_leg/s ở mọi nhóm, và chuẩn hoá cục bộ cho 27.0) |
| Luật (c): chỉ đánh dấu khi nhảy đi rồi quay lại trong ≤ 5 frame | Một bước nhảy đơn lẻ không cho biết phía nào sai |
| Thêm luật collapse | Hai cổ chân chập một chỗ trong khi tham chiếu pha nói phải cách xa. Hoán đổi không sửa được; hiếm (6 frame toàn dataset) |
| L_leg theo thời gian (trung vị ±0.5 s) | Người chạy lớn gần gấp đôi trong khối ở F10_1, F10_2 |
| Viterbi đi xuyên ranh giới `speed_part_id` nhưng chỉ bằng vị trí, không dùng vận tốc | Cần giữ định danh A/B trong cả khối theo 3.1. Không làm mượt hay lọc qua ranh giới, validator có kiểm tra |
| `E_flip_*` tính trên toạ độ trừ tâm hông | Không tìm thấy định nghĩa Phase 1 trong repo nên dùng định nghĩa trong đề bài. Toạ độ ảnh dễ thành tung đồng xu khi camera lia nhanh hơn độ tách 2 chân. Cột `*_raw` là trước sửa |
| Khoảng trống ≤ 3 frame được nội suy nhưng `valid_mask` vẫn False | Giữ đúng "outlier không phải quan sát"; bước sau tự chọn có dùng `interp_mask` hay không |
| Biểu đồ dùng xanh lá `#167A3C` và cam `#F08A24` | Cặp màu đầu tiên không đủ tách với người mù màu đỏ (ΔE 5); cặp này ΔE 14 |

## Phát hiện về dữ liệu

**Tốc độ đổi trong part** đã xử lý ở Bước 1b: 12 part ở 11 video (C17, C24, C26, C27, F10_2, F17, F19, F20, F21, F26, F27).

**`s_final` của F10_2** đã được người kiểm tra sửa thành −1. Near/far khớp metadata 101/113 frame; khối 1 (12 frame) vẫn lệch.

**Bỏ luật (d)** vì nó loại cả bàn chân đúng: `foot_valid_frac` chân gần tăng ở C38_8 (0.83 → 0.93), C01 (0.83 → 0.89), C11 (0.91 → 0.97), C04_3 (0.77 → 0.82).

**Luật (b) heel→toe vừa bắt được lỗi thật, vừa nhầm.** Nó bắt được ngón chân xa bị đặt lên ống chân (C13_1, C13_2), nhưng cũng đánh dấu nhầm bàn chân đang đu bị co ngắn trong ảnh (F22_4, C04_1).

## Giới hạn

- Không có ground truth nhãn L/R. Số tiêm swap đo độ nhạy trên chuỗi đã sửa, giả định chuỗi nền đúng.
- Viterbi hoán đổi cả chân. Lỗi chỉ ở bàn chân (ngón nhảy sang chân kia) không được hoán đổi, chỉ bị luật (b) hoặc (c) loại.
- Ở video có `phase_periodic_frac` thấp, `swap_rate_resid` chỉ còn kiểm tra bước nhảy, nên 0 ở đó là bằng chứng yếu hơn.
- Trên video ramp, tham chiếu pha phụ thuộc `t_real_s_v2`, mà trục này dựng từ chính các lần chéo chân của v1. PASS ở nhóm này vì vậy yếu hơn PASS ở video thường.
- `C27`: chuỗi hoán đổi f76–f117 gần trùng đoạn người kiểm tra thấy đổi màu (từ f83), nên có thể do Viterbi sửa nhầm. Không sửa theo quyết định đã chốt.
- `occlusion` dùng score làm đại diện, không phân tích pixel, và là bằng chứng yếu nhất (khớp metadata 77%, so với `leg_size` 95%).

## Quyết định sau review (05/11)

- **`time_feature_flags.csv`:** C20 part 1 (`speed_part_id` = 1, f76–243, 126 frame unique) có `exclude_time_features` = True. Tỉ số d_max/d_min = 3,78 nên gần như chắc là ramp, nhưng chỉ có 2 bước nên không dựng được k(t), và k = 1/5 có độ tin cậy thấp. Không dùng đặc trưng thời gian của đoạn này. File `footkp/C20_footkp.npz` không đổi; cờ theo từng frame nằm ở `step4_footmask/masks_post/C20.npz`. File cờ do `step4_footmask/code/s4_postprocess.py` sinh ra, không bị `run_step3.py` ghi đè.
- **Ngưỡng vận tốc:** giữ 28,65 L_leg/s, không đổi.
- **F10_2 khối 1 (f523–583):** `near_track` = B với `near_conf` = 1,0 ở đây là sai. Ảnh f545, f550 cho thấy chân A ở gần, metadata cũng kỳ vọng A, và `near_conf` chỉ từ 1 chunk vì khối có 12 frame. File Bước 3 không đổi; Bước 4 không dùng nhãn near/far ở khối này. `near_conf` của các khối rất ngắn không đáng tin nói chung.
