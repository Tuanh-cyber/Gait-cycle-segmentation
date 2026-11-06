# Đặc tả Bước 6: Phát hiện IC/TO và cắt chu kỳ dáng chạy

**Trạng thái:** chưa thực hiện. Đây là đặc tả đã thống nhất; người làm có thể điều chỉnh, nhưng phải ghi lại mọi điểm khác đặc tả, kèm lý do, trong `data/step6_events/README.md`.

## Mục tiêu

- Với từng chân, tìm **IC** (initial contact: bàn chân chạm đất) và **TO** (toe-off: rời đất).
- Cắt thành **chu kỳ** IC → IC kế tiếp cùng chân. Mỗi chu kỳ là một instance cho phân loại MIL: video là bag, chu kỳ là instance.

## Output

```
data/step6_events/
  events/<vid>.csv     mọi IC/TO: foot, type, t (s), method, confidence, ci90_ms, strike_region
  cycles/<vid>.csv     mỗi dòng 1 chu kỳ
  cycles_all.csv       gộp toàn dataset
  qc_events.csv        QC 1 dòng/video, có gate
  plots/, review/      plot theo thời gian; ảnh frame quanh sự kiện
  README.md, code/, s6_validate.py
```

## Nguyên tắc

1. **Không dùng nhãn CORRECT/INCORRECT** ở bất kỳ đâu khi fit hay chọn tham số.
2. **Không làm mượt feature** trước mô hình, ngoại trừ nội suy lên lưới thời gian.
3. Mọi thời gian theo `t_real_s_v2`.
4. Prior thời lượng phải **rộng**. GCT, duty factor và flight time là feature để phân loại, nên prior hẹp sẽ ép mọi người chạy về cùng một chuẩn.
5. Không sửa thư mục của các bước trước.

## Input

- `step5_flow/features/<vid>.npz` (V1): `slow_ratio` 3 vùng ở các ngưỡng 1,0 / 1,5 / 2,0 / 3,0, `v_med`, `v_p25`, `v_p75`, `vx_med`, `vy_med`, `flow_valid`, `cam_fit_bad`, `frame_valid_mask`, `label`, `view_lateral`, `exclude_time_features`.
- `step3_footkp/footkp/<vid>_footkp.npz`: `kp_clean`, `L_leg`, `near_track_reliable`.
- `step0_audit/frames/<vid>_frames.csv`: `t_real_s_v2`, `continuity_id`, `speed_part_id`.
- `step0_audit/step0_metadata.csv`: `abs_time_ok`, `s_final`, `cadence_real_spm_v2`; `step3_footkp/qc_footkp.csv`: `stride_T_median_s`.

## Ba phương pháp

### A. FOOF-rule (Ceriola)

- Với mỗi chân, stance candidate khi max(`slow_ratio` 3 vùng, ngưỡng 2,0) > 0,5. Đóng khoảng hở ≤ max(1 frame, 30 ms).
- **IC** là thời điểm đầu tiên có một vùng chuyển sang chậm, nội suy tuyến tính dưới mức frame tại mức 0,5.
- **TO** là thời điểm vùng cuối cùng chuyển sang nhanh.
- Ghi lại vùng chạm đầu tiên (heel/mid/toe). Đây là foot-strike pattern.

### B. Zeni (chỉ keypoint, độc lập với flow)

- Toạ độ đã nhân `s_final` (+x là hướng chạy) và trừ tâm hông, dùng `kp_clean`.
- **IC** tại cực đại x(gót) − x(hông); **TO** tại cực tiểu x(mũi) − x(hông).
- Khoảng cách tối thiểu giữa 2 cực trị cùng loại là 0,5 × `stride_T`.

### C. HSMM hai chân (phương pháp chính)

**Lưới thời gian:** lưới đều 200 Hz trên `t_real_s_v2` trong từng khối continuity. Cờ nội suy nearest, giá trị số nội suy tuyến tính. Frame không hợp lệ, occluded hoặc `cam_fit_bad` được **marginalize** (không có quan sát), không coi là swing.

**Trạng thái** đi theo vòng:

```
S_A (chân A chống) → FL1 (bay) → S_B → FL2 → S_A
```

Thêm DS (cả hai chân chống) với xác suất thấp, và cho phép bỏ qua FL (thời gian bay = 0).

**Emission** cho từng chân:
- Dùng `slow_ratio` 3 vùng (ngưỡng 2,0) và log(`v_med`) cả bàn chân.
- Mô hình đơn giản: Beta hoặc logistic cho `slow_ratio`, Gaussian cho log v.
- Fit bằng EM chung toàn dataset, không theo nhãn. Sau đó điều chỉnh nhẹ theo video (MAP với prior toàn dataset).

**Thời lượng:**
- Log-normal theo giây, prior rộng: stance 0,08–0,50 s, flight 0–0,25 s.
- Học bằng EM toàn dataset.
- Cho phép co giãn theo `stride_T` của từng video, để không ép mọi video về cùng một cadence.

**Giải mã:**
- Viterbi cho đường tốt nhất.
- Forward–backward cho posterior: độ tin cậy của từng sự kiện và khoảng tin cậy 90% (ms).
- IC/TO là thời điểm vào/ra trạng thái chống của từng chân.

## Cắt chu kỳ

Chu kỳ là IC → IC kế tiếp cùng chân, nằm trọn trong **một `continuity_id` và một `speed_part_id`**. Mỗi dòng gồm:

| Nhóm | Cột |
|---|---|
| Định danh | `video_id`, `label` (chỉ để xuất), `foot` (near/far/A/B) |
| Sự kiện | `t_IC`, `t_TO`, `t_IC_next` |
| Thời lượng | `stride_time`, `stance_time`, `swing_time`, `duty_factor`, `flight_time` (từ chân kia) |
| Kiểu tiếp đất | `strike_region` (từ phương pháp A) |
| Chất lượng | `confidence` (posterior nhỏ nhất của các sự kiện), `frac_valid_frames`, `method_agreement` (\|ΔIC\|, \|ΔTO\| giữa C với A và B, ms) |
| Cờ | `abs_time_ok`, `exclude_time_features`, `view_lateral`, `near_track_reliable`, `cycle_valid` |

`cycle_valid` là True khi: `confidence` ≥ ngưỡng, `frac_valid` ≥ 0,7, độ khớp giữa các phương pháp nằm trong ngưỡng, và thời lượng nằm trong khoảng sinh lý rộng.

## QC theo video (`qc_events.csv`)

**Chỉ số:**
- Số chu kỳ hợp lệ / tổng; CV của `stride_time`.
- Trung vị `stance_time` và `duty_factor` của mỗi chân.
- Tỉ lệ vi phạm luân phiên (IC hai chân không xen kẽ).
- Độ khớp C–A và C–B: trung vị \|ΔIC\|, \|ΔTO\| (ms), và tỉ lệ sự kiện ghép cặp được trong ±50 ms.
- Cadence tính từ sự kiện, theo từng speed_part, so với tham chiếu.

**Gate:**
- FAIL khi < 2 chu kỳ hợp lệ, hoặc cadence lệch > 15% (theo part, có xét độ phân giải tần số).
- WARN khi trung vị \|ΔIC\| C–A > 40 ms, CV stride > 0,15, hoặc vi phạm luân phiên > 10%.

**Kiểm tra thiên lệch theo nhãn:** so tỉ lệ FAIL/WARN và số chu kỳ hợp lệ giữa hai nhãn. Phép so này chỉ để chẩn đoán, **không dùng để chỉnh tham số**.

## Review

- Mỗi video một plot theo thời gian: `slow_ratio` 3 vùng mỗi chân, trạng thái HSMM, IC/TO của A, B, C, và tốc độ keypoint.
- Với 6 video thử: mỗi video 4 IC và 4 TO của C, mỗi sự kiện 3 frame (trước, tại, sau), crop quanh bàn chân, có ghi thời điểm.

## Kế hoạch chạy

1. Chạy thử 6 video, mỗi video thử một trường hợp:

| Video | Trường hợp |
|---|---|
| C30 | Tín hiệu tốt, clip dài |
| F02 | Slow-mo dày frame |
| F29 | Đã sửa bù camera |
| C38_9 | Nền còn trôi nhẹ |
| F22_4 | Chân gần bỏ sót pha chống |
| C17 | Speed ramp |

2. Review ảnh frame quanh sự kiện bằng mắt.
3. Nếu đạt thì chạy toàn bộ 66 video.

## Rủi ro đã biết

- Tín hiệu yếu ở C04_3, C04_2, C09_2 (Bước 5 FAIL), C38_8 và C27.
- Video ngắn như bộ C28 và F28 chỉ có khoảng 2 sải.
- C09_1, C09_2 có hệ số `k` = 1/8 với độ tin cậy thấp, nên thời lượng tuyệt đối không tin được.
- Chỉ số pha chống của Bước 5 mới đạt cả 3 khoảng sinh lý ở 12/66 video. Đây là kiểm tra định nghĩa đơn giản, không phải kết quả của mô hình.
