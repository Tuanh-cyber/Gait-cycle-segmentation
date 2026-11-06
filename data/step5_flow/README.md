# Bước 5: Optical flow theo vùng bàn chân

Tính optical flow GMA giữa các frame unique, trừ chuyển động camera bằng chính mặt đất quanh bàn chân, rồi lấy feature tốc độ theo từng chân và từng vùng gót / giữa / mũi. Đầu vào là `step4_footmask/masks_post/`.

Kết quả hiện tại (biến thể mặc định V1): **45 PASS, 18 WARN, 3 FAIL** trên 66 video, 8331 cặp frame. FAIL chỉ là cờ, không chặn pipeline.

**Các bước sau đọc `features/<vid>.npz` (V1) và các cột không tiền tố của `qc_flow.csv`.** V2 được giữ để ablation. Gate không kiểm tra độ đúng của pha chống, xem [Chỉ số pha chống](#chỉ-số-pha-chống).

## Chạy

Chạy từ `ROOT/data`, dùng `.venv-track`:

```bash
caffeinate -i ../.venv-track/bin/python step5_flow/code/run_step5.py --store-only   # chỉ chạy GMA vào flow_store, khoảng 45 phút
../.venv-track/bin/python step5_flow/code/run_step5.py                              # tính feature 2 biến thể, tự chạy GMA nếu kho thiếu, khoảng 5 phút
../.venv-track/bin/python step5_flow/code/run_step5.py --videos C17,F30             # vài video
../.venv-track/bin/python step5_flow/code/run_step5.py --force                      # tính lại feature dù đã xong
S5_TRIAL=1 ../.venv-track/bin/python step5_flow/code/run_step5.py --videos F29      # thử cấu hình mới: ghi vào _trial/, không đụng output chính
../.venv-track/bin/python step5_flow/code/s5_validate.py                            # kiểm tra output, sai thì thoát mã 1
```

GMA chỉ chạy khi `flow_store/<vid>.npz` chưa có hoặc có `flow_hash` khác. Đổi cách bù camera chỉ cần tính lại feature, không chạy lại GMA. Một video được coi là xong khi đủ output và có `config` trùng với hằng số trong `code/s5_common.py`.

Cài đặt: GMA ở `ROOT/.third_party/GMA` (commit `2f1fd29`), checkpoint `checkpoints/gma-sintel.pth`, thêm gói `einops`. Không sửa mã GMA. MPS cho kết quả khớp CPU, nên không cần chuyển sang RAFT.

## Cấu hình

| Mục | Giá trị |
|---|---|
| Model | GMA sintel, 12 vòng lặp |
| Cặp frame | 2 frame unique liền nhau trong cùng `continuity_id` và `shot_id`, từ danh sách frame của Bước 3 |
| Crop | Nửa dưới bbox Bước 2 của 2 frame, nới 25% chiều cao bbox 2 bên, 15% phía dưới |
| Resize | Cố định theo khối: L_foot ≥ 48 px, cạnh dài ≤ 640 px (được tới 800 px nếu cần giữ L_foot) |
| FB | \|f(x) + b(x + f(x))\| ≤ 1 px + 5% \|f\| trên lưới flow. **Chỉ dùng cho QC và bản `fb_*`** |
| Chuẩn hoá | v = \|residual\| / L_leg(t) / Δt; vx nhân `s_final` nên dương là hướng chạy; vy dương là xuống dưới ảnh |
| `slow_ratio` chính | Tỉ lệ pixel có v < 2,0 L_leg/s |

## Bù camera

### A. Bù trực tiếp (mọi video)

Homography toàn ảnh cũ thiếu ở 33/66 video: mặt đường ít chi tiết làm flow gần 0 dù camera đang bám người chạy, hoặc phép fit bám vào hàng rào xa. Cách hiện tại:

1. **Dải đất:** từ 0,5 L_foot phía trên tới 1,0 L_foot phía dưới điểm chân thấp nhất, trừ mask chân đã nới và ống chân.
2. **Mặt nạ chữ cho cả video:** ô ảnh có pixel chi tiết cao và đứng yên tuyệt đối (< 0,5 px) lặp lại ở ít nhất 3 cặp là chữ gắn trên ảnh. Không áp dụng cho video camera đứng yên (`cam_static`), vì ở đó mọi nền đều đứng yên.
3. **Chỉ pixel có chi tiết theo cả 2 hướng** (trị riêng nhỏ của structure tensor), thử mức cao trước. Lấy tối đa 3 cụm độ dịch, chọn cụm đủ lớn có độ dịch lớn nhất, vì làm trơn của flow chỉ kéo mặt đất về 0.
4. **Mô hình tịnh tiến**, không dùng affine, vì dải đất hẹp làm affine ngoại suy kém.
5. **Không có chi tiết** thì dùng cả dải nếu flow trên đó đang chuyển động nhất quán. Nếu pixel chi tiết đứng yên chiếm ít nhất 2% dải thì coi là nền đứng yên. Còn lại là không đo được.
6. **Theo thời gian:** loại giá trị đổi hướng hoặc tụt quá nửa tốc độ so với trung vị trượt (không loại phía tăng, vì frame bị bỏ làm độ dịch nhảy gấp đôi là thật); chuỗi đo thưa được lọc trung vị; cặp không đo được thì nội suy.

`cam_dx` của Bước 0 không dùng được làm nguồn thứ hai: tương quan với độ dịch trực tiếp có trung vị 0,26 (−0,49 đến 0,98), và độ lớn chỉ bằng khoảng 7% độ dịch đó. Vẫn ghi `cam_dx_rho`, `cam_dx_ratio`, `cam_dx_sum` để đối chiếu.

### B. Thành phần chậm (chỉ khi A vẫn thiếu)

Khi P10 của vx cả bàn chân trên toàn video nhỏ hơn ngưỡng của biến thể, thêm hiệu chỉnh c(t) theo từng khối: spline bậc 3 có nút cách 1 stride (`stride_T` của Bước 3), fit theo P10 của vx trên cả 2 chân trong cửa sổ 1 stride, và bị chặn trong khoảng P10 đã đo. Khối ngắn hơn 2 stride dùng hằng số. c(t) được trừ khỏi cả pixel và keypoint, chỉ theo hướng x.

Bước B **không phải ràng buộc chân chống đứng yên theo từng frame**: nó không thay đổi nhanh hơn 1 stride, nên không tự tạo pha chống. Nhưng nó vẫn giả định P10 của vx bằng 0 trong mỗi stride, nên có thể làm giảm khác biệt thật nếu video có lỗi kiểu bàn chân trượt.

`comp_method` là `direct` hoặc `direct+slow`. `comp_slow` ghi lượng hiệu chỉnh theo frame (L_leg/s).

### 2 biến thể

| | Ngưỡng kích hoạt B | Video `direct+slow` | CORRECT (trên 39) | INCORRECT (trên 27) | File |
|---|---|---:|---:|---:|---|
| **V1 (mặc định)** | P10 vx < −1 | 25 | 15 (38%) | 10 (37%) | `features/<vid>.npz` |
| V2 (để ablation) | P10 vx < −2 | 16 | 7 (18%) | 9 (33%) | `features/<vid>__v2.npz` |

**Lý do chọn V1:** V1 dùng bước B cân bằng giữa 2 nhãn (38% và 37%), còn V2 lệch (18% và 33%). Chọn V1 để tránh tiền xử lý khác nhau theo nhãn.

Cột không có tiền tố trong `qc_flow.csv` là V1; cột `v2_*` là V2. 9 video khác cách bù có thêm `plots/<vid>__v2.png`: C17, C23_3, C26, C28_1, C28_3, C29, C38_6, C38_9, F28_3.

## Quyết định khác

**FB chỉ dùng cho QC.** Trên bàn chân mảnh và nhanh, FB loại đúng những pixel flow đúng và giữ pixel bám theo nền. Bản lọc FB vẫn lưu ở `fb_*`.

**Ngưỡng 2,0 L_leg/s.** Các ngưỡng 1,0 / 1,5 / 3,0 và Otsu được lưu để Bước 6 so sánh; tất cả tính từ histogram.

**Δt: VFR giả.** pts của nhiều video không phản ánh thời điểm chụp thật. Mỗi video được quyết định `dt_mode`:

1. Tính Δt và độ dịch chia cho trung vị trượt 9 cặp.
2. `dt_jitter` < 0,05: giữ `raw` (28 video).
3. rho lớn nhất giữa Δt và độ dịch (camera, `kp_raw`) ≥ 0,4: giữ `raw` (1 video: C09_2).
4. Còn lại: `smooth` (37 video), trung bình trượt 5 cặp trong từng (khối, shot, part), giữ nguyên tổng thời gian.

Phép thử dùng `kp_raw`, vì `kp_clean` đã lọc trên trục `t_real_s_v2` nên độ dịch của nó tỉ lệ với Δt theo cấu tạo. Video dùng `smooth`: xem cột `dt_mode`.

**Cờ `cam_fit_bad` theo frame:** `flow_valid` = False và mọi feature là NaN.

| Bit `cam_fit_reason` | Điều kiện | Số cặp (V1) | Làm `cam_fit_bad` |
|---|---|---:|---|
| 1 | Cặp đo được có tỉ lệ inlier < 0,15 | 40 | Có |
| 2 | `bg_residual` > P95 của video, > 5 px và > 25% độ dịch camera | 172 | Có |
| 4 | Camera đang pan, cả 2 chân đều > 15 hoặc đều < 0,2 L_leg/s | 25 | Có |
| 8 | `bg_residual` > P95 nhưng dưới sàn | 151 | **Không** |
| 16 | \|`comp_slow`\| > 6 L_leg/s và > 2 × trung vị của khối: bù trực tiếp sai cục bộ, không phải nền trôi chậm | 14 (chỉ C38_3, f0–13) | Có |

Tổng 232 trên 8331 cặp (2,8%). Cặp không đo được đã được nội suy nên không tự động bị gắn cờ.

**Cadence theo speed_part.** Lomb–Scargle của `slow_ratio` gót tính riêng từng part. Video nhiều part so với chu kỳ stride của part đó ở Bước 3 (`cadence_ref_source` = `part_T`, 13 video); video 1 part so với `cadence_real_spm_v2` (44 video), hoặc 120 / `stride_T_median_s` nếu không có (8 video). F25 không có tham chiếu. Bảng theo part ở `qc/<vid>_cadence_parts.csv`. Giá trị của video là trung bình theo số frame. Đỉnh phải là cực đại cục bộ.

## Chỉ số pha chống

Chỉ để kiểm tra: **không vào gate và không thay đổi feature.** Theo quyết định, không làm mượt hay lọc trễ feature; việc gộp đoạn dưới đây chỉ áp dụng khi tính các cột này. `_first` là near hoặc A, `_second` là far hoặc B (`foot_labels` cho biết video có nhãn nào).

| Định nghĩa | Tiền tố cột | Chân đang chống khi | Xử lý đoạn |
|---|---|---|---|
| **Mới** | không có | max(`slow_ratio` gót, giữa, mũi) > 0,5, ngưỡng 2,0 L_leg/s | Gộp 2 đoạn cách nhau ≤ max(1 frame unique, 30 ms thật); rồi bỏ đoạn chỉ có 1 frame |
| Mới, chỉ gót | `heel_` | `slow_ratio` gót > 0,5 | Như trên |
| Cũ | `old_` | `slow_ratio` gót > 0,5 | Chỉ nối qua đoạn không hợp lệ ≤ 2 frame |

| Cột | Định nghĩa | Kỳ vọng sinh lý |
|---|---|---|
| `duty_factor_*` | Tỉ lệ thời gian (theo `t_real_s_v2`) đang chống | 0,25–0,40 khi chạy; có thể cao hơn ở video chạy chậm |
| `stance_run_median_s_*`, `stance_run_median_frames_*` | Trung vị độ dài đoạn chống | khoảng 0,15–0,30 s |
| `n_stance_per_stride_*` | Số đoạn chống / số stride theo cadence tham chiếu (theo part nếu video nhiều part) | khoảng 1 mỗi chân |
| `kp_speed_min_p10_*` | P10 tốc độ keypoint bàn chân sau bù, L_leg/s; không phụ thuộc định nghĩa | gần 0 |

Bỏ frame `exclude_time_features`. Đoạn chạm mép khối bị cắt nên không dùng để tính độ dài nếu có đoạn khác, nhưng vẫn được đếm.

Kết quả V1 trên 66 video, trung vị / số video ngoài khoảng (dùng 0,75–1,25 cho `n_stance_per_stride`):

| Chỉ số | Chân | Cũ | Mới, chỉ gót | **Mới** |
|---|---|---:|---:|---:|
| `duty_factor` | near/A | 0,23 / 42 | 0,22 / 43 | **0,29 / 30** |
| | far/B | 0,25 / 35 | 0,24 / 37 | **0,32 / 35** |
| `stance_run_median_s` | near/A | 0,13 / 46 | 0,15 / 30 | **0,19 / 20** |
| | far/B | 0,10 / 52 | 0,15 / 38 | **0,19 / 19** |
| `n_stance_per_stride` | near/A | 1,32 / 41 | 1,05 / 24 | **1,11 / 26** |
| | far/B | 1,68 / 45 | 1,12 / 22 | **1,19 / 24** |
| Đạt cả 3 khoảng, cả 2 chân | | 0 | 5 | **12** |

`kp_speed_min_p10` không đổi: trung vị 0,84 và 0,94, có 27 và 30 video trên 1,0.

Một phần cải thiện đến từ chính định nghĩa mới, vì max của 3 vùng và việc gộp đoạn đều nới điều kiện. Vì vậy bảng này không chứng minh `slow_ratio` đã đúng; nó cho thấy đoạn vụn phần lớn là do cách đếm. Đa số video vẫn chưa đạt cả 3 khoảng. Bước sau muốn dùng đoạn chống phải tự xác định lại từ `slow_ratio`, vì không có mảng đoạn chống trong features.

## Output

```
flow_store/<vid>.npz            flow forward thô (float16, 1/16 px) và mặt nạ FB, trên hộp bao mask chân + dải đất. Tổng 781 MB
features/<vid>.npz              feature biến thể mặc định (V1)
features/<vid>__v2.npz          feature biến thể V2, để ablation
qc_flow.csv                     1 dòng mỗi video, có gate; cột v2_* cho V2
qc/<vid>_qc.json                bản đầy đủ của dòng QC
qc/<vid>_cadence_parts.csv      cadence theo speed_part (V1); __v2 cho V2
plots/<vid>.png                 slow_ratio gót và mũi của near/far, kèm tốc độ keypoint (V1)
plots/<vid>__v2.png             như trên cho V2, chỉ ở 9 video khác cách bù
review/<vid>.jpg                8 frame có residual flow trên mask (V1)
baseline_v1/                    feature và QC trước khi đổi cách bù camera, để so sánh. Tên "v1" ở đây không liên quan đến biến thể V1
logs/                           log các lần chạy
```

Kho lưu flow thô thay vì residual: residual = flow − mô hình camera, còn đổi mô hình thì cần flow thô.

### `features/<vid>.npz`

N = số frame của Bước 3. Feature của frame a tính trên cặp (a, a+1), nên frame cuối mỗi khối là NaN. Trục 2 là chân: 0 = slot A, 1 = slot B; dùng `label` để biết near/far. Trục 3 là vùng: 0 gót, 1 giữa, 2 mũi, 3 cả bàn chân (hợp 3 vùng, không gồm phần mask ngoài vùng).

| Khoá | Kích thước | Nội dung |
|---|---|---|
| `v_med`, `v_p25`, `v_p75` | N×2×4 | Tốc độ, L_leg/s, theo Δt đã chọn, trên mọi pixel |
| `vx_med`, `vy_med` | N×2×4 | Thành phần, L_leg/s |
| `n_pixels`, `frac_fb_valid` | N×2×4 | Số pixel trên lưới flow; tỉ lệ pixel qua FB |
| `v_hist`, `hist_edges` | N×2×4×B, B+1 | Histogram v, để đổi ngưỡng không cần tính lại |
| `slow_ratio` | N×2×4 | Tỉ lệ pixel v < 2,0 L_leg/s |
| `slow_ratio_1p0`, `_1p5`, `_3p0`, `_otsu`; `otsu_thr` | N×2×4 | Các ngưỡng khác |
| `fb_v_med`, `fb_v_p25`, `fb_v_p75`, `fb_vx_med`, `fb_vy_med`, `fb_n_pixels`, `fb_v_hist`, `fb_slow_ratio`, `fb_slow_ratio_otsu`, `fb_otsu_thr` | | Bản lọc FB, chỉ để đối chiếu |
| `v_rawdt`, `v_smoothdt`, `slow_ratio_rawdt`, `slow_ratio_smoothdt` | N×2×4 | Cả 2 cách tính Δt |
| `kp_speed`, `kp_speed_rawdt`, `kp_speed_smoothdt` | N×2 | Tốc độ tâm 3 điểm bàn chân (`kp_clean`) sau bù, L_leg/s |
| `kp_raw_speed_rawdt`, `kp_raw_speed_smoothdt` | N×2 | Như trên, từ `kp_raw` |
| `comp_variant`, `comp_trigger`, `comp_method` | | Biến thể, ngưỡng, `direct` hoặc `direct+slow` |
| `comp_tx`, `comp_ty` | N | Tịnh tiến camera, px trên lưới flow |
| `comp_direct_v`, `comp_slow` | N | Phần bù trực tiếp và phần bù chậm, L_leg/s |
| `comp_state`, `comp_source` | N | 0 đo được, 1 nền đứng yên, 2 nội suy, 3 không có nguồn; nguồn đo |
| `comp_inlier_frac`, `comp_n_inlier`, `comp_n_tex`, `bg_residual`, `cam_flow`, `cam_dx_sum` | N | Chất lượng bù camera |
| `dt_real`, `dt_smooth`, `dt_used`, `dt_mode` | N | Δt gốc, làm mượt, đang dùng; `raw` hoặc `smooth` |
| `flow_valid`, `cam_fit_bad`, `cam_fit_reason`, `has_flow` | N | Cờ theo cặp |
| `frame_valid_mask`, `invalid_reason`, `state`, `label` | N×2 | Từ Bước 4 |
| `frame`, `t_real_s_v2`, `continuity_id`, `shot_id`, `speed_part_id`, `L_leg` | N | Trục thời gian và khối |
| `view_lateral`, `exclude_time_features` | N | Từ Bước 4 |
| `crop`, `scale`, `frac_fb_all`, `v_foot_any` | | Lưới flow; `v_foot_any` là v **trước khi lọc**, chỉ để chẩn đoán |
| `cadence_ref`, `cadence_ref_source`, `s_final`, `config`, `flow_hash`, `post_config`, `model` | | Metadata |

Feature là NaN khi `frame_valid_mask` hoặc `flow_valid` là False, và khi có dưới 10 pixel.

## Gate và cột của `qc_flow.csv`

| Mức | Điều kiện |
|---|---|
| FAIL | `spearman_flow_kp` < 0,4; hoặc `cadence_dev` > 15% **và** `cadence_dev_spm` ≥ `cadence_resolution_spm` |
| WARN | `spearman_flow_kp` < 0,7; `bg_residual_median` > max(3 px, 25% độ dịch camera); `frac_fb_valid` < 0,7; hoặc `cam_undercomp` |
| PASS | Còn lại |

`cam_undercomp` là P10 của vx sau bù < −2. Sau bước B, P10 về 0 theo cấu tạo nên cờ này không còn bật ở video nào; để biết video nào phải dựa vào B, xem `comp_method` và `comp_slow_median`.

| Cột | Ý nghĩa |
|---|---|
| `comp_method`, `comp_slow_median`, `comp_slow_max`, `comp_direct_v_median` | Cách bù; hiệu chỉnh chậm; độ dịch camera, L_leg/s |
| `frac_comp_measured`, `frac_comp_static`, `frac_comp_filled`, `frac_comp_none` | Tỉ lệ cặp đo được, nền đứng yên, nội suy, không có nguồn |
| `vx_p10`, `vx_p10_direct`, `frac_foot_backward`, `cam_undercomp` | P10 vx sau bù, sau riêng A; tỉ lệ vx < −1; cờ |
| `spearman_flow_kp`, `spearman_first`, `spearman_second`, `n_spearman` | Spearman giữa `v_med` cả bàn chân và `kp_speed` |
| `spearman_fb`, `spearman_rawdt`, `spearman_smoothdt`, `spearman_direct` | Với bản FB; từng cách tính Δt; chỉ A |
| `cadence_ref`, `cadence_ref_source`, `cadence_ref_used`, `cadence_flow`, `cadence_fb`, `cadence_flow_direct` | Tham chiếu của video; nguồn; tham chiếu đã dùng theo part; cadence từ `slow_ratio` |
| `cadence_dev`, `cadence_dev_direct`, `cadence_dev_spm`, `cadence_ratio`, `cadence_resolution_spm`, `cadence_below_resolution`, `n_cadence_parts`, `ls_power`, `ls_span_s` | Độ lệch và độ phân giải |
| `minima_per_stride`, `minima_per_stride_direct`, `power_ratio_2f`, `power_ratio_2f_direct` | Số cực tiểu \|v\| mỗi stride; công suất ở 2f so với f. Nền trôi làm các giá trị này tăng |
| `alternation_rho`, `alternation_rho_heel`, `alternation_rho_heel_direct`, `frac_both_high`, `frac_both_low`, `frac_both_still` | Luân phiên, chỉ để tham khảo |
| `dt_mode`, `dt_mode_reason`, `dt_jitter`, `dt_rho_cam`, `dt_rho_kp`, `dt_cv` | Quyết định Δt |
| `n_cam_fit_bad`, `n_cam_bad_inlier`, `n_cam_bad_bg`, `n_cam_bad_tail`, `n_cam_bad_slow`, `n_cam_bg_below_floor`, `frac_flow_valid` | Cờ fit camera |
| `bg_residual_median`, `bg_residual_p95`, `bg_residual_v`, `bg_inlier_frac`, `cam_flow_median`, `cam_grid_median`, `overlay_frac`, `cam_static`, `cam_dx_rho`, `cam_dx_ratio` | Bù camera |
| `frac_fb_valid`, `otsu_thr`, `fb_otsu_thr`, `v_foot_p10`, `v_foot_p50`, `v_foot_p90`, `slow_ratio_median`, `slow_ratio_median_direct` | Phân phối v |
| `duty_factor_*`, `stance_run_median_s_*`, `stance_run_median_frames_*`, `n_stance_per_stride_*`, `kp_speed_min_p10_*`; `heel_*`, `old_*` | Chỉ số pha chống, xem mục trên |
| `scale_min`, `scale_max`, `l_foot_grid`, `crop_edge_max`, `frac_usable_first`, `frac_usable_second`, `n_frames`, `n_pairs`, `n_pairs_no_crop` | Lưới flow và số lượng |
| `abs_time_ok`, `device`, `model`, `time_s`, `fps`, `config`, `default_variant`, `foot_labels` | Metadata |
| `v2_*` | Các cột chính của V2 |

## Giới hạn đã biết

1. **Chỉ số pha chống chưa đạt ở đa số video,** chỉ 12/66 video đạt cả 3 khoảng ở cả 2 chân, và kết quả phụ thuộc cách đếm. Nguyên nhân phần còn lại chưa được xác định.
2. **Tham số bù camera được chỉnh trên 12 video thử,** rồi áp dụng cho 54 video còn lại. 25 video (V1) phải dựa vào bước B.
3. **Bước B có thể làm giảm khác biệt thật** nếu bàn chân trượt. V1 cân bằng số video dùng B giữa 2 nhãn, nhưng vẫn cần kiểm tra ảnh hưởng khi phân cụm; so với V2 để ablation.
4. **Spearman ở video `direct+slow` có thể được nâng một phần,** vì keypoint và flow cùng bị trừ c(t).
5. **C38_3:** 14 frame đầu có độ dịch trực tiếp sai (+55 px, ngược hướng), đã bị loại bằng bit 16. Quy tắc chỉ bắt lỗi lớn; lỗi tương tự với `comp_slow` dưới 6 L_leg/s vẫn lọt.
6. **Không đo được mặt đất ở C28_2, F28_2, F28_4** (`frac_comp_none` = 1): toàn bộ bù camera đến từ B. 10 video khác có hơn 50% cặp là nội suy (`frac_comp_filled`).
7. **`alternation_rho` bị kéo về 0 khi đa số frame có slow_ratio bằng 0 ở cả 2 chân** (nhiều giá trị trùng), nên có thể đánh giá thấp luân phiên. Ví dụ C38_9 theo V1: plot luân phiên rõ mà `alternation_rho_heel` chỉ −0,08.
8. **Δt làm mượt vẫn là ước lượng,** không khôi phục thời điểm chụp thật. Với C09_1 và C09_2 (k = 1/8 độ tin cậy thấp), độ dài đoạn chống theo giây không có ý nghĩa.
9. **Cadence của part rất ngắn không đáng tin** (ví dụ C19 part 1, 0,62 s).
10. **`s5_validate.py`** chỉ xác nhận output nhất quán và đúng định nghĩa, không xác nhận flow đúng về hình ảnh.

## Mã nguồn

| File | Vai trò |
|---|---|
| `code/run_step5.py` | Chạy pipeline, resume, 2 biến thể |
| `code/s5_common.py` | Hằng số trong config hash, nạp dữ liệu, cặp frame, crop |
| `code/s5_store.py` | Kho flow: chạy GMA và đọc lại |
| `code/s5_flow.py` | GMA và FB |
| `code/s5_ground.py` | Bù trực tiếp trên dải đất, mặt nạ chữ, thành phần chậm |
| `code/s5_features.py` | Feature, quyết định Δt, cờ fit camera, ngưỡng |
| `code/s5_checks.py` | Spearman, Lomb–Scargle theo part, cực tiểu mỗi stride, chỉ số pha chống, gate |
| `code/s5_report.py` | Plot và ảnh review |
| `code/s5_validate.py` | Kiểm tra output |
