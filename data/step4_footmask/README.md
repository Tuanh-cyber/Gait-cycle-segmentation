# Bước 4: Foot mask bằng SAM2

Tách mask từng bàn chân theo thời gian bằng SAM2.1, chia mask thành 3 vùng gót / giữa / mũi, gán trạng thái che khuất và nhãn near/far, rồi cho mỗi video một gate.

Kết quả hiện tại: **59 PASS, 5 WARN, 2 FAIL** trên 66 video, 8407 frame. Bảng chi tiết ở `qc_footmask.csv`.

**Các bước sau đọc `masks_post/`, không đọc `masks/`,** và chỉ dùng frame có `frame_valid_mask` = True. `masks/` là output gốc của SAM2 (config `ed4897ca59`), giữ nguyên để có thể làm lại hậu kỳ mà không chạy SAM2 khoảng 1,5 giờ. `masks_post/` là bản đã clip ống chân, sửa nhãn và gắn cờ (post_config `7b8f8caa7d`).

## Chạy

Chạy từ `ROOT/data`, dùng `.venv-track`:

```bash
../.venv-track/bin/python step4_footmask/code/run_step4.py                     # cả 66 video, tự bỏ qua video đã xong
../.venv-track/bin/python step4_footmask/code/run_step4.py --videos C30,F26    # vài video
../.venv-track/bin/python step4_footmask/code/run_step4.py --force             # chạy lại dù đã xong
../.venv-track/bin/python step4_footmask/code/s4_validate.py                   # kiểm tra masks/, sai thì thoát mã 1
../.venv-track/bin/python step4_footmask/code/s4_postprocess.py                # hậu kỳ cả 66 video, vẽ lại ảnh, kiểm tra; không chạy SAM2
../.venv-track/bin/python step4_footmask/code/s4_postprocess.py --check        # chỉ kiểm tra masks_post/ và qc_footmask.csv
../.venv-track/bin/python step4_footmask/code/regen_review.py                  # vẽ lại ảnh WARN/FAIL từ masks_post/
```

Một video được coi là xong SAM2 khi đủ output và có config hash trùng với hằng số trong `code/s4_common.py`. Đổi bất kỳ hằng số nào trong file đó thì mọi video sẽ chạy lại SAM2; vì vậy hằng số hậu kỳ nằm trong `s4_postprocess.py`. **`run_step4.py` ghi đè `qc_footmask.csv` bằng bảng gốc của SAM2 và vẽ lại ảnh từ `masks/`,** nên sau khi chạy nó phải chạy lại `s4_postprocess.py`.

Nên tắt chế độ ngủ của máy khi chạy cả dataset, ví dụ `caffeinate -i ../.venv-track/bin/python ...`. Lần chạy trước, máy ngủ từ 03:41 làm F24 và F25 mỗi video mất gần 1 giờ; mask của chúng vẫn qua validate, nhưng `time_s` và `fps` của 2 dòng này không phản ánh tốc độ thật.

Cài đặt: SAM2 ở `ROOT/.third_party/sam2` (commit `2b90b9f`), `pip install --no-deps -e` với `SAM2_BUILD_CUDA=0`, thêm `hydra-core` và `iopath`. Không có C extension nên bước lấp lỗ của SAM2 không chạy. Checkpoint ở `models/` (khoảng 500 MB).

## Cấu hình cuối

| Mục | Giá trị | Ghi chú |
|---|---|---|
| Checkpoint | SAM2.1 `hiera_base_plus` | `models/sam2.1_hiera_base_plus.pt` |
| `image_size` | 768 | SAM2 mặc định 1024 |
| Thiết bị | MPS, `PYTORCH_ENABLE_MPS_FALLBACK=1` | Lỗi thì tự chuyển CPU và in cảnh báo; cả 66 video đều chạy MPS |
| Crop | Nới 15% mỗi phía, cố định theo khối, giữ độ phân giải gốc | `crop_scale` luôn bằng 1 |
| Prompt | Box nới 35% L_foot, điểm dương gót + ngón cái, điểm âm trên chân kia và giữa ống chân của chính nó | |
| Điều kiện trôi | `coverage_tol` < 0,5 hoặc tỉ lệ diện tích ngoài [0,4; 2,5] | Dung sai `COV_TOL` = 0,15 L_foot |
| Vòng re-prompt | Đoạn lỗi tối thiểu 3, 3, 1 frame | Không re-prompt khi box 2 bàn chân có IoU > 0,1 |
| Bỏ mảnh rời | Mảnh ≥ 5% diện tích mask và ≥ 20 px, thuộc về chân có keypoint gần tâm mảnh hơn; xa quá 1 L_foot thì bỏ | Không xét khi 2 bàn chân đang chồng nhau |
| Trạng thái | occluded: tỉ lệ diện tích < 0,25, rỗng, hoặc đã dính sang chân kia; partial: 0,25–0,6 hoặc IoU > 0,3 | |
| Vùng | gót [−0,15; 0,33], giữa (0,33; 0,66], mũi (0,66; 1,15] L_foot | Giới hạn vuông góc ±0,6 L_foot |

Lý do chọn checkpoint, đo trên video thử:

| Cấu hình | Tốc độ | Nhận xét |
|---|---|---|
| `small` @1024 | 0,9 fps | Hơn 2,6 giờ cho cả dataset, vượt ngân sách 2 giờ |
| `base_plus` @1024 | 0,74–0,8 fps | Hơn 2,6 giờ, vượt ngân sách |
| `small` @768 | 2,3 fps | `coverage_tol`: F26 0,901, C38_8 0,609 |
| `base_plus` @768 | 2,0 fps | `coverage_tol`: F26 0,973, C38_8 0,696. **Đã chọn** |

So với 1024, mask ở 768 có IoU khoảng 0,96–0,97 trên F26 và 0,87–0,89 trên C38_8 (người nhỏ). Thời gian thực tế cả dataset khoảng 1,45 giờ, chưa tính 2 video bị máy ngủ.

## Hậu kỳ

`code/s4_postprocess.py` đọc `masks/`, tính lại mọi thứ trên mask đã clip, không dùng SAM2. Kiểm tra `--check` chạy trên mọi frame: mask mới nằm trong mask gốc, RLE khớp diện tích, 3 vùng rời nhau và nằm trong mask, các cờ đúng định nghĩa.

### Ankle clip

Bỏ pixel mask ở phía ống chân, vượt quá **đường song song với trục bàn chân** đi qua cổ chân (`kp_clean`) cộng lề 0,15 L_foot.

Quyết định ban đầu ghi "đường vuông góc với trục đi qua ankle". Làm đúng nghĩa đen thì đường đó cắt ngang bàn chân chứ không cắt ống chân: cổ chân nằm ở khoảng 0,13–0,18 L_foot dọc trục, nên 56–66% diện tích mask nằm ở phía mũi và 34–44% ở phía gót (đo trên F17, C13_2, C09_1, C20). Khoảng cách vẫn được đo theo hướng vuông góc với trục, đúng với mục đích của quyết định.

- **Phía ống chân:** theo gối nếu gối lệch trục ít nhất 0,2 L_foot; nếu không thì theo cổ chân, khi cổ chân lệch trục ít nhất 0,05 L_foot.
- **Thiếu cổ chân:** dùng trung vị độ cao cổ chân của khối.
- **Giới hạn:** đường cắt không bao giờ thấp hơn trục, nên keypoint cổ chân lệch không cắt mất thân giày.
- **Không clip:** khi không biết phía ống chân, hoặc trục ngắn hơn 0,4 L_foot. Có 82 trên 16814 cặp (frame, object) như vậy (`ankle_clip_skipped`).

Clip bỏ trung vị 1% diện tích mask mỗi video, nhiều nhất 45% ở F17. `mean_coverage_tol` giảm nhiều nhất 0,004, tức gần như không mất điểm trên bàn chân.

### Xung đột ID lấy từ cả mask trước clip

`mask_id_conflict` là hợp của cờ trước và sau clip. Nếu chỉ tính sau clip, số frame xung đột giảm từ 180 xuống 111 và không còn video FAIL. Nhưng đó không phải sửa thật: clip chỉ cắt bớt phần mask phủ keypoint chân kia nên cờ tắt đi, trong khi mask vẫn nằm sai chân (C04_3 f14, f23, f87; ở các video có nhiều cờ bị mất nhất, trung vị `coverage_tol` của chúng bằng 0).

### `frame_valid_mask` và `invalid_reason`

Cả hai là mảng N×2, theo từng chân. `frame_valid_mask` = True khi `invalid_reason` không có bit nào trong `INVALID_BITS`. Không loại cả video.

| Bit | Tên | Điều kiện | Làm False |
|---|---|---|---|
| 1 | occluded | Chân đó occluded | Có |
| 2 | id_conflict | Frame có xung đột ID ở bất kỳ chân nào (đặt cho cả 2 chân) | Có |
| 4 | area_high | `area_ratio_norm` > 1,6 | Có |
| 8 | area_low | `area_ratio_norm` < 0,4 và mask không rỗng | Có |
| 16 | no_keypoint | Cả gót, ngón cái, ngón út (`kp_clean`) đều cách mask quá 0,2 L_foot | Có |
| 32 | perp_thick | `perp_thickness` > 0,7 **và** > 1,3 × trung vị độ dày của chân đó trong khối | Có |
| 64 | literal_only | Chỉ luật nguyên văn bắt, 2 luật đã chuẩn hoá không bắt | **Không** |

Toàn dataset: 94,1% cặp (frame, object) hợp lệ. 3 kiểm tra mới loại thêm 546 trên 16814 cặp (3,2%); theo video trung vị 1,6%, lớn nhất 14,1% (C04_3). Gate không dùng các kiểm tra này.

**Khác với quyết định review.** Các ngưỡng 1,6 / 0,4 / 0,7 / 0,2 giữ nguyên, nhưng có 2 chuẩn hoá vì luật nguyên văn bắt nhầm hàng loạt mask đúng:

| | Luật nguyên văn | Đã dùng |
|---|---|---|
| Diện tích | so với trung vị khối | `area` / L_leg(t)², rồi so với trung vị khối. Người lại gần hoặc đi xa camera trong khối làm F10_1 bị 26%, F10_2 bị 19% |
| Độ dày | > 0,7 L_foot | Thêm điều kiện > 1,3 × trung vị của chân đó. L_foot đo từ gót tới ngón nên ngắn hơn giày thật: F30 bị 26%, C30 bị 48% |
| Loại thêm theo video | trung vị 3,5%, lớn nhất 71% | trung vị 1,6%, lớn nhất 14% |
| Danh sách 23 frame lỗi | bắt 14 | bắt 11 |

Hệ số 1,3 được chọn trên chính danh sách review, chưa kiểm định trên tập độc lập. Muốn dùng luật nguyên văn: thêm `R_LITERAL_ONLY` vào `INVALID_BITS` trong `s4_postprocess.py` rồi chạy lại hậu kỳ; `frac_removed_new_literal` cho biết tỉ lệ loại khi đó.

**Đối chiếu với danh sách review:**

| Kết quả | Frame |
|---|---|
| Kiểm tra mới bắt | C26 f96, F26 f0, F22_4 f8/f59, C28_3 f5, F28_4 f3 |
| Đã False từ trước (xung đột ID) | F25 f0/f15, C11 f55, C30 f204/f26 |
| Chỉ luật nguyên văn bắt (bit 64) | F10_2 f84, F20 f26/f115 |
| Không luật nào bắt | F10_2 f14, F22_4 f25, F20 f221, C28_2 f4/f21, F28_4 f12/f29, F12 f33/f149 |
| Frame đúng, không bị bắt | C38_3 f11/f29/f83, C38_4 f4/f23 |

9 frame không luật nào bắt có số đo nằm trong vùng bình thường. Ví dụ F12 f33 có tỉ lệ diện tích 0,87 và độ dày 0,37, thấp hơn trung vị dataset. Ở các frame này mask lan theo chiều dọc trục hoặc nằm trên phần mờ, nên kiểm tra hình học không phân biệt được.

**Vẫn có bắt nhầm:** `perp_thick` bắt C38_3 f1/f56 và C38_4 f13 (giày dựng đứng pha swing), F30 f5/f54. Trong 16 frame mẫu ở 7 video có tỉ lệ bị bắt cao, xem bằng mắt thì khoảng 9 frame mask sai thật. F17 có 15 frame bị bắt; 4 frame đã xem đều là bàn chân bị mép ảnh cắt.

### F10_2

| Khối | Frame | Nhãn | `view_lateral` | Lý do |
|---|---|---|---|---|
| 0 | f0–143 (52) | near/far | True | Bước 3 khớp metadata |
| 1 | f523–583 (12) | **A/B** | True | Bước 3 không tin cậy ở khối này, xem dưới |
| 2 | f985–1051 (26) | near/far | True | Bước 3 khớp metadata |
| 2 | **f1052–1078 (23)** | A/B | **False** | Quay lưng |

**Khối 1 không đạt điều kiện "Bước 3 tin cậy".** Bước 3 cho `near_track` = B, trong khi ảnh f545 và f550 cho thấy chân A đè lên chân B, tức A mới là chân gần. Metadata cũng kỳ vọng A (`match` = False trong `step3_footkp/qc/F10_2_blocks.csv`). `near_conf` = 1,0 chỉ là bootstrap trên 1 chunk, vì khối có 12 frame. Quyết định review ngày 05/11: giữ A/B, không đổi near.

**Đoạn quay lưng** là hậu tố của khối 2 có trung vị trượt 5 frame của |x hông trái − x hông phải| / chiều dài thân ≥ 0,30; f985–1017 có khoảng 0,13, còn f1072–1078 quay lưng rõ có 0,33–0,38. Người chạy xoay dần nên ranh giới không sắc: ngưỡng 0,25 cho f1041, 0,28 cho f1048, 0,33 cho f1053. Kết quả lưu ở `qc_post/F10_2_view.json`.

`view_lateral` chỉ được xét ở F10_2. Các video khác mặc định True, không có nghĩa là đã kiểm tra góc quay.

### Cờ thời gian của Bước 3

C20 part 1 (f76–243, 126 frame unique) có `exclude_time_features` = True trong `masks_post/C20.npz`, theo `step3_footkp/time_feature_flags.csv`. Các video khác toàn False.

## Khác với đề bài

| Đề bài | Đã làm | Lý do |
|---|---|---|
| `image_size` mặc định | 768 | Giữ 1024 vượt ngân sách thời gian |
| Tối đa 5 re-prompt/khối, nối tiếp memory | Không giới hạn; mỗi đoạn lỗi được prompt lại bằng memory mới, chỉ object đó, chỉ trong đoạn đó | Giày giống nhau làm object nhảy sang chân kia ở mỗi lần chéo chân; 5 lần dùng hết rất sớm (C13_1 cần 179) |
| Coverage đúng nghĩa đen | Thêm `coverage_tol` để phát hiện trôi và xung đột; `coverage` gốc vẫn được lưu | Keypoint hay nằm sát mép giày nên coverage gốc báo sai |
| Điểm âm trên chân kia | Thêm điểm âm giữa ống chân | Chân trần làm mask lan lên cẳng chân |
| Vùng chỉ theo trục dọc | Thêm giới hạn vuông góc ±0,6 L_foot | Ống chân chiếu lên trục sẽ rơi vào vùng gót |
| Không có | Bỏ mảnh mask rời nằm trên chân kia | Một object có thể phủ cả 2 bàn chân; xem `n_mask_trim_frames` |
| Không có | Mask gần trùng (IoU > 0,8) mà `coverage_tol` < 0,5 tính là occluded | Mask đang ở chân kia, không phải chân này |
| Crop có scale | Giữ nguyên độ phân giải | SAM2 đã tự resize |
| Gate theo near/far | Video không có nhãn dùng chân thấy rõ hơn thay near, chân còn lại thay far | C27, C04_3 không có near/far |

## Output

```
masks_post/<vid>.npz            bản dùng cho các bước sau: mask đã clip, vùng, trạng thái, các cờ
masks/<vid>.npz                 output gốc của SAM2
qc_footmask.csv                 1 dòng mỗi video, gate mới
qc_post/<vid>_qc.json           bản đầy đủ của dòng QC
qc_post/<vid>_conflicts.csv     đoạn xung đột ID ≥ 3 frame
qc_post/F10_2_view.json         đoạn quay lưng của F10_2
qc/<vid>_reprompt.csv           mọi lần re-prompt hoặc bỏ qua
qc/<vid>_qc.json                dòng QC gốc của SAM2, gồm cả gate cũ
review/<vid>.jpg                8 frame: 4 cách đều + tối đa 4 frame chồng lấn cao nhất
review/_conflict_<vid>_<k>.jpg  đầu, giữa, cuối đoạn xung đột, tối đa 5 ảnh mỗi video
footmask_review.csv             cột video_id, verdict, notes, để người kiểm tra điền
logs/                           log các lần chạy
```

Ảnh review cố ý ưu tiên frame chồng lấn nặng nhất, nên trông xấu hơn chất lượng trung bình của video. Màu: xanh lá = near, cam = far, xanh dương/tím = A/B khi không có nhãn near/far.

### `masks_post/<vid>.npz`

`masks/<vid>.npz` có cùng cấu trúc nhưng thiếu các cột hậu kỳ. N = số frame unique, không freeze. Chiều thứ 2 là object: chỉ số 0 = slot A, 1 = slot B của Bước 3.

| Khoá | Kích thước | Nội dung |
|---|---|---|
| `frame`, `t`, `continuity_id` | N | Chỉ số frame, thời gian thật (`t_real_s_v2`), khối |
| `W`, `H` | | Kích thước frame gốc |
| `rle_counts`, `rle_ptr` | M, N×2×4×2 | RLE của mask và 3 vùng, xem dưới |
| `state` | N×2 | 0 visible, 1 partial, 2 occluded |
| `label` | N×2 | `near`, `far`, hoặc `A`, `B` nếu khối không tin cậy |
| `area`, `area_ratio` | N×2 | Diện tích (px), tỉ lệ so với trung vị của object trong khối |
| `coverage`, `coverage_tol` | N×2 | Tỉ lệ điểm gót, ngón cái, ngón út trúng mask: đúng nghĩa đen / có dung sai |
| `origin`, `axis` | N×2×2 | Gốc (gót) và vector gót → giữa 2 ngón, toạ độ frame gốc |
| `axis_borrowed` | N×2 | Trục mượn từ frame hợp lệ gần nhất, cách ≤ 3 frame |
| `l_foot` | N×2 | L_foot của khối (px) |
| `region_area` | N×2×3 | Diện tích gót, giữa, mũi. NaN khi occluded hoặc không có trục |
| `iou` | N | IoU của 2 mask |
| `reprompt_flag`, `mask_trimmed`, `mask_id_conflict` | N×2 | Frame được prompt lại, đã bỏ mảnh rời, xung đột ID |
| `frame_valid_mask`, `invalid_reason` | N×2 | **Hậu kỳ.** Frame và chân dùng được; lý do dạng bit, xem mục Hậu kỳ |
| `area_ratio_norm`, `kp_mask_dist`, `perp_thickness` | N×2 | **Hậu kỳ.** Tỉ lệ diện tích đã chuẩn hoá theo L_leg(t)²; khoảng cách keypoint gần nhất tới mask; độ dày P95 − P5 (2 cột sau tính theo L_foot) |
| `view_lateral` | N | **Hậu kỳ.** False ở đoạn quay lưng của F10_2 |
| `exclude_time_features` | N | **Hậu kỳ.** True ở C20 part 1 |
| `ankle_clipped`, `ankle_clip_skipped` | N×2 | **Hậu kỳ.** Đã bị cắt pixel; không clip được |
| `crop_window`, `crop_scale` | N×4, 1 | x0, y0, x1, y1 của crop; scale luôn bằng 1 |
| `init_frame` | N | Frame khởi tạo của khối |
| `config`, `checkpoint`, `post_config` | | Config hash SAM2, tên checkpoint, hash hậu kỳ |

Mask đã ở toạ độ frame gốc. Giải mã:

```python
import numpy as np
z = np.load("step4_footmask/masks_post/C30.npz")
shape = (int(z["H"]), int(z["W"]))

def decode(a, o, kind):
    """a: vị trí frame, o: 0/1, kind: 0 mask, 1 gót, 2 giữa, 3 mũi."""
    s, e = z["rle_ptr"][a, o, kind]
    out, p = np.zeros(shape[0] * shape[1], bool), 0
    for i, r in enumerate(z["rle_counts"][s:e]):      # row-major, đoạn đầu là số 0
        if i % 2:
            out[p:p + int(r)] = True
        p += int(r)
    return out.reshape(shape)
```

Không nên dùng vùng của frame có `frame_valid_mask` = False. Frame occluded không được nội suy mask, và `region_area` là NaN. Frame xung đột ID vẫn có `region_area`, nên phải lọc bằng cờ này.

## Cột của `qc_footmask.csv`

Các cột diện tích, coverage, trạng thái, leak và vùng được tính trên mask sau clip. Hậu tố `_before` là giá trị trước clip.

| Cột | Ý nghĩa |
|---|---|
| `gate`, `gate_reasons` | Kết quả gate và lý do |
| `n_frames`, `n_blocks` | Số frame unique, số khối liên tục |
| `frac_{visible,partial,occluded}_{near,far,A,B}` | Tỉ lệ frame ở từng trạng thái theo nhãn; NaN nếu video không có nhãn đó |
| `frac_frame_valid`, `n_frames_conflict_invalid` | Tỉ lệ cặp (frame, object) có `frame_valid_mask` = True; số frame có xung đột ID |
| `frac_valid_{near,far,A,B}` | Tỉ lệ frame hợp lệ theo nhãn, sau các kiểm tra mới; NaN nếu video không có nhãn đó |
| `frac_removed_new`, `frac_removed_new_literal` | Tỉ lệ cặp bị loại thêm so với chỉ dùng occluded và xung đột ID: luật đã dùng; luật nguyên văn |
| `n_flag_area`, `n_flag_no_keypoint`, `n_flag_thick` | Số cặp có từng cờ, kể cả cặp đã False từ trước |
| `mean_coverage`, `mean_coverage_tol`, `mean_coverage_tol_before` | Coverage trung bình, không và có dung sai; có dung sai trước clip |
| `n_reprompt`, `n_reprompt_skipped` | Số lần prompt lại; số đoạn lỗi không prompt được vì 2 bàn chân đang chồng nhau |
| `max_reprompt_block` | Số lần prompt lại nhiều nhất trong 1 khối; chỉ để tham khảo |
| `reprompt_per100`, `reprompt_per100_max_block` | `n_reprompt` / `n_frames` × 100, gate dùng cột này; giá trị lớn nhất tính theo từng khối |
| `n_mask_trim_frames` | Số frame có ít nhất 1 mask bị bỏ mảnh rời |
| `frac_mask_leak`, `frac_mask_leak_before` | Tỉ lệ cặp (frame, object) có hơn 30% pixel mask nằm ngoài cả 3 vùng, sau và trước clip. Mẫu số là mọi cặp, kể cả occluded. Gate dùng giá trị sau clip |
| `n_clip_frames`, `n_clip_skipped`, `frac_area_clipped` | Số frame bị cắt pixel; số cặp (frame, object) không clip được; tỉ lệ diện tích mask bị bỏ |
| `n_id_conflict_frames`, `n_id_conflict_runs` | Số frame có mask nằm chủ yếu trên keypoint chân kia khi 2 chân cách nhau ≥ 0,5 L_foot, trước hoặc sau clip; số đoạn ≥ 3 frame |
| `median_mask_iou` | Trung vị IoU **chỉ trên frame có IoU > 0**, nên có thể cao dù rất ít frame chồng nhau (C38_9) |
| `frac_labelled_near_far` | Tỉ lệ frame có nhãn near/far |
| `frac_view_lateral`, `frac_exclude_time_features` | Tỉ lệ frame có `view_lateral` = True; có `exclude_time_features` = True |
| `frac_no_axis`, `frac_axis_short` | Tỉ lệ cặp không occluded không có trục; có trục ngắn hơn 0,7 L_foot |
| `frac_outside_regions`, `perp_extent_median` | Trung vị tỉ lệ pixel ngoài 3 vùng; trung vị độ rộng vuông góc với trục, theo L_foot |
| `region_area_frac_{heel,mid,toe}`, `region_area_frac_heel_before` | Tỉ phần diện tích từng vùng trên cả video; vùng gót trước clip |
| `device`, `checkpoint`, `config`, `post_config` | Thiết bị, checkpoint, config hash SAM2, hash hậu kỳ |
| `time_s`, `fps` | Thời gian chạy thực của video; tính cả lúc máy ngủ |

## Gate

| Mức | Điều kiện |
|---|---|
| FAIL | `n_id_conflict_frames` > 10% số frame, hoặc `frac_visible_near` < 0,6 |
| WARN | `reprompt_per100` ≥ 20, `frac_mask_leak` ≥ 0,1, hoặc `frac_visible_far` < 0,4 |
| PASS | Còn lại |

So với gate ban đầu:

- **Bỏ `max_reprompt_block` ≥ 3.** Sau khi bỏ giới hạn 5 re-prompt/khối, 42/42 WARN cũ đều chỉ vì lý do này, từ C23_1 (`frac_visible` 1,00/1,00) đến C13_2 (158 re-prompt).
- **Thêm `reprompt_per100` và `frac_mask_leak`.** Phân phối `reprompt_per100`: trung vị 4,09, P75 9,31, P90 18,26, P95 25,21, lớn nhất 37,42.
- **Giữ 2 điều kiện `frac_visible`** vì không có quyết định bỏ. Chúng hiện không kích hoạt ở video nào: thấp nhất là near 0,76 và far 0,75.
- **Video không có nhãn near/far** (C27, C04_3) dùng chân thấy rõ hơn thay near, chân còn lại thay far.
- **Bỏ `near_far_occlusion_agreement` và `near_far_agreement_area` khỏi mọi đánh giá.** Có trọng số theo frame, độ khớp chỉ 0,51 (logit) và 0,53 (diện tích) trên 526 frame của 33 video, tức gần như tung đồng xu. SAM2 suy luận từng object độc lập nên logit của 2 object không nói được chân nào ở trước. Các cột này không còn trong `qc_footmask.csv`; giá trị cũ vẫn ở `qc/<vid>_qc.json`.

Hai video nằm sát ngưỡng: F22_4 có `reprompt_per100` đúng 20,00 (WARN) và xung đột ID đúng 10,0% (không FAIL); C04_2 có `reprompt_per100` 19,57.

## Giới hạn đã biết

1. **PASS không có nghĩa mọi frame đều đúng, và `frame_valid_mask` = True cũng không đảm bảo mask đúng.** 9 trong 23 frame lỗi mà review liệt kê vẫn có cờ True. Gate tính theo video và không dùng 3 kiểm tra mới.
2. **Clip chỉ cắt phần ống chân của chính chân đó.** Mask vẫn có thể lan sang chân kia (F17 f240: mask far phủ một phần ống chân near), và phần ngay trên cổ chân trong lề 0,15 L_foot vẫn được giữ.
3. **Clip không giảm nhiều vùng gót.** `region_area_frac_heel` ở các video lan nhiều chỉ giảm 0,01–0,03 (C13_1 từ 0,479 còn 0,453; C09_1 từ 0,480 còn 0,466). Phần còn lại không chủ yếu do ống chân: vùng gót rộng 0,48 L_foot, vùng giữa chỉ rộng 0,33 L_foot, và phần cổ giày nằm trong vùng gót. So tỉ phần vùng gót giữa các video có `frac_mask_leak_before` cao và thấp vẫn cần thận trọng.
4. **`frac_mask_leak` còn cao ở C04_3 (0,24), C01 (0,17), C04_2 (0,11)** vì nguyên nhân không phải ống chân: mờ chuyển động làm mask và keypoint lệch nhau. Clip không xử lý được loại này.
5. **Trục ngắn làm clip yếu đi.** Bàn chân nhìn thẳng từ mũi hoặc gót có trục không đáng tin nên không được clip. F10_2 còn `frac_axis_short` khoảng 0,2.
6. **Keypoint Bước 3 có frame dồn cả 2 chân vào 1 bàn chân.** Thấy ở F22_4 (f15, f47) và C04_3 (f14, f33, f87): bàn chân kia nhìn thấy được nhưng không có mask. Bước 4 không tự sửa được.
7. **Lúc 2 bàn chân chồng nhau, hai mask thường dính lại.** Mask sai ở đây được đánh dấu partial hoặc occluded, chứ không được sửa. Quy tắc bỏ mảnh rời và re-prompt cũng không chạy trong lúc chồng nhau.
8. **Trạng thái của frame phụ thuộc trung vị diện tích sau clip.** Trung vị đã bớt lệch (F17: `frac_visible_far` từ 0,67 lên 0,81), nhưng vẫn là so sánh tương đối trong từng khối.
9. **`s4_validate.py` chỉ kiểm tra `masks/`** và kiểm RLE trên tối đa 12 frame mỗi video. `s4_postprocess.py --check` kiểm mọi frame của `masks_post/`. Cả hai chỉ xác nhận output nhất quán, không xác nhận mask đúng về hình ảnh.

## Mã nguồn

| File | Vai trò |
|---|---|
| `code/run_step4.py` | Chạy pipeline, gộp khối, tính QC |
| `code/s4_common.py` | Hằng số, nạp dữ liệu, crop, RLE |
| `code/s4_sam.py` | Prompt, lan truyền, re-prompt, bỏ mảnh rời |
| `code/s4_regions.py` | Trạng thái, trục, chia vùng, xung đột ID, nhãn |
| `code/s4_report.py` | Gate, ảnh review |
| `code/s4_postprocess.py` | Ankle clip, gate mới, `frame_valid_mask`, sửa nhãn F10_2, cờ thời gian, kiểm tra |
| `code/s4_validate.py` | Kiểm tra `masks/` |
| `code/regen_review.py` | Vẽ lại ảnh review từ `masks_post/` |

`footkp.npz` của Bước 3 lưu `time_model` dạng object nên phải đọc với `allow_pickle=True`; `load_clip` đã xử lý việc này.
