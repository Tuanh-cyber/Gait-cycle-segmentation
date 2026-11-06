# Bước 1b: k(t) cho slow-mo dạng ramp

Một số video đổi tốc độ ngay trong một speed part (speed ramp kiểu TikTok), trong khi Bước 1 gán một `k` hằng, nên `t_real_s` sai ngay trong part. Bước này ước lượng `k(t)` cho các part đó.

```bash
../.venv-track/bin/python step1_wu/code/ramp_time.py --dry-run   # chỉ tính và in
../.venv-track/bin/python step1_wu/code/ramp_time.py             # sao lưu rồi ghi
```

## Cảnh báo phương pháp

**Giả định thời lượng bước thực là hằng trong clip, nên biến thiên cadence thật trong part ramp bị xoá.** Đặc trưng cadence và thời gian tuyệt đối chỉ dùng video `abs_time_ok = True`. Với video ramp, `t_real_s_v2` chỉ đúng về tỉ lệ tương đối giữa các pha.

**CV < 0.15 gần như đúng theo cách dựng,** vì `k_i` được chọn để mọi bước dài bằng nhau. Nó chỉ cho biết phép nội suy không làm sai lệch, không chứng minh ramp là thật. Cột `corr_k_image_rate` mới là kiểm chứng độc lập: tương quan giữa `k` và cường độ đổi ảnh `diff_prev` theo giây video, không dùng keypoint.

## Cách làm

1. **Tín hiệu:** x cổ chân near − far (× L_leg) từ `step3_footkp/footkp_v1/`, trên trục `pts_s`.
2. **Bước:** mỗi khoảng giữa 2 lần chéo chân, có bộ kích Schmitt (15% biên độ p95) để bỏ nhiễu quanh 0, nội suy thời điểm chéo dưới mức frame. Không có bước nào vắt qua khoảng trống > 3 frame.
3. **Tiêu chí ramp:** `max(d)/min(d)` > 1.8 với ≥ 4 bước.
4. **T_step:** median `d` của phần `speed_k = 1`, `k_confidence = cao` nếu có (`realtime_ref`). Nếu không có thì median các bước `d` ≤ 1.3 × d nhỏ nhất (`fastest_steps`), và đặt `abs_time_ok = False`.
5. **k_i = T_step / d_i:** chặn tối đa 1, snap vào {1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/8} nếu trong 15%.
6. **Nội suy:** PCHIP theo `pts_s`.
7. **t_real_s_v2:** cùng công thức cộng dồn của `write_time.py`; video không có ramp cho đúng `t_real_s` cũ (script tự kiểm tra và dừng nếu lệch).

## Khác với đề bài

| Thay đổi | Lý do |
|---|---|
| Mỗi bước có 2 mốc PCHIP cùng giá trị `k_i` (15% và 85% thời lượng bước), không phải 1 mốc giữa bước | Mốc giữa bước làm chuyển tiếp trải dài hết bước dài, tích phân lệch khỏi T_step: 7/12 part không đạt CV < 0.15. Với 2 mốc, cả 12 đạt |
| Bước `d` < 0.25 s bị loại khỏi mốc và khỏi tính CV | Trên 240 bước/phút là không thể nếu video không tua nhanh; đây là chéo giả do nhiễu (F26). Nếu không loại, T_step của F26 thành 0.23 s thay vì 0.32 s |
| Chỉ hạ `abs_time_ok`, không nâng | Đề bài chỉ quy định chiều hạ; hiện không có part ramp nào có mốc thật |

## Giới hạn

- **Không kiểm được:** 33/85 part có < 4 bước. C20 part 1 có tỉ số 3.78 nhưng chỉ 2 bước, gần như chắc là ramp, vẫn giữ `k` cũ. 9 part có < 2 bước: C11/1, C12/1, C19/1, C21/1, F10_2/1, F11/1, F11/2, F11/3, F24/1 (`ramp_tested_parts.csv`).
- **C27 chưa được xác nhận:** `corr_k_image_rate` = 0.05 trong khi 11 part còn lại ≥ 0.79. Ảnh có chữ và icon động, và nhãn chân ở đoạn cuối clip không chắc chắn.
- **Không có mốc thật:** cả 12 part đều dùng `fastest_steps`. F20, C24, F27 cho cadence 127–138 bước/phút, dưới dải 140–215, nên đoạn nhanh nhất có thể vẫn là slow-mo.
- **Chạy lại Bước 1 gốc:** `write_time.py` ghi lại frames CSV nhưng không xoá cột mới; khi đó cần chạy lại `ramp_time.py` để cột mới khớp.

## File

- `ramp_parts.csv`: part ramp, gồm `cv_old`, `cv_v2`, `corr_k_image_rate`.
- `ramp_tested_parts.csv`: toàn bộ 85 part đã kiểm.
- `plots_ramp/<vid>.png`: thời lượng bước và `k` trước/sau.
- `step0_audit/frames_backup_20261104/`, `step0_audit/step0_metadata_backup_20261104.csv/.xlsx`: bản trước khi ghi.
