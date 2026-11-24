# Step 6 — RFF-GP-HSMM bốn pha

Luồng chính cho Overground dùng `scripts/evaluate_cyclic_overground.py` để fit và xuất khoảng IC/TO, và `scripts/render_overground_review.py --intervals` để dựng video đối chứng. Lệnh tiền xử lý và ví dụ đầy đủ nằm trong [README gốc](../../../README.md).

| Module | Vai trò |
|---|---|
| `event_data.py` | Chọn một slot chân, kiểm tra trục frame và tách đoạn quan sát liên tục |
| `hsmm_inputs.py` | Đọc flow theo pixel, tạo embedding vùng và whitening có điều chuẩn |
| `rff_hsmm.py` | GP emissions, thời lượng, Viterbi và forward–backward |
| `cyclic_hsmm.py` | Chu kỳ IC → Flat → TO → Swing, prior thời lượng và khoảng posterior của biên |
| `mask_overlay.py` | Giải nén mask Step 4, tô màu vùng và căn khung hình |

## Đầu vào và tham số

- Dùng foot mask Step 4 và pixel flow được bù chuyển động ở Step 5. Chọn near-slot đầu lượt, giữ slot trong cả lượt. Mỗi vùng cần ít nhất 10 pixel qua kiểm tra flow thuận–nghịch.
- Embedding lấy trung bình RFF của `[xi, eta, signedlog(vx), signedlog(vy)]` trong mỗi vùng, ghép thành 96 hoặc 192 chiều. Tọa độ `xi/eta` được chuẩn hóa theo trục bàn chân. Vận tốc đã bù có đơn vị chiều dài chân mỗi giây.
- `--embedding-per-region 32 64` chọn số chiều theo vùng. `--durations` nhận JSON ánh xạ mỗi tên pha tới `[min, median, max]` tính bằng giây. Preset theo điều kiện nằm trong `cyclic_hsmm.py`. `--evidence-seconds` mặc định là 0,10 s cho mỗi quan sát flow hiệu dụng.
- Nếu thiếu cache GMA, chạy lại Bước 5 với `--store-only`. Xem [đặc tả](../../../docs/STEP6_SPEC.md) để diễn giải khoảng pha và khoảng mốc IC/TO.

Kiểm thử từ thư mục gốc bằng `py -3.11 -m unittest discover -s tests -v`.
