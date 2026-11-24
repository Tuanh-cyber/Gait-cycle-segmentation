# Step 6 — HSMM bốn pha với tập pixel

Runner Overground dùng scripts/evaluate_cyclic_overground.py để fit mô hình và xuất khoảng IC/TO. Dùng scripts/render_overground_review.py --intervals để dựng video với dải bất định của toàn pha IC/TO. Các lệnh đầy đủ nằm trong [README gốc](../../../README.md).

| Module | Vai trò |
|---|---|
| event_data.py | Chọn slot cố định, kiểm tra trục frame và tách đoạn quan sát liên tục |
| hsmm_inputs.py | Đọc optical flow theo pixel và gắn tọa độ trong mask |
| pixel_emission.py | Chấm tập pixel theo phân bố vận tốc có điều kiện |
| cyclic_hsmm.py | Chu kỳ IC → Flat → TO → Swing và học thời lượng pha |
| hsmm.py | Viterbi và forward–backward |
| mask_overlay.py | Giải nén và tô màu mask Step 4 |

Dùng foot mask Step 4 và pixel flow đã bù ở Step 5. Mỗi vùng cần ít nhất 10 pixel qua kiểm tra flow thuận–nghịch. Giá trị vận tốc và vị trí của tất cả pixel hợp lệ tham gia vào điểm emission. Xem [đặc tả](../../../docs/STEP6_SPEC.md) để biết công thức và định dạng đầu ra.
