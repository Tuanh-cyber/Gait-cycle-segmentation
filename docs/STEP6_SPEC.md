# Đặc tả Step 6 — bốn pha tuần hoàn

## Dữ liệu

Một slot bàn chân được chọn lúc đầu và giữ trong cả lượt. Step 4 chia mask thành heel, mid và toe. Step 5 cung cấp optical flow GMA, bù chuyển động và thời gian từ AVI. Mỗi vùng yêu cầu ít nhất 10 pixel hợp lệ. Mô hình xử lý từng đoạn flow liên tục.

Hai nhánh nhận cùng pixel hợp lệ. Embedding lấy trung bình RFF mô tả phân bố chung của vị trí và vận tốc trong mỗi vùng. Ghép 32 hoặc 64 giá trị mỗi vùng cho vector 96 hoặc 192 chiều. Đặc trưng được chuẩn hóa và whitening có điều chuẩn.

## Mô hình và thời gian

- Thứ tự duy nhất là **IC → Flat → TO → Swing → IC**. IC diễn giải giai đoạn tiếp đất, Flat diễn giải giai đoạn vùng giữa bàn chân duy trì tiếp xúc, TO diễn giải giai đoạn mũi nhấc và rời sàn, Swing diễn giải giai đoạn chân ở trên không.
- Các pha được ước lượng từ chuyển động vùng, embedding và prior thời lượng trong `cyclic_hsmm.py`. Min/max áp dụng cho pha quan sát trọn vẹn. Pha ở đầu hoặc cuối đoạn flow có thể bị cắt cụt.
- Mốc IC là biên Swing → IC. Mốc TO là biên TO → Swing. Khoảng biên chứa 90% posterior có điều kiện quanh lần chuyển pha ở chu kỳ đó. Posterior chưa hiệu chuẩn theo ground truth.

## Đầu ra

| Tệp | Nội dung |
|---|---|
| `phase_intervals.csv` | Khoảng frame của từng pha và cờ pha bị cắt cụt |
| `event_intervals.csv` | Khoảng frame của pha IC/TO và khoảng posterior cho mốc chạm/rời sàn |
| `event_confidence_frames.csv` | Xác suất pha và biên trên các frame nguồn có bằng chứng flow |
| `interval_metrics.csv` | Số mốc CSV nằm trong khoảng dự đoán, ghép một-một cùng loại và bên chân |
| `metrics.csv` | Chẩn đoán mốc đại diện tại các dung sai thời gian |
| `phase_intervals.png` | Biểu đồ pha, khoảng biên và ground truth |
| `*_four_phase_review.mp4` | Đối chứng mask và các khoảng IC/TO trên AVI gốc |

[Hướng dẫn chạy](../README.md#chạy-trên-video-overground) · [Các module Step 6](../data/step6_fixed_foot_events/code/README.md)
