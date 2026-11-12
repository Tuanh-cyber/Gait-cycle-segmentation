# Ba sample: C13_1, C38_8 và F10_2

| Video | Slot cố định | Fréchet IC / TO | HSMM IC / TO | File |
|---|---|---:|---:|---|
| C13_1 | 0 / A | 7 / 12 | 18 / 16 | [MP4](videos/C13_1.mp4) · [flow/pha](plots/C13_1.png) · [events](events/C13_1.csv) |
| C38_8 | 1 / B | 0 / 0 | 0 / 0 | [MP4](videos/C38_8.mp4) · [flow/pha](plots/C38_8.png) · [events](events/C38_8.csv) |
| F10_2 | 0 / A | 3 / 1 | 3 / 2 | [MP4](videos/F10_2.mp4) · [flow/pha](plots/F10_2.png) · [events](events/F10_2.csv) |

C13 có hai file nguồn C13_1 và C13_2; bộ sample này chọn C13_1. Các sự kiện trong bảng là **candidate**, chưa xác nhận bằng ground truth. Số IC và TO có thể lệch vì quan sát thiếu/ngắt đoạn hoặc một phía ranh giới còn UNKNOWN.

## Cách xem

- Thanh trên: Fréchet CPD. Thanh dưới: RFF-GP-HSMM.
- Pha: xanh lá = stance-like; xanh dương = swing-like; xám = UNKNOWN/không đủ dữ liệu.
- Mốc IC màu trắng, TO màu vàng. Vạch trắng dài là thời gian nguồn đang phát.
- **Mask thật Bước 4**: gót tím, giữa vàng cam, mũi cyan; viền trắng là hợp các vùng được hiển thị. Màu mask là vùng giải phẫu, khác màu của phase.
- Không dựng mask bằng đa giác keypoint. Keypoint chỉ hỗ trợ chọn vùng zoom nếu mask không có; không vẽ đa giác đó thành mask thật.
- Chỉ overlay mask đúng frame unique, hoặc frame trùng cùng `unique_id`; không giữ mask cũ qua frame khác, freeze hoặc vùng không có quan sát tương ứng. Có mask trên hình không đồng nghĩa có flow đủ hợp lệ để suy phase; thanh pha vẫn có thể xám.

## Phương pháp và phạm vi

Dùng trực tiếp feature GMA theo vùng trong Bước 5, chọn near-slot lúc đầu rồi giữ nguyên slot A/B cho cả video. Bỏ qua hoàn toàn tracker `step5_single_foot`. Tách suy luận tại ranh giới continuity/shot/speed part và tại khoảng thiếu dữ liệu; không ghép IC/TO qua gap.

HSMM là **nguyên model của bản C30**: 2 latent states, 12 RFF, phase-switch penalty 0, đã fit unsupervised trên C30/F29/F02/C38_9. **Không fit lại trên C13_1, C38_8 hoặc F10_2**. SHA256 file model nguồn và model dùng trong bộ này giống nhau. Mapping của model giữ nguyên: class 0 stance-like, class 1 swing-like; đây là diễn giải theo slow-ratio, không phải nhãn ground truth.

Fréchet giữ cùng cấu hình C30: local window 0.6 giây trên trục thời gian Bước 5, 99 bootstrap, alpha 0.05, min-duration 0.08 giây. P-value local là bằng chứng thay đổi feature; bootstrap mặc định độc lập chưa hiệu chuẩn đầy đủ cho phụ thuộc thời gian hoặc nhiều phép kiểm định.

## Lưu ý từng video

- **C13_1:** 516/662 quan sát đầu vào hợp lệ. Hai phương pháp khác nhau đáng kể ở các đoạn ngắn và chỗ gián đoạn; cần đối chiếu tiếp xúc trong video, không chọn phương pháp chỉ theo tổng số sự kiện.
- **C38_8:** chỉ 43/84 quan sát đầu vào hợp lệ, bị chia thành nhiều đoạn ngắn. Fréchet có một boundary mơ hồ; HSMM có hai boundary nội swing. Không boundary nào đủ điều kiện thành IC/TO, dù video rõ ràng có chuyển động chạy. Kết quả 0 thể hiện detector chưa xác định được sự kiện trên đầu vào này, không phải không có contact thực.
- **F10_2:** 67/113 quan sát đầu vào hợp lệ; có nhiều vùng UNKNOWN. `abs_time_ok=False`: thời gian sinh học tuyệt đối từ trục Bước 5 chưa chắc chắn. Frame nguồn và source PTS của video xuất vẫn được bảo toàn; ưu tiên kiểm tra event theo frame.

## Kiểm tra xuất video

| Video | Frame nguồn = frame xuất | Sai lệch PTS lớn nhất | Frame có overlay mask |
|---|---:|---:|---:|
| C13_1 | 672 | < 0.000001 giây | 669 |
| C38_8 | 90 | < 0.000001 giây | 82 |
| F10_2 | 1.079 | < 0.000001 giây | 299 |

Đã giải mã toàn bộ ba MP4 bằng ffmpeg với `-xerror`, không gặp lỗi. Mỗi video có provenance, hash đầu vào và verification trong `records/<video_id>.json`; cấu hình model đóng băng nằm trong [manifest.json](manifest.json). Tỷ lệ mask hiển thị tính trên frame nguồn; tỷ lệ input hợp lệ tính trên danh sách quan sát feature, không phải cùng mẫu số. Video không kèm audio.

## Tái tạo

```powershell
py -3.11 -m pip install -r data/step6_fixed_foot_events/code/requirements.txt
py -3.11 data/step6_fixed_foot_events/code/run_events.py --videos C13_1 C38_8 F10_2 --classes 2 --phase-switch-penalty 0 --model-from data/step6_fixed_foot_events/three_samples --out data/step6_fixed_foot_events/reproduced_samples
```

Model đóng băng đã có ngay trong [model.npz](model.npz); không cần cache `_trial_two` hoặc thư mục C30 preview. Lệnh trên ghi vào thư mục khác để giữ nguyên ba sample đã xuất. Cần `ffmpeg` và `ffprobe` trên PATH; xem [hướng dẫn code](../code/README.md).

Trước khi xuất đã pass 7 test trong workspace phát triển, gồm kiểm tra thống kê Fréchet, HSMM/posterior so với duyệt toàn bộ đường phân đoạn nhỏ, chặn event qua gap và giải mã region RLE đúng canvas/crop. Bộ test ở thư mục gốc `tests/` không thuộc phạm vi hai thư mục sample/code này.
