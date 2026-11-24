# Step 6 — Suy luận pha và khoảng IC/TO bằng pixel-set HSMM

Step 6 nhận optical flow đã tính ở Step 5 cho **một bàn chân cố định trong mỗi lượt**, rồi ước lượng chuỗi pha **IC → Flat → TO → Swing → IC**. Mỗi quan sát là ba tập pixel của cùng một cặp frame, tương ứng với gót, giữa bàn chân và mũi. Mô hình học emission và thời lượng từ video đang xét. Sau khi fit, mô hình suy luận cả qua các khoảng thiếu flow nằm trong đoạn tracking liên tục. Đầu ra gồm chuỗi pha đại diện, posterior theo frame, khoảng khả tín của toàn pha IC/TO và khoảng khả tín của thời điểm chuyển pha.

## 1. Input và đơn vị quan sát

| Dữ liệu | Step 6 sử dụng |
|---|---|
| Video AVI, FPS, `passages.csv` | Chỉ số frame gốc, thời gian và phạm vi của từng lượt. BD004 dùng 50 fps, tương ứng 20 ms mỗi frame nguồn. |
| Slot bàn chân và keypoint từ Step 3, mask từ Step 4 | Chọn slot gắn nhãn gần camera trong tối đa 10 quan sát đầu lượt và giữ slot đó. Nếu không có nhãn gần camera trong cửa sổ này, mã dùng slot A và ghi cờ fallback. Mask cung cấp vùng `heel`, `mid`, `toe`, gốc tọa độ, trục và chiều dài bàn chân. |
| Cache flow GMA và thông số Step 5 | Flow thuận, cờ nhất quán thuận–nghịch, crop, tỷ lệ resize, thời gian giữa hai ảnh, chiều dài chân tham chiếu và các thành phần bù chuyển động. |
| CSV ground truth | Runner dùng timestamp IC/TO theo lượt và bên chân sau khi suy luận để chấm điểm và dựng video đối chứng. |

[Mã đọc đầu vào](../data/step6_fixed_foot_events/code/event_data.py) · [Mã ghép các lượt trên trục AVI](../scripts/evaluate_overground.py)

### 1.1. Một observation lấy từ hai frame

GMA dùng ảnh $I_t$ và $I_{t+1}$ để ước lượng độ dịch chuyển $f_i=(f_{x,i},f_{y,i})$ của pixel $i$ từ vị trí $q_i$ ở ảnh đầu. Vị trí $q_i$ và điểm đến $q_i+f_i$ nằm trên cùng lưới flow. Step 6 lấy mask ở $I_t$ để gán pixel cho gót, giữa bàn chân hoặc mũi, rồi gắn toàn bộ quan sát vào **frame nguồn $t$**. Ví dụ, cặp 390–391 trong video 50 fps cách nhau 0,02 s và tạo observation ở frame 390. Frame cuối của một đoạn không có cặp tiếp theo hợp lệ sẽ thiếu observation.

Cặp chỉ được dùng khi hai ảnh có mask hợp lệ của cùng slot, có góc nhìn ngang, cùng lượt, continuity, shot và speed part, có trục thời gian tăng, và có flow đạt các cờ kiểm tra của Step 5. Ở mức pixel, điểm đến phải nằm trong ảnh và sai số khứ hồi phải thỏa

$$
\left\|f_i+b(q_i+f_i)\right\|
\le 1+0{,}05\|f_i\|.
$$

$b$ là flow nghịch nội suy tại điểm đến. Ngưỡng tính theo pixel trên lưới flow. Sau khi lọc thêm các giá trị vận tốc không hữu hạn, **mỗi vùng phải còn ít nhất 10 pixel**. Chỉ một vùng thiếu pixel cũng đủ để đánh dấu toàn bộ observation thiếu. Mask có thể vẫn được vẽ trên video vì điều kiện lấy mask và điều kiện chấp nhận flow là hai phép kiểm tra khác nhau. [Kiểm tra thuận–nghịch](../data/step5_flow/code/s5_flow.py) · [Tạo tập pixel theo vùng](../data/step6_fixed_foot_events/code/hsmm_inputs.py)

### 1.2. Vector của từng pixel

Một pixel hợp lệ được biểu diễn bằng $u_i=[\xi_i,\eta_i,v_{x,i},v_{y,i}]$. Gọi $p_i$ là vị trí của pixel sau khi quy về ảnh gốc, $o_t$ là gốc mask của bàn chân, $L_\mathrm{foot}$ là chiều dài bàn chân trên ảnh, $a$ là vector đơn vị hướng gót → mũi và $a_\perp=(-a_y,a_x)$.

| Thành phần | Cách tính và ý nghĩa |
|---|---|
| $\xi_i=(p_i-o_t)\cdot a/L_\mathrm{foot}$ | Vị trí dọc bàn chân. Nếu trục hướng sang phải, $L_\mathrm{foot}=40$ px và pixel cách gốc 20 px sang phải thì $\xi_i=0{,}5$. |
| $\eta_i=(p_i-o_t)\cdot a_\perp/L_\mathrm{foot}$ | Vị trí vuông góc với trục bàn chân. Trong ví dụ trên, pixel nằm cao hơn gốc 4 px thì $\eta_i=-0{,}1$. |
| $v_{x,i}$ | Vận tốc ngang ảnh sau bù chuyển động, đã thống nhất dấu theo hướng di chuyển của lượt. |
| $v_{y,i}$ | Vận tốc dọc ảnh sau bù chuyển động. Dấu dương hướng xuống ảnh, dấu âm hướng lên. |

Gọi $s$ là tỷ lệ resize sang lưới flow, $(c_x,c_y)$ là dịch chuyển camera trên lưới đó, $L_\mathrm{leg}$ là chiều dài chân tham chiếu trên ảnh, $\Delta t$ là thời gian giữa hai ảnh và $\sigma_x$ là hệ số dấu ngang. `comp_slow` là hiệu chỉnh vận tốc ngang thêm của Step 5. Công thức vận tốc là

$$
v_{x,i}=\frac{\sigma_x(f_{x,i}-c_x)}{sL_\mathrm{leg}\Delta t}-\mathrm{comp\_slow},
\qquad
v_{y,i}=\frac{f_{y,i}-c_y}{sL_\mathrm{leg}\Delta t}.
$$

Hai vận tốc có đơn vị **chiều dài chân mỗi giây**. Chẳng hạn, với $s=1$, $L_\mathrm{leg}=100$ px, $\Delta t=0{,}02$ s và độ dịch ngang sau bù là 3 px, ta có $v_x=3/(100\times 0{,}02)=1{,}5$ nếu $\sigma_x=1$ và `comp_slow` bằng 0. Tọa độ $\xi,\eta$ theo trục bàn chân, còn vận tốc theo trục ảnh. [Mã dựng vector pixel](../data/step6_fixed_foot_events/code/hsmm_inputs.py)

## 2. Cách một observation đi vào HSMM

### 2.1. Ba tập điểm tạo điểm emission cho một frame

Observation ở frame $t$ là $O_t=(U_{t,\mathrm{heel}},U_{t,\mathrm{mid}},U_{t,\mathrm{toe}})$. Mỗi $U_{t,r}$ có kích thước $N_{t,r}\times4$. Ví dụ, nếu gót có 200 pixel hợp lệ, toàn bộ 200 hàng đi vào phép tính điểm vùng gót. 200 pixel này đóng góp vào **cùng một bước thời gian** của HSMM.

HSMM xét từng đoạn pha ứng viên $[a,b)$, với frame đầu $a$, frame cuối $b-1$ và pha $k$. Tại frame $t$ trong đoạn, đặt $\tau_t=T_t-T_a$, tính bằng giây kể từ đầu đoạn ứng viên. Với pixel $i$ thuộc vùng $r$, emission chấm hai thành phần vận tốc quan sát được bằng Gaussian hai chiều có kỳ vọng phụ thuộc vào vị trí và $\tau_t$.

$$
(v_{x,i},v_{y,i})
\mid \xi_i,\eta_i,\tau_t,k,r
\sim
\mathcal N\!\left([1,\xi_i,\eta_i,\tau_t]B_{k,r},\Sigma_{k,r}\right).
$$

$B_{k,r}$ có kích thước $4\times2$, còn $\Sigma_{k,r}$ có kích thước $2\times2$ và cho phép sai số ngang, dọc tương quan. Kỳ vọng Gaussian là vận tốc điển hình mà pha ứng viên gán cho pixel tại vị trí và thời gian đó. Mô hình so vận tốc đã đo với kỳ vọng này. Cùng observation ở frame 395 có thể được chấm với $\tau_t=0{,}10$ s nếu pha bắt đầu ở 390, hoặc $\tau_t=0{,}04$ s nếu pha bắt đầu ở 393, tại 50 fps.

$$
S_t(k,\tau_t)=
\sum_{r\in\{\mathrm{heel,mid,toe}\}}
\frac{1}{N_{t,r}}\sum_{i\in U_{t,r}}
\log p_{k,r}\!\left(v_{x,i},v_{y,i}\mid \xi_i,\eta_i,\tau_t\right).
$$

Emission lấy **trung bình log mật độ trong mỗi vùng** rồi cộng ba vùng. Vì vậy, một vùng 200 pixel có cùng trọng số vùng với một vùng 80 pixel. Trong mã, với $w_i=[1,\xi_i,\eta_i,v_{x,i},v_{y,i}]^\top$, mỗi vùng được tính ma trận moment $M_{t,r}=\frac{1}{N_{t,r}}\sum_i w_iw_i^\top$. Gaussian với kỳ vọng tuyến tính chỉ cần các tổng tích bậc hai này để tính đúng $S_t$. Ma trận là dạng tính toán chính xác của điểm cho emission hiện tại. Kết quả giữ ảnh hưởng của mọi pixel vào điểm và bất biến với thứ tự liệt kê pixel. Phần covariance được điều chuẩn bằng một quan sát giả từ covariance gộp theo vùng và một mức sàn số học. [Mã tính moment, fit và chấm emission](../data/step6_fixed_foot_events/code/pixel_emission.py)

### 2.2. Bốn pha và điểm gợi ý

| Pha | Ý nghĩa trong gait cycle | Tín hiệu ưu tiên |
|---|---|---|
| **IC** | Giai đoạn tiếp đất từ lúc gót chạm sàn đến khi bàn chân tiếp đất hoàn toàn. | Gót chậm hơn mũi và có thành phần đi xuống. |
| **Flat** | Giai đoạn vùng giữa bàn chân duy trì tiếp xúc với sàn. | Vùng giữa chậm, gót và mũi tương đối ổn định. |
| **TO** | Giai đoạn từ lúc mũi bắt đầu nhấc đến lúc bàn chân rời sàn. | Mũi chậm tương đối so với gót và có thành phần đi lên. |
| **Swing** | Giai đoạn bàn chân ở trên không. | Tỷ lệ pixel chậm thấp khi xét cả ba vùng. |

Điểm gợi ý $C_t(k)$ dùng tỷ lệ pixel chậm và median vận tốc dọc. Với mỗi vùng, tỷ lệ chậm là trung bình của hai tỷ lệ $\|v_i\|<1$ và $\|v_i\|<2$, theo đơn vị chiều dài chân mỗi giây. Vận tốc dọc được đổi thang bằng $\tanh(v_y^{\mathrm{median}}/s_v)$, với $s_v=\max(0{,}35,Q_{0.8}(|v_y^{\mathrm{median}}|))$ tính trên các observation hợp lệ của video và cả ba vùng. Các hệ số tổ hợp tín hiệu trong `flow_evidence` được đặt trong mã. Chúng giúp gắn ý nghĩa IC, Flat, TO, Swing cho bốn trạng thái khi khởi tạo và tiếp tục góp điểm khi suy luận. Mô hình học phân đoạn từ flow của video với các gợi ý này và thứ tự pha đã định trước. [Công thức điểm gợi ý theo pha](../data/step6_fixed_foot_events/code/cyclic_hsmm.py)

## 3. Điểm đoạn, thời lượng và các vòng fit

### 3.1. Điểm của một đoạn pha

HSMM cho phép pha đầu của mỗi đoạn suy luận là một trong bốn pha với điểm khởi đầu bằng nhau. Khi kết thúc một pha, chỉ được sang pha kế tiếp trong chu kỳ. Việc ở cùng pha qua nhiều frame được biểu diễn bằng **thời lượng đoạn**. Với $G$ là tập frame có flow usable, điểm đoạn $[a,b)$ của pha $k$ sau khi fit là

$$
\mathrm{score}(a,b,k)
=
\log p_{\mathrm{dur}}(b-a\mid k,\text{vị trí đoạn})
+\sum_{t\in[a,b)\cap G}\big[S_t(k,T_t-T_a)+C_t(k)\big].
$$

Mỗi observation hợp lệ có trọng số 1. Viterbi tối đa hóa tổng điểm của cả chuỗi phân đoạn, có xét thứ tự chuyển pha. Forward–backward cộng trọng số của tất cả chuỗi hợp lệ để tính posterior. Vì hai phép tính trả lời hai câu hỏi khác nhau, pha trên đường Viterbi ở một frame có thể khác pha có posterior biên lớn nhất ở frame đó. [Mã quy hoạch động](../data/step6_fixed_foot_events/code/hsmm.py)

### 3.2. Phân bố thời lượng học từ video

Với pha $k$, mô hình học xác suất rời rạc $p_k(d)$ của độ dài $d$ frame. Miền hỗ trợ kỹ thuật chạy từ 1 đến độ dài dải fit dài nhất trong video. Gọi $c_k(d)$ là số đếm pha trọn vẹn và $q(d)$ là histogram gộp qua bốn pha, chuẩn hóa sau khi thêm tổng trọng số một lượt phân đều trên miền độ dài. Mỗi pha dùng một quan sát giả từ $q$ để làm trơn.

$$
p_k(d)=\frac{c_k(d)+q(d)}{\sum_j c_k(j)+1}.
$$

Đoạn pha tiếp xúc mép dải có thể chỉ quan sát được một phần. Vì vậy, histogram chỉ nhận pha có đủ hai biên bên trong dải fit. Khi chấm ứng viên, đoạn pha nằm giữa dải dùng $p_k(d)$, đoạn ở mép trái dùng phân bố thời gian còn lại, đoạn ở mép phải dùng xác suất kéo dài ít nhất $d$ frame. Đoạn chạm cả hai mép dùng xác suất sống sót của thời gian còn lại. Cách này học thời lượng riêng cho video đang chạy, với cùng quy tắc cho các điều kiện walk. [Mã học và chấm thời lượng](../data/step6_fixed_foot_events/code/cyclic_hsmm.py)

### 3.3. Từ khởi tạo đến giải mã cuối

Các lượt trong một AVI dùng chung tham số. Trong từng lượt, fit dùng dải các observation usable liền nhau. Mặc định runner đặt `--iterations 4`.

1. **Khởi tạo.** Chạy Viterbi trên các dải fit với điểm gợi ý $C_t(k)$ và điểm thời lượng bằng nhau cho mọi đoạn ứng viên. Đếm trực tiếp thời lượng của các pha trọn vẹn trên đường khởi tạo.
2. **Fit tham số mỗi vòng.** Cập nhật histogram thời lượng từ số đếm đang có. Dùng pha và thời điểm bắt đầu trên đường Viterbi hiện tại để fit $B_{k,r}$ và $\Sigma_{k,r}$. Mỗi frame đóng góp các moment pixel theo vùng với $\tau_t$ xác định từ đường này.
3. **Cập nhật phân đoạn giữa hai vòng fit.** Chấm lại bằng emission mới, điểm gợi ý và thời lượng vừa học. Viterbi cung cấp đường mới cho lần fit emission tiếp theo. Forward–backward cung cấp **số đếm kỳ vọng** của từng độ dài cho lần fit thời lượng kế tiếp, bằng cách cộng xác suất hậu nghiệm của các đoạn ứng viên có đủ hai biên. Mặc định có ba lần cập nhật trung gian.
4. **Suy luận cuối.** Sau vòng fit cuối, giữ tham số cố định và chạy Viterbi cùng forward–backward trên trục frame dành cho suy luận, bao gồm những khoảng thiếu hợp lệ theo mục 4.

Emission vì vậy được fit từ gán pha trên một đường Viterbi, còn phân bố thời lượng sau lượt fit đầu dùng số đếm kỳ vọng. Runner ghi số đường thay đổi giữa hai vòng để theo dõi và chạy theo số vòng đã cấu hình. [Mã vòng lặp fit và suy luận](../data/step6_fixed_foot_events/code/cyclic_hsmm.py)

## 4. Suy luận qua frame thiếu flow

Bước suy luận giữ các frame nguồn liên tiếp trong cùng lượt, continuity, shot và speed part. Trong mỗi khối như vậy, trục suy luận chạy từ observation usable đầu tiên đến observation usable cuối cùng, gồm cả các frame thiếu nằm bên trong. Các khối được giải độc lập. Vùng trước observation đầu tiên, sau observation cuối cùng, hoặc cả khối không có flow usable sẽ để trống posterior.

Tại frame thiếu, Step 6 giữ bước thời gian và cộng **0** cho cả log-emission và điểm gợi ý, tương ứng hệ số nhân \(1\) trong trọng số của đường phân đoạn. Cách xử lý áp dụng nguyên tắc lấy biên qua quan sát thiếu ở mục 3.2 của Yu và Kobayashi, phương trình (6), (10) và (14), với giả định bản thân cờ thiếu không cung cấp thông tin về trạng thái. Thời lượng vẫn đếm các frame thiếu. Thời gian $\tau_t$ vẫn theo đồng hồ AVI, nên observation sau khoảng thiếu được chấm với thời gian đã trôi qua đầy đủ. [Yu và Kobayashi, 2003](https://doi.org/10.1016/S0165-1684(02)00378-X) · [Mã tạo trục suy luận](../data/step6_fixed_foot_events/code/cyclic_hsmm.py)

Ví dụ với ba frame 303, 304, 305, chỉ 303 và 305 có flow usable. HSMM chấm flow ở 303 và 305, rồi so các chuỗi mà pha tiếp tục qua 304 hoặc có biên chuyển ở đó. Posterior tại 304 đến từ các chuỗi này, nhờ bằng chứng hai phía, thứ tự pha và thời lượng đã học. Giá trị $(v_x,v_y)$ của frame 304 vẫn thiếu. [Mã bỏ đóng góp emission tại frame thiếu](../data/step6_fixed_foot_events/code/pixel_emission.py)

## 5. Posterior và hai loại khoảng khả tín

### 5.1. Hai câu hỏi theo frame

- **Frame $t$ thuộc pha nào?** $P(s_t=k\mid O_G)$ cộng trọng số của tất cả đoạn pha $k$ chứa $t$. Bốn posterior pha tại một frame được suy luận có tổng bằng 1.
- **Có mốc IC hoặc TO tại $t$?** Mốc IC là biên **Swing → IC**, ghi tại frame đầu IC. Mốc TO là biên **TO → Swing**, ghi tại frame đầu Swing, ngay sau frame cuối của pha TO. Mỗi score là xác suất của riêng chuyển tiếp đó giữa $t-1$ và $t$. Frame đầu của dải suy luận có posterior pha nhưng không có score biên vì thiếu frame đứng trước trong dải.

Do đó, một frame có thể có $P(s_t=\mathrm{TO})$ gần 1 và score mốc TO gần 0. Ví dụ đó là trường hợp frame nằm trong pha nhấc mũi, còn thời điểm rời sàn được mô hình đặt ở frame khác. [Mã posterior theo pha và chuyển tiếp](../data/step6_fixed_foot_events/code/hsmm.py)

### 5.2. Khoảng khả tín của toàn pha IC/TO

Với mỗi đoạn IC hoặc TO trên đường Viterbi, lấy **frame giữa đoạn** làm mốc. Xét các đoạn cùng pha bao phủ frame giữa này. Forward–backward cho trọng số của từng ứng viên và phân bố hậu nghiệm của hai biên sau khi điều kiện hóa trên việc frame giữa thuộc pha đó.

- Cận trái $L$ là phân vị 5% của frame bắt đầu.
- Cận phải $R$ là phân vị 95% của **frame cuối còn thuộc pha**.
- Dải $[L,R]$ tính cả hai đầu. Xác suất hậu nghiệm có điều kiện để **toàn pha** nằm trong dải ít nhất 90% theo mô hình, vì xác suất bắt đầu trước $L$ và kết thúc sau $R$ mỗi bên tối đa 5%. Với dữ liệu frame rời rạc, độ bao phủ có thể lớn hơn 90%.
- `anchor_probability` là posterior của pha tại frame giữa dùng để điều kiện hóa. `inferred_gaps` liệt kê các frame thiếu nằm trong dải. Dải thể hiện bất định về hai biên, nên một frame nằm trong dải vẫn có thể được Viterbi gán cho pha bên cạnh.

Trong video Walk Comfortable, IC ở lượt 1 có đoạn Viterbi **273–275**, còn dải khả tín là **273–276**. Ở một đoạn TO khác của cùng lượt, cả hai dải đều là **302–306**, với frame **304** tô xám vì thiếu flow. [Ví dụ BD004 theo từng pha](../outputs/overground_BD004/pixel_set_missing_flow/Walk_Comfortable/phase_uncertainty_intervals.csv) · [Mã dựng khoảng toàn pha](../data/step6_fixed_foot_events/code/cyclic_hsmm.py)

### 5.3. Khoảng khả tín của thời điểm mốc IC/TO

Với mỗi mốc IC hoặc TO trên đường Viterbi, tạo cửa sổ $W$ trong cùng dải suy luận. Biên của các cửa sổ nằm ở trung điểm giữa các mốc Viterbi liên tiếp cùng loại. Nếu chỉ có một mốc loại đó, cửa sổ bao phủ các frame có thể chứa biên trong dải. Từng chuỗi phân đoạn có thể có nhiều mốc cùng loại trong $W$, nên pipeline chỉ tính **mốc đầu tiên** của mỗi chuỗi trong cửa sổ.

Đặt $f_W(t)=P(\text{mốc đầu tiên trong }W\text{ ở }t\mid O_G)$ và $m=\sum_{t\in W} f_W(t)$. Khi đó $m$ là xác suất có ít nhất một mốc loại đó trong $W$. Lấy hai phân vị 5% và 95% của $f_W(t)/m$ để xuất `event_start_frame` và `event_end_frame`. Khoảng này chứa ít nhất 90% posterior **có điều kiện trên việc có sự kiện trong cửa sổ**. `local_boundary_mass` lưu $m$. `confidence` lưu xác suất chuyển pha tại đúng frame đại diện Viterbi. Hai đại lượng này bổ sung cho độ rộng của khoảng. [Mã phân bố biên đầu tiên](../data/step6_fixed_foot_events/code/hsmm.py) · [Mã xuất sự kiện](../scripts/evaluate_cyclic_overground.py)

## 6. Output và cách đọc video

| Tệp | Dùng để đọc |
|---|---|
| `phase_intervals.csv` | Chuỗi Viterbi của cả bốn pha, frame đầu/cuối, posterior pha trung bình và cờ chạm mép dải. `end_frame` tính cả frame cuối, `end_exclusive_s` là mép thời gian bên phải. |
| `phase_uncertainty_intervals.csv` | Khoảng toàn pha IC/TO từ đầu p05 đến cuối p95, frame đầu/cuối Viterbi, `anchor_probability`, cờ chạm mép và `inferred_gaps`. |
| `event_intervals.csv` | Khoảng thời điểm mốc IC/TO, frame đại diện, frame đầu/cuối của pha Viterbi đi kèm, `confidence`, `local_boundary_mass`, cờ flow ở biên và kết quả ghép GT. |
| `event_confidence_frames.csv` | Một dòng trên mỗi frame AVI. Các cột `pixel_set_p_phase_*` lưu bốn posterior pha. `pixel_set_ic_confidence` và `pixel_set_to_confidence` lưu posterior biên. Các hàng ngoài dải suy luận để trống score. |
| `pixel_set/model.npz`, `pixel_set/duration_distribution.csv`, `interval_settings.json` | Tham số emission, phân bố và số đếm thời lượng, cấu hình suy luận. |
| `predictions.csv`, `ground_truth_selected_foot.csv`, `matched_events_100ms.csv`, `metrics.csv`, `interval_metrics.csv` | Mốc Viterbi, nhãn tham chiếu của chân được chọn và số liệu chấm theo mốc hoặc khoảng. `foot_selection.csv` và `passages.csv` phục vụ kiểm tra ánh xạ chân và lượt. |
| `phase_uncertainty.png`, `*_phase_uncertainty.mp4` | Biểu đồ và video đối chứng. Video giữ frame nguồn trong các lượt đã chọn và phát ở 0,5×. |

Trong video, **thanh pha trên** của dòng Pixel-set HSMM tô chuỗi Viterbi. **Box dưới** tô khoảng bất định toàn pha IC màu tím hoặc TO màu cam. Phần xám trong box đánh dấu frame thiếu flow nằm trong khoảng. Xem thanh pha trên hoặc posterior theo frame để biết mô hình gán pha nào tại vùng xám. Thanh **Usable flow** thể hiện observation hợp lệ trên trục thời gian.

Ba cờ trong CSV giúp đọc cùng thông tin ở mức frame. `pixel_set_flow_observed` cho biết cặp ảnh gắn với frame đó có flow usable. `pixel_set_phase_inferred_without_flow` cho biết frame thiếu flow nằm trong dải có posterior. `pixel_set_flow_supported` cho biết biên ứng viên giữa $t-1$ và $t$ có observation usable ở cả hai phía. [Mã dựng video](../scripts/render_overground_review.py) · [Mã xuất CSV](../scripts/evaluate_cyclic_overground.py)

## 7. Đối chiếu với ground truth và các giới hạn

Pipeline ghép mốc một-một theo lượt, loại sự kiện và bên chân. **Chấm theo mốc** so các timestamp đại diện của Viterbi với CSV ở dung sai ±40, ±100 và ±160 ms. MAE và bias chỉ tính trên các cặp đã ghép. **Chấm theo khoảng** kiểm tra frame gần timestamp nhất, $\lfloor time_s\times FPS+0{,}5\rfloor$, có nằm trong khoảng **thời điểm mốc** hay không. Đây là phép chấm tương ứng với nhãn IC/TO dạng timestamp. [Mã ghép mốc](../scripts/evaluate_overground.py) · [Mã chấm khoảng](../scripts/evaluate_cyclic_overground.py)

Kết quả bản suy luận qua thiếu flow trên BD004 Session 1. Mỗi ô mốc ghi số khớp theo dung sai ±100 ms trên số mốc tham chiếu. Hai cột cuối ghi MAE trên các cặp đã ghép.

| Điều kiện | Lượt | Frame thiếu được suy luận | IC khớp | TO khớp | MAE IC (ms) | MAE TO (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Walk Slow | 8 | 90 | 0/30 | 24/30 | — | 75,0 |
| Walk Comfortable | 6 | 26 | 2/17 | 0/18 | 35,0 | — |
| Walk Fast | 8 | 64 | 3/24 | 0/19 | 66,7 | — |

Tổng cộng, mô hình suy luận pha qua 180 frame thiếu flow. Số mốc khớp tăng từ 28/138 ở bản tách dải tại frame thiếu lên 29/138, với một IC khớp thêm ở Walk Fast. Khi chấm bằng khoảng mốc p05–p95, chỉ một IC ở Walk Fast chứa frame ground truth. Các kết quả này đo một người và một phiên, trên các video dùng để fit. [Bảng kết quả và baseline](../outputs/overground_BD004/pixel_set_missing_flow/summary.csv)

- **Độ tin cậy phụ thuộc vào mô hình.** Điểm emission gộp log mật độ pixel theo vùng, thêm điểm gợi ý và dùng thời lượng đã học. Covariance mô hình hóa tương quan $v_x,v_y$ của một pixel. Tương quan giữa các pixel lân cận và giữa các cặp ảnh liên tiếp chưa được mô hình hóa trực tiếp. Các khoảng 90% là theo posterior của mô hình và chưa được hiệu chuẩn với ground truth.
- **Đoạn thiếu làm giảm bằng chứng trực tiếp.** Nếu mất flow hay xảy ra ở một tư thế cụ thể do che khuất, giả định trung tính có thể làm posterior qua đoạn thiếu bị lệch. Đoạn thiếu dài làm suy luận phụ thuộc nhiều hơn vào thứ tự pha và phân bố thời lượng.
- **Thời lượng phụ thuộc vào phần nhìn rõ.** Pha dài dễ chạm mép dải fit và bị loại khỏi số đếm pha trọn vẹn. Khi một video có ít chu kỳ đủ bằng chứng, kết quả nhạy hơn với gợi ý khởi tạo và phân bố gộp.
- **Ground truth kiểm chứng mốc.** CSV có thời điểm chạm và rời sàn để chấm hai biên ngoài của chu kỳ tiếp xúc. Việc kết luận đúng biên IC → Flat hoặc Flat → TO cần thêm nhãn. Sai định danh slot, mờ chuyển động, phối cảnh và sai số bù camera cũng ảnh hưởng đến kết quả.

[Hướng dẫn chạy Overground](../README.md#chạy-trên-video-overground) · [Các module Step 6](../data/step6_fixed_foot_events/code/README.md)
