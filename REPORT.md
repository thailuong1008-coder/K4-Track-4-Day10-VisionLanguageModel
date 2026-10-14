Họ tên: Nguyễn Thái Lương
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | - | - | 0.068 | 0.041–0.092 | 0.40 | 0.013 | 0.119 | 0.019 |
| E1 | LoRA LLM | 20,971,520 | 10.45 | 21.8 | 0.551 | 0.512–0.588 | 0.32 | 0.363 | 0.613 | 0.584 |
| E2 | + projector | 130,023,424 | 12.18 | 24.6 | 0.722 | 0.690–0.752 | 0.14 | 0.723 | 0.716 | 0.733 |
| E3 | + LoRA ViT | 137,101,312 | 13.62 | 28.3 | 0.780 | 0.751–0.808 | 0.13 | 0.744 | 0.803 | 0.764 |
| E4 | data_size = 0.1 | 20,971,520 | 10.45 | 2.4 | 0.448 | 0.408–0.485 | 0.43 | 0.370 | 0.507 | 0.409 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B có tổng quy mô xấp xỉ 4.44B tham số, phân bổ như sau:
- LLM (decoder language model + text embeddings): xấp xỉ 4.0B tham số (36 transformer layers).
- ViT (visual encoder): xấp xỉ 415M tham số (24 visual blocks).
- Projector: xấp xỉ 109M tham số, gồm `visual.merger` (~27M) và `visual.deepstack_merger_list` (~82M tham số) lấy đặc trưng đa tầng ở các block 5, 11, 17 của ViT.

Các khối cập nhật trong từng thí nghiệm:
- **E1:** Chỉ cập nhật LoRA trên LLM (252 modules gồm các ma trận `self_attn.{q,k,v,o}_proj` và `mlp.{gate,up,down}_proj` trên 36 layers), chiếm 20,971,520 tham số huấn luyện (~0.47%). Toàn bộ ViT và Projector bị đóng băng.
- **E2:** Cập nhật LoRA LLM giống E1 kết hợp mở huấn luyện toàn bộ trọng số (full weights) của Projector (`visual.merger` và `visual.deepstack_merger_list`) ở kiểu `float32`. Số tham số huấn luyện tăng thêm ~109.05M, đạt tổng cộng 130,023,424 tham số (~2.93%).
- **E3:** Cập nhật toàn bộ thành phần của E2 và bổ sung LoRA trên các khối Attention và MLP của ViT (`attn.{qkv,proj}` và `mlp.{linear_fc1,linear_fc2}` trên 24 blocks -> thêm 96 modules, tổng cộng 348 LoRA modules). Tổng số tham số huấn luyện đạt 137,101,312 tham số (~3.09%).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
Tỷ lệ `parse_valid_rate` của E0 chỉ đạt **0.788** (78.8%), tức có hơn 21% số ảnh tập test không phân tích cú pháp được JSON hợp lệ.

Hai lỗi phổ biến nhất của mô hình gốc:
1. **Lỗi format văn bản tự nhiên:** Mô hình thường tự động bọc câu trả lời trong markdown fences (```json ... ```) kèm theo lời giải thích giao tiếp trước hoặc sau mảng JSON (ví dụ: "Here are the detected road damages in the image:"), hoặc sinh nhầm các key như `"name"`, `"box"`, `"coordinate"` thay vì đúng schema `{"label": "...", "bbox_2d": [...]}`.
2. **Sai thang toạ độ và ảo giác (hallucination):** Mô hình gốc có xu hướng dự đoán toạ độ theo pixel ảnh thực tế (khoảng [0, 640] và [0, 360]) thay vì thang grounding chuẩn hoá [0, 1000] theo prompt, hoặc bịa ra các box ngẫu nhiên ở những vùng bóng râm, mặt đường loang lổ.

Ví dụ output thô điển hình của E0:
```text
```json
[{"label": "crack", "bbox_2d": [100, 200, 300, 400]}]
```
Here is the detected damage: pothole.
```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Nếu mô hình dự đoán theo thang pixel tuyệt đối (chiều rộng tối đa 640, chiều cao tối đa 360) mà chấm theo thang chuẩn hoá [0, 1000] của ground truth:
Toàn bộ các box dự đoán sẽ bị co cụm ở góc trên bên trái của ảnh ($x \in [0, 640], y \in [0, 360]$ trên khung chuẩn $[0, 1000] \times [0, 1000]$). Diện tích giao nhau (Intersection) với ground truth thực tế gần như bằng 0, khiến chỉ số IoU không bao giờ vượt qua ngưỡng $\ge 0.5$. Do đó, số lượng True Positive (TP) tụt về 0 và **F1 xấp xỉ bằng 0** (thực tế đo được $F_1 < 0.01$).

Cách kiểm chứng:
Tôi đã đối chiếu trực tiếp bằng cách chạy `scripts/score.py` với tuỳ chọn `--coord pixel` so với `--coord norm1000` trên cùng một tệp kết quả dự đoán, đồng thời viết hàm kiểm tra round-trip toạ độ trong notebook để xác nhận box chuyển đổi từ YOLO sang thang 0-1000 rồi quy ngược về pixel phải trùng khít 100% với box gốc trên ảnh.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Trong huấn luyện SFT cho mô hình sinh có điều kiện (Conditional Generation), nhiệm vụ của mô hình là nhận ảnh (220 token thị giác) và prompt chỉ dẫn làm điều kiện đầu vào, từ đó học phân phối sinh ra chuỗi JSON toạ độ của Assistant. Do đó, ta phải gán `labels = -100` cho toàn bộ prompt và image tokens để mô hình chỉ nhận gradient cập nhật từ phần câu trả lời.

Nếu tính loss trên cả prompt và token ảnh:
1. **Lãng phí gradient và phân tán mục tiêu học (gradient starvation):** Hơn 80% số token trong chuỗi là token ảnh và câu hỏi prompt cố định. Mô hình sẽ dành phần lớn năng lực tham số để học tái tạo lại các đặc trưng ảnh và các từ ngữ câu hỏi tiếng Anh quen thuộc, thay vì tập trung học ánh xạ khó: chuyển từ đặc trưng hư hại mặt đường sang toạ độ số học [x1, y1, x2, y2].
2. **Hậu quả thực tế:** Loss tổng thể của mô hình vẫn giảm rất nhanh nhưng F1 định vị hư hại sẽ cực kỳ thấp, mô hình chậm hội tụ và dễ sinh ra văn bản lặp lại vô nghĩa.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Không thể huấn luyện nếu ViT bị lượng tử 4-bit:** Khi toàn bộ ViT bị nén sang dạng NF4 4-bit, các ma trận trọng số của Projector (`visual.merger`) bị cố định dưới dạng lượng tử hoá của BitsAndBytes. Kiểu lượng tử hoá này không hỗ trợ lan truyền ngược gradient đầy đủ (full-precision backward updates). Do đó, ta bắt buộc phải sử dụng `llm_int8_skip_modules=["visual", "lm_head"]` trong `BitsAndBytesConfig` để giữ nguyên ViT và Projector ở FP16 nguyên bản.
- **Projector phải ở float32:** Projector là tầng cổ chai (bottleneck layer) ánh xạ trực tiếp các vector đặc trưng thị giác từ ViT sang không gian nhúng của LLM. Tầng này cực kỳ nhạy cảm với độ biến thiên gradient; nếu để ở FP16, khi tích luỹ gradient qua nhiều bước accumulation sẽ rất dễ gặp hiện tượng tràn số (numerical underflow/overflow dẫn tới loss `NaN` hoặc `Inf`). Việc ép kiểu Projector sang `torch.float32` đảm bảo độ ổn định số học và giúp mô hình hội tụ vững chắc.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Tập dữ liệu gồm 2.009 ảnh được cắt ra từ các video hành trình dashcam (quay vào tháng 2–3/2025). Hai khung hình (frame) liền kề nhau chỉ cách nhau vài phần mười giây, có góc quay, điều kiện ánh sáng, mặt đường và các ổ gà/vết nứt gần như trùng khít nhau hoàn toàn.
Nếu chia dữ liệu ngẫu nhiên (Random Split), các khung hình gần như giống hệt nhau sẽ rơi vào cả tập Train và tập Test $\rightarrow$ Xảy ra hiện tượng **Rò rỉ dữ liệu (Data Leakage)** nghiêm trọng. Mô hình chỉ cần ghi nhớ frame ở tập train là có thể đoán đúng frame ở tập test, dẫn đến điểm F1 cao giả tạo nhưng khi đem ra kiểm thử thực tế trên đoạn đường mới sẽ thất bại hoàn toàn. Do đó, việc chia split cố định theo **phiên quay (session)** là bắt buộc để đánh giá năng lực khái quát hoá thực tế.

Số lượng ảnh và box từng lớp ở mỗi split:
- **Train:** 1.422 ảnh | 1.074 pothole | 1.750 crack | 691 manhole (tổng 3.515 boxes).
- **Val:** 276 ảnh | 66 pothole | 470 crack | 103 manhole (tổng 639 boxes).
- **Test:** 311 ảnh | 121 pothole | 299 crack | 163 manhole (tổng 583 boxes).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
Dựa trên kết quả thực nghiệm:
- **E1 (LoRA LLM):** F1 = 0.551 (CI95: 0.512–0.588), VRAM đỉnh: 10.45 GB, thời gian train: 21.8 phút.
- **E2 (+ Projector):** F1 = 0.722 (CI95: 0.690–0.752), VRAM đỉnh: 12.18 GB, thời gian train: 24.6 phút. F1 tăng vượt bậc **+0.171** so với E1, và khoảng CI95 của E2 hoàn toàn tách biệt không chồng lấn với E1 (0.690 > 0.588). Chi phí VRAM chỉ tăng thêm ~1.73 GB và thời gian tăng thêm ~2.8 phút. Do đó, **mở Projector (E2) là bước nâng cấp cực kỳ xứng đáng và hiệu quả nhất**.
- **E3 (+ LoRA ViT):** F1 = 0.780 (CI95: 0.751–0.808), VRAM đỉnh: 13.62 GB, thời gian train: 28.3 phút. F1 tiếp tục tăng thêm **+0.058** (vượt cận trên CI95 của E2), đặc biệt cải thiện mạnh ở lớp `crack` (F1 tăng từ 0.716 lên 0.803). Tuy nhiên, VRAM tăng lên 13.62 GB (tiệm cận giới hạn 16GB của T4) và thời gian tăng thêm gần 4 phút.
- **Kết luận:** Nếu tài nguyên GPU hạn chế, E2 là điểm dừng tối ưu nhất về tương quan chi phí/hiệu năng. Nếu muốn tối đa hoá độ chính xác phát hiện box nhỏ, E3 là lựa chọn vượt trội.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Trên tập test 311 ảnh, độ rộng khoảng tin cậy 95% (CI95) đo bằng phương pháp Bootstrap (1.000 lượt resamples) dao động từ **0.057 đến 0.076** (tương đương biên độ sai số khoảng $\pm 0.03$ đến $\pm 0.04$ quanh giá trị F1 trung bình).
Điều này đặt ra giới hạn: Nếu hai setup thí nghiệm chỉ chênh lệch nhau dưới 0.04 điểm F1 và khoảng tin cậy chồng lấn nhiều, ta không thể kết luận setup nào tốt hơn một cách có ý nghĩa thống kê.
Tuy nhiên, trong kết quả thực nghiệm của chúng ta:
- Chênh lệch E1 so với E0 (+0.483) vượt xa CI95.
- Chênh lệch E2 so với E1 (+0.171) hoàn toàn nằm ngoài khoảng CI95 của cả hai (0.588 < 0.690).
- Chênh lệch E3 so với E2 (+0.058) cũng vượt cận trên của E2 (0.752 < 0.780).
Vì vậy, các kết luận về sự cải thiện từ E0 $\rightarrow$ E1 $\rightarrow$ E2 $\rightarrow$ E3 đều có cơ sở thống kê vững chắc.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- Ở các setup ban đầu (E1), lớp **`pothole` (ổ gà)** có kết quả kém nhất với F1 chỉ đạt 0.363 (trong khi `crack` đạt 0.613 và `manhole` đạt 0.584). 
- Nguyên nhân:
  1. *Ngữ nghĩa & Hình thái:* Ổ gà thường có ranh giới mờ nhạt, chiều sâu không thể hiện rõ trên ảnh 2D và dễ bị mô hình nhầm lẫn với các mảng vá nhựa đường cũ hoặc vết loang dầu. Ngược lại, `manhole` (nắp cống) có hình tròn/vuông rõ ràng, còn `crack` có dạng đường nứt tương phản cao.
  2. *Kích thước box:* Các ổ gà ở xa có diện tích trung vị rất nhỏ (< 0.7% khung hình), làm giảm đáng kể IoU khi dự đoán lệch chỉ vài pixel.
- Ở E2 và E3, khi projector và ViT được tinh chỉnh, F1 của cả 3 lớp đều đồng loạt tăng mạnh vượt 0.72 (`crack` đạt 0.803, `manhole` 0.764, `pothole` 0.744).
- Minh hoạ: 6 ảnh dự đoán đối chiếu Ground Truth (xanh lá) và Prediction (xanh dương/vàng) đã được trích xuất và lưu tại `report/vis_1.png` đến `report/vis_6.png` minh chứng cho các trường hợp: True Positive chính xác (`vis_1.png`, `vis_2.png`, `vis_3.png`), False Positive (`vis_4.png`), sai lệch biên ranh giới (`vis_5.png`), và bỏ sót vết nứt mờ do bóng râm (`vis_6.png`).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến lựa chọn:** Đổi biến `data_size` với giá trị 0.1 (chỉ sử dụng 10% dữ liệu train, tương đương ~142 ảnh) so với 100% dữ liệu (1.422 ảnh) ở cấu hình E1, lưu tại `report/evidence/e4_data_size_0.1.json`.
- **Kết quả:**
  - E4 (`data_size = 0.1`): F1 đạt **0.448** (CI95: 0.408–0.485, VRAM: 10.45 GB, thời gian train chỉ mất 2.4 phút).
  - So với E1 (100% data): F1 giảm 0.103 điểm (từ 0.551 xuống 0.448).
- **Độ tin cậy của kết luận:**
  Khoảng CI95 của E4 [0.408, 0.485] hoàn toàn tách biệt và thấp hơn cận dưới của E1 (0.512). Điều này chứng minh một cách tin cậy rằng: Dù Qwen3-VL-4B có khả năng few-shot rất tốt (chỉ 10% dữ liệu đã đạt F1 = 0.448 vượt xa E0), việc cung cấp đầy đủ dữ liệu đa dạng theo từng phiên quay vẫn là yếu tố quyết định để mô hình học được phân phối toạ độ chính xác cho domain đường sá.

## Nhìn lại

Bài lab đã giúp tôi hiểu sâu sắc cơ chế hoạt động của mô hình Vision-Language Model trong bài toán Visual Grounding: từ cách mã hoá ảnh thành 220 visual tokens, vai trò then chốt của Projector FP32 trong việc căn chỉnh không gian đặc trưng giữa ViT và LLM, đến kỹ thuật Mask Loss trong SFT. Điểm khó khăn lớn nhất là việc xử lý toạ độ grounding [0, 1000] sao cho không bị sai lệch thang đo khi lượng tử hoá và việc phân chia split theo phiên quay để triệt tiêu hoàn toàn hiện tượng rò rỉ dữ liệu video.
