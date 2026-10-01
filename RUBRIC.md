# RUBRIC — Tiêu chí chấm Lab Day 2

**Tổng điểm: 100** (cộng tối đa 10 điểm thưởng, tổng không vượt quá 110).
Điểm thưởng không bù được điều kiện tiên quyết bị vi phạm.

Bài lab chấm **quy trình thực nghiệm** ngang với **kết quả**. Một bài có điểm test thấp nhưng thiết kế chặt chẽ, phân tích trung thực thì điểm cao hơn một bài có điểm test cao nhưng chọn cấu hình bằng test hay không có bằng chứng.

---

## 0. Điều kiện tiên quyết

Thiếu một trong các mục sau thì bài **chưa được chấm** (hoặc bị giới hạn điểm, ghi trong ngoặc):

| # | Điều kiện | Hậu quả nếu thiếu |
|---|---|---|
| P1 | Có đủ 5 sản phẩm theo `README.md` mục 4: `results.xlsx`, báo cáo, `curves/`, `code/`, README riêng với link chạy lại | Trả bài, yêu cầu bổ sung |
| P2 | Số liệu đến từ lần chạy thật; truy ngược được `exp_id` → log/ảnh biểu đồ | Phần có số liệu bịa hoặc không truy ngược được **bị 0 điểm** và báo cáo vi phạm học thuật |
| P3 | Dùng đúng dataset DeepWeeds và fold 0 chia sẵn | Giới hạn tối đa 60 điểm nếu tự chia lại mà không nêu lý do |

---

## 1. Bảng điểm

### A. Thiết lập và tính chặt chẽ của pipeline (15 điểm)

| Tiêu chí | Điểm |
|---|---|
| Chia dữ liệu đúng (fold 0), chọn mô hình/siêu tham số **chỉ trên val**, test chạy **một lần** ở cuối, có nêu rõ trong báo cáo | 5 |
| EDA: phân bố lớp (biểu đồ), ảnh mẫu, kiểm tra trùng/rò rỉ, nêu nhận xét về mất cân bằng | 3 |
| Kiểm tra pipeline trước khi chạy thật: loss ban đầu ≈ ln 9, overfit 1 batch nhỏ, kiểm tra ảnh sau augmentation (có bằng chứng trong notebook hoặc báo cáo) | 3 |
| Cố định seed, ghi cấu hình đầy đủ (tag trọng số, hyperparameter, version thư viện); dùng `eval()` đúng lúc | 4 |

### B. So sánh backbone (15 điểm)

| Tiêu chí | Điểm |
|---|---|
| **≥ 5 backbone**, đủ ràng buộc: có ResNet, có ResNeXt hoặc ConvNeXt, có ít nhất 1 transformer, có ít nhất 1 mạng nhẹ (thiếu mỗi ràng buộc trừ 1 điểm; dưới 5 backbone: tối đa 6/15 cho mục này) | 5 |
| **Công bằng:** cùng công thức nền, cùng split, cùng seed; ghi rõ tag trọng số | 4 |
| Báo cáo đủ: params, GMAC, macro-F1/top-1 val, thời gian train, độ trễ sơ bộ | 3 |
| Chọn backbone đi tiếp **có lý do dựa trên số liệu** (không chỉ "vì F1 cao nhất" nếu có đánh đổi khác) | 3 |

### C. Công thức huấn luyện (20 điểm)

| Tiêu chí | Điểm |
|---|---|
| **≥ 3 trục** (khởi tạo, augmentation, loss, sampler, LR/optimizer, chính quy hoá, độ phân giải...), mỗi trục ≥ 2 giá trị; **ít nhất một trục là loss hoặc augmentation** | 6 |
| **Thiết kế có kiểm soát:** mỗi lần chạy chỉ khác nền một yếu tố, bảng ghi rõ khác ở điểm nào; nếu dùng cách tham lam theo trục thì nêu rõ | 5 |
| Cài đặt đúng các kỹ thuật (ví dụ focal `γ=0` ≡ CE; Mixup/CutMix trộn cả nhãn; weight decay không áp dụng cho norm/bias; EMA đánh giá đúng trọng số) | 4 |
| Thử **ít nhất một kết hợp** các yếu tố tốt và nhận xét cộng dồn hay triệt tiêu | 2 |
| Phân tích so sánh Δ với nhiễu (std); không kết luận từ chênh lệch nhỏ hơn nhiễu | 3 |

### D. Suy luận (15 điểm)

| Tiêu chí | Điểm |
|---|---|
| **≥ 4 phương pháp suy luận** ngoài mốc 1-view (ví dụ: TTA lật, multi-crop/scale, dò độ phân giải, gộp xác suất vs logit, ensemble, EMA/soup, temperature scaling, gộp BN/FP16) | 5 |
| Đo độ trễ **đúng cách**: warmup, `cuda.synchronize` (hoặc CUDA event), ≥ 50 lần, báo cáo p50/p95/p99, ghi rõ GPU/dtype/batch | 4 |
| Có **hiệu chuẩn:** ECE trước và sau temperature scaling, T khớp trên val | 2 |
| Biểu đồ/bảng **đánh đổi độ chính xác và độ trễ**, nhận xét phương pháp nào hợp ngoại tuyến, phương pháp nào hợp thời gian thực | 4 |

### E. Bảng so sánh `results.xlsx` (10 điểm)

| Tiêu chí | Điểm |
|---|---|
| Đủ các sheet và cột bắt buộc (`GUIDE.md` mục 6.1); mọi thí nghiệm có mặt | 4 |
| Nhất quán: `exp_id` khớp ảnh biểu đồ và báo cáo; đơn vị rõ; mean ± std đúng cho cấu hình chung kết | 3 |
| Sheet `Summary` rõ ràng, dễ đọc, làm nổi bật cấu hình tốt nhất và so sánh với mốc | 3 |

### F. Biểu đồ training (5 điểm)

| Tiêu chí | Điểm |
|---|---|
| Mỗi thí nghiệm huấn luyện (B, T, F) có một ảnh riêng, đủ loss train/val và metric val theo epoch | 3 |
| Đọc được: tiêu đề, nhãn trục, chú thích, tên khớp `exp_id`; có nhận xét về hiện tượng đáng chú ý (quá khớp, nhiễu, hội tụ chậm...) trong báo cáo | 2 |

### G. Báo cáo kết luận (15 điểm)

| Tiêu chí | Điểm |
|---|---|
| Đủ cấu trúc theo `GUIDE.md` mục 6.3; tóm tắt nêu rõ cấu hình tốt nhất và con số test cuối cùng kèm std | 4 |
| **Kết luận có bằng chứng:** trả lời được yếu tố nào đóng góp nhiều nhất (backbone, huấn luyện hay suy luận) và cấu hình nào tốt nhất, mỗi khẳng định gắn với số trong xlsx | 5 |
| **Phân tích lỗi:** ma trận nhầm lẫn, F1 từng lớp, xem ảnh bị đoán sai và đưa ra giả thuyết | 3 |
| **Trung thực và hạn chế:** nêu số seed, một fold, giảm bớt do ngân sách GPU, thí nghiệm thất bại, rủi ro lệch phân phối; khuyến nghị triển khai thực tế (ngân sách thời gian) | 3 |

### H. Code và khả năng tái lập (5 điểm)

| Tiêu chí | Điểm |
|---|---|
| Code tự viết, có cấu trúc rõ (model, data/augment, train, loss, inference, benchmark), đọc được, một hàm train dùng chung cho mọi cấu hình | 3 |
| README riêng có link notebook Colab/Kaggle, version thư viện, thứ tự chạy; người khác chạy lại được và ra kết quả cùng mức | 2 |

---

## 2. Điểm thưởng (tối đa +10)

| Việc làm thêm | Điểm |
|---|---|
| Linear probe với DINOv2 (hoặc mô hình nền tảng khác) đóng băng, so sánh với CNN tinh chỉnh | +2 |
| Chạy nhiều fold (ví dụ 3–5 fold) cho cấu hình cuối, báo cáo mean ± std qua fold | +3 |
| Chưng cất tri thức (giáo viên lớn → học sinh nhỏ) và so sánh với huấn luyện thường | +2 |
| Grad-CAM hoặc bản đồ attention để giải thích lỗi | +1 |
| Test-time adaptation đơn giản (chuẩn hoá lại thống kê BN, hoặc Tent) trên tập lệch miền tự tạo (ví dụ ảnh làm tối/nhiễu) | +2 |
| Xuất ONNX và so sánh độ trễ với PyTorch | +1 |
| Phân tích riêng về lệch phân phối: đánh giá trên ảnh bị nhiễu/làm mờ/thiếu sáng, ECE trước/sau | +2 |

Điểm thưởng chỉ tính khi phần làm thêm có số liệu thật và nằm trong báo cáo.

---

## 3. Lỗi bị trừ điểm

| Lỗi | Trừ |
|---|---|
| Chọn cấu hình, siêu tham số hoặc phương pháp suy luận **dựa vào điểm test**, hoặc chạy test nhiều lần rồi chọn lần tốt nhất | −10 đến −20 (nghiêm trọng, tuỳ mức) |
| Thiếu `model.eval()` khi đánh giá, hoặc đo độ trễ không `synchronize` / không warmup (số liệu sai) | −3 mỗi loại, tối đa −6 |
| Kết luận "A tốt hơn B" khi chênh lệch nhỏ hơn std hoặc chỉ có 1 seed mà không nêu hạn chế | −1 mỗi chỗ, tối đa −5 |
| Chỉ báo accuracy, không có macro-F1 hoặc F1 từng lớp | −3 |
| Thí nghiệm không có ảnh biểu đồ training | −0,5 mỗi thí nghiệm, tối đa −5 |
| Số liệu trong báo cáo mâu thuẫn với xlsx | −1 mỗi chỗ, tối đa −5 |
| Commit dataset hoặc checkpoint lớn vào git | −2 |
| Nộp trễ | Theo quy định của giảng viên |
| **Bịa số liệu, chép bài, hoặc chép số từ bài báo/slide như kết quả của mình** | **0 điểm phần liên quan; báo cáo vi phạm học thuật** |

---

## 4. Mức điểm tham khảo

| Điểm | Mô tả |
|---|---|
| **90–100+** | Đủ ≥ 5 backbone, ≥ 3 trục huấn luyện có kiểm soát, ≥ 4 phương pháp suy luận với độ trễ đo đúng; cấu hình cuối ≥ 3 seed; test một lần; báo cáo kết luận có bằng chứng và phân tích lỗi; code tái lập được |
| **75–89** | Đạt các ngưỡng tối thiểu, nhưng thiếu một số điểm chặt chẽ (ví dụ ít seed ở vòng cuối, phân tích nông, bảng thiếu cột) |
| **60–74** | Đủ sản phẩm nhưng một phần thí nghiệm thiếu kiểm soát hoặc kết luận không đủ bằng chứng |
| **< 60** | Thiếu ngưỡng tối thiểu (số backbone, số trục, số phương pháp), hoặc vi phạm quy tắc val/test, hoặc không tái lập được |

---

## 5. Danh sách tự kiểm trước khi nộp

- [ ] ≥ 5 backbone, cùng công thức nền, ghi tag trọng số.
- [ ] ≥ 3 trục công thức huấn luyện, mỗi lần chạy khác nền một yếu tố.
- [ ] ≥ 4 phương pháp suy luận, độ trễ p50/p95/p99 đo đúng cách.
- [ ] Cấu hình cuối chạy ≥ 3 seed, báo cáo mean ± std; test chạy **một lần**.
- [ ] `results.xlsx` đủ sheet, `exp_id` khớp ảnh trong `curves/`.
- [ ] Mỗi thí nghiệm huấn luyện có ảnh biểu đồ riêng.
- [ ] Báo cáo có tóm tắt, bảng so sánh, ma trận nhầm lẫn, kết luận, hạn chế.
- [ ] Code đầy đủ, README riêng có link notebook chạy lại được.
- [ ] Không commit dataset hoặc checkpoint lớn.
- [ ] Mọi số liệu đến từ lần chạy thật của bạn.
