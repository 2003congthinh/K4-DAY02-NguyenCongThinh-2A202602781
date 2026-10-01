# Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

> Track 4 · Ngày 2 · *Tích chập, chuỗi, attention · backbone · huấn luyện · suy luận*
> Bài lab này mở rộng **Lab #2** trong slide Day 2. Slide chỉ yêu cầu 1 backbone, 3 cách khởi tạo, có/không CutMix và TTA. Ở đây bạn làm đầy đủ: **≥ 5 backbone**, **nhiều công thức huấn luyện**, **nhiều cách suy luận**, rồi chọn cấu hình tốt nhất và báo cáo.

Thư mục này chỉ có **hướng dẫn và tiêu chí chấm**. **Không có code mẫu**: bạn tự viết model, train, inference và đo độ trễ. Làm vậy để bạn hiểu từng thành phần trong slide, không chỉ chạy lại code có sẵn.

| File | Dùng để làm gì |
|---|---|
| `README.md` (file này) | Tổng quan, dataset, sản phẩm phải nộp, cách nộp bài |
| [`GUIDE.md`](GUIDE.md) | Quy trình từng bước, danh sách thí nghiệm, cấu trúc file xlsx và báo cáo, bẫy thường gặp |
| [`RUBRIC.md`](RUBRIC.md) | Thang điểm 100, tiêu chí đạt, lỗi bị trừ điểm |

---

## 1. Mục tiêu học tập

Sau bài lab, bạn có thể:

1. So sánh công bằng nhiều backbone (CNN và transformer) trên cùng một bài toán, cùng một công thức huấn luyện.
2. Đo riêng đóng góp của từng yếu tố trong **công thức huấn luyện** (khởi tạo, augmentation, loss, optimizer/LR, EMA). Slide chương 5 nhấn mạnh công thức quan trọng ngang kiến trúc.
3. So sánh các **kỹ thuật suy luận** (TTA, độ phân giải kiểm tra, ensemble, hiệu chuẩn, gộp BatchNorm/FP16) bằng cả độ chính xác lẫn **độ trễ đo đúng cách**.
4. Phân biệt chênh lệch thật với **nhiễu** do hạt giống (seed) bằng mean ± std.
5. Rút ra kết luận có bằng chứng, nêu rõ cấu hình nào tốt nhất và vì sao.

## 2. Dataset: DeepWeeds

| Thuộc tính | Giá trị |
|---|---|
| Bài toán | Phân loại ảnh cỏ dại ngoài đồng (robot nông nghiệp, Queensland, Úc) |
| Số ảnh | 17.509 ảnh RGB 256×256 |
| Số lớp | 9: 8 loài cỏ dại + `Negative` (thực vật không phải loài mục tiêu) |
| Đặc điểm | **Mất cân bằng lớp** (`Negative` chiếm khoảng một nửa số ảnh, mỗi loài còn lại khoảng 1.000 ảnh; bạn tự kiểm tra con số chính xác ở bước EDA) |
| Dung lượng | Khoảng 490 MB |
| Giấy phép | CC BY 4.0 |
| Bài báo | Olsen et al., *DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning*, Scientific Reports 9, 2058 (2019), [doi:10.1038/s41598-018-38343-3](https://doi.org/10.1038/s41598-018-38343-3) |
| Baseline tham khảo | Bài báo báo cáo ResNet-50 đạt khoảng 95,7% accuracy trung bình |

**Nơi tải:**

- Ảnh: [Zenodo, DOI 10.5281/zenodo.7939060](https://zenodo.org/record/7939059), file `images.zip`.
  - Link trực tiếp: `https://zenodo.org/records/7939060/files/images.zip?download=1`
  - MD5: `b7b30f96d466fba86016aa5a26606e0f`. Hãy kiểm tra checksum sau khi tải.
- Nhãn và các fold chia sẵn: [github.com/AlexOlsen/DeepWeeds](https://github.com/AlexOlsen/DeepWeeds), thư mục `labels/` gồm `labels.csv`, `train_subset{0-4}.csv`, `val_subset{0-4}.csv`, `test_subset{0-4}.csv` (cột `Filename, Label, Species`; chia 60/20/20).
- Ngoài ra có trong [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/deep_weeds), nhưng nên dùng file CSV ở trên để mọi người dùng **cùng một cách chia**.

**Quy ước bắt buộc về dữ liệu:** dùng **fold 0** (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`) cho mọi thí nghiệm chính. Chọn mô hình và siêu tham số chỉ trên **val**. **Test chỉ được dùng đúng một lần ở bước cuối** (xem GUIDE).

## 3. Môi trường gợi ý: Google Colab hoặc Kaggle

| Nền tảng | Ưu điểm | Lưu ý |
|---|---|---|
| **Kaggle Notebooks** (khuyên dùng) | GPU miễn phí theo hạn mức hằng tuần, phiên chạy ổn định, tắt trình duyệt vẫn chạy được bằng *Save & Run All* | Kiểm tra hạn mức GPU hiện hành trong tài khoản của bạn; cần bật Internet để tải dataset |
| **Google Colab** | Khởi động nhanh, gắn Google Drive để lưu checkpoint | GPU miễn phí không được đảm bảo và có thể bị ngắt; phải lưu checkpoint và log ra Drive thường xuyên |

Gợi ý chung:

- Dùng GPU (T4 trở lên), bật mixed precision (AMP).
- Lưu **mọi log, đường cong, logit dự đoán và checkpoint** ra ổ bền (Drive, hoặc Kaggle output) ngay sau mỗi lần chạy, để phiên bị ngắt không mất kết quả.
- Ước lượng ngân sách tính toán và mẹo tiết kiệm nằm ở [`GUIDE.md`](GUIDE.md), mục 7.

## 4. Bạn phải nộp những gì

| # | Sản phẩm | Yêu cầu tối thiểu |
|---|---|---|
| 1 | `results.xlsx` | Bảng so sánh **tất cả** thí nghiệm (backbone, training, inference, kết quả cuối, độ trễ). Cấu trúc sheet và cột ở GUIDE mục 6.1 |
| 2 | `report.md` (hoặc `report.pdf`) | Báo cáo kết luận: thiết lập, kết quả, phân tích, cấu hình tốt nhất, hạn chế. Dàn ý ở GUIDE mục 6.3 |
| 3 | `curves/` | **Ảnh biểu đồ training của từng thí nghiệm** (loss và metric theo epoch, train và val), một ảnh `.png` cho mỗi `exp_id` |
| 4 | `code/` | **Toàn bộ code** do bạn viết: định nghĩa model, dataset/augmentation, train loop, các loss, inference/TTA/ensemble, đo độ trễ, tạo bảng và biểu đồ |
| 5 | `README.md` riêng của bạn | Link notebook Colab/Kaggle chạy lại được, phiên bản thư viện, lệnh/thứ tự chạy, seed đã dùng |

Số thí nghiệm tối thiểu (chi tiết ở GUIDE):

- **≥ 5 backbone** (có ít nhất 1 họ transformer và ít nhất 1 mạng nhẹ).
- **≥ 3 trục công thức huấn luyện**, mỗi trục thử ít nhất 2–3 giá trị (ví dụ: khởi tạo, augmentation, loss).
- **≥ 4 phương pháp suy luận** và đo độ trễ p50/p95/p99.
- Cấu hình cuối cùng chạy **≥ 3 seed**, báo cáo mean ± std.

## 5. Cách nộp bài

1. `git pull` repo này để lấy thư mục `Day2/`.
2. Tạo thư mục bài làm của bạn, đặt đúng cấu trúc sau:

```
Day2/submissions/<mssv>_<ho_ten_khong_dau>/
├── README.md          # link Colab/Kaggle, cách chạy lại
├── results.xlsx
├── report.md          # hoặc report.pdf
├── curves/
│   ├── B01_resnet50.png
│   ├── T03_cutmix.png
│   └── ...            # mỗi exp_id một ảnh
└── code/
    ├── *.ipynb
    └── *.py           # model, train, inference, benchmark, ...
```

3. **Không commit** dataset (`images.zip`, ảnh) và checkpoint lớn. Chỉ commit code, `results.xlsx`, báo cáo và ảnh biểu đồ. Nếu cần chia sẻ checkpoint, đặt link ở README riêng của bạn.
4. Nộp bài theo cách giảng viên thông báo (ví dụ Pull Request vào repo này, hoặc nén thư mục và nộp lên hệ thống của lớp). Hạn nộp do giảng viên công bố.

## 6. Quy tắc trung thực học thuật

- Mọi con số trong `results.xlsx` và báo cáo phải **đến từ log chạy thật** của bạn, truy ngược được tới một `exp_id` và một dòng log.
- Không bịa số, không chép số từ bài báo hay slide rồi ghi như kết quả của mình. Số tham khảo từ nguồn khác phải ghi rõ là *trích dẫn* và có nguồn.
- Được thảo luận ý tưởng với bạn học, nhưng code, bảng và báo cáo phải của riêng bạn.
- Kết quả xấu, thí nghiệm thất bại, hoặc phát hiện "kỹ thuật X không giúp ích" **vẫn được điểm** nếu có phân tích tốt. Xem `RUBRIC.md`.

## 7. Tài liệu đọc trước

- Slide Day 2: chương 4 (backbone), chương 5 (huấn luyện), chương 6 (suy luận) và trang "Lab #2".
- *ResNet strikes back* (Wightman et al., arXiv:2110.00476): công thức huấn luyện quan trọng ngang kiến trúc.
- CutMix (arXiv:1905.04899), Focal Loss (arXiv:1708.02002), Temperature scaling (Guo et al., arXiv:1706.04599), FixRes (arXiv:1906.06423).
- Karpathy, *A Recipe for Training Neural Networks*: checklist gỡ lỗi khi huấn luyện không hội tụ.
