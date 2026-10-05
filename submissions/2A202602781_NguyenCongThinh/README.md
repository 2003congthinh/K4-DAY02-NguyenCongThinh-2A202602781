# Lab Day 2 – DeepWeeds: bài nộp của Nguyễn Công Thịnh (2A202602781)

Bài nộp gồm code hoàn chỉnh (`code/`, phát triển từ bộ khung `starter/`) và các sản phẩm do code sinh ra:
`results.xlsx`, `report.md`, `curves/`, `predictions/`, `figures/`, `results/`.

> Phần code được viết với sự hỗ trợ của AI assistant (Claude). Mọi con số trong báo cáo đến từ các lần chạy thật
> trên máy được khai báo ở mục "Môi trường".

## Kết quả chính (test fold 0, 3 seed, mean ± std)

| Cấu hình | macro-F1 | top-1 | recall Chinee apple | recall Snake weed | ECE |
|---|---|---|---|---|---|
| **F01**: ConvNeXt-T + TrivialAugment + EMA, suy luận 288 px + temperature scaling | **0.9779 ± 0.0012** | **0.9823 ± 0.0010** | 0.9558 ± 0.0117 | 0.9608 ± 0.0130 | 0.0043 ± 0.0013 |
| Mốc T00 + I00: ConvNeXt-T, công thức nền, 1 view 224 px | 0.9695 ± 0.0004 | 0.9757 ± 0.0004 | 0.9381 ± 0.0160 | 0.9493 ± 0.0075 | 0.0106 ± 0.0001 |

Độ trễ F01 batch 1 trên RTX 4060: p50 11.1 ms, p95 15.0 ms. Phân tích đầy đủ trong [`report.md`](report.md), mọi
bảng trong [`results.xlsx`](results.xlsx).

## Link chạy lại

- Notebook: [`code/lab_day2.ipynb`](code/lab_day2.ipynb) (chạy được trên Kaggle/Colab hoặc Jupyter ở máy có GPU).
- Mở trên Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/2003congthinh/K4-DAY02-NguyenCongThinh-2A202602781/blob/main/submissions/2A202602781_NguyenCongThinh/code/lab_day2.ipynb)
  (chọn *Runtime → Change runtime type → GPU*). Ô đầu tiên tự `git clone` repo này, ô thứ hai tải ảnh từ Zenodo
  (kiểm tra MD5) và nhãn từ GitHub của tác giả; sau đó chạy lần lượt các ô (Bước 0 → 5).
  Kaggle: *File → Import Notebook → GitHub*, dán cùng đường dẫn, bật GPU và Internet.

## Môi trường

Kết quả trong bài được chạy trên máy cá nhân sau (phiên bản chính xác của từng run nằm trong
`runs/<exp>/seed<k>/config.json`, khoá `env`):

| Thành phần | Phiên bản / giá trị |
|---|---|
| Hệ điều hành | Windows 11 Pro (26200) |
| GPU / CPU | NVIDIA GeForce RTX 4060 8 GB / Intel Core i7-12700K |
| Python | 3.10.18 |
| PyTorch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 |
| timm | 1.0.30 |
| onnxruntime | 1.23.2 |
| Seed | 0 cho Bước 1–3; 0, 1, 2 cho chung kết `F01` và mốc `T00` |

Cài đặt (Windows/Linux, máy có GPU NVIDIA):

```bash
conda create -n lab2 python=3.10 -y && conda activate lab2
# hoặc: python -m venv .venv  (Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install timm scikit-learn pandas matplotlib openpyxl onnx onnxruntime
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`openpyxl` là bắt buộc cho giai đoạn `results` (ghi `results.xlsx`); nếu thiếu, `run_all.py all` dừng ở bước cuối.

Dữ liệu (không commit): ảnh giải nén vào `images\` ở **thư mục gốc repo** (17.509 file .jpg, MD5 của
`images.zip` = `b7b30f96d466fba86016aa5a26606e0f`), các file CSV của Alex Olsen (`labels.csv`, `train/val/test_subset0.csv`) ở `data\labels\` cũng ở thư mục gốc repo
(thư mục `data/` bị .gitignore nên phải tự chép/tải). Code tìm hai thư mục này theo vị trí `eval.py`, nên chạy từ đâu cũng được.
Nếu để chỗ khác, đặt biến môi trường `LAB_IMAGES` và `LAB_LABELS`.

## Thứ tự chạy

Mọi lệnh chạy từ thư mục này (`submissions/2A202602781_NguyenCongThinh/`). Code tự dùng GPU khi
`torch.cuda.is_available()` (profile `gpu`: 224 px, 12 epoch, batch 64, AMP, đúng công thức nền GUIDE 1.4).

```bash
python code/run_all.py all          # chạy hết; dừng giữa chừng thì chạy lại lệnh này, phần đã xong được bỏ qua
```

hoặc từng giai đoạn:

| # | Lệnh | Việc làm | Sinh ra |
|---|---|---|---|
| 0 | `python code/run_all.py eda` | EDA, kiểm tra split, loss ban đầu ≈ ln 9, overfit 1 batch | `results/eda.json`, `figures/eda_*.png` |
| 1 | `python code/run_all.py backbones` | 6 backbone B01–B06, công thức nền, seed 0 | `runs/B0x/`, `curves/B0x_*.png` |
| 1 | `python code/run_all.py choose_backbone` | độ trễ sơ bộ + chọn backbone (chỉ val) | `results/backbone_latency.csv`, `results/decisions.json` |
| 2 | `python code/run_all.py training` | T00 + 11 ablation T01–T11 (một yếu tố mỗi run) | `runs/T*/`, `curves/T*.png` |
| 2 | `python code/run_all.py choose_combo` → `combo` → `choose_final` | kết hợp yếu tố thắng (T12), chốt công thức | `results/decisions.json` |
| 3 | `python code/run_all.py inference` | TTA, multi-crop, độ phân giải, ensemble, EMA, TS, gộp BN, AMP, ONNX + độ trễ | `results/inference.csv`, `results/latency.csv` |
| 4 | `python code/run_all.py final_train` | F01 seed 0/1/2, T00 seed 1/2 | `runs/F01/`, `runs/T00/` |
| 4 | `python code/run_all.py final_test` | **test một lần mỗi seed**, `eval.py score` + `grade` | `predictions/`, `results/eval/` |
| 5 | `python code/run_all.py results` | bảng tổng hợp | `results.xlsx`, `results/tables.md` |

Kiểm tra code (không cần GPU): `cd code && python -m unittest test_codes -v` (24 test: focal γ=0 ≡ CE,
CutMix λ đúng diện tích, gộp BN chính xác, warmup + cosine, EMA, temperature scaling, ...).

Lưu ý trên Windows:

- `eval.py grade` ghi `grade_I.json` (có tiếng Việt) mà không chỉ định encoding, nên trên Windows lỗi cp1252.
  `final.py` tự gọi eval.py với `PYTHONUTF8=1`; khi tự chạy eval.py bằng tay hãy đặt `set PYTHONUTF8=1`
  (PowerShell: `$env:PYTHONUTF8=1`). Test gốc của repo cũng cần biến này: `PYTHONUTF8=1 python -m unittest discover -s tests`.
- Nếu GPU hết bộ nhớ (8 GB với Swin-T/ConvNeXt-T), giảm batch cho **mọi** thí nghiệm: `set LAB_BATCH=32`
  trước khi chạy, và ghi lại trong báo cáo (GUIDE 1.4).

Lựa chọn giữa các bước được ghi tự động (kèm lý do) trong `results/decisions.json`, theo luật chỉ dùng **val**
(xem docstring của `code/run_all.py`). Muốn chọn khác: ghi khoá vào `results/decisions_override.json`.

## Cấu trúc

```
report.md      báo cáo kết luận (GUIDE 6.3)
results.xlsx   bảng so sánh mọi thí nghiệm (GUIDE 6.1)
code/          dataset.py model.py losses.py train.py inference.py benchmark.py   (bộ khung đã hoàn thiện)
               experiments.py run_all.py eda.py infer_experiments.py final.py make_results.py test_codes.py
               lab_day2.ipynb
curves/        một ảnh .png cho mỗi exp_id (B, T, F)
predictions/   <exp_id>_seed<k>_test.csv / _val.csv  (định dạng eval.py)
figures/       EDA, ma trận nhầm lẫn, ảnh lỗi, đánh đổi độ chính xác–độ trễ, reliability diagram
results/       decisions.json, eda.json, inference.csv, latency.csv, eval/ (đầu ra eval.py), tables.md
runs/          checkpoint, logit, history (KHÔNG commit, đã .gitignore)
```
