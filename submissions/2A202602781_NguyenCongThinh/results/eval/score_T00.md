```
$ python eval.py score --pred predictions/T00_seed*_test.csv --test-csv C:\Users\ADMIN\Desktop\code_folders\VinUni-AI20K\track4-ComputerVision\K4-DAY02-NguyenCongThinh-2A202602781\data\labels\test_subset0.csv --labels C:\Users\ADMIN\Desktop\code_folders\VinUni-AI20K\track4-ComputerVision\K4-DAY02-NguyenCongThinh-2A202602781\data\labels\labels.csv --tag T00 --out results\eval
```

### T00 (3 seed: [0, 1, 2]; 3507 ảnh)

| Chỉ số | mean ± std |
|---|---|
| top-1 accuracy | 0.9757 ± 0.0004 |
| macro-F1 | 0.9695 ± 0.0004 |
| balanced accuracy | 0.9709 ± 0.0031 |
| ECE (15 bin) | 0.0106 ± 0.0001 |
| NLL | 0.0758 ± 0.0025 |

| Lớp | Precision | Recall | F1 | Số ảnh |
|---|---|---|---|---|
| Chinee apple | 0.961 ± 0.006 | 0.938 ± 0.016 | 0.949 ± 0.008 | 226 |
| Lantana | 0.961 ± 0.024 | 0.978 ± 0.003 | 0.969 ± 0.011 | 213 |
| Parkinsonia | 0.981 ± 0.005 | 0.979 ± 0.007 | 0.980 ± 0.002 | 207 |
| Parthenium | 0.992 ± 0.003 | 0.976 ± 0.010 | 0.984 ± 0.005 | 205 |
| Prickly acacia | 0.923 ± 0.008 | 0.973 ± 0.005 | 0.947 ± 0.002 | 213 |
| Rubber vine | 0.985 ± 0.000 | 0.974 ± 0.003 | 0.979 ± 0.001 | 202 |
| Siam weed | 0.979 ± 0.014 | 0.989 ± 0.003 | 0.984 ± 0.006 | 215 |
| Snake weed | 0.951 ± 0.014 | 0.949 ± 0.007 | 0.950 ± 0.003 | 204 |
| Negative | 0.985 ± 0.004 | 0.982 ± 0.003 | 0.983 ± 0.000 | 1822 |

Đã lưu kết quả vào results\eval/T00_*
