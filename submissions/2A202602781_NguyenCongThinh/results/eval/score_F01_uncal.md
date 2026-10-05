```
$ python eval.py score --pred predictions/F01_uncal_seed*_test.csv --test-csv C:\Users\ADMIN\Desktop\code_folders\VinUni-AI20K\track4-ComputerVision\K4-DAY02-NguyenCongThinh-2A202602781\data\labels\test_subset0.csv --labels C:\Users\ADMIN\Desktop\code_folders\VinUni-AI20K\track4-ComputerVision\K4-DAY02-NguyenCongThinh-2A202602781\data\labels\labels.csv --tag F01_uncal --out results\eval
```

### F01_uncal (3 seed: [0, 1, 2]; 3507 ảnh)

| Chỉ số | mean ± std |
|---|---|
| top-1 accuracy | 0.9823 ± 0.0010 |
| macro-F1 | 0.9779 ± 0.0012 |
| balanced accuracy | 0.9749 ± 0.0021 |
| ECE (15 bin) | 0.0059 ± 0.0020 |
| NLL | 0.0561 ± 0.0005 |

| Lớp | Precision | Recall | F1 | Số ảnh |
|---|---|---|---|---|
| Chinee apple | 0.977 ± 0.004 | 0.956 ± 0.012 | 0.966 ± 0.007 | 226 |
| Lantana | 0.992 ± 0.003 | 0.969 ± 0.003 | 0.980 ± 0.001 | 213 |
| Parkinsonia | 0.973 ± 0.003 | 0.992 ± 0.003 | 0.982 ± 0.003 | 207 |
| Parthenium | 0.998 ± 0.003 | 0.972 ± 0.012 | 0.985 ± 0.007 | 205 |
| Prickly acacia | 0.961 ± 0.003 | 0.966 ± 0.015 | 0.963 ± 0.009 | 213 |
| Rubber vine | 0.988 ± 0.006 | 0.977 ± 0.006 | 0.983 ± 0.000 | 202 |
| Siam weed | 0.988 ± 0.007 | 0.991 ± 0.005 | 0.989 ± 0.001 | 215 |
| Snake weed | 0.969 ± 0.003 | 0.961 ± 0.013 | 0.965 ± 0.007 | 204 |
| Negative | 0.984 ± 0.002 | 0.991 ± 0.002 | 0.988 ± 0.001 | 1822 |

Đã lưu kết quả vào results\eval/F01_uncal_*
