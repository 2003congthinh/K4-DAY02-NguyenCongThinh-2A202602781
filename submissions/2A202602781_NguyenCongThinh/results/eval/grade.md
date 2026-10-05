```
$ python eval.py grade --final predictions/F01_seed*_test.csv --baseline predictions/T00_seed*_test.csv --uncal predictions/F01_uncal_seed*_test.csv --final-val predictions/F01_seed*_val.csv --val-csv C:\Users\ADMIN\Desktop\code_folders\VinUni-AI20K\track4-ComputerVision\K4-DAY02-NguyenCongThinh-2A202602781\data\labels\val_subset0.csv --latency-p95-ms 15.04 --latency-method proper --test-csv C:\Users\ADMIN\Desktop\code_folders\VinUni-AI20K\track4-ComputerVision\K4-DAY02-NguyenCongThinh-2A202602781\data\labels\test_subset0.csv --labels C:\Users\ADMIN\Desktop\code_folders\VinUni-AI20K\track4-ComputerVision\K4-DAY02-NguyenCongThinh-2A202602781\data\labels\labels.csv --out results\eval
```

## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 98.23% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 4 | 5 | final 0.9779, mốc 0.9695, Δ=+0.0084, s=0.0012 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 95.6% (mốc 88.5%), Snake Weed 96.1% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.0059, sau 0.0043 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9762, test 0.9779, chênh 0.0018 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 15.0 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 19 / 20** (phần I tối đa 20).

Ngưỡng điểm là TẠM THỜI (xem khối hằng số đầu file eval.py và RUBRIC.md mục I).
