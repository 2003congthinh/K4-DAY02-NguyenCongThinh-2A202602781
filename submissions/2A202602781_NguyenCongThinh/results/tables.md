# Bảng kết quả (tự sinh bởi code/make_results.py, profile gpu)

## Backbones

| exp_id | backbone | weight tag (timm) | params (M) | GMAC | img size (px) | epochs | seed | macro-F1 val | top-1 val | best epoch | train time / epoch (s) | latency batch-1 p50 (ms) | device | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B01 | resnet50 | resnet50.a1_in1k | 23.5265 | 4.0872 | 224 | 12 | 0 | 0.7944 | 0.8500 | 12 | 220.0899 | 5.1513 | NVIDIA GeForce RTX 4060 |  |
| B02 | convnext_tiny | convnext_tiny.in12k_ft_in1k | 27.8270 | 4.4548 | 224 | 12 | 0 | 0.9681 | 0.9754 | 11 | 37.1980 | 3.5925 | NVIDIA GeForce RTX 4060 | CHỌN đi tiếp |
| B03 | deit_small_patch16_224 | deit_small_patch16_224.fb_in1k | 21.6691 | 4.2408 | 224 | 12 | 0 | 0.9480 | 0.9612 | 12 | 25.9167 | 3.2075 | NVIDIA GeForce RTX 4060 |  |
| B04 | swin_tiny_patch4_window7_224 | swin_tiny_patch4_window7_224.ms_in1k | 27.5263 | 4.4898 | 224 | 12 | 0 | 0.9621 | 0.9712 | 7 | 46.3373 | 6.8025 | NVIDIA GeForce RTX 4060 |  |
| B05 | efficientnet_b0 | efficientnet_b0.ra_in1k | 4.0191 | 0.3845 | 224 | 12 | 0 | 0.8510 | 0.8880 | 11 | 355.6184 | 6.6538 | NVIDIA GeForce RTX 4060 |  |
| B06 | mobilenetv3_large_100 | mobilenetv3_large_100.ra_in1k | 4.2136 | 0.2153 | 224 | 12 | 0 | 0.7798 | 0.8343 | 12 | 72.4453 | 5.3401 | NVIDIA GeForce RTX 4060 |  |

## Training

| exp_id | backbone | axis (A-F) | khác T00 ở | overrides | seed | macro-F1 val | top-1 val | Δ macro-F1 vs T00 | F1 Chinee apple val | F1 Snake weed val | best epoch | train time / epoch (s) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T00 | convnext_tiny | - | công thức nền | {} | 0 | 0.9681 | 0.9754 | 0.0000 | 0.9447 | 0.9268 | 11 | 37.1980 | cùng run với B02 |
| T01 | convnext_tiny | A | init=scratch (khởi tạo ngẫu nhiên) | {"init": "scratch"} | 0 | 0.2962 | 0.5513 | -0.6719 | 0.2267 | 0.2682 | 12 | 37.1182 | 1 seed |
| T02 | convnext_tiny | A | init=frozen (đóng băng backbone, chỉ train head) | {"init": "frozen"} | 0 | 0.8495 | 0.8812 | -0.1186 | 0.8287 | 0.7752 | 11 | 13.8634 | 1 seed |
| T03 | convnext_tiny | B | aug=trivial (TrivialAugmentWide) | {"aug": "trivial"} | 0 | 0.9729 | 0.9791 | 0.0048 | 0.9543 | 0.9343 | 12 | 36.5156 | 1 seed |
| T04 | convnext_tiny | B | mix=cutmix (alpha=1) | {"mix": "cutmix", "mix_alpha": 1.0} | 0 | 0.9632 | 0.9717 | -0.0049 | 0.9433 | 0.9234 | 11 | 36.8670 | 1 seed |
| T05 | convnext_tiny | B | aug=flipv (thêm lật dọc) | {"aug": "flipv"} | 0 | 0.9698 | 0.9763 | 0.0017 | 0.9488 | 0.9383 | 12 | 36.6871 | 1 seed |
| T06 | convnext_tiny | C | loss=ls (label smoothing eps=0.1) | {"loss": "ls", "label_smoothing": 0.1} | 0 | 0.9677 | 0.9749 | -0.0004 | 0.9358 | 0.9242 | 12 | 37.1120 | 1 seed |
| T07 | convnext_tiny | C | loss=focal (gamma=2) | {"loss": "focal", "focal_gamma": 2.0} | 0 | 0.9629 | 0.9709 | -0.0052 | 0.9436 | 0.9242 | 12 | 35.4036 | 1 seed |
| T08 | convnext_tiny | C | loss=ce_weighted (trọng số 1/n_c) | {"loss": "ce_weighted"} | 0 | 0.9653 | 0.9723 | -0.0028 | 0.9440 | 0.9242 | 10 | 37.3008 | 1 seed |
| T09 | convnext_tiny | D | sampler=balanced (oversample lớp hiếm) | {"sampler": "balanced"} | 0 | 0.9613 | 0.9686 | -0.0068 | 0.9488 | 0.9135 | 12 | 36.2485 | 1 seed |
| T10 | convnext_tiny | E | lr_head = lr_backbone = 1e-4 (không x10) | {"lr_head": 0.0001} | 0 | 0.9627 | 0.9706 | -0.0054 | 0.9324 | 0.9200 | 12 | 35.7364 | 1 seed |
| T11 | convnext_tiny | F | ema_decay=0.999 (đánh giá bằng trọng số EMA) | {"ema_decay": 0.999} | 0 | 0.9687 | 0.9760 | 0.0006 | 0.9447 | 0.9291 | 12 | 37.9600 | 1 seed |
| T12 | convnext_tiny | combo | kết hợp: T03, T11 | {"aug": "trivial", "ema_decay": 0.999} | 0 | 0.9735 | 0.9797 | 0.0053 | 0.9543 | 0.9366 | 12 | 38.5065 | 1 seed |

## Inference

| exp_id | phương pháp | mô hình/checkpoint | K | macro-F1 val | top-1 val | ECE val | F1 Chinee apple val | F1 Snake weed val | p50 batch-1 (ms) | p95 batch-1 (ms) | p99 batch-1 (ms) | throughput batch-32 (ảnh/s) | note | chi phí tương đối vs I00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I00 | 1 view (resize + center crop) | T12 seed0 (convnext_tiny) | 1.0000 | 0.9735 | 0.9797 | 0.0079 | 0.9543 | 0.9366 | 10.9747 | 15.7904 | 17.9589 | 521.3238 |  | 1.0000 |
| I01 | TTA lật ngang, gộp xác suất | T12 seed0 (convnext_tiny) | 2.0000 | 0.9737 | 0.9800 | 0.0081 | 0.9565 | 0.9340 | 20.7831 | 29.3139 | 32.1892 | 260.4205 |  | 1.8937 |
| I03 | TTA lật ngang, gộp logit | T12 seed0 (convnext_tiny) | 2.0000 | 0.9737 | 0.9800 | 0.0084 | 0.9565 | 0.9340 | 20.7831 | 29.3139 | 32.1892 | 260.4205 |  | 1.8937 |
| I02 | TTA 5 crop, gộp xác suất | T12 seed0 (convnext_tiny) | 5.0000 | 0.9734 | 0.9794 | 0.0057 | 0.9543 | 0.9340 | 57.2441 | 64.4243 | 74.2343 | 103.9818 |  | 5.2160 |
| I02b | TTA 10 crop (5 crop + lật), gộp xác suất | T12 seed0 (convnext_tiny) | 10.0000 | 0.9746 | 0.9806 | 0.0050 | 0.9567 | 0.9363 | 159.8995 | 217.5963 | 223.3347 | 52.8827 |  | 14.5699 |
| I03b | TTA 10 crop, gộp logit | T12 seed0 (convnext_tiny) | 10.0000 | 0.9743 | 0.9803 | 0.0081 | 0.9567 | 0.9363 | 159.8995 | 217.5963 | 223.3347 | 52.8827 |  | 14.5699 |
| I04_192 | 1 view ở độ phân giải 192 (train 224) | T12 seed0 (convnext_tiny) | 1.0000 | 0.9699 | 0.9774 | 0.0110 | 0.9490 | 0.9327 | 3.9728 | 4.4977 | 4.5596 | 726.0442 | resize 219 + center crop 192 | 0.3620 |
| I04_256 | 1 view ở độ phân giải 256 (train 224) | T12 seed0 (convnext_tiny) | 1.0000 | 0.9778 | 0.9829 | 0.0064 | 0.9507 | 0.9461 | 3.6784 | 4.4703 | 5.3812 | 404.0113 | resize 293 + center crop 256 | 0.3352 |
| I04_288 | 1 view ở độ phân giải 288 (train 224) | T12 seed0 (convnext_tiny) | 1.0000 | 0.9787 | 0.9831 | 0.0073 | 0.9596 | 0.9531 | 3.9595 | 4.8739 | 6.0538 | 304.2541 | resize 329 + center crop 288 | 0.3608 |
| I04_320 | 1 view ở độ phân giải 320 (train 224) | T12 seed0 (convnext_tiny) | 1.0000 | 0.9746 | 0.9803 | 0.0049 | 0.9548 | 0.9450 | 4.5530 | 6.6855 | 7.3021 | 226.4938 | resize 366 + center crop 320 | 0.4149 |
| I05 | Ensemble 3 mô hình (trung bình xác suất) | T12 + B04 + B03 | 3.0000 | 0.9756 | 0.9823 | 0.0175 | 0.9498 | 0.9356 | 28.4563 | 39.6685 | 45.3714 | 64.4076 | độ trễ = tổng các mô hình chạy tuần tự | 2.5929 |
| I06_raw | T11: trọng số raw (không EMA) | T11 seed0 | 1.0000 | 0.9679 | 0.9754 | 0.0106 | 0.9423 | 0.9242 | 10.9747 | 15.7904 | 17.9589 | 521.3238 | cùng epoch với checkpoint EMA | 1.0000 |
| I06 | T11: trọng số EMA (miễn phí lúc suy luận) | T11 seed0 | 1.0000 | 0.9687 | 0.9760 | 0.0104 | 0.9447 | 0.9291 | 10.9747 | 15.7904 | 17.9589 | 521.3238 |  | 1.0000 |
| I07 | Temperature scaling (T=1.299 khớp trên val) | T12 seed0 (convnext_tiny) | 1.0000 | 0.9735 | 0.9797 | 0.0050 | 0.9543 | 0.9366 | 10.9747 | 15.7904 | 17.9589 | 521.3238 | ECE trước 0.0079 -> sau 0.0050 (2-fold); in-sample 0.0045 | 1.0000 |
| I08 | Gộp BN |  |  |  |  |  |  |  |  |  |  |  | không áp dụng: convnext_tiny không có BatchNorm |  |
| I08b | AMP (autocast FP16) | T12 seed0 (convnext_tiny) | 1.0000 | 0.9735 | 0.9797 | 0.0081 | 0.9543 | 0.9366 | 14.7613 | 19.1017 | 21.1418 | 931.1813 |  | 1.3450 |

## Final

| exp_id | cấu hình | seed | macro-F1 val | macro-F1 test | top-1 test | balanced acc test | ECE test | recall Chinee apple test | recall Snake weed test |
|---|---|---|---|---|---|---|---|---|---|
| F01 | convnext_tiny + T12 recipe {"aug": "trivial", "ema_decay": 0.999} + I04_288 (none, prob) + temperature scaling | 0 | 0.9787 | 0.9766 | 0.9812 | 0.9737 | 0.0058 | 0.9602 | 0.9755 |
| F01 | convnext_tiny + T12 recipe {"aug": "trivial", "ema_decay": 0.999} + I04_288 (none, prob) + temperature scaling | 1 | 0.9759 | 0.9783 | 0.9826 | 0.9736 | 0.0032 | 0.9425 | 0.9510 |
| F01 | convnext_tiny + T12 recipe {"aug": "trivial", "ema_decay": 0.999} + I04_288 (none, prob) + temperature scaling | 2 | 0.9740 | 0.9789 | 0.9832 | 0.9773 | 0.0040 | 0.9646 | 0.9559 |
| F01 mean ± std | 3 seed, std mẫu ddof=1 | all | 0.9762 ± 0.0024 | 0.9779 ± 0.0012 | 0.9823 ± 0.0010 | 0.9749 ± 0.0021 | 0.0043 ± 0.0013 | 0.9558 ± 0.0117 | 0.9608 ± 0.0130 |
| T00 | convnext_tiny + công thức nền T00 + 1 view (I00), không TS | 0 | 0.9681 | 0.9690 | 0.9752 | 0.9684 | 0.0106 | 0.9248 | 0.9559 |
| T00 | convnext_tiny + công thức nền T00 + 1 view (I00), không TS | 1 | 0.9721 | 0.9696 | 0.9758 | 0.9698 | 0.0106 | 0.9336 | 0.9412 |
| T00 | convnext_tiny + công thức nền T00 + 1 view (I00), không TS | 2 | 0.9708 | 0.9699 | 0.9760 | 0.9744 | 0.0108 | 0.9558 | 0.9510 |
| T00 mean ± std | 3 seed, std mẫu ddof=1 | all | 0.9703 ± 0.0020 | 0.9695 ± 0.0004 | 0.9757 ± 0.0004 | 0.9709 ± 0.0031 | 0.0106 ± 0.0001 | 0.9381 ± 0.0160 | 0.9493 ± 0.0075 |
| F01_uncal | F01 trước temperature scaling | 0 |  | 0.9766 | 0.9812 | 0.9737 | 0.0076 | 0.9602 | 0.9755 |
| F01_uncal | F01 trước temperature scaling | 1 |  | 0.9783 | 0.9826 | 0.9736 | 0.0037 | 0.9425 | 0.9510 |
| F01_uncal | F01 trước temperature scaling | 2 |  | 0.9789 | 0.9832 | 0.9773 | 0.0065 | 0.9646 | 0.9559 |
| F01_uncal mean ± std | 3 seed, std mẫu ddof=1 | all |  | 0.9779 ± 0.0012 | 0.9823 ± 0.0010 | 0.9749 ± 0.0021 | 0.0059 ± 0.0020 | 0.9558 ± 0.0117 | 0.9608 ± 0.0130 |

## PerClass

| cấu hình | lớp | số ảnh test | precision | recall | F1 |
|---|---|---|---|---|---|
| F01 | Chinee apple | 226 | 0.9774 ± 0.0045 | 0.9558 ± 0.0117 | 0.9664 ± 0.0069 |
| F01 | Lantana | 213 | 0.9920 ± 0.0027 | 0.9687 ± 0.0027 | 0.9802 ± 0.0014 |
| F01 | Parkinsonia | 207 | 0.9731 ± 0.0027 | 0.9919 ± 0.0028 | 0.9825 ± 0.0028 |
| F01 | Parthenium | 205 | 0.9983 ± 0.0029 | 0.9724 ± 0.0123 | 0.9851 ± 0.0075 |
| F01 | Prickly acacia | 213 | 0.9610 ± 0.0032 | 0.9656 ± 0.0151 | 0.9633 ± 0.0091 |
| F01 | Rubber vine | 202 | 0.9884 ± 0.0057 | 0.9769 ± 0.0057 | 0.9826 ± 0.0001 |
| F01 | Siam weed | 215 | 0.9877 ± 0.0070 | 0.9907 ± 0.0047 | 0.9892 ± 0.0013 |
| F01 | Snake weed | 204 | 0.9687 ± 0.0029 | 0.9608 ± 0.0130 | 0.9647 ± 0.0073 |
| F01 | Negative | 1822 | 0.9838 ± 0.0022 | 0.9912 ± 0.0019 | 0.9875 ± 0.0008 |
| T00 | Chinee apple | 226 | 0.9608 ± 0.0065 | 0.9381 ± 0.0160 | 0.9492 ± 0.0075 |
| T00 | Lantana | 213 | 0.9605 ± 0.0239 | 0.9781 ± 0.0027 | 0.9691 ± 0.0109 |
| T00 | Parkinsonia | 207 | 0.9807 ± 0.0046 | 0.9791 ± 0.0074 | 0.9798 ± 0.0015 |
| T00 | Parthenium | 205 | 0.9917 ± 0.0028 | 0.9756 ± 0.0098 | 0.9836 ± 0.0052 |
| T00 | Prickly acacia | 213 | 0.9229 ± 0.0082 | 0.9734 ± 0.0054 | 0.9475 ± 0.0019 |
| T00 | Rubber vine | 202 | 0.9850 ± 0.0000 | 0.9736 ± 0.0029 | 0.9793 ± 0.0015 |
| T00 | Siam weed | 215 | 0.9787 ± 0.0137 | 0.9891 ± 0.0027 | 0.9838 ± 0.0060 |
| T00 | Snake weed | 204 | 0.9511 ± 0.0137 | 0.9493 ± 0.0075 | 0.9501 ± 0.0032 |
| T00 | Negative | 1822 | 0.9850 ± 0.0040 | 0.9815 ± 0.0030 | 0.9832 ± 0.0005 |

## Latency

| cấu hình | GPU / thiết bị | dtype | batch | gộp BN | p50 (ms) | p95 (ms) | p99 (ms) | ảnh/s | torch |
|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny 224px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 10.9747 | 15.7904 | 17.9589 | 91.1191 | 2.13.0+cu130 |
| convnext_tiny 224px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 61.3822 | 62.5499 | 65.1270 | 521.3238 | 2.13.0+cu130 |
| convnext_tiny 224px K=2 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 20.7831 | 29.3139 | 32.1892 | 48.1160 | 2.13.0+cu130 |
| convnext_tiny 224px K=2 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 122.8782 | 125.0894 | 128.9986 | 260.4205 | 2.13.0+cu130 |
| convnext_tiny 224px K=5 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 57.2441 | 64.4243 | 74.2343 | 17.4690 | 2.13.0+cu130 |
| convnext_tiny 224px K=5 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 307.7461 | 311.0258 | 336.6795 | 103.9818 | 2.13.0+cu130 |
| convnext_tiny 224px K=10 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 159.8995 | 217.5963 | 223.3347 | 6.2539 | 2.13.0+cu130 |
| convnext_tiny 224px K=10 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 605.1131 | 640.3914 | 653.6307 | 52.8827 | 2.13.0+cu130 |
| convnext_tiny 192px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 3.9728 | 4.4977 | 4.5596 | 251.7085 | 2.13.0+cu130 |
| convnext_tiny 192px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 44.0745 | 45.3134 | 46.1545 | 726.0442 | 2.13.0+cu130 |
| convnext_tiny 256px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 3.6784 | 4.4703 | 5.3812 | 271.8610 | 2.13.0+cu130 |
| convnext_tiny 256px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 79.2057 | 81.6086 | 82.6231 | 404.0113 | 2.13.0+cu130 |
| convnext_tiny 288px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 3.9595 | 4.8739 | 6.0538 | 252.5571 | 2.13.0+cu130 |
| convnext_tiny 288px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 105.1752 | 112.2262 | 113.5629 | 304.2541 | 2.13.0+cu130 |
| convnext_tiny 320px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 4.5530 | 6.6855 | 7.3021 | 219.6354 | 2.13.0+cu130 |
| convnext_tiny 320px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 141.2842 | 152.2665 | 153.4617 | 226.4938 | 2.13.0+cu130 |
| B04 swin_tiny_patch4_window7_224 (ensemble member) 224px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 7.4424 | 12.8997 | 14.3290 | 134.3653 | 2.13.0+cu130 |
| B04 swin_tiny_patch4_window7_224 (ensemble member) 224px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 83.2326 | 100.0152 | 100.7856 | 384.4647 | 2.13.0+cu130 |
| B03 deit_small_patch16_224 (ensemble member) 224px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 10.0392 | 10.9784 | 13.0834 | 99.6090 | 2.13.0+cu130 |
| B03 deit_small_patch16_224 (ensemble member) 224px K=1 | NVIDIA GeForce RTX 4060 | fp32 | 32 | False | 352.2208 | 370.0442 | 379.4617 | 90.8521 | 2.13.0+cu130 |
| convnext_tiny 224px K=1 | NVIDIA GeForce RTX 4060 | amp | 1 | False | 14.7613 | 19.1017 | 21.1418 | 67.7447 | 2.13.0+cu130 |
| convnext_tiny 224px K=1 | NVIDIA GeForce RTX 4060 | amp | 32 | False | 34.3650 | 36.2548 | 36.4594 | 931.1813 | 2.13.0+cu130 |
| convnext_tiny 224px K=1 (CPU) | CPU: 12th Gen Intel(R) Core(TM) i7-12700K | fp32 | 1 | False | 93.6141 | 98.7089 | 101.1310 | 10.6822 | 2.13.0+cu130 |
| convnext_tiny 224px ONNX Runtime (CPU) | CPU: 12th Gen Intel(R) Core(TM) i7-12700K | fp32 | 1 | onnx | 395.9432 | 452.0223 | 464.5686 | 2.5256 | onnxruntime 1.23.2 |
| B01 resnet50 (Bước 1, sơ bộ) | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 5.1513 | 5.9295 |  | 194.1258 |  |
| B02 convnext_tiny (Bước 1, sơ bộ) | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 3.5925 | 4.4565 |  | 278.3577 |  |
| B03 deit_small_patch16_224 (Bước 1, sơ bộ) | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 3.2075 | 4.2533 |  | 311.7644 |  |
| B04 swin_tiny_patch4_window7_224 (Bước 1, sơ bộ) | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 6.8025 | 7.8175 |  | 147.0048 |  |
| B05 efficientnet_b0 (Bước 1, sơ bộ) | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 6.6538 | 7.6875 |  | 150.2912 |  |
| B06 mobilenetv3_large_100 (Bước 1, sơ bộ) | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 5.3401 | 6.3180 |  | 187.2642 |  |
| F01 convnext_tiny none K=1 288px (chung kết) | NVIDIA GeForce RTX 4060 | fp32 | 1 | False | 11.1191 | 15.0369 | 21.5988 | 89.9353 |  |

## Summary

| hạng | exp_id | loại | mô tả | macro-F1 val | top-1 val | p50 batch-1 (ms) |
|---|---|---|---|---|---|---|
| 1 | I04_288 | inference | 1 view ở độ phân giải 288 (train 224) | 0.9787 | 0.9831 | 3.9595 |
| 2 | I04_256 | inference | 1 view ở độ phân giải 256 (train 224) | 0.9778 | 0.9829 | 3.6784 |
| 3 | I05 | inference | Ensemble 3 mô hình (trung bình xác suất) | 0.9756 | 0.9823 | 28.4563 |
| 4 | I04_320 | inference | 1 view ở độ phân giải 320 (train 224) | 0.9746 | 0.9803 | 4.5530 |
| 5 | I02b | inference | TTA 10 crop (5 crop + lật), gộp xác suất | 0.9746 | 0.9806 | 159.8995 |
| 6 | I03b | inference | TTA 10 crop, gộp logit | 0.9743 | 0.9803 | 159.8995 |
| 7 | I01 | inference | TTA lật ngang, gộp xác suất | 0.9737 | 0.9800 | 20.7831 |
| 8 | I03 | inference | TTA lật ngang, gộp logit | 0.9737 | 0.9800 | 20.7831 |
| 9 | T12 | training | kết hợp: T03, T11 | 0.9735 | 0.9797 |  |
| 10 | I00 | inference | 1 view (resize + center crop) | 0.9735 | 0.9797 | 10.9747 |
|  | TEST | chung kết vs mốc | F01 (chung kết) | 0.9779 ± 0.0012 (test) | 0.9823 ± 0.0010 (test) |  |
|  | TEST | chung kết vs mốc | T00 + I00 (mốc) | 0.9695 ± 0.0004 (test) | 0.9757 ± 0.0004 (test) |  |
|  | TEST | Δ | Δ macro-F1 = +0.0084, std lớn hơn s = 0.0012 | vượt nhiễu |  |  |

## Lựa chọn giữa các bước (results/decisions.json)

```json
{
  "chosen_backbone": "B02",
  "reasons": {
    "backbone": "B02 (convnext_tiny): macro-F1 val 0.9681 (tốt nhất 0.9681), p50 batch 1 = 3.59 ms; nhóm ngang nhau (cách tốt nhất < 0.01): B02, B04 -> chọn cái nhanh nhất",
    "combo": "ít hơn 2 trục thắng rõ: lấy 2 trục có Δ dương lớn nhất: T03 (aug=trivial (TrivialAugmentWide), Δ=+0.0048), T11 (ema_decay=0.999 (đánh giá bằng trọng số EMA), Δ=+0.0006)",
    "final": "macro-F1 val (seed 0): T00 0.9681, T03 0.9729, T12 0.9735 -> T12",
    "inference": "I04_288 có macro-F1 val 0.9787 (I00 0.9735); luật: cao nhất trên val, phải hơn I00 >= 0.002, sau đó temperature scaling khớp trên val"
  },
  "profile": "gpu",
  "combo": {
    "aug": "trivial",
    "ema_decay": 0.999
  },
  "combo_from": [
    "T03",
    "T11"
  ],
  "final_recipe": {
    "aug": "trivial",
    "ema_decay": 0.999
  },
  "final_source": "T12",
  "inference": {
    "exp_id": "I04_288",
    "method": "none",
    "space": "prob",
    "img_size": 288,
    "temperature_scaling": true
  }
}
```
