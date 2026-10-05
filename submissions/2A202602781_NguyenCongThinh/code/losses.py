"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Đã hoàn thiện mọi hàm/lớp của bộ khung `starter/`.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

# imports for the implementation below.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    TODO: tạo đúng loss, hoặc gọi các lớp bên dưới.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight=tensor[9] (dùng class_weights trên số ảnh train)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(kw["weight"], dtype=torch.float32))
    raise ValueError(f"loss không hợp lệ: {kind!r}")


class LabelSmoothingCE(nn.Module):  # TODO: kế thừa torch.nn.Module
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    TODO: tự cài đặt hoặc dùng torch.nn.CrossEntropyLoss(label_smoothing=eps), rồi ghi rõ
    bạn đã chọn cách nào. Kiểm tra: eps = 0 phải cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        # TỰ CÀI ĐẶT (không dùng tham số label_smoothing của PyTorch).
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):  # TODO: kế thừa torch.nn.Module
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    TODO:
      - tính log_softmax, lấy p_t của lớp đúng, nhân (1 - p_t)^gamma, lấy trung bình batch
      - alpha: None hoặc vector trọng số theo lớp
    BẮT BUỘC viết một kiểm tra nhỏ: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt).clamp(min=0.0) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp

    TODO: trả về tensor độ dài 9. Chỉ dùng số liệu của train, không dùng val hay test.
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mỗi lớp phải có ít nhất 1 ảnh train")
    if beta == 0:
        w = 1.0 / n
    else:
        if not 0.0 < beta < 1.0:
            raise ValueError("beta phải trong [0, 1)")
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    w = w * len(n) / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]

    TODO: tự cài đặt. Kiểm tra bằng mắt: vẽ vài ảnh sau khi trộn và in lam.
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        H, W = x.shape[-2:]
        cut = np.sqrt(1.0 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
        x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(H * W)
    else:
        raise ValueError(f"mode không hợp lệ: {mode!r}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    TODO. Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
