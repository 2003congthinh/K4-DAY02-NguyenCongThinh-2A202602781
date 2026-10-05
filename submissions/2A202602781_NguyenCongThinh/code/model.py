"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Đã hoàn thiện mọi hàm của bộ khung `starter/`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

# imports for the implementation below.
import torch
import torch.nn as nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}

# backbone nhỏ dùng cho ngân sách CPU của bài nộp này
# (máy không có GPU CUDA; xem report.md mục 2). Vẫn đủ 4 ràng buộc của GUIDE mục 2.1.
CPU_BACKBONES = {
    "resnet18": "resnet18",                     # họ ResNet (mốc)
    "convnext_atto": "convnext_atto",           # họ ConvNeXt
    "deit_tiny": "deit_tiny_patch16_224",       # transformer (DeiT)
    "mobilenetv3": "mobilenetv3_large_100",     # mạng nhẹ
    "efficientnet_b0": "efficientnet_b0",       # mạng nhẹ
}
# họ kiến trúc cần biết kích thước ảnh lúc tạo model (positional embedding / cửa sổ attention)
_NEEDS_IMG_SIZE = ("vit", "deit", "swin", "eva", "beit")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", img_size: int | None = None,
                drop_path_rate: float = 0.0, head_init: str = "zero"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    TODO:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)
    """
    import timm

    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init!r}")
    if init == "scratch":
        pretrained = False
    kw = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate:
        kw["drop_path_rate"] = drop_path_rate
    if img_size is not None and any(k in name for k in _NEEDS_IMG_SIZE):
        kw["img_size"] = img_size
    model = timm.create_model(name, **kw)

    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch = cfg.get("architecture", name)
    tag = cfg.get("tag")
    model.weight_tag = (f"{arch}.{tag}" if tag else arch) if pretrained else f"{name} (scratch)"
    if head_init == "zero":
        head = model.get_classifier()
        nn.init.zeros_(head.weight)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
    elif head_init != "default":
        raise ValueError(f"head_init không hợp lệ: {head_init!r}")
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    TODO:
      - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
      - lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
        Hãy nghĩ nơi nào trong train loop phải gọi lại model.train() mà vẫn giữ BN ở eval.
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen_backbone = True


def head_module_names(model) -> set[str]:
    """Tên (theo named_modules) của head và các module con của nó.

    Input : model (timm nn.Module). Output: set[str] tên module thuộc head (dùng để giữ backbone ở eval).
    """
    head = model.get_classifier()
    return {n for n, m in model.named_modules() if m is head or any(m is c for c in head.modules())}


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    TODO:
      - bỏ qua tham số requires_grad == False
      - trả về list[dict] dạng {"params": [...], "lr": ..., "weight_decay": ...}
      - (trục E) mở rộng: LR theo tầng nếu bạn muốn thử
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    buckets = {"backbone_decay": [], "backbone_no_decay": [], "head": [], "head_no_decay": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        no_decay = p.ndim <= 1
        key = ("head" if is_head else "backbone") + ("_no_decay" if no_decay else ("" if is_head else "_decay"))
        buckets[key].append(p)
    settings = {
        "backbone_decay": (lr_backbone, weight_decay),
        "backbone_no_decay": (lr_backbone, 0.0),
        "head": (lr_head, weight_decay),
        "head_no_decay": (lr_head, 0.0),
    }
    return [{"name": k, "params": v, "lr": settings[k][0], "weight_decay": settings[k][1]}
            for k, v in buckets.items() if v]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng. TODO."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    TODO: dùng thư viện đếm (fvcore, ptflops, thop...) hoặc tự đếm bằng hook.
    Ghi rõ công cụ đã dùng; số có thể lệch vài phần trăm giữa các công cụ.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 2 / 1e9
