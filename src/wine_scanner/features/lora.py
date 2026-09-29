"""LoRA-адаптер для attention-проекций vision-башни SigLIP.

Формат весов совместим с обученным ранее models/lora_v2/lora.pt:
ключи `{layer}.{proj}.lora_A|lora_B`, r=8, alpha=16. Адаптер можно включать
и выключать на лету (`enabled`) — одна модель в памяти обслуживает и
каталог (без адаптера), и запросы (с адаптером).
"""

from __future__ import annotations

from pathlib import Path

import torch


class LoraSwitch:
    """Общий флаг для всех LoRA-слоёв модели."""

    def __init__(self) -> None:
        self.enabled = True


class LoraLinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, switch: LoraSwitch, r: int = 8,
                 alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.switch = switch
        self.scaling = alpha / r
        dtype, device = base.weight.dtype, base.weight.device
        self.lora_A = torch.nn.Parameter(
            (torch.randn(base.in_features, r, device=device) * 0.02).to(dtype)
        )
        self.lora_B = torch.nn.Parameter(torch.zeros(r, base.out_features, device=device,
                                                     dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.switch.enabled:
            out = out + (x @ self.lora_A) @ self.lora_B * self.scaling
        return out


def wrap_vision(model, switch: LoraSwitch, r: int = 8, alpha: float = 16.0):
    """Заменяет nn.Linear в self_attn каждого слоя vision-энкодера на LoraLinear."""
    layers = []
    for index, layer in enumerate(model.vision_model.encoder.layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        for name, module in list(attention.named_children()):
            if isinstance(module, torch.nn.Linear):
                wrapped = LoraLinear(module, switch, r=r, alpha=alpha)
                setattr(attention, name, wrapped)
                layers.append((index, name, wrapped))
    return layers


def apply_lora(model, path: Path) -> LoraSwitch:
    switch = LoraSwitch()
    state = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for index, name, layer in wrap_vision(model, switch):
            layer.lora_A.copy_(state[f"{index}.{name}.lora_A"].to(layer.lora_A.dtype))
            layer.lora_B.copy_(state[f"{index}.{name}.lora_B"].to(layer.lora_B.dtype))
    for parameter in model.parameters():
        parameter.requires_grad = False
    return switch
