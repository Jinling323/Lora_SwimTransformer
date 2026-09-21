import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import (
    ShiftedWindowAttention,
    SwinTransformer,
    shifted_window_attention,
)

from .transformer_cosine_multibatch import TransformerEncoder, TransformerEncoderLayer


__all__ = ["swin_large_trans"]


class LoRAShiftedWindowAttention(ShiftedWindowAttention):
    """Shifted-window attention with low-rank qkv and projection updates."""

    def __init__(self, dim, window_size, shift_size, num_heads, rank=8,
                 alpha=16.0, attention_dropout=0.0, dropout=0.0):
        super().__init__(
            dim=dim,
            window_size=window_size,
            shift_size=shift_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.scaling = alpha / rank
        self.qkv_lora_a = nn.Parameter(torch.empty(rank, dim))
        self.qkv_lora_b = nn.Parameter(torch.zeros(3 * dim, rank))
        self.proj_lora_a = nn.Parameter(torch.empty(rank, dim))
        self.proj_lora_b = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.qkv_lora_a, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.proj_lora_a, a=5 ** 0.5)

    def forward(self, x):
        relative_position_bias = self.get_relative_position_bias()
        qkv_weight = self.qkv.weight + self.scaling * (
            self.qkv_lora_b @ self.qkv_lora_a
        )
        proj_weight = self.proj.weight + self.scaling * (
            self.proj_lora_b @ self.proj_lora_a
        )
        return shifted_window_attention(
            x,
            qkv_weight,
            proj_weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


def _build_swin_large():
    """Build the Swin-Large-Patch4-Window12-384 configuration used by CCST."""
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=[12, 12],
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        stochastic_depth_prob=0.1,
        num_classes=1000,
    )


def _map_official_key(key):
    """Map official Microsoft Swin checkpoint keys to torchvision names."""
    for prefix in ("module.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix):]

    if key.startswith("patch_embed.proj."):
        return key.replace("patch_embed.proj.", "features.0.0.", 1)
    if key.startswith("patch_embed.norm."):
        return key.replace("patch_embed.norm.", "features.0.2.", 1)
    if not key.startswith("layers."):
        return None

    parts = key.split(".")
    stage = int(parts[1])
    if parts[2] == "blocks":
        block = parts[3]
        suffix = ".".join(parts[4:])
        suffix = suffix.replace("mlp.fc1.", "mlp.0.")
        suffix = suffix.replace("mlp.fc2.", "mlp.3.")
        return f"features.{2 * stage + 1}.{block}.{suffix}"
    if parts[2] == "downsample" and stage < 3:
        suffix = ".".join(parts[3:])
        return f"features.{2 * stage + 2}.{suffix}"
    return None


def _load_swin_22k(backbone, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "CCST requires the Swin-Large 22K checkpoint, but it was not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model_state = backbone.state_dict()
    mapped = {}
    for key, value in state_dict.items():
        mapped_key = _map_official_key(key)
        if mapped_key in model_state and model_state[mapped_key].shape == value.shape:
            mapped[mapped_key] = value

    if not mapped:
        raise RuntimeError("No compatible Swin-Large weights were found in the checkpoint")

    incompatible = backbone.load_state_dict(mapped, strict=False)
    logging.info(
        "loaded %d Swin-Large tensors from %s (%d model tensors left initialized)",
        len(mapped), checkpoint_path, len(incompatible.missing_keys)
    )


def _add_lora_to_swin(module, rank, alpha, dropout):
    if dropout != 0:
        logging.warning(
            "torchvision fused shifted-window attention does not support standard "
            "LoRA input dropout; lora_dropout is ignored"
        )
    for name, child in list(module.named_children()):
        if isinstance(child, ShiftedWindowAttention):
            replacement = LoRAShiftedWindowAttention(
                dim=child.qkv.in_features,
                window_size=child.window_size,
                shift_size=child.shift_size,
                num_heads=child.num_heads,
                rank=rank,
                alpha=alpha,
                attention_dropout=child.attention_dropout,
                dropout=child.dropout,
            ).to(device=child.qkv.weight.device, dtype=child.qkv.weight.dtype)
            replacement.load_state_dict(child.state_dict(), strict=False)
            for parameter in replacement.qkv.parameters():
                parameter.requires_grad = False
            for parameter in replacement.proj.parameters():
                parameter.requires_grad = False
            replacement.relative_position_bias_table.requires_grad = False
            setattr(module, name, replacement)
        else:
            _add_lora_to_swin(child, rank, alpha, 0.0)


class SwinLargeTrans(nn.Module):
    """CCST Swin-Large backbone with the original density-counting backend."""

    def __init__(self, stage="baseline", pretrained_path="", baseline_checkpoint="",
                 lora_rank=8, lora_alpha=16.0, lora_dropout=0.0):
        super().__init__()
        if stage not in ("baseline", "lora"):
            raise ValueError(f"unknown Swin training stage: {stage}")
        self.stage = stage
        self.backbone = _build_swin_large()
        if stage == "baseline" and pretrained_path:
            _load_swin_22k(self.backbone, pretrained_path)

        # Swin-Large stage 4 has 1536 channels; the existing backend expects 512.
        self.channel_adapter = nn.Conv2d(1536, 512, kernel_size=1)

        d_model = 512
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=None)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        if stage == "lora":
            if not baseline_checkpoint:
                raise ValueError("LoRA stage requires a baseline checkpoint")
            checkpoint = torch.load(baseline_checkpoint, map_location="cpu",
                                    weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            if not isinstance(state_dict, dict):
                raise ValueError("baseline checkpoint does not contain a model state dict")
            if any("lora_" in key for key in state_dict):
                raise ValueError("expected a baseline checkpoint without LoRA parameters")
            # Strict loading also checks the adapter, Transformer, and density head.
            self.load_state_dict(state_dict, strict=True)
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            _add_lora_to_swin(self.backbone, lora_rank, lora_alpha,
                              lora_dropout)

    def forward(self, x):
        _, _, input_h, input_w = x.shape
        if input_h != 384 or input_w != 384:
            raise ValueError(
                "CCST Swin-Large-Patch4-Window12 expects 384x384 input, "
                f"but received {input_h}x{input_w}"
            )

        # torchvision Swin keeps features in channels-last format.
        x = self.backbone.features(x)  # [B, 12, 12, 1536]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.channel_adapter(x)

        batch_size, channels, height, width = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, features = self.encoder(x, (height, width))
        x = x.permute(1, 2, 0).reshape(batch_size, channels, height, width)

        x = F.interpolate(
            x,
            size=(input_h // 16, input_w // 16),
            mode="bilinear",
            align_corners=False,
        )
        x = self.reg_layer_0(x)
        return torch.relu(x), features


def swin_large_trans(stage="baseline", pretrained_path="",
                     baseline_checkpoint="", lora_rank=8, lora_alpha=16.0,
                     lora_dropout=0.0):
    return SwinLargeTrans(
        stage=stage,
        pretrained_path=pretrained_path,
        baseline_checkpoint=baseline_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
