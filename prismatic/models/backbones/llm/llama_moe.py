"""
Utilities to inject, freeze, save, and load MoE adapters into OpenVLA's LLaMA backbone.
"""
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn

from .moe_adapter import MoEAdapter, LlamaDecoderLayerWithMoE


# ── Injection ──────────────────────────────────────────────────────────────────

def inject_moe_adapters(
    model: nn.Module,
    num_experts: int = 4,
    bottleneck_dim: int = 256,
    num_tasks: int = 1,
    adapter_layer_fraction: float = 1 / 3,
) -> List[MoEAdapter]:
    """
    Replace the last `adapter_layer_fraction` of LLaMA decoder layers with
    LlamaDecoderLayerWithMoE wrappers. Returns list of MoEAdapter instances.
    """
    layers = _get_llama_layers(model)
    n      = len(layers)
    start  = int(n * (1.0 - adapter_layer_fraction))  # e.g. 21 for 32 layers
    d_model = _infer_d_model(layers[0])

    adapters: List[MoEAdapter] = []
    for idx in range(start, n):
        adapter = MoEAdapter(
            d_model=d_model,
            num_experts=num_experts,
            bottleneck_dim=bottleneck_dim,
            num_tasks=num_tasks,
        )
        layers[idx] = LlamaDecoderLayerWithMoE(layers[idx], adapter)
        adapters.append(adapter)

    print(
        f"[MoE] Injected adapters into layers {start}–{n-1} "
        f"({len(adapters)} layers | {num_experts} experts | "
        f"bottleneck={bottleneck_dim} | d_model={d_model})"
    )
    return adapters


# ── Freeze / Unfreeze ──────────────────────────────────────────────────────────

def freeze_base_model(model: nn.Module) -> None:
    """Freeze every parameter in the model (call before inject)."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_moe_adapters(adapters: List[MoEAdapter]) -> None:
    """Unfreeze only adapter experts + task routers."""
    for adapter in adapters:
        for p in adapter.parameters():
            p.requires_grad = True


# ── Task management ────────────────────────────────────────────────────────────

def set_train_task_all(adapters: List[MoEAdapter], task_id: int) -> None:
    for a in adapters:
        a.set_train_task(task_id)


def add_task_router_all(adapters: List[MoEAdapter], task_id: int) -> None:
    for a in adapters:
        a.add_task_router(task_id)


def build_inference_router_all(
    adapters: List[MoEAdapter], strategy: str = "mean"
) -> None:
    for a in adapters:
        a.build_inference_router(strategy)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_moe_checkpoint(model: nn.Module, path: str) -> None:
    """Save only adapter weights (small, task-incremental checkpoints)."""
    state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "moe_adapter" in k
    }
    torch.save(state, path)
    print(f"[MoE] Saved {len(state)} tensors → {path}")


def load_moe_checkpoint(model: nn.Module, path: str) -> None:
    """Load adapter weights into an already-injected model."""
    state     = torch.load(path, map_location="cpu")
    cur_state = model.state_dict()
    matched   = 0
    for k, v in state.items():
        if k in cur_state:
            cur_state[k].copy_(v)
            matched += 1
    print(f"[MoE] Loaded {matched}/{len(state)} tensors from {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_llama_layers(model: nn.Module) -> nn.ModuleList:
    """Navigate common nesting paths to find the LLaMA decoder layer list."""
    for path in [
        "language_model.model.layers",   # HF OpenVLA
        "llm_backbone.model.layers",      # Prismatic
        "model.layers",
        "model.model.layers",
    ]:
        obj, ok = model, True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            print(f"[MoE] Found LLaMA layers at: {path} (n={len(obj)})")
            return obj
    raise AttributeError(
        "Could not locate LLaMA decoder layers. "
        "Add your model's path to _get_llama_layers()."
    )


def _infer_d_model(layer: nn.Module) -> int:
    # Unwrap if already wrapped
    if hasattr(layer, "original_layer"):
        layer = layer.original_layer
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "gate_proj"):
            return mlp.gate_proj.in_features
        if hasattr(mlp, "fc1"):
            return mlp.fc1.in_features
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
        return layer.self_attn.q_proj.in_features
    raise ValueError("Cannot infer d_model from layer.")


def print_trainable_params(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[MoE] Trainable params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.3f}%)"
    )