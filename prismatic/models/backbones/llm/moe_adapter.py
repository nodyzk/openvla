"""
MoE Adapter module.
- AdapterExpert: single bottleneck FFN expert
- TaskRouter: linear softmax router
- MoEAdapter: full MoE block (parallel to LLaMA FFN)
- LlamaDecoderLayerWithMoE: wraps an existing frozen LLaMA layer
"""
from __future__ import annotations
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterExpert(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.up   = nn.Linear(bottleneck_dim, d_model, bias=False)
        self.act  = nn.GELU()
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)   # zero-init → identity at start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class TaskRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, T, E)
        return F.softmax(self.linear(x), dim=-1)


class MoEAdapter(nn.Module):
    """
    Parallel MoE adapter inserted alongside the FFN of a LLaMA layer.
    For this sanity-check experiment: num_tasks=1, single shared router.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        bottleneck_dim: int = 256,
        num_tasks: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        self.experts = nn.ModuleList([
            AdapterExpert(d_model, bottleneck_dim) for _ in range(num_experts)
        ])

        # One router per task — for sanity check, just one
        self.task_routers = nn.ModuleDict({
            str(i): TaskRouter(d_model, num_experts) for i in range(num_tasks)
        })

        self._current_task_id: int = 0
        self._inference_router: Optional[TaskRouter] = None

    def set_train_task(self, task_id: int) -> None:
        self._current_task_id = task_id
        self._inference_router = None

    def add_task_router(self, task_id: int) -> None:
        key = str(task_id)
        if key not in self.task_routers:
            device = next(self.parameters()).device
            dtype  = next(self.parameters()).dtype
            self.task_routers[key] = TaskRouter(
                self.d_model, self.num_experts
            ).to(device=device, dtype=dtype)
        self._inference_router = None

    def build_inference_router(self, strategy: str = "mean") -> None:
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        merged = TaskRouter(self.d_model, self.num_experts).to(device=device, dtype=dtype)
        with torch.no_grad():
            stacked = torch.stack(
                [r.linear.weight.data for r in self.task_routers.values()], dim=0
            )
            merged.linear.weight.data = stacked.mean(0)
        self._inference_router = merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pick router
        if self.training:
            router = self.task_routers[str(self._current_task_id)]
        else:
            if self._inference_router is None:
                self.build_inference_router()
            router = self._inference_router

        # Ensure x matches the module's parameter dtype (e.g. bfloat16 vs float32)
        input_dtype = x.dtype
        param_dtype = next(self.parameters()).dtype
        x = x.to(dtype=param_dtype)

        weights = router(x)                          # (B, T, E)
        expert_outs = torch.stack(
            [e(x) for e in self.experts], dim=-1
        )                                            # (B, T, D, E)
        out = (expert_outs * weights.unsqueeze(-2)).sum(-1)  # (B, T, D)
        return out.to(dtype=input_dtype)             # restore original activation dtype


class LlamaDecoderLayerWithMoE(nn.Module):
    """
    Wraps a frozen LlamaDecoderLayer and adds a MoEAdapter parallel to the FFN.

    Original:  out = layer(x)
    Modified:  out = layer(x) + moe_adapter(norm2(x_after_attn))

    We capture the FFN input via a pre-hook on original_layer.mlp.
    """
    def __init__(self, original_layer: nn.Module, moe_adapter: MoEAdapter):
        super().__init__()
        self.original_layer = original_layer
        self.moe_adapter    = moe_adapter

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        **kwargs,
    ):
        # Capture the input to MLP (post-norm2 hidden state) via pre-hook
        _mlp_input: List[torch.Tensor] = []

        def _hook(module, args, kwargs_fwd):
            inp = args[0] if args else kwargs_fwd.get(
                "hidden_states", kwargs_fwd.get("x")
            )
            _mlp_input.append(inp.detach() if not self.training else inp)

        hook = self.original_layer.mlp.register_forward_pre_hook(
            _hook, with_kwargs=True
        )
        try:
            outputs = self.original_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            hook.remove()

        layer_out = outputs[0]

        if _mlp_input:
            layer_out = layer_out + self.moe_adapter(_mlp_input[0])

        return (layer_out,) + outputs[1:]