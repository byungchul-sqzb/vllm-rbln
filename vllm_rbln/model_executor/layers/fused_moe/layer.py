# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward,
    _moe_forward_shared,
)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def _rbln_moe_select_forward(self):
    """Use the unwrapped MoE forward functions instead of the opaque
    ``torch.ops.vllm.moe_forward{,_shared}`` custom ops.

    On RBLN the traced decode graph is lowered by rebel-compiler (not
    Inductor), which cannot lower the opaque MoE custom op -- with it in the
    graph, decode compile fails ("moe_forward_shared not implemented") and
    silently falls back to CPU eager, losing NPU acceleration. Calling
    ``_forward_impl`` directly (via the raw module-level functions) lets the
    MoE decompose into primitive ops rebel-compiler can lower. Upstream
    already does exactly this for CPU/TPU in ``MoERunner._select_forward``;
    the opaque-op path is only load-bearing for the MoE-LoRA dual-stream path,
    which the RBLN Unlimited-OCR runtime does not use.
    """
    return _moe_forward if self._shared_experts is None else _moe_forward_shared


# Applied at import (register_ops) time, before any MoERunner is constructed
# during model build, so every runner picks up the unwrapped entry.
MoERunner._select_forward = _rbln_moe_select_forward


fused_moe_upstream__init__ = FusedMoE.__init__


def fused_moe_custom__init__(self, *args, **kwargs):
    fused_moe_upstream__init__(self, *args, **kwargs)

    # RBLN overrides the fused expert kernel only.  vLLM's overlapped shared-
    # expert path expects FusedMoE.forward to return ``(shared_out, fused_out)``,
    # while this backend returns the fused routed-expert tensor.  Disable the
    # overlap optimization so SharedFusedMoE computes shared experts in the
    # standard path and adds them with the routed output, preserving semantics.
    if getattr(self, "_shared_experts", None) is not None:
        self.use_overlapped = False

    self.expert_map_const = (
        self.expert_map.tolist() if self.expert_map is not None else None
    )


# Define custom_moe_glu op based on environment variable
# VLLM_RBLN_MOE_USE_OPT_KERNEL: uses pre-masked routing weights + hidden_act
# VLLM_RBLN_MOE_CUSTOM_KERNEL: uses expert_select_count parameter
if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:

    @torch.library.custom_op(
        "rbln_custom_ops::custom_moe_glu",
        mutates_args=(),
    )
    def custom_moe_glu(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        hidden_act: str,
        expert_map: torch.Tensor | None = None,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        n_group: int | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        """RBLN compiler-compatible optimized MoE GLU custom op.

        The installed RBLN converter expects routing to be computed externally
        and passed as experts-first ``masked_routing_weight`` with shape
        ``[num_experts_global, num_tokens]``.
        """
        if hidden_act.lower() not in {"silu", "swish"} and "gelu" not in hidden_act.lower():
            raise ValueError(f"Unsupported hidden_act={hidden_act!r}")
        if gate_proj_bias is not None or up_proj_bias is not None or down_proj_bias is not None:
            raise ValueError("Biased MoE GLU is not supported by this RBLN path")
        del n_group, topk_group

        valid_weight = masked_routing_weight.to(torch.float32)
        if expert_map is not None:
            global_indices = torch.nonzero(expert_map >= 0).flatten()
            valid_weight = valid_weight[global_indices, :]

        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        for i in range(expert_cnt):
            gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
            up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
            if "gelu" in hidden_act.lower():
                mul = torch.nn.functional.gelu(gate) * up
            else:
                mul = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(mul, down_proj_weight[i])
            out += down * valid_weight[i].unsqueeze(-1).to(down.dtype)
        return out

    @custom_moe_glu.register_fake
    def custom_moe_glu_fake(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        hidden_act: str,
        expert_map: torch.Tensor | None = None,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        n_group: int | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

else:

    @torch.library.custom_op(
        "rbln_custom_ops::custom_moe_glu",
        mutates_args=(),
    )
    def custom_moe_glu(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        expert_select_count: torch.Tensor,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Customized MoE GLU operation (custom kernel version).

        Expected tensor shapes:
        - hidden_states: [batch * seq_len, hidden_size]
        - gate_proj_weight: [num_experts, intermediate_size, hidden_size]
        - up_proj_weight: [num_experts, intermediate_size, hidden_size]
        - down_proj_weight: [num_experts, hidden_size, intermediate_size]
        - masked_routing_weight: [batch * seq_len, num_experts]
        - expert_select_count: [num_experts]

        Returns:
            torch.Tensor: [batch * seq_len, hidden_size]
        """
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        for i in range(expert_cnt):
            gate = torch.nn.functional.linear(hidden_states, gate_proj_weight[i])
            up = torch.nn.functional.linear(hidden_states, up_proj_weight[i])
            mul = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(mul, down_proj_weight[i])
            out += down * masked_routing_weight[:, i : i + 1]
        return out

    @custom_moe_glu.register_fake
    def custom_moe_glu_fake(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        expert_select_count: torch.Tensor,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)


def unquantized_fused_moe_method_rbln(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
):
    # selected_experts
    w1 = layer.w13_weight
    w2 = layer.w2_weight

    orig_shape = x.shape  # noqa: F841
    hidden_size = x.shape[-1]
    num_tokens = x.shape[:-1].numel()  # noqa: F841
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = x.dtype
    top_k = layer.top_k

    hidden_states = x
    gating_output = router_logits
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights = topk_weights.to(torch.float)
    topk_weights, selected_experts = topk_weights.topk(top_k, dim=-1)
    if layer.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if layer.expert_map is not None:
        selected_experts = layer.expert_map[selected_experts]

    final_hidden_states = None

    # 1. build expert_mask & expert_weights
    # 2. FFN
    # topk_weights, expert_weights, expert_mask.shape = [b, seq, top_k]
    # NOTE - convert for loop scalar operation into tensor compare

    # [1,num_tokens,hidden_size]
    hidden_states = hidden_states.reshape(1, num_tokens, -1)
    # [num_experts,1,1,1]
    expert_idx_array = torch.arange(0, num_experts).reshape(num_experts, 1, 1, 1)
    # [1,1,num_tokens,topk]
    selected_experts_array = selected_experts.reshape(-1, 1, num_tokens, top_k)
    # [num_experts,1,num_tokens,topk]
    expert_mask_array = selected_experts_array == expert_idx_array
    # [num_experts,1,num_tokens,topk]
    topk_weights_array = topk_weights.reshape(-1, 1, num_tokens, top_k)
    # [num_experts,1,num_tokens,1]
    expert_weights_array = (topk_weights_array * expert_mask_array).sum(
        dim=-1, keepdim=True
    )
    # [1,num_tokens,1]
    temp_expert_weights = expert_weights_array[0]
    # NOTE - make explicit dependence between hidden_states and expert_weights
    # [1,num_tokens,hidden_size]
    # [1,num_tokens,1] <- broadcast add
    hidden_states = hidden_states + temp_expert_weights - temp_expert_weights
    # [num_experts,1,num_tokens,1] -> [num_experts,1,num_tokens,hidden_size]
    hidden_states = hidden_states.to(dtype)
    expert_weights_array = expert_weights_array.broadcast_to(
        (num_experts, 1, num_tokens, hidden_size)
    ).to(dtype)
    # solution1. make custom operation for expert loop
    # solution2. add dummy use of expert_weights_array
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_weights = expert_weights_array[expert_idx]
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[..., :intermediate_size])
        x = x[..., intermediate_size:] * gate
        x = F.linear(x, expert_w2)

        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    assert final_hidden_states is not None
    return final_hidden_states.reshape(orig_shape)


def get_tokens_mask(num_tokens: int, left=1.0, right=0.0, device=None):
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        # Standalone/eager single-DP probes do not populate DP metadata.
        # In that case every local token is valid, so the mask is all-left.
        tokens_mask = torch.full((num_tokens, 1), left, dtype=torch.float32)
        if device is not None:
            tokens_mask = tokens_mask.to(device)
        return tokens_mask

    num_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu
    num_tokens_across_dp = num_tokens_across_dp.unsqueeze(1)
    if num_tokens_across_dp.size(0) == 1:
        max_pad = num_tokens
    else:
        max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
    pos = torch.arange(max_pad, dtype=torch.int32).unsqueeze(0)  # [1, max_pad]
    tokens_mask = torch.where(
        pos < num_tokens_across_dp, left, right
    )  # [dp_size, max_pad]
    tokens_mask = tokens_mask.reshape(-1, 1)  # [dp_size * max_pad, 1]
    if device is not None:
        tokens_mask = tokens_mask.to(device)
    return tokens_mask


# based on custom fused moe expert kernel
def get_masked_routing_weights(
    router_logits, top_k, renormalize, expert_map, scoring_func="softmax"
):
    # routing_weights: (batch * sequence_length, n_experts)
    # selected_experts: (batch * sequence_length, top_k)
    router_logits = router_logits.to(torch.float)
    if scoring_func == "softmax":
        if renormalize:
            selected_weights, selected_experts = torch.topk(
                router_logits, k=top_k, dim=-1
            )
            selected_weights = torch.nn.functional.softmax(selected_weights, dim=1)
        else:
            routing_weights = torch.nn.functional.softmax(router_logits, dim=1)
            selected_weights, selected_experts = torch.topk(
                routing_weights, k=top_k, dim=-1
            )
    elif scoring_func == "sigmoid":
        routing_weights = router_logits.sigmoid()
        selected_weights, selected_experts = torch.topk(
            routing_weights, k=top_k, dim=-1
        )
        if renormalize:
            selected_weights = selected_weights / (
                selected_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
    else:
        raise ValueError(f"Unsupported MoE scoring_func={scoring_func!r}")

    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(
            router_logits.shape[0], 1.0, 0.0, device=selected_weights.device
        )
        selected_weights = selected_weights * tokens_mask

    n_expert = router_logits.shape[1]
    if expert_map is not None:
        expert_map_within_bounds = torch.where(
            expert_map < 0, n_expert - 1, expert_map
        ).to(torch.int64)
        selected_experts = expert_map_within_bounds[selected_experts]

    # masked_routing_weights=selected_weights w/ non selected indicies zeros
    # selected_weights      = [..., top_k]
    # masked_routing_weights= [..., n_experts], selected_experts has only value
    masked_routing_weights = torch.zeros_like(router_logits, dtype=torch.float32)
    masked_routing_weights.scatter_(1, selected_experts, selected_weights)

    ## count selected tokens for each expert index from selected_experts
    zeros = torch.zeros(n_expert, dtype=torch.int32)

    if use_moe_tokens_mask:
        ones = torch.ones_like(selected_experts, dtype=torch.int32)
        tokens_mask = tokens_mask.to(torch.int32)
        ones = ones * tokens_mask
        ones = ones.view(-1)
    else:
        ones = torch.ones_like(selected_experts.view(-1), dtype=torch.int32)

    expert_select_count = torch.scatter_add(
        zeros, dim=0, index=selected_experts.view(-1), src=ones
    )

    return masked_routing_weights, expert_select_count


def unquantized_fused_moe_method_custom(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
):
    # selected_experts
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    # w13_weight- merged weight for gate_proj(w1_weight) and up_proj (w3_weight)
    # w2_weight - down_proj
    # gate_proj_weight - first half, layer.w13_weight[:intermediate_size]
    # up_proj_weight - second half, layer.w13_weight[intermediate_size:]
    # down_proj_weights = layer.w2_weight
    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :].contiguous()
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :].contiguous()
    down_proj_weight = layer.w2_weight.contiguous()

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)
    router_logits = router_logits.reshape(num_tokens, -1)

    masked_routing_weights, expert_select_count = get_masked_routing_weights(
        router_logits,
        layer.top_k,
        layer.renormalize,
        layer.expert_map,
        getattr(layer, "scoring_func", "softmax"),
    )

    tokens_mask = None
    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(num_tokens, device=router_logits.device)

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        masked_routing_weights,
        expert_select_count,
        None,
        None,
        None,
        tokens_mask,
    )
    return final_hidden_states.reshape(orig_shape)


def unquantized_fused_optimize_moe_method_custom(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
):
    # selected_experts
    # w1 : gate_proj, w2 : down_proj, w3 : up_proj
    orig_shape = x.shape  # noqa: F841
    num_tokens = orig_shape[:-1].numel()  # noqa: F841
    intermediate_size = layer.w2_weight.shape[-1]

    # w13_weight- merged weight for gate_proj(w1_weight) and up_proj (w3_weight)
    # w2_weight - down_proj
    # gate_proj_weight - first half, layer.w13_weight[:intermediate_size]
    # up_proj_weight - second half, layer.w13_weight[intermediate_size:]
    # down_proj_weights = layer.w2_weight
    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :].contiguous()
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :].contiguous()
    down_proj_weight = layer.w2_weight.contiguous()

    # expected tensor shape - [num_tokens, -1]
    hidden_states = x.reshape(num_tokens, -1)
    router_logits = router_logits.reshape(num_tokens, -1)

    masked_routing_weights, _expert_select_count = get_masked_routing_weights(
        router_logits,
        layer.top_k,
        layer.renormalize,
        layer.expert_map,
        getattr(layer, "scoring_func", "softmax"),
    )
    # RBLN custom_moe_glu lowering expects experts-first routing layout.
    masked_routing_weights = masked_routing_weights.transpose(0, 1).contiguous()

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        masked_routing_weights,
        "silu",
        layer.expert_map,
        None,
        None,
        None,
        None,
        None,
    )
    return final_hidden_states.reshape(orig_shape)


def fused_moe_forward_rbln(
    self: FusedMoE,
    hidden_states: torch.Tensor,
    router: torch.nn.Module | None = None,
    router_logits: torch.Tensor | None = None,
    **_kwargs,
) -> torch.Tensor:
    assert self.quant_method is not None

    if router_logits is None:
        if router is None:
            raise TypeError("FusedMoE RBLN forward requires router or router_logits")
        router_logits = router(hidden_states)

    if self.moe_parallel_config.dp_size > 1:
        org_hidden_shape = hidden_states.shape

        # input broadcast - all DPs broadcast hidden_states & router_logits
        # example) DP2, TP/EP2
        # dp_group = {{0, 2}, {1, 3}}
        # tp_group = {{0, 1}, {2, 3}}
        # 1. initially, each DP hidden_states = [1, 128, 1024]
        # 2. after multicast, all DPs hidden_states = [dp_size, 128, 1024]
        # - all DP ranks broadcast inputs to process group
        # 3. DP x TP/EP expert parallel
        # ex) 0, 1, 2, 3 has its own hidden_states = [dp_size, 128, 1024]
        # 4. dp_group all reduce - {0+2}, {1+3}, {0+2}, {1+3}
        # 5. select each DP rank output
        # 6. to_group all reduce - {0+2+1+3}, {0+2+1+3}, {0+2+1+3}, {0+2+1+3}
        hidden_states = self.naive_multicast(hidden_states)

    # Matrix multiply.
    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=router_logits,
    )

    if self.moe_parallel_config.dp_size > 1:
        # output all_reduce == dp all_reduce + tp all_reduce
        if envs.VLLM_RBLN_MOE_REDUCE_SCATTER:
            hidden_shape_dp = (-1, 1, org_hidden_shape[-1])
            all_hidden_states = final_hidden_states.reshape(hidden_shape_dp)
            assert all_hidden_states.shape[0] % self.moe_parallel_config.dp_size == 0

            hidden_states = get_dp_group().reduce_scatter(all_hidden_states, dim=0)
            max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
            assert hidden_states.shape[0] == max_pad

            num_tokens = org_hidden_shape[:-1].numel()  # noqa: F841
            final_hidden_states = hidden_states[:num_tokens]
        else:
            all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
            hidden_shape_dp = (-1, 1, org_hidden_shape[-1])
            final_hidden_states = all_hidden_states.reshape(hidden_shape_dp)

            max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
            num_tokens = org_hidden_shape[:-1].numel()  # noqa: F841
            start = self.moe_parallel_config.dp_rank * max_pad
            end = start + num_tokens
            final_hidden_states = final_hidden_states[start:end]

        final_hidden_states = final_hidden_states.reshape(org_hidden_shape)

    return final_hidden_states


def fused_moe_naive_multicast_rbln(self: FusedMoE, x: torch.Tensor):
    # as-is : [num_tokens, hidden_size]
    # to-be : buffer = [data_parallel_size*batch, seq, hidden_size], broadcast
    #         hidden = [batch, seq, hidden_size]
    # x.shape = [1, seq, hidden_size]
    # assert len(x.shape) == 3

    x = x.reshape(1, -1, x.size(-1))
    max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
    num_tokens = x.size(1)
    num_repeat = max_pad // num_tokens
    # TODO: evaluate various padding approaches
    x = x.repeat(num_repeat, 1, 1)
    x = x.reshape(1, max_pad, -1)

    if not envs.VLLM_RBLN_DP_INPUT_ALL_GATHER:
        # each DP rank gather all inputs via torch.distributed.all_reduce
        # broadcast(value) == all_reduce(value for me or zeros for others)
        all_buffer = None
        zeros = x - x
        for rank in range(get_dp_group().world_size):
            rank_tensor = x if rank == self.moe_parallel_config.dp_rank else zeros
            all_buffer = (
                torch.cat((all_buffer, rank_tensor), dim=0)
                if all_buffer is not None
                else rank_tensor
            )
        output = get_dp_group().all_reduce(all_buffer)
        return output
    else:
        # gather all inputs via torch.distributed.all_gather
        all_gather_buffer = get_dp_group().all_gather(x, dim=0)
        return all_gather_buffer


def _unquantized_fused_moe_method_is_monolithic_rbln(self) -> bool:
    # Upstream's is_monolithic falls back to
    # `self.experts_cls.is_monolithic()` when `self.moe_kernel` is still
    # unset, but out-of-tree backends (RBLN included) get `experts_cls=None`
    # from `select_unquantized_moe_backend` (see
    # fused_moe/oracle/unquantized.py), so that call raises AttributeError
    # on None -- which Python's property-getter-swallows-AttributeError
    # behavior then misreports as "no attribute 'is_monolithic'" entirely.
    # RBLN's kernels (below) compute routing internally from router_logits
    # in one call -- the "monolithic" calling convention -- so report True
    # here (and route through `.apply_monolithic()`, which RBLN overrides
    # below) rather than falling through to the experts_cls-based upstream
    # logic.
    if self.moe_kernel is None and self.experts_cls is None:
        return True
    return _upstream_is_monolithic.__get__(self)


_upstream_is_monolithic = UnquantizedFusedMoEMethod.is_monolithic

FusedMoE.__init__ = fused_moe_custom__init__
FusedMoE.forward_oot = fused_moe_forward_rbln
UnquantizedFusedMoEMethod.is_monolithic = property(
    _unquantized_fused_moe_method_is_monolithic_rbln
)


# NOTE(RBLN): these kernels compute routing internally from router_logits in
# one call (the "monolithic" convention: layer, x, router_logits, input_ids),
# not the topk_weights/topk_ids-precomputed `.apply()` convention -- wire
# them to apply_monolithic (dispatched to when is_monolithic is True, see
# above), not apply.
if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:
    logger.info("[RBLN] fused moe, RBLN optimize moe custom kernel")
    UnquantizedFusedMoEMethod.apply_monolithic = unquantized_fused_optimize_moe_method_custom
elif envs.VLLM_RBLN_MOE_CUSTOM_KERNEL:
    logger.info("[RBLN] fused moe, RBLN moe custom kernel")
    UnquantizedFusedMoEMethod.apply_monolithic = unquantized_fused_moe_method_custom
else:
    logger.info("[RBLN] fused moe, pytorch native kernel")
    UnquantizedFusedMoEMethod.apply_monolithic = unquantized_fused_moe_method_rbln
FusedMoE.naive_multicast = fused_moe_naive_multicast_rbln
