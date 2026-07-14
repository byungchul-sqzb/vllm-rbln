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

import vllm_rbln.rbln_envs as envs


def register():
    """Register the RBLN platform."""
    return "vllm_rbln.platform.RblnPlatform"


def register_model():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "UnlimitedOCRForCausalLM",
        "vllm_rbln.model_executor.models.unlimited_ocr:UnlimitedOCRForCausalLM",
    )

    if not envs.VLLM_RBLN_USE_VLLM_MODEL:
        ModelRegistry.register_model(
            "T5WithLMHeadModel",
            "vllm_rbln.model_executor.models.optimum.t5:RBLNT5ForConditionalGeneration",
        )
        ModelRegistry.register_model(
            "T5ForConditionalGeneration",
            "vllm_rbln.model_executor.models.optimum.t5:RBLNT5ForConditionalGeneration",
        )
        ModelRegistry.register_model(
            "T5EncoderModel",
            "vllm_rbln.model_executor.models.optimum.encoder:RBLNOptimumForEncoderModel",
        )
        ModelRegistry.register_model(
            "Gemma3ForConditionalGeneration",
            "vllm_rbln.model_executor.models.optimum.gemma3:RBLNOptimumGemma3ForConditionalGeneration",
        )


def register_ops():
    import vllm_rbln.distributed.ec_transfer.ec_connector.factory  # noqa

    # Always apply: needed by both the standard engine path and the custom
    # Unlimited-OCR experimental runtime (which calls register_ops() with
    # VLLM_RBLN_USE_VLLM_MODEL=False). vLLM's ViT `torch_sdpa_wrapper` has a
    # fake-impl signature bug: the real op declares `scale=None,
    # cu_seqlens=None` defaults but `torch_sdpa_wrapper_fake` omits them, so
    # fake-tensor propagation during torch.compile of the ViT/CLIP encoder
    # fails ("torch_sdpa_wrapper_fake() missing 1 required positional
    # argument: 'cu_seqlens'"). This blocks compiling the vision frontend on
    # RBLN. The fake can't be re-registered (already registered), so bypass
    # the opaque custom op entirely: point the mm-encoder's wrapper at the
    # plain `torch_sdpa_wrapper` function, which decomposes to aten SDPA
    # in-graph (no custom op, no fake). Behaviorally identical in eager;
    # makes the ViT graph rebel-compilable.
    try:
        import vllm.model_executor.layers.attention.mm_encoder_attention as _mea
        import vllm.v1.attention.ops.vit_attn_wrappers as _vaw

        _mea.vit_torch_sdpa_wrapper = _vaw.torch_sdpa_wrapper
    except Exception:
        pass

    if envs.VLLM_RBLN_USE_VLLM_MODEL:
        # Disable vLLM's IR-op torch custom-op wrapping. With it enabled,
        # `vllm.ir` ops (rms_norm, fused_add_rms_norm, ...) appear in the
        # traced graph as opaque `vllm_ir::*` custom ops. RBLN lowers graphs
        # with rebel-compiler (not Inductor), which cannot lower those opaque
        # ops -- so compile fails and falls back to CPU eager. Disabling the
        # wrapper makes IR ops dispatch straight to their native torch
        # implementation, decomposing into primitive ops rebel-compiler can
        # lower. This is the wrapper's documented use case ("avoiding the need
        # for lowering for platforms not using Inductor").
        try:
            from vllm.ir.op import set_default_torch_wrap

            set_default_torch_wrap(False)
        except ImportError:
            pass

        import vllm_rbln.model_executor.layers.attention.attention  # noqa
        import vllm_rbln.distributed.kv_transfer.kv_connector.factory  # noqa
        import vllm_rbln.forward_context  # noqa
        import vllm_rbln.lora.layer  # noqa
        import vllm_rbln.model_executor.layers.fused_moe.layer  # noqa

        try:
            import vllm_rbln.model_executor.layers.fused_moe.shared_fused_moe  # noqa
        except ImportError:
            # vLLM merged SharedFusedMoE's shared-expert handling directly
            # into FusedMoE (see fused_moe/runner/shared_experts.py); the
            # standalone class this patch targets no longer exists on newer
            # vLLM versions. None of our currently supported models
            # (Unlimited-OCR/DeepSeekV2 included) construct SharedFusedMoE
            # directly, so skip rather than block all RBLN model loading.
            pass
        import vllm_rbln.model_executor.layers.logits_processor  # noqa

        try:
            import vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision  # noqa
        except ImportError:
            # compressed_tensors reorganized its compressor submodules
            # (quantized_compressors -> pack_quantized/naive_quantized/etc)
            # on newer versions; this mixed-precision quant kernel patch
            # isn't needed for Unlimited-OCR's unquantized bf16 model, so
            # skip rather than block all RBLN model loading.
            pass
        import vllm_rbln.model_executor.layers.quantization.mxfp4  # noqa

        try:
            import vllm_rbln.model_executor.layers.quantization.fp8  # noqa
        except ImportError:
            # vLLM renamed/reorganized fp8_utils helpers on newer versions
            # (e.g. maybe_post_process_fp8_weight_block ->
            # deepgemm_post_process_fp8_weight_block); this fp8 quant patch
            # isn't needed for Unlimited-OCR's unquantized bf16 model, so
            # skip rather than block all RBLN model loading.
            pass
        import vllm_rbln.model_executor.layers.rotary_embedding.base  # noqa
        import vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope  # noqa
        import vllm_rbln.model_executor.layers.vocab_parallel_embedding  # noqa
        import vllm_rbln.model_executor.model_loader.weight_loader  # noqa
        import vllm_rbln.models.deepseek_v2  # noqa
        import vllm_rbln.models.gpt_oss  # noqa
        import vllm_rbln.models.qwen2  # noqa
        import vllm_rbln.models.qwen2_moe  # noqa
        import vllm_rbln.models.qwen3  # noqa
        import vllm_rbln.models.qwen3_moe  # noqa
        import vllm_rbln.models.minimax_m2  # noqa
        import vllm_rbln.models.utils  # noqa
        from vllm_rbln.triton_kernels import attention  # noqa
        from vllm_rbln.triton_kernels import causal_attention  # noqa
        from vllm_rbln.triton_kernels import flash_attention  # noqa
        from vllm_rbln.triton_kernels import flash_causal_attention  # noqa
        from vllm_rbln.triton_kernels import sliding_window_attention  # noqa
