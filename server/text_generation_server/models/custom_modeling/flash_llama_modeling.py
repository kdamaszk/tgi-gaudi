# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN

from text_generation_server.layers.attention import PREFILL_IN_KV_CACHE
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.layers.attention import (
    paged_attention,
    attention,
    reshape_and_cache,
    Seqlen,
)
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    TensorParallelMultiAdapterLinear,
    TensorParallelAdapterRowLinear,
)
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)
from text_generation_server.utils.weights import (
    Weights,
)
from text_generation_server.layers.fp8 import HybridFP8UnquantLoader

if SYSTEM == "rocm":
    try:
        from vllm import _custom_C
    except Exception as e:
        raise ImportError(f"Could not load `vllm._custom_C`. Full error: {e}")


def load_attention(config, prefix: str, weights, layer_id):
    # Only defined in granite.
    bias = getattr(config, "attention_bias", False)
    head_size = config.hidden_size // config.num_attention_heads
    sizes = None
    prefixes = None

    if config.model_type == "phi3":
        base_layer = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.qkv_proj",
            weights=weights,
            bias=bias,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )
        prefixes = ["qkv_proj"]
    elif config.model_type == "baichuan":
        prefix = f"{prefix}.W_pack"
        base_layer = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=prefix,
            weights=weights,
            bias=bias,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )
        prefixes = [prefix]
    else:
        prefixes = ["q_proj", "k_proj", "v_proj"]
        sizes = [
            head_size * config.num_attention_heads,
            head_size * config.num_key_value_heads,
            head_size * config.num_key_value_heads,
        ]
        base_layer = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=bias,
        )

    return TensorParallelMultiAdapterLinear.load(
        base_layer=base_layer,
        layer_id=layer_id,
        layer_names=prefixes,
        sizes=sizes,
        process_group=weights.process_group,
    )


@contextmanager
def no_fp8(weights: Weights):
    """De-activate fp8 auto conversion for the duration of this context manager"""
    weights_loader = weights.weights_loader
    if isinstance(weights_loader, HybridFP8UnquantLoader) and weights_loader.to_fp8:
        weights_loader = HybridFP8UnquantLoader(
            weights_loader.activation_scale_ub, to_fp8=False
        )

    with weights.use_loader(weights_loader):
        yield


class FlashLlamaAttention(torch.nn.Module):
    def __init__(
        self,
        index: int,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        # Setting defaults for baichuan custom config which doesn't apply them.
        config.rope_theta = getattr(config, "rope_theta", 10000)
        config.num_key_value_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_size,
            base=config.rope_theta,
            device=weights.device,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        if config.num_key_value_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_key_value_heads` must be divisible by `num_shards` (got `num_key_value_heads`: {config.num_key_value_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )

        self.query_key_value = load_attention(config, prefix, weights, index)
        self.index = index

        o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

        self.o_proj = TensorParallelAdapterRowLinear.load(
            o_proj,
            index,
            "o_proj",
            process_group=weights.process_group,
        )

        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        seqlen,
        max_s,
        adapter_data,
    ):
        qkv = self.query_key_value(hidden_states, adapter_data)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)

        reshape_and_cache(kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = attention(
                query,
                kv_cache[0] if PREFILL_IN_KV_CACHE else kv[:, 0],
                kv_cache[1] if PREFILL_IN_KV_CACHE else kv[:, 1],
                seqlen,
                block_tables,
                self.softmax_scale,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                seqlen,
                max_s,
            )

        return self.o_proj(
            attn_output.view(-1, self.num_heads * self.head_size), adapter_data
        )


class LlamaMLP(nn.Module):
    def __init__(self, prefix, config, weights, index):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.act = (
            ACT2FN[self.hidden_act]
            if "gelu" not in self.hidden_act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh"
                    if self.hidden_act in ["gelu_fast", "gelu_pytorch_tanh"]
                    else "none"
                ),
            )
        )
        prefixes = None
        sizes = None

        # Fuse gate and up proj
        bias = getattr(config, "mlp_bias", False)
        if config.model_type == "phi3":
            gate_up_proj = TensorParallelColumnLinear.load_gate_up(
                config,
                prefix=f"{prefix}.gate_up_proj",
                weights=weights,
                bias=bias,
            )
        else:
            prefixes = ["gate_proj", "up_proj"]
            sizes = [
                config.intermediate_size,
                config.intermediate_size,
            ]
            gate_up_proj = TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                dim=0,
                bias=bias,
            )

        self.gate_up_proj = TensorParallelMultiAdapterLinear.load(
            gate_up_proj,
            index,
            layer_names=prefixes,
            sizes=sizes,
            process_group=weights.process_group,
        )

        down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=bias,
        )

        self.down_proj = TensorParallelAdapterRowLinear.load(
            down_proj,
            index,
            "down_proj",
            process_group=weights.process_group,
        )

        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

        # TODO: This is a hotfix to be removed & properly refactored.
        self.quantize = config.quantize

    def forward(self, hidden_states, adapter_data):
        if (
            SYSTEM == "rocm"
            and self.hidden_act == "silu"
            and hidden_states.shape[0] == 1
            and not self.quantize
        ):
            out = torch.empty(
                hidden_states.shape[0],
                self.intermediate_size,
                dtype=hidden_states.dtype,
                device="cuda",
            )
            _custom_C.LLMM_Silu(
                self.gate_up_proj.base_layer.linear.weight, hidden_states, out, 8
            )
            return self.down_proj(out, adapter_data)
        else:
            gate_up_states = self.gate_up_proj(hidden_states, adapter_data)
            gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
            return self.down_proj(
                self.act(gate_up_states[:, 0]) * gate_up_states[:, 1], adapter_data
            )


class FlashLlamaLayer(nn.Module):
    def __init__(self, index, prefix, config, weights):
        super().__init__()

        with no_fp8(weights):
            self.self_attn = FlashLlamaAttention(
                index=index,
                prefix=f"{prefix}.self_attn",
                config=config,
                weights=weights,
            )

        self.mlp = LlamaMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, index=index
        )

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        seqlen,
        max_s,
        adapter_data,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            adapter_data,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output, adapter_data)

        return mlp_output, attn_res


class FlashLlamaModel(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        # Skip fp8 quant for first and last layers
        self.layers = nn.ModuleList()
        with no_fp8(weights):
            self.layers.append(
                FlashLlamaLayer(
                    index=0,
                    prefix=(
                        "model.layers.0" if not prefix else f"{prefix}.model.layers.0"
                    ),
                    config=config,
                    weights=weights,
                )
            )

        self.layers.extend(
            [
                FlashLlamaLayer(
                    index=layer_id,
                    prefix=(
                        f"model.layers.{layer_id}"
                        if not prefix
                        else f"{prefix}.model.layers.{layer_id}"
                    ),
                    config=config,
                    weights=weights,
                )
                # Skip first and last layers
                for layer_id in range(1, config.num_hidden_layers - 1)
            ]
        )

        with no_fp8(weights):
            last_layer_id = config.num_hidden_layers - 1
            self.layers.append(
                FlashLlamaLayer(
                    index=last_layer_id,
                    prefix=(
                        f"model.layers.{last_layer_id}"
                        if not prefix
                        else f"{prefix}.model.layers.{last_layer_id}"
                    ),
                    config=config,
                    weights=weights,
                )
            )

        self.norm = FastRMSNorm.load(
            prefix="model.norm" if not prefix else f"{prefix}.model.norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        true_max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
        adapter_data,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                seqlen,
                max_s,
                adapter_data,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashLlamaForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        with no_fp8(weights):
            self.embed_tokens = TensorParallelEmbedding(
                prefix=(
                    "model.embed_tokens"
                    if not prefix
                    else f"{prefix}.model.embed_tokens"
                ),
                weights=weights,
            )
        self.model = FlashLlamaModel(prefix, config, weights)
        if config.tie_word_embeddings:
            suffix = "model.embed_tokens"
        else:
            suffix = "lm_head"

        with no_fp8(weights):
            self.lm_head = SpeculativeHead.load(
                config,
                prefix=suffix if not prefix else f"{prefix}.{suffix}",
                weights=weights,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            true_max_s=max_s,
            prefill_cache_indices=prefill_cache_indices,
            adapter_data=adapter_data,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
