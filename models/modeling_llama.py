"""JAX/Pallas LLaMA model for SVD-Flash on TPU.

This is the TPU counterpart of ``NeuronMM/models/llama/modeling_llama.py``. The
programming pattern is deliberately the same: one class per transformer
component, a module registry, an inference config with derived attributes and a
per-layer "updated configs" pass, and an MLP whose ``forward`` dispatches on
``mlp_kernel_enabled`` between a fused low-rank kernel and the plain reference
math. A ``metadata["svd_llama"]`` flag picks the low-rank MLP over the dense
one, exactly as ``llama_inference.py`` does when it runs the two arms of the
speedup comparison.

What changes is the substrate:

===============================================  ==========================================
NeuronMM (Trainium, PyTorch)                     this port (TPU, JAX)
===============================================  ==========================================
``torch.nn.Module``                              :class:`TpuModule` (arrays + submodules)
``nki.jit`` kernels in ``modeling_llama.py``      ``jax.experimental.pallas`` kernels in
                                                 ``svdflash_tpu/kernels``
``fused_mlp_up_T``                                ``pallas_ops.svd_mlp`` (Gate/Up kernel)
``fused_three_mm_XTUV``                           ``pallas_ops.svd_mlp`` (Down XUV kernel)
``nki_matmul_fully_optimized_``                   ``pallas_ops.dense_mlp``
``get_fused_mlp_up_T_params``                     ``kernels.config.gate_up_block_params``
``torch.autograd.Function`` wrappers              dropped -- inference only, no gradients
tensor parallelism (``tp_degree``)                single device; the kernels are
                                                 TensorCore-mesh parallel internally
``torch.int32`` rank tensors                      not needed, no collectives
===============================================  ==========================================

Two behaviours of the original are intentionally *not* reproduced:

* The dense baseline in NeuronMM still runs an NKI matmul kernel
  (``NeuronLlamaMLP._native_mlp`` is replaced by ``nki_matmul_fully_optimized_``),
  so its reported speedup is kernel-versus-kernel. Here ``_native_mlp`` is the
  real dense math and ``_kernel_enabled_mlp`` is the Pallas kernel, so the two
  arms are distinguishable.
* Attention is plain JAX. No TPU attention kernel exists in this port yet (see
  the README's "后续工作"), but the QKV layout, the RoPE tables, the GQA repeat
  and the cache layout are written so one can be dropped in behind
  :class:`NeuronLlamaAttention` later.

Checkpoint compatibility: parameter names are the Hugging Face keys minus a
trailing ``.weight`` for the projection matrices (see
:meth:`NeuronLlamaForCausalLM.convert_hf_to_tpu_state_dict`). Every projection
is stored as the natural matmul operand ``(in, out)`` -- the transpose of
PyTorch's ``Linear.weight`` -- which is what the Pallas kernels consume.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np

from svdflash_tpu.kernels.config import (
    MODELS,
    GateUpBlocks,
    ModelShape,
    gate_up_block_params,
    rank_for,
)
from svdflash_tpu.kernels.pallas_ops import (
    contracted_svd_mlp,
    dense_mlp,
    svd_mlp,
)
from svdflash_tpu.kernels.reference import low_rank_dim

logger = logging.getLogger("SvdFlashTpu")


###############################################################################
# Module registry
###############################################################################

_LLAMA_MODULE_MAP: Dict[str, Type["TpuModule"]] = {}


def _register_module(key: str, cls: Type["TpuModule"]) -> None:
    _LLAMA_MODULE_MAP[key] = cls


def register_module(key: str):
    """
    Register a module for use in the Llama model.

    Arguments:
        key: String used to identify the module

    Example:
        @register_module("NeuronLlamaAttention")
        class NeuronLlamaAttention(TpuModule):
            ...
    """

    def inner(cls: Type[TpuModule]):
        _register_module(key, cls)
        return cls

    return inner


def get_rmsnorm_cls() -> Type["LlamaRMSNorm"]:
    """Initialize to the appropriate implementation of RMSNorm.

    NeuronMM switches between a fused kernel norm and the Hugging Face one
    depending on the device; this port has a single JAX implementation, so the
    indirection is kept only because the decoder layer still calls through it.
    """
    return LlamaRMSNorm


###############################################################################
# Parameter container
###############################################################################

class TpuModule:
    """A minimal stand-in for ``torch.nn.Module``.

    Parameters are the ``jax.Array`` attributes assigned in ``__init__``;
    submodules are nested :class:`TpuModule` instances, and layers live in plain
    lists. :meth:`named_parameters` walks both so that a checkpoint state dict
    can be mapped onto dotted names of the same shape NeuronMM uses.

    There is no autograd here. This file is inference-only, which is why the
    ``torch.autograd.Function`` wrappers around NeuronMM's fused linear layers
    have no counterpart: JAX arrays are immutable, so a parameter update is a
    ``setattr`` on the module tree rather than an in-place copy.
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Any]]:
        for name, value in vars(self).items():
            if isinstance(value, (jax.Array, np.ndarray)):
                yield f"{prefix}{name}", value
            elif isinstance(value, TpuModule):
                yield from value.named_parameters(f"{prefix}{name}.")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, TpuModule):
                        yield from item.named_parameters(f"{prefix}{name}.{index}.")
                    elif isinstance(item, (jax.Array, np.ndarray)):
                        yield f"{prefix}{name}.{index}", item

    def parameters(self) -> Dict[str, Any]:
        return dict(self.named_parameters())

    def _descend(self, path: str) -> Any:
        """Walk a dotted path to the module or container that owns it."""
        target: Any = self
        if not path:
            return target
        for part in path.split("."):
            if isinstance(target, (list, tuple)):
                target = target[int(part)]
            else:
                target = getattr(target, part)
        return target

    def _set_parameter(self, path: str, value: Any) -> None:
        *parents, leaf = path.split(".")
        owner = self._descend(".".join(parents))
        if isinstance(owner, list):
            owner[int(leaf)] = value
        else:
            setattr(owner, leaf, value)

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        *,
        strict: bool = True,
        dtype: Optional[Any] = None,
    ) -> List[str]:
        """Copy ``state_dict`` onto the module tree.

        Returns the list of parameters the checkpoint did not provide. With
        ``strict`` they raise instead, which is the default because a silently
        un-loaded parameter is a zero-filled one.
        """
        expected = dict(self.named_parameters())
        missing = [name for name in expected if name not in state_dict]
        if missing and strict:
            preview = ", ".join(missing[:8])
            raise KeyError(
                f"{len(missing)} parameters missing from the checkpoint "
                f"(first few: {preview})"
            )

        for name, target in expected.items():
            if name not in state_dict:
                continue
            value = jnp.asarray(state_dict[name])
            if value.shape != target.shape:
                raise ValueError(
                    f"{name}: checkpoint shape {value.shape} does not match "
                    f"module shape {target.shape}"
                )
            if dtype is not None:
                value = value.astype(dtype)
            self._set_parameter(name, value)

        extra = [name for name in state_dict if name not in expected]
        if extra:
            logger.debug("ignoring %d checkpoint tensors unused by the model", len(extra))
        return missing

    def init_random(self, key: jax.Array, scale: float = 0.02) -> "TpuModule":
        """Fill every parameter with ``N(0, scale)``, except norm gains.

        Only the smoke tests use this; real weights come from a checkpoint.
        """
        expected = dict(self.named_parameters())
        names = list(expected)
        keys = jax.random.split(key, max(len(names), 1))
        for name, subkey in zip(names, keys):
            target = expected[name]
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                value = jnp.ones(target.shape, target.dtype)
            else:
                value = (jax.random.normal(subkey, target.shape, jnp.float32) * scale)
                value = value.astype(target.dtype)
            self._set_parameter(name, value)
        return self


def _empty(shape: Sequence[int], dtype: Any) -> jax.Array:
    """A zero-valued placeholder parameter, overwritten by the checkpoint."""
    return jnp.zeros(tuple(shape), dtype=dtype)


###############################################################################
# RMSNorm
###############################################################################

class LlamaRMSNorm(TpuModule):
    """Root-mean-square layer norm, computed in fp32 like Hugging Face.

    NeuronMM routes this through ``CustomRMSNorm`` (a kernel) on device and
    ``LlamaRMSNorm`` on CPU. On TPU the elementwise form is already cheap
    relative to the projections it precedes, so there is one implementation.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.weight = jnp.ones((hidden_size,), jnp.float32)
        self.variance_epsilon = eps

    def forward(self, hidden_states: jax.Array) -> jax.Array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return (self.weight.astype(jnp.float32) * hidden_states).astype(input_dtype)


###############################################################################
# Config
###############################################################################

# Mirrors ``LlamaInferenceConfig.get_required_attributes``.
_REQUIRED_HF_ATTRIBUTES = (
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "hidden_act",
)


@dataclass
class LlamaTpuConfig:
    """Model shape plus the runtime switches NeuronMM keeps in ``NeuronConfig``.

    The split is the same one ``LlamaInferenceConfig`` / ``NeuronConfig`` makes:
    shape attributes come from the Hugging Face config, and the kernel flags,
    dtype and metadata come from the launcher (``llama_inference.py`` passes
    ``metadata={"svd_llama": ..., "compress_ratio": ...}``).
    """

    # ----------------------------------------------------------- shape (HF)
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int = 131072
    head_dim: int = 0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    model_name: str = "custom"

    # --------------------------------------------------------- runtime (NeuronConfig)
    dtype: Any = jnp.bfloat16
    metadata: Optional[Dict[str, Any]] = None
    compress_ratio: float = 0.8
    mlp_kernel_enabled: bool = False
    contracted_mlp_enabled: bool = False
    mixed_precision: bool = True
    modules_to_not_convert: Optional[List[str]] = None
    attn_cls: str = "NeuronLlamaAttention"
    # Static KV-cache length. The attention score tensor is
    # ``batch * num_heads * q_len * max_sequence_length`` fp32, so this is the
    # single biggest memory knob: at the full 131072 context a 128-token prefill
    # on this model would allocate a 2 GB score tensor per layer. Leave it at 0
    # to size the cache automatically per call (see ``empty_cache``), or set it
    # to ``prompt + max_new_tokens`` for a fixed decode budget.
    max_sequence_length: int = 0
    interpret: bool = False

    # ------------------------------------------------------------- derived
    num_kv_groups: int = field(default=0, init=False)
    low_rank: int = field(default=0, init=False)

    def add_derived_config(self) -> None:
        """Compute the fields the components read but nobody configures."""
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be a "
                f"multiple of num_key_value_heads={self.num_key_value_heads}"
            )
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads
        self.low_rank = rank_for(self.get_model_shape(), self.compress_ratio)
        if self.low_rank <= 0:
            raise ValueError(
                f"compress_ratio={self.compress_ratio} rounds to rank "
                f"{self.low_rank}; it is too small to survive 128-alignment"
            )
        if self.metadata is None:
            self.metadata = {"svd_llama": False, "compress_ratio": self.compress_ratio}

    def get_model_shape(self) -> ModelShape:
        """The ``kernels.config`` description of this model.

        Registered shapes are reused verbatim so the rank arithmetic cannot
        drift between the model and the benchmarks; an unregistered model gets
        an equivalent shape built from its own dimensions. Set
        ``model_name="llama-3.1-8b"`` (or another key of
        ``kernels.config.MODELS``) to take the registered path.
        """
        registered = MODELS.get(self.model_name)
        if registered is not None and registered.hidden_size == self.hidden_size:
            return registered
        return ModelShape(
            name=self.model_name,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            rank_at_080=low_rank_dim(self.hidden_size, self.intermediate_size, 0.8),
        )

    @property
    def svd_llama(self) -> bool:
        """Whether the launcher asked for the low-rank MLP (NeuronMM metadata)."""
        return bool(self.metadata and self.metadata.get("svd_llama"))

    def replace(self, **overrides) -> "LlamaTpuConfig":
        """``dataclasses.replace`` plus a derived-config refresh."""
        updated = replace(self, **overrides)
        updated.add_derived_config()
        return updated

    def get_required_attributes(self) -> List[str]:
        return list(_REQUIRED_HF_ATTRIBUTES)

    @classmethod
    def from_hf_config(cls, hf_config: Dict[str, Any], **overrides) -> "LlamaTpuConfig":
        """Build a config from a Hugging Face ``config.json`` mapping."""
        missing = [name for name in _REQUIRED_HF_ATTRIBUTES if name not in hf_config]
        if missing:
            raise ValueError(f"HF config is missing required attributes: {missing}")

        hidden_size = int(hf_config["hidden_size"])
        num_heads = int(hf_config["num_attention_heads"])
        kwargs: Dict[str, Any] = {
            "hidden_size": hidden_size,
            "intermediate_size": int(hf_config["intermediate_size"]),
            "num_hidden_layers": int(hf_config["num_hidden_layers"]),
            "num_attention_heads": num_heads,
            "num_key_value_heads": int(hf_config["num_key_value_heads"]),
            "vocab_size": int(hf_config["vocab_size"]),
            "max_position_embeddings": int(hf_config["max_position_embeddings"]),
            "head_dim": int(hf_config.get("head_dim") or hidden_size // num_heads),
            "rms_norm_eps": float(hf_config["rms_norm_eps"]),
            "rope_theta": float(hf_config.get("rope_theta", 10000.0)),
            "rope_scaling": hf_config.get("rope_scaling"),
            "hidden_act": hf_config["hidden_act"],
            "attention_bias": bool(hf_config.get("attention_bias", False)),
            "mlp_bias": bool(hf_config.get("mlp_bias", False)),
            "tie_word_embeddings": bool(hf_config.get("tie_word_embeddings", False)),
            "pad_token_id": int(hf_config.get("pad_token_id") or 0),
            "bos_token_id": int(hf_config.get("bos_token_id") or 1),
            "eos_token_id": int(hf_config.get("eos_token_id") or 2),
            "model_name": str(hf_config.get("_name_or_path") or hf_config.get("model_type") or "custom"),
        }
        kwargs.update(overrides)
        config = cls(**kwargs)
        config.add_derived_config()
        return config

    @classmethod
    def from_pretrained(cls, model_path: str | Path, **overrides) -> "LlamaTpuConfig":
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing checkpoint config: {config_path}")
        hf_config = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.from_hf_config(hf_config, **overrides)


def get_modules_to_not_convert(config: LlamaTpuConfig) -> Optional[List[str]]:
    return getattr(config, "modules_to_not_convert", None)


def get_updated_configs(config: LlamaTpuConfig) -> List[LlamaTpuConfig]:
    """Generate one config per hidden layer.

    Mirrors ``NeuronMM``'s ``get_updated_configs``: a layer whose MLP is listed
    in ``modules_to_not_convert`` gets the fused kernels switched off, so the
    model can mix kernel and native layers (useful for bisecting an accuracy
    regression by layer).
    """
    updated_configs: List[LlamaTpuConfig] = []
    modules_to_not_convert = get_modules_to_not_convert(config) or []

    for layer_index in range(config.num_hidden_layers):
        module_pattern = f"layers.{layer_index}.mlp"
        if any(module_pattern in module for module in modules_to_not_convert):
            updated_configs.append(
                config.replace(mlp_kernel_enabled=False, contracted_mlp_enabled=False)
            )
        else:
            updated_configs.append(config)
    return updated_configs


def gate_up_blocks_for(seq_len: int, config: LlamaTpuConfig) -> GateUpBlocks:
    """Block shape for the fused Gate/Up kernel at this sequence length.

    This is the port of NeuronMM's ``get_fused_mlp_up_T_params``: NKI chose how
    many 128-tiles to group per block, Pallas takes the block shape directly.
    The row block is additionally clamped to the TPU sublane granularity,
    because for ``9 <= seq_len < 64`` the tuning table can return a ``bs`` that
    is neither a multiple of the 8 sublanes nor a divisor of the padded row
    count, which the kernel launcher rejects.
    """
    blocks = gate_up_block_params(seq_len, config.get_model_shape(), config.compress_ratio)
    padded_rows = ((max(seq_len, 8) + 7) // 8) * 8
    row_block = min(max(8, ((blocks.bs + 7) // 8) * 8), padded_rows)
    return blocks._replace(bs=row_block)


###############################################################################
# Rotary embeddings
###############################################################################

class LlamaRotaryEmbedding:
    """RoPE tables, with the Hugging Face scaling modes.

    ``Llama3RotaryEmbedding`` in NeuronMM implements only the Llama-3.1
    low/high-frequency blend. The checkpoint used by this port declares
    ``rope_type: linear``, so both are implemented here and selected from
    ``rope_scaling``; ``linear`` is the plain ``inv_freq / factor``.

    This is deliberately *not* a :class:`TpuModule`: ``cos``/``sin`` are derived
    from the config, not loaded from a checkpoint, and a plain class keeps them
    out of :meth:`TpuModule.named_parameters` so a strict load does not demand
    them.
    """

    def __init__(self, config: LlamaTpuConfig):
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.dtype = config.dtype

        inv_freq = self._compute_inv_freq(config)
        positions = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos_table = jnp.cos(emb).astype(jnp.float32)
        self.sin_table = jnp.sin(emb).astype(jnp.float32)

    def _compute_inv_freq(self, config: LlamaTpuConfig) -> jax.Array:
        freq_indices = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))

        rope_scaling = config.rope_scaling or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type is None:
            return inv_freq
        factor = float(rope_scaling.get("factor", 1.0))
        if rope_type == "linear":
            return inv_freq / factor
        if rope_type != "llama3":
            raise ValueError(f"unsupported rope_type {rope_type!r}")

        # The blend from Llama 3.1 (HF modeling_rope_utils.get_llama3_parameters),
        # vectorized where NeuronMM looped.
        old_context = float(rope_scaling["original_max_position_embeddings"])
        low_freq_factor = float(rope_scaling["low_freq_factor"])
        high_freq_factor = float(rope_scaling["high_freq_factor"])
        low_freq_wavelen = old_context / low_freq_factor
        high_freq_wavelen = old_context / high_freq_factor

        wavelen = 2 * jnp.pi / inv_freq
        smooth = (old_context / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        blended = (1 - smooth) * inv_freq / factor + smooth * inv_freq
        return jnp.where(
            wavelen < high_freq_wavelen,
            inv_freq,
            jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, blended),
        )

    def cos_sin(self, position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Return ``(cos, sin)`` shaped ``(batch, 1, seq, head_dim)``.

        The leading singleton broadcasts over the head axis of the
        ``(batch, heads, seq, head_dim)`` query/key tensors.
        """
        position_ids = jnp.asarray(position_ids)
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]
        cos = jnp.take(self.cos_table, position_ids, axis=0)
        sin = jnp.take(self.sin_table, position_ids, axis=0)
        return cos[:, None, :, :], sin[:, None, :, :]


def rotate_half(x: jax.Array) -> jax.Array:
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def apply_rotary_pos_emb(
    query: jax.Array, key: jax.Array, cos: jax.Array, sin: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Apply RoPE in fp32, then cast back, matching Hugging Face."""
    dtype = query.dtype
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    cos = cos.astype(jnp.float32)
    sin = sin.astype(jnp.float32)
    query_rotated = query * cos + rotate_half(query) * sin
    key_rotated = key * cos + rotate_half(key) * sin
    return query_rotated.astype(dtype), key_rotated.astype(dtype)


###############################################################################
# KV cache
###############################################################################

@dataclass
class LlamaKVCache:
    """A fixed-size KV cache, laid out ``(batch, kv_heads, length, head_dim)``.

    Slots are filled in token order and ``length`` records how many are live.
    Attention reads the whole buffer and masks the dead slots, so no dynamic
    slicing is needed and the shapes stay static under ``jax.jit``.
    """

    key: jax.Array
    value: jax.Array
    length: jax.Array

    @classmethod
    def empty(
        cls,
        batch_size: int,
        num_kv_heads: int,
        max_length: int,
        head_dim: int,
        dtype: Any,
    ) -> "LlamaKVCache":
        shape = (batch_size, num_kv_heads, max_length, head_dim)
        return cls(
            key=jnp.zeros(shape, dtype),
            value=jnp.zeros(shape, dtype),
            length=jnp.zeros((), jnp.int32),
        )

    @property
    def max_length(self) -> int:
        return int(self.key.shape[2])

    def write(self, key_states: jax.Array, value_states: jax.Array) -> "LlamaKVCache":
        """Append ``seq`` new tokens at the current write position."""
        start = self.length
        seq_len = key_states.shape[2]
        if self.max_length and int(seq_len) + int(start) > self.max_length:
            raise ValueError(
                f"KV cache overflow: {seq_len} new tokens at offset {start} "
                f"exceed max_sequence_length={self.max_length}"
            )
        index = (0, 0, start, 0)
        key = jax.lax.dynamic_update_slice(self.key, key_states, index)
        value = jax.lax.dynamic_update_slice(self.value, value_states, index)
        return LlamaKVCache(key=key, value=value, length=start + seq_len)


###############################################################################
# Attention
###############################################################################

@register_module("NeuronLlamaAttention")
class NeuronLlamaAttention(TpuModule):
    """Grouped-query self-attention with rotary embeddings and a KV cache.

    NeuronMM delegates the projections to ``BaseGroupQueryAttention`` plus the
    QKV kernel; there is no TPU attention kernel in this port yet, so the math
    is plain JAX. The parts that a future kernel would need are kept explicit:
    the ``(batch, heads, seq, head_dim)`` layout, the fp32 score accumulation,
    and the ``(batch, kv_heads, length, head_dim)`` cache.
    """

    def __init__(self, config: LlamaTpuConfig):
        if config.attention_bias:
            raise NotImplementedError(
                "attention_bias is not supported: the reference checkpoint and "
                "the fused kernels both assume bias-free projections"
            )
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.scaling = self.head_dim ** -0.5

        query_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Stored as (in, out): the natural matmul operand.
        self.q_proj = _empty((self.hidden_size, query_size), config.dtype)
        self.k_proj = _empty((self.hidden_size, kv_size), config.dtype)
        self.v_proj = _empty((self.hidden_size, kv_size), config.dtype)
        self.o_proj = _empty((query_size, self.hidden_size), config.dtype)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        past_key_value: Optional[LlamaKVCache] = None,
    ) -> Tuple[jax.Array, Optional[LlamaKVCache]]:
        batch_size, q_len, _ = hidden_states.shape

        query_states = hidden_states @ self.q_proj
        key_states = hidden_states @ self.k_proj
        value_states = hidden_states @ self.v_proj

        query_states = query_states.reshape(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, q_len, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, q_len, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(q_len, dtype=jnp.int32)[None, :], (batch_size, q_len)
            )
        cos, sin = self.rotary_emb.cos_sin(position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is None:
            query_start = jnp.zeros((), jnp.int32)
            present = None
            kv_len = q_len
            kv_valid = jnp.ones((kv_len,), dtype=bool)
        else:
            query_start = past_key_value.length
            present = past_key_value.write(key_states, value_states)
            key_states = present.key
            value_states = present.value
            kv_len = present.max_length
            kv_valid = jnp.arange(kv_len) < present.length

        # Repeat each KV head over the query heads that share it (HF's repeat_kv).
        key_states = jnp.repeat(key_states, self.num_kv_groups, axis=1)
        value_states = jnp.repeat(value_states, self.num_kv_groups, axis=1)

        attention_weights = jnp.matmul(
            query_states, key_states.swapaxes(-1, -2), preferred_element_type=jnp.float32
        )
        attention_weights = attention_weights * self.scaling

        query_positions = query_start + jnp.arange(q_len, dtype=jnp.int32)
        kv_positions = jnp.arange(kv_len, dtype=jnp.int32)
        mask = kv_positions[None, :] <= query_positions[:, None]
        mask = mask[None, None, :, :] & kv_valid[None, None, None, :]
        if attention_mask is not None:
            # The mask covers the tokens currently being written, which occupy
            # cache slots [0, length); pad it out to the buffer length so the
            # broadcast against the score tensor lines up.
            padding = jnp.asarray(attention_mask)[:, :kv_len].astype(bool)
            if padding.shape[1] < kv_len:
                padding = jnp.pad(padding, ((0, 0), (0, kv_len - padding.shape[1])))
            mask = mask & padding[:, None, None, :]
        attention_weights = jnp.where(
            mask, attention_weights, jnp.finfo(jnp.float32).min
        )

        attention_weights = jax.nn.softmax(attention_weights, axis=-1).astype(
            query_states.dtype
        )
        attn_output = jnp.matmul(
            attention_weights, value_states, preferred_element_type=jnp.float32
        ).astype(query_states.dtype)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, q_len, self.num_heads * self.head_dim
        )
        return attn_output @ self.o_proj, present


###############################################################################
# MLP -- low-rank (SVD-Flash) arm
###############################################################################

ACT2FN = {"silu": jax.nn.silu}


def _activation(name: str):
    if name not in ACT2FN:
        raise NotImplementedError(f"unsupported hidden_act {name!r}")
    return ACT2FN[name]


class SVD_LlamaMLP(TpuModule):
    """The plain low-rank MLP, factored exactly as NeuronMM's ``SVD_LlamaMLP``.

    Kept as the readable reference for the two-kernel path: same factors, same
    order (``v_proj`` applied first, then ``u_proj``), same ``SiLU(gate) * up``.
    In this port the *first* projection of each pair is stored as ``(hidden, rank)``
    and the second as ``(rank, intermediate)``, which is the ``u``/``v`` naming
    the Pallas kernels use -- hence ``u_gate``/``v_gate`` rather than
    ``gate_v_proj``/``gate_u_proj``.
    """

    def __init__(self, config: LlamaTpuConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = _activation(config.hidden_act)

        # NeuronMM: round(I * H * ratio / ((I + H) * 128)) * 128. Same formula,
        # shared with kernels.config so the model and the benchmarks agree.
        low_rank = config.low_rank
        if low_rank <= 0:
            raise ValueError(f"compress_ratio={config.compress_ratio} yields rank {low_rank}")

        self.u_gate = _empty((self.hidden_size, low_rank), config.dtype)
        self.v_gate = _empty((low_rank, self.intermediate_size), config.dtype)
        self.u_up = _empty((self.hidden_size, low_rank), config.dtype)
        self.v_up = _empty((low_rank, self.intermediate_size), config.dtype)
        self.u_down = _empty((self.intermediate_size, low_rank), config.dtype)
        self.v_down = _empty((low_rank, self.hidden_size), config.dtype)

    def forward(self, x: jax.Array) -> jax.Array:
        up = (x @ self.u_up) @ self.v_up
        gate = (x @ self.u_gate) @ self.v_gate
        return ((self.act_fn(gate) * up) @ self.u_down) @ self.v_down


class NeuronLlamaMLP_SVD(TpuModule):
    """The kernel-enabled low-rank MLP.

    ``forward`` dispatches on ``mlp_kernel_enabled`` exactly as NeuronMM's does:
    the fused Pallas kernels, the contracted variant, or the plain reference
    math. The dense fallback in the original was replaced by a hard-coded
    ``nki_matmul_fully_optimized_`` call; here ``_native_mlp`` really is the
    unfused low-rank math, so a speedup measured against it is meaningful.
    """

    def __init__(self, config: LlamaTpuConfig):
        if config.mlp_bias:
            raise NotImplementedError(
                "mlp_bias is not supported: the fused kernels have no bias epilogue"
            )
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.low_rank = config.low_rank
        self.act_fn = _activation(config.hidden_act)
        self.mlp_kernel_enabled = config.mlp_kernel_enabled
        self.contracted_mlp_enabled = config.contracted_mlp_enabled
        self.mixed_precision = config.mixed_precision
        self.interpret = config.interpret

        self.u_gate = _empty((self.hidden_size, self.low_rank), config.dtype)
        self.v_gate = _empty((self.low_rank, self.intermediate_size), config.dtype)
        self.u_up = _empty((self.hidden_size, self.low_rank), config.dtype)
        self.v_up = _empty((self.low_rank, self.intermediate_size), config.dtype)
        self.u_down = _empty((self.intermediate_size, self.low_rank), config.dtype)
        self.v_down = _empty((self.low_rank, self.hidden_size), config.dtype)

    def _native_mlp(self, x: jax.Array) -> jax.Array:
        """Unfused low-rank math, the honest baseline for the fused kernels."""
        logger.debug("MLP: native compiler")
        up = (x @ self.u_up) @ self.v_up
        gate = (x @ self.u_gate) @ self.v_gate
        return ((self.act_fn(gate) * up) @ self.u_down) @ self.v_down

    def _svd_baseline(self, x: jax.Array) -> jax.Array:
        """Same math as ``_native_mlp``; kept as the named comparison arm."""
        logger.debug("-" * 30 + " svd baseline mlp " + "-" * 30)
        return self._native_mlp(x)

    def _kernel_enabled_mlp(self, x: jax.Array) -> jax.Array:
        """Gate/Up/SwiGLU kernel followed by the Down XUV kernel.

        NeuronMM's ``_neuron_mm`` calls ``fused_mlp_up_T`` and then routes the
        result through ``SVDRowParallelLinear`` (``fused_three_mm_XTUV``); this
        is the same two-kernel dataflow behind one Pallas launcher.
        """
        logger.debug("-" * 30 + " neuron_mm mlp " + "-" * 30)
        batch_size, seq_len, hidden = x.shape
        blocks = gate_up_blocks_for(seq_len, self.config)
        rows = x.reshape(-1, hidden)
        output = svd_mlp(
            rows,
            self.u_gate,
            self.v_gate,
            self.u_up,
            self.v_up,
            self.u_down,
            self.v_down,
            gate_up_blocks=blocks,
            mixed_precision=self.mixed_precision,
            interpret=self.interpret,
        )
        return output.reshape(batch_size, seq_len, hidden)

    def _contracted_mlp(self, x: jax.Array) -> jax.Array:
        """Fuse Gate/Up/SwiGLU through Down-U, so the HBM boundary is Down-XU.

        The experimental path from the README: ``hidden`` never reaches HBM,
        the second kernel consumes the smaller ``(rows, down_rank)`` tensor.
        """
        batch_size, seq_len, hidden = x.shape
        blocks = gate_up_blocks_for(seq_len, self.config)
        rows = x.reshape(-1, hidden)
        down_xu = contracted_svd_mlp(
            rows,
            self.u_gate,
            self.v_gate,
            self.u_up,
            self.v_up,
            self.u_down,
            self.v_down,
            gate_up_blocks=blocks,
            mixed_precision=self.mixed_precision,
            interpret=self.interpret,
        )
        return down_xu.reshape(batch_size, seq_len, hidden)

    def forward(self, x: jax.Array) -> jax.Array:
        if self.mlp_kernel_enabled:
            if self.contracted_mlp_enabled:
                return self._contracted_mlp(x)
            return self._kernel_enabled_mlp(x)
        return self._svd_baseline(x)


###############################################################################
# MLP -- dense arm
###############################################################################

class NeuronLlamaMLP(TpuModule):
    """The dense SwiGLU MLP, with the Pallas dense kernel as its fast path.

    Selected when ``metadata["svd_llama"]`` is false, which is the first arm of
    the ``llama_inference.py`` speedup comparison. The checkpoint used by this
    port only ships SVD factors, so this arm needs a dense checkpoint to load.
    """

    def __init__(self, config: LlamaTpuConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = _activation(config.hidden_act)
        self.mlp_kernel_enabled = config.mlp_kernel_enabled
        self.mixed_precision = config.mixed_precision
        self.interpret = config.interpret

        self.gate_proj = _empty((self.hidden_size, self.intermediate_size), config.dtype)
        self.up_proj = _empty((self.hidden_size, self.intermediate_size), config.dtype)
        self.down_proj = _empty((self.intermediate_size, self.hidden_size), config.dtype)

    def _native_mlp(self, x: jax.Array) -> jax.Array:
        logger.debug("MLP: native compiler")
        gate_proj_output = x @ self.gate_proj
        up_proj_output = x @ self.up_proj
        return (self.act_fn(gate_proj_output) * up_proj_output) @ self.down_proj

    def _kernel_enabled_mlp(self, x: jax.Array) -> jax.Array:
        logger.debug("-" * 30 + " dense mlp kernel " + "-" * 30)
        batch_size, seq_len, hidden = x.shape
        output = dense_mlp(
            x.reshape(-1, hidden),
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            mixed_precision=self.mixed_precision,
            interpret=self.interpret,
        )
        return output.reshape(batch_size, seq_len, hidden)

    def forward(self, x: jax.Array) -> jax.Array:
        if self.mlp_kernel_enabled:
            return self._kernel_enabled_mlp(x)
        return self._native_mlp(x)


###############################################################################
# Decoder layer
###############################################################################

class NeuronLlamaDecoderLayer(TpuModule):
    """
    Just replace the attention with the TPU version, and MLP with the SVD one
    when the launcher asked for it.
    """

    def __init__(self, config: LlamaTpuConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        self.self_attn = _LLAMA_MODULE_MAP[config.attn_cls](config=config)

        if config.svd_llama:
            self.mlp = NeuronLlamaMLP_SVD(config)
        else:
            self.mlp = NeuronLlamaMLP(config)

        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} "
            f"and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        past_key_value: Optional[LlamaKVCache] = None,
    ) -> Tuple[jax.Array, Optional[LlamaKVCache]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


###############################################################################
# Model
###############################################################################

class NeuronLlamaModel(TpuModule):
    """
    The TPU version of the LlamaModel.
    """

    def __init__(self, config: LlamaTpuConfig):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = _empty((config.vocab_size, config.hidden_size), config.dtype)

        updated_configs = get_updated_configs(config)
        self.layers = [NeuronLlamaDecoderLayer(conf) for conf in updated_configs]
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def empty_cache(
        self,
        batch_size: int,
        max_length: Optional[int] = None,
        dtype: Any = None,
    ) -> Tuple[LlamaKVCache, ...]:
        """One zeroed cache per layer.

        ``max_length`` defaults to ``config.max_sequence_length``; because the
        attention score tensor scales with it, a caller that leaves both unset
        gets an error rather than a silent multi-gigabyte allocation.
        """
        length = max_length or self.config.max_sequence_length
        if not length:
            raise ValueError(
                "cache length is unset: pass max_length here or set "
                "config.max_sequence_length"
            )
        dtype = dtype or self.config.dtype
        return tuple(
            LlamaKVCache.empty(
                batch_size=batch_size,
                num_kv_heads=self.config.num_key_value_heads,
                max_length=length,
                head_dim=self.config.head_dim,
                dtype=dtype,
            )
            for _ in self.layers
        )

    def forward(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        past_key_values: Optional[Sequence[LlamaKVCache]] = None,
    ) -> Tuple[jax.Array, Tuple[Optional[LlamaKVCache], ...]]:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len)
            )
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        hidden_states = self.embed_tokens[input_ids]

        present_key_values: List[Optional[LlamaKVCache]] = []
        for index, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[index]
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            present_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        return hidden_states, tuple(present_key_values)


###############################################################################
# Checkpoint conversion
###############################################################################

# Projections stored as (in, out) = transpose(PyTorch Linear.weight).
_TRANSPOSED_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
_MLP_FACTOR_PAIRS = (
    # checkpoint suffix            internal name   checkpoint (out, in) -> internal
    ("gate_v_proj.weight", "u_gate"),   # (rank, hidden)     -> (hidden, rank)
    ("gate_u_proj.weight", "v_gate"),   # (inter, rank)      -> (rank, inter)
    ("up_v_proj.weight", "u_up"),
    ("up_u_proj.weight", "v_up"),
    ("down_v_proj.weight", "u_down"),   # (rank, inter)      -> (inter, rank)
    ("down_u_proj.weight", "v_down"),   # (hidden, rank)     -> (rank, hidden)
)


def add_state_dict_for_mlp(state_dict: Dict[str, Any], config: LlamaTpuConfig) -> Dict[str, Any]:
    """Rename the six SVD factors and drop the ``.weight`` suffix.

    Mirrors NeuronMM's ``add_state_dict_for_mlp``, which renamed
    ``gate_v_proj.weight`` to ``gate_proj.weight_u`` and so on. This port names
    the pair after the operand order the Pallas kernels take -- ``u`` applied
    first, ``v`` second -- so the checkpoint's ``*_v_proj`` becomes ``u_*`` and
    ``*_u_proj`` becomes ``v_*``, each transposed to ``(in, out)``.
    """
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp."
        for suffix, target in _MLP_FACTOR_PAIRS:
            source_key = prefix + suffix
            if source_key not in state_dict:
                continue
            value = np.asarray(state_dict.pop(source_key))
            if value.ndim != 2:
                raise ValueError(f"{source_key} must be 2D, got {value.shape}")
            state_dict[prefix + target] = value.T
    return state_dict


# Norm gains are the one parameter owned by a submodule (``LlamaRMSNorm`` has a
# ``weight`` attribute), so their ``.weight`` suffix stays; every other tensor
# is a bare array on its module and loses it.
_KEEP_WEIGHT_SUFFIXES = ("layernorm.weight", "norm.weight")


def _parameter_name(key: str) -> Tuple[str, bool]:
    """Map one checkpoint key to ``(internal parameter name, transpose)``.

    Only the attention and dense-MLP projections are transposed: the embedding
    table is already ``(vocab, hidden)``, the LM head is ``(vocab, hidden)`` and
    the norms are 1D.
    """
    if key == "lm_head.weight":
        return "lm_head", True
    transpose = any(key.endswith(suffix) for suffix in _TRANSPOSED_SUFFIXES)
    keeps_weight = key.endswith(_KEEP_WEIGHT_SUFFIXES)
    if key.endswith(".weight") and not keeps_weight:
        return key[: -len(".weight")], transpose
    return key, transpose


###############################################################################
# Causal LM
###############################################################################

class NeuronLlamaForCausalLM(TpuModule):
    """
    This class extends LlamaForCausalLM and creates the TPU blocks.

    Args:
        config (LlamaTpuConfig): the model and runtime configuration.
    """

    _model_cls = NeuronLlamaModel

    def __init__(self, config: LlamaTpuConfig):
        if not config.low_rank:
            config.add_derived_config()
        self.config = config
        self.model = self._model_cls(config)
        self.lm_head = _empty((config.hidden_size, config.vocab_size), config.dtype)

    @staticmethod
    def load_hf_model(model_path: str | Path, **kwargs) -> Dict[str, Any]:
        """Return the Hugging Face config mapping for ``model_path``."""
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing checkpoint config: {config_path}")
        return json.loads(config_path.read_text(encoding="utf-8"))

    @staticmethod
    def load_hf_state_dict(model_path: str | Path) -> Dict[str, Any]:
        """Read every safetensors shard of an HF checkpoint into host memory.

        Uses the shard index when present and falls back to scanning the
        directory, so both chunked and single-file checkpoints work.
        """
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "loading a checkpoint requires safetensors; install it with "
                "`python -m pip install safetensors`"
            ) from error

        model_path = Path(model_path)
        index_path = model_path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(index["weight_map"].values()))
        else:
            shard_names = sorted(path.name for path in model_path.glob("*.safetensors"))
        if not shard_names:
            raise FileNotFoundError(f"no safetensors shards found under {model_path}")

        state_dict: Dict[str, Any] = {}
        for name in shard_names:
            with safe_open(str(model_path / name), framework="numpy") as shard:
                for key in shard.keys():
                    state_dict[key] = shard.get_tensor(key)
        return state_dict

    @staticmethod
    def convert_hf_to_tpu_state_dict(
        state_dict: Dict[str, Any], config: LlamaTpuConfig
    ) -> Dict[str, Any]:
        """Map a Hugging Face state dict onto this port's parameter names.

        Every projection is transposed to ``(in, out)``; the SVD factors are
        renamed by :func:`add_state_dict_for_mlp`. Layer norms, the embedding
        table and the LM head keep their HF names (the head loses only the
        ``.weight`` suffix, since it is a bare array here).
        """
        if config.svd_llama:
            state_dict = add_state_dict_for_mlp(state_dict, config)

        converted: Dict[str, Any] = {}
        for key, value in state_dict.items():
            name, transpose = _parameter_name(key)
            converted[name] = np.asarray(value).T if transpose else value
        return converted

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        config: Optional[LlamaTpuConfig] = None,
        strict: bool = True,
        **overrides,
    ) -> "NeuronLlamaForCausalLM":
        """Build the model and load a checkpoint in one step."""
        if config is None:
            config = LlamaTpuConfig.from_pretrained(model_path, **overrides)
        model = cls(config)
        state_dict = cls.load_hf_state_dict(model_path)
        model.load_state_dict(
            cls.convert_hf_to_tpu_state_dict(state_dict, config),
            strict=strict,
            dtype=config.dtype,
        )
        return model

    @classmethod
    def get_config_cls(cls) -> Type[LlamaTpuConfig]:
        return LlamaTpuConfig

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        past_key_values: Optional[Sequence[LlamaKVCache]] = None,
    ) -> Tuple[jax.Array, Tuple[Optional[LlamaKVCache], ...]]:
        hidden_states, present_key_values = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = hidden_states @ self.lm_head
        return logits.astype(jnp.float32), present_key_values

    def build_jitted_forward(self):
        """A ``jax.jit``-ed forward; weights are closed over, so the compiled
        executable is invalidated if they are replaced."""
        return jax.jit(self.forward)

    # ---------------------------------------------------------------- generate

    def generate(
        self,
        input_ids: np.ndarray | jax.Array,
        max_new_tokens: int,
        *,
        eos_token_id: Optional[int] = None,
        jit: bool = False,
    ) -> jax.Array:
        """Greedy decode with the KV cache: one prefill, then one step per token.

        This is the modelling-side analogue of ``llama_inference.py``'s
        ``run_generation``: the benchmark harness in ``bench/`` covers latency,
        this covers correctness end to end.
        """
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        batch_size, prompt_len = input_ids.shape

        step = jax.jit(self.forward) if jit else self.forward
        caches = self.model.empty_cache(batch_size, max_length=prompt_len + max_new_tokens)
        if any(cache.max_length < prompt_len + max_new_tokens for cache in caches):
            raise ValueError("KV cache is too small for this prompt and budget")

        attention_mask = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
        logits, caches = step(input_ids, attention_mask, None, caches)
        next_token = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            if bool(jnp.all(next_token == eos_token_id)):
                break
            length = int(caches[0].length)
            position_ids = jnp.full((batch_size, 1), length, dtype=jnp.int32)
            attention_mask = jnp.ones((batch_size, length + 1), dtype=jnp.int32)
            logits, caches = step(
                next_token[:, None], attention_mask, position_ids, caches
            )
            next_token = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)
            generated.append(next_token)

        tokens = jnp.stack(generated, axis=1)
        return jnp.concatenate((input_ids, tokens), axis=1)
