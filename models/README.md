# TPU LLaMA model for SVD-Flash

`modeling_llama.py` is the TPU counterpart of `NeuronMM/models/llama/modeling_llama.py`.

## Structure

The programming pattern deliberately mirrors NeuronMM:

- **Config split**: `LlamaTpuConfig` combines the Hugging Face shape attributes with runtime switches (dtype, kernel flags, metadata) the same way `LlamaInferenceConfig` + `NeuronConfig` does.
- **Module registry**: `@register_module("NeuronLlamaAttention")` decorator, used by `NeuronLlamaDecoderLayer.__init__` to look up `config.attn_cls`.
- **Updated configs**: `get_updated_configs` produces one config per layer, so layers listed in `modules_to_not_convert` get the fused kernels disabled (useful for bisecting accuracy regressions).
- **MLP dispatch**: `NeuronLlamaMLP_SVD.forward` routes on `mlp_kernel_enabled` between the fused Pallas kernels and the plain reference math, exactly as the NeuronMM original does. The `metadata["svd_llama"]` flag picks the low-rank MLP over the dense one.
- **Checkpoint conversion**: `convert_hf_to_tpu_state_dict` transposes every projection to `(in, out)` and renames the six SVD factors through `add_state_dict_for_mlp`, mirroring the NeuronMM state-dict hooks.

## What changes

| NeuronMM (Trainium, PyTorch) | this port (TPU, JAX) |
|------------------------------|----------------------|
| `torch.nn.Module` | `TpuModule` (arrays + submodules) |
| `nki.jit` kernels in `modeling_llama.py` | `jax.experimental.pallas` kernels in `svdflash_tpu/kernels` |
| `fused_mlp_up_T` | `pallas_ops.svd_mlp` (Gate/Up kernel) |
| `fused_three_mm_XTUV` | `pallas_ops.svd_mlp` (Down XUV kernel) |
| `nki_matmul_fully_optimized_` | `pallas_ops.dense_mlp` |
| `get_fused_mlp_up_T_params` | `kernels.config.gate_up_block_params` |
| `torch.autograd.Function` wrappers | dropped — inference only |
| tensor parallelism (`tp_degree`) | single device; kernels are TensorCore-mesh parallel internally |
| `torch.int32` rank tensors | not needed, no collectives |

## Components

- **`TpuModule`**: minimal stand-in for `torch.nn.Module`. Parameters are `jax.Array` attributes, submodules are nested `TpuModule` instances. `named_parameters()` walks both so a state dict can be mapped onto dotted names.
- **`LlamaRMSNorm`**: root-mean-square layer norm, computed in fp32 like Hugging Face.
- **`LlamaTpuConfig`**: model shape plus runtime switches. The `from_pretrained` classmethod reads `config.json` and the launcher passes `metadata={"svd_llama": ..., "compress_ratio": ...}`.
- **`LlamaRotaryEmbedding`**: RoPE tables with Llama-3.1 scaling and linear mode (the checkpoint declares `rope_type: linear`). Deliberately *not* a `TpuModule` so the tables stay out of `named_parameters()`.
- **`LlamaKVCache`**: fixed-size cache `(batch, kv_heads, length, head_dim)`. Slots are filled in token order; attention reads the whole buffer and masks the dead slots.
- **`NeuronLlamaAttention`**: grouped-query self-attention with rotary embeddings and a KV cache. Plain JAX (no TPU attention kernel yet), but the layout and fp32 score accumulation are kept explicit for a future kernel.
- **`SVD_LlamaMLP`**: plain low-rank MLP, kept as the readable reference.
- **`NeuronLlamaMLP_SVD`**: kernel-enabled low-rank MLP. `forward` dispatches on `mlp_kernel_enabled` between the fused Pallas kernels (`_kernel_enabled_mlp`), the contracted variant (`_contracted_mlp`), and the plain reference math (`_native_mlp`).
- **`NeuronLlamaMLP`**: dense SwiGLU MLP, with the Pallas dense kernel as its fast path. Selected when `metadata["svd_llama"]` is false.
- **`NeuronLlamaDecoderLayer`**: transformer block. Instantiates the attention and MLP (SVD or dense, depending on `config.svd_llama`), plus two RMSNorms.
- **`NeuronLlamaModel`**: the base model. Embedding table, decoder layers, final norm.
- **`NeuronLlamaForCausalLM`**: causal language model. Wraps `NeuronLlamaModel` and adds the LM head. `from_pretrained` loads a checkpoint in one step. `generate` is greedy decode with the KV cache.

## Checkpoint compatibility

Parameter names are the Hugging Face keys with:

- Trailing `.weight` dropped for projections and the LM head (bare arrays here).
- Trailing `.weight` kept for norm gains (they are attributes of `LlamaRMSNorm` submodules).
- Every projection transposed to `(in, out)` — the natural matmul operand, which is `Linear.weight.T` in PyTorch.
- SVD factors renamed: checkpoint `gate_v_proj.weight` → internal `u_gate`, `gate_u_proj.weight` → `v_gate`, each transposed to `(in, out)`.

The conversion is `convert_hf_to_tpu_state_dict`, mirroring NeuronMM's state-dict hooks.

## Verification

`tests/check_model_state_dict.py` stubs JAX and imports the model on CPU, confirming every parameter name and shape matches the converted checkpoint. Run it with:

```bash
python -m svdflash_tpu.tests.check_model_state_dict
```

Expected output:

```
hidden=4096 heads=32 kv_heads=8 head_dim=128
low_rank=2816 (checkpoint rank is 2816)
num_kv_groups=4
model parameters: 387
converted checkpoint tensors: 387

all parameter names and shapes match the checkpoint
```

## Usage

```python
from svdflash_tpu.models.modeling_llama import (
    NeuronLlamaForCausalLM,
    LlamaTpuConfig,
)

model_path = "path/to/llama3-8b_0.9_svd"
config = LlamaTpuConfig.from_pretrained(
    model_path,
    model_name="llama-3.1-8b",
    compress_ratio=0.9,
    mlp_kernel_enabled=True,
    max_sequence_length=512,  # prompt + max_new_tokens
)
config.metadata = {"svd_llama": True, "compress_ratio": 0.9}

model = NeuronLlamaForCausalLM.from_pretrained(model_path, config=config)

# Greedy decode
import jax.numpy as jnp
input_ids = jnp.array([[1, 2, 3, 4]])  # (batch, seq)
output_ids = model.generate(input_ids, max_new_tokens=16, jit=True)
```

## What is not reproduced

Two behaviors of NeuronMM are intentionally different:

1. **Dense baseline**: In NeuronMM, `NeuronLlamaMLP._native_mlp` is replaced by `nki_matmul_fully_optimized_`, so the reported speedup is kernel-versus-kernel. Here `_native_mlp` is the real dense math and `_kernel_enabled_mlp` is the Pallas kernel, so the two arms are distinguishable.
2. **Attention**: plain JAX. No TPU attention kernel exists in this port yet, but the QKV layout, RoPE tables, GQA repeat, and cache layout are written so one can be dropped in behind `NeuronLlamaAttention` later.

## Next steps

- Wire this model into `bench/all_mlp_layers.py` so the 32-layer MLP sweep uses real checkpoint weights through this loader instead of random factors.
- Add a prefill/decode harness that runs the full model (not MLP-only) and measures perplexity on a validation set.
- Port the remaining components: quantization, speculative decoding, bucketing.
- Build a TPU attention kernel and replace the JAX math in `NeuronLlamaAttention.forward`.
