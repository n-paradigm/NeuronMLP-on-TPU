"""Load NeuronMLP SVD factors from a sharded Hugging Face checkpoint."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

import jax


FACTOR_SUFFIXES = (
    "gate_v_proj.weight",
    "gate_u_proj.weight",
    "up_v_proj.weight",
    "up_u_proj.weight",
    "down_v_proj.weight",
    "down_u_proj.weight",
)


def checkpoint_metadata(model_path: str | Path) -> tuple[dict, dict]:
    """Read and minimally validate config and safetensors shard metadata."""
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if "weight_map" not in index:
        raise ValueError(f"safetensors index has no weight_map: {index_path}")
    return config, index


def factor_keys(layer_index: int) -> tuple[str, ...]:
    if layer_index < 0:
        raise ValueError(f"layer index must be non-negative, got {layer_index}")
    prefix = f"model.layers.{layer_index}.mlp."
    return tuple(prefix + suffix for suffix in FACTOR_SUFFIXES)


def validate_factor_shapes(
    factors: Sequence[jax.Array],
    *,
    hidden_size: int,
    intermediate_size: int,
) -> int:
    """Validate checkpoint orientation after transpose and return its rank."""
    if len(factors) != 6:
        raise ValueError(f"expected six MLP factors, got {len(factors)}")
    rank = factors[0].shape[1]
    expected = (
        (hidden_size, rank),
        (rank, intermediate_size),
        (hidden_size, rank),
        (rank, intermediate_size),
        (intermediate_size, rank),
        (rank, hidden_size),
    )
    actual = tuple(tuple(value.shape) for value in factors)
    if actual != expected:
        raise ValueError(
            "invalid transposed MLP factor shapes; "
            f"expected {expected}, got {actual}"
        )
    dtypes = {value.dtype for value in factors}
    if len(dtypes) != 1:
        raise TypeError(f"MLP factor dtypes differ: {sorted(map(str, dtypes))}")
    return rank


def inspect_mlp_checkpoint(
    model_path: str | Path, layer_indices: Sequence[int] | None = None,
) -> dict:
    """Validate real shard headers without loading the weight payloads."""
    from safetensors import safe_open
    import jax.numpy as jnp

    model_path = Path(model_path)
    config, index = checkpoint_metadata(model_path)
    if config.get("hidden_act", "silu") != "silu" or config.get("mlp_bias", False):
        raise ValueError("the fused kernel requires bias-free SwiGLU MLPs")
    num_layers = int(config["num_hidden_layers"])
    hidden, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
    selected = tuple(range(num_layers)) if layer_indices is None else tuple(layer_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select at least one layer, without duplicates")
    if min(selected) < 0 or max(selected) >= num_layers:
        raise ValueError(f"layer indices must be in [0, {num_layers})")
    weight_map = index["weight_map"]
    keys = tuple(key for layer in selected for key in factor_keys(layer))
    missing = [key for key in keys if key not in weight_map]
    if missing:
        raise KeyError(f"checkpoint is missing MLP factors: {missing}")
    names = sorted({weight_map[key] for key in keys})
    layers = []
    dtype_map = {"BF16": jnp.bfloat16, "F32": jnp.float32}
    with ExitStack() as stack:
        shards = {}
        for name in names:
            path = model_path / name
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint shard: {path}")
            shards[name] = stack.enter_context(safe_open(str(path), framework="numpy"))
        for layer in selected:
            descriptors, tensor_info = [], []
            for key in factor_keys(layer):
                view = shards[weight_map[key]].get_slice(key)
                shape, dtype = tuple(view.get_shape()), view.get_dtype()
                if len(shape) != 2 or dtype not in dtype_map or min(shape) <= 0:
                    raise ValueError(f"unsupported MLP tensor {key}: {shape}, {dtype}")
                descriptors.append(jax.ShapeDtypeStruct(shape[::-1], dtype_map[dtype]))
                tensor_info.append({"key": key, "shape": list(shape), "dtype": dtype,
                                    "shard": weight_map[key]})
            rank = validate_factor_shapes(
                descriptors, hidden_size=hidden, intermediate_size=intermediate)
            nbytes = sum(a.size * a.dtype.itemsize for a in descriptors)
            layers.append({"layer": layer, "rank": rank, "dtype": tensor_info[0]["dtype"],
                           "factor_bytes": nbytes, "tensors": tensor_info})
    return {"model_path": str(model_path.resolve()), "num_hidden_layers": num_layers,
            "hidden_size": hidden, "intermediate_size": intermediate,
            "selected_layers": list(selected), "shards": names, "layers": layers}


def load_mlp_factors(
    model_path: str | Path,
    layer_indices: Sequence[int],
) -> tuple[tuple[jax.Array, ...], ...]:
    """Load selected layers into device memory in the Pallas U/V convention.

    PyTorch Linear stores weights as ``[output, input]``. NeuronMM names the
    first projection ``v_proj`` and the second projection ``u_proj``. The
    Pallas API instead consumes natural matmul operands, so every checkpoint
    tensor is transposed and the v/u name order is deliberately reversed.
    """
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "loading real factors requires safetensors; install it with "
            "`python -m pip install safetensors`"
        ) from error

    model_path = Path(model_path)
    config, index = checkpoint_metadata(model_path)
    weight_map = index["weight_map"]
    layer_indices = tuple(layer_indices)
    if not layer_indices:
        raise ValueError("at least one layer must be selected")

    all_keys = tuple(key for layer in layer_indices for key in factor_keys(layer))
    missing_keys = tuple(key for key in all_keys if key not in weight_map)
    if missing_keys:
        raise KeyError(f"checkpoint is missing MLP factors: {missing_keys}")

    shard_names = {weight_map[key] for key in all_keys}
    missing_shards = tuple(
        sorted(name for name in shard_names if not (model_path / name).is_file())
    )
    if missing_shards:
        raise FileNotFoundError(
            "checkpoint shards required by the selected layers are missing: "
            + ", ".join(missing_shards)
        )

    layers = []
    with ExitStack() as stack:
        shards = {
            name: stack.enter_context(
                safe_open(str(model_path / name), framework="flax")
            )
            for name in shard_names
        }
        for layer_index in layer_indices:
            tensors = []
            for key in factor_keys(layer_index):
                tensor = shards[weight_map[key]].get_tensor(key)
                transposed = jax.device_put(tensor.T)
                jax.block_until_ready(transposed)
                tensors.append(transposed)
                del tensor
            factors = tuple(tensors)
            validate_factor_shapes(
                factors,
                hidden_size=int(config["hidden_size"]),
                intermediate_size=int(config["intermediate_size"]),
            )
            layers.append(factors)

    jax.block_until_ready(tuple(layers))
    return tuple(layers)
