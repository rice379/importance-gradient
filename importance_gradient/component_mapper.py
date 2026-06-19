"""Utilities for mapping model parameters to Transformer components."""

from __future__ import annotations

from typing import Optional, Tuple


ComponentId = Tuple[int, str]


def parse_transformer_component(param_name: str) -> Optional[ComponentId]:
    """Map common Hugging Face LLM parameter names to paper-level components.

    Returned component names follow the paper notation: Q, K, V, O, QKV, FC1,
    and FC2. Non-transformer parameters such as embeddings return None.
    """
    parts = param_name.split(".")

    layer_id = None
    for token in ("layers", "h", "block"):
        if token in parts:
            idx = parts.index(token)
            if idx + 1 < len(parts):
                try:
                    layer_id = int(parts[idx + 1])
                    break
                except Exception:
                    pass
    if layer_id is None and "transformer" in parts and "h" in parts:
        idx = parts.index("h")
        if idx + 1 < len(parts):
            try:
                layer_id = int(parts[idx + 1])
            except Exception:
                layer_id = None
    if layer_id is None:
        return None

    joined = ".".join(parts)
    if "q_proj" in parts:
        return layer_id, "Q"
    if "k_proj" in parts:
        return layer_id, "K"
    if "v_proj" in parts:
        return layer_id, "V"
    if "out_proj" in parts or "o_proj" in parts or ("dense" in parts and "attention" in parts):
        return layer_id, "O"
    if "query_key_value" in parts or "query_key_value" in joined or "qkv_proj" in parts:
        return layer_id, "QKV"
    if "fc1" in parts or "dense_h_to_4h" in parts or "gate_proj" in parts or "up_proj" in parts:
        return layer_id, "FC1"
    if "fc2" in parts or "dense_4h_to_h" in parts or "down_proj" in parts:
        return layer_id, "FC2"
    return None


def parse_opt_component(param_name: str) -> Optional[ComponentId]:
    """Backward-compatible OPT parser used by the runtime gate."""
    return parse_transformer_component(param_name)
