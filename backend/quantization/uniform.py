"""Teaching quantizer. Integer values use int8 storage, not packed GGUF/MLX weights."""
import numpy as np


def quantize(weights, bits=4, axis=None):
    weights = np.asarray(weights, dtype=np.float32)
    if bits not in range(2, 9) or weights.size == 0 or not np.isfinite(weights).all():
        raise ValueError("Expected finite nonempty weights and 2–8 bits")
    if axis is not None and not -weights.ndim <= axis < weights.ndim:
        raise ValueError("Channel axis is outside tensor dimensions")
    limit = 2 ** (bits - 1) - 1
    reduction = None if axis is None else tuple(i for i in range(weights.ndim) if i != axis % weights.ndim)
    maximum = np.max(np.abs(weights), axis=reduction, keepdims=True)
    scale = np.where(maximum == 0, 1.0, maximum / limit)
    values = np.clip(np.rint(weights / scale), -limit, limit).astype(np.int8)
    restored = values.astype(np.float32) * scale
    return values, scale, {"bits": bits, "channel_axis": axis, "mse": float(np.mean((weights - restored) ** 2)),
                           "max_error": float(np.max(np.abs(weights - restored)))}
