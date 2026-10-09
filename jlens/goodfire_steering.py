"""Tensor-only loading of released Goodfire SAE decoder directions."""

from collections.abc import Sequence
from pathlib import Path

import torch


def _goodfire_decoder(path: str | Path, d_model: int) -> torch.Tensor:
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    decoder = state.get("decoder_linear.weight")
    if not torch.is_tensor(decoder) or decoder.ndim != 2 or decoder.shape[0] != d_model:
        raise ValueError(
            "Expected Goodfire decoder_linear.weight [d_model, n_features]"
        )
    return decoder


def goodfire_decoder_norms(path: str | Path, *, d_model: int) -> torch.Tensor:
    """Read CPU column norms; zero columns are removed features, not directions.

    Memory-map the trusted tensor-only state dict and scan in feature chunks.
    Nonfinite columns are returned as nonfinite norms for explicit exclusion.
    """
    decoder = _goodfire_decoder(path, d_model)
    return torch.cat(
        [
            decoder[:, start : start + 1024].float().norm(dim=0)
            for start in range(0, decoder.shape[1], 1024)
        ]
    )


def load_goodfire_directions(
    path: str | Path, feature_ids: Sequence[int], *, d_model: int
) -> torch.Tensor:
    """Read selected Goodfire decoder columns as CPU fp32 unit direction rows."""
    if not feature_ids or len(set(feature_ids)) != len(feature_ids):
        raise ValueError("feature_ids must be nonempty and unique")
    decoder = _goodfire_decoder(path, d_model)
    if any(type(i) is not int or not 0 <= i < decoder.shape[1] for i in feature_ids):
        raise ValueError("feature ID outside Goodfire decoder")
    rows = decoder[:, list(feature_ids)].T.float().contiguous()
    norms = rows.norm(dim=1, keepdim=True)
    if (
        not torch.isfinite(rows).all()
        or not torch.isfinite(norms).all()
        or (norms <= 0).any()
    ):
        raise ValueError("Selected feature is removed, nonfinite, or zero")
    return rows / norms
