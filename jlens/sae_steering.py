"""Batched residual-direction steering and multiple-choice capability evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


@dataclass(frozen=True)
class SteeringCondition:
    """One direction-bank index and signed residual-relative steering strength."""

    name: str
    direction: int
    strength: float


def load_gemma_scope_directions(
    path: str | Path, feature_ids: Sequence[int], *, d_model: int
) -> torch.Tensor:
    """Read selected decoder rows from an official Gemma Scope NPZ archive.

    Returns CPU fp32 unit vectors. This is additive decoder-direction steering;
    it neither reconstructs the residual nor clamps latents. The caller must
    check model identity and the residual hook site against the SAE metadata.
    """
    if not feature_ids or len(set(feature_ids)) != len(feature_ids):
        raise ValueError("feature_ids must be nonempty and unique")
    with np.load(path, allow_pickle=False) as archive:
        decoder = archive["W_dec"]
        if decoder.ndim != 2 or decoder.shape[1] != d_model:
            raise ValueError("W_dec must have shape [n_features, d_model]")
        if any(
            not isinstance(i, int) or not 0 <= i < len(decoder) for i in feature_ids
        ):
            raise ValueError("feature ID outside W_dec")
        selected = torch.from_numpy(decoder[list(feature_ids)].astype(np.float32))
    norms = selected.norm(dim=-1, keepdim=True)
    if not torch.isfinite(selected).all() or (norms <= 0).any():
        raise ValueError("Decoder directions must be finite and nonzero")
    return selected / norms


class BatchedDirectionSteering(AbstractContextManager):
    """Add per-row unit directions at one block output, with a token-position mask.

    Each selected token gets ``strength * ||h||_2 * unit_direction``. Single-layer
    steering uses the clean incoming residual norm. Zero strength is an exact
    no-op. All constants are on-device before the forward pass.
    """

    def __init__(
        self,
        model: LensModel,
        layer: int,
        directions: torch.Tensor,
        strengths: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if not 0 <= layer < model.n_layers:
            raise ValueError("layer is outside the model")
        if directions.ndim != 2 or directions.shape[1] != model.d_model:
            raise ValueError("directions must have shape [batch, d_model]")
        if strengths.shape != (directions.shape[0],):
            raise ValueError("strengths must have shape [batch]")
        if positions.ndim != 2 or positions.shape[0] != directions.shape[0]:
            raise ValueError("positions must have shape [batch, sequence]")
        if positions.dtype != torch.bool:
            raise ValueError("positions must be boolean")
        if len({directions.device, strengths.device, positions.device}) != 1:
            raise ValueError("Hook tensors must share a device")
        self._block = model.layers[layer]
        self._directions = directions.float()
        self._strengths = strengths.float()
        self._positions = positions
        self._handle = None

    def _hook(self, module, inputs, output):
        h = output if torch.is_tensor(output) else output[0]
        if h.shape[:2] != self._positions.shape or h.device != self._positions.device:
            raise ValueError("Hook batch shape/device differs from prepared tensors")
        delta = (
            h.float().norm(dim=-1, keepdim=True)
            * self._strengths[:, None, None]
            * self._directions[:, None, :]
        )
        selected = self._positions & self._strengths[:, None].ne(0)
        patched = torch.where(selected[..., None], (h.float() + delta).to(h.dtype), h)
        return patched if torch.is_tensor(output) else (patched, *output[1:])

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("Steering context is already active")
        self._handle = self._block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def encode_choice_prompts(
    tokenizer, prompts: Sequence[str], *, max_seq_len: int = 2048
) -> tuple[list[list[int]], list[int]]:
    """Prepend one BOS; verify context-stable, single-token A/B/C/D continuations.

    Prompts end in ``Answer:``; completions are `` A`` through `` D``. Never
    truncate questions or silently take the last token of a multi-token label.
    """
    if tokenizer.bos_token_id is None:
        raise ValueError("This prompt protocol requires a BOS token")
    choice_ids = None
    sequences = []
    for prompt in prompts:
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        if not prefix or prefix[0] == tokenizer.bos_token_id:
            raise ValueError("Supply nonempty prompts without an embedded BOS")
        candidates = []
        for letter in "ABCD":
            full = tokenizer.encode(prompt + " " + letter, add_special_tokens=False)
            if full[:-1] != prefix or len(full) != len(prefix) + 1:
                raise ValueError("Answer labels must be single context-stable tokens")
            candidates.append(full[-1])
        if len(set(candidates)) != 4:
            raise ValueError("Answer label token IDs must be distinct")
        if choice_ids is not None and candidates != choice_ids:
            raise ValueError("Answer label token IDs vary across prompts")
        choice_ids = candidates
        ids = [tokenizer.bos_token_id, *prefix]
        if len(ids) > max_seq_len:
            raise ValueError(f"Prompt has {len(ids)} tokens; limit is {max_seq_len}")
        sequences.append(ids)
    if choice_ids is None:
        raise ValueError("At least one prompt is required")
    return sequences, choice_ids


@torch.inference_mode()
def evaluate_choice_steering(
    model: LensModel,
    sequences: Sequence[Sequence[int]],
    choice_ids: Sequence[int],
    directions: torch.Tensor,
    conditions: Sequence[SteeringCondition],
    *,
    layer: int,
    pad_token_id: int,
    batch_size: int = 8,
    max_batch_tokens: int = 4096,
    positions: str = "all",
) -> pd.DataFrame:
    """Batch prompt × condition pairs with length sorting and right padding.

    ``model`` must support ``forward(..., attention_mask=...)`` and independent
    batch rows in eval mode. Input sequences begin with BOS. ``all`` patches
    all real non-BOS tokens; ``last`` patches each row's final real token.
    The final block's last real residual is decoded with the native model
    normalization/head/logit transforms. Returns full-vocabulary log probabilities
    of the four labels and the full-vocabulary top token ID.
    """
    if positions not in {"all", "last"}:
        raise ValueError("positions must be 'all' or 'last'")
    if batch_size < 1 or max_batch_tokens < 1:
        raise ValueError("Batch limits must be positive")
    if not sequences or any(len(s) < 2 for s in sequences):
        raise ValueError("Supply nonempty BOS-prefixed sequences")
    if max(map(len, sequences)) > max_batch_tokens:
        raise ValueError("One sequence exceeds max_batch_tokens; no truncation allowed")
    if len(choice_ids) != 4 or len(set(choice_ids)) != 4:
        raise ValueError("Exactly four distinct choice IDs are required")
    if not conditions or len({c.name for c in conditions}) != len(conditions):
        raise ValueError("Condition names must be nonempty and unique")
    if directions.ndim != 2 or directions.shape[1] != model.d_model:
        raise ValueError("Direction bank must have shape [directions, d_model]")
    if any(
        not 0 <= c.direction < len(directions) or not math.isfinite(c.strength)
        for c in conditions
    ):
        raise ValueError("Invalid condition direction or strength")
    device = model.input_device
    bank = directions.to(device=device, dtype=torch.float32)
    if not torch.isfinite(bank).all() or not torch.allclose(
        bank.norm(dim=-1), torch.ones(len(bank), device=device), atol=1e-5
    ):
        raise ValueError("Direction bank must contain finite unit vectors")
    strength_bank = torch.tensor([c.strength for c in conditions], device=device)
    direction_index = torch.tensor([c.direction for c in conditions], device=device)
    labels = torch.tensor(choice_ids, device=device)
    pairs = sorted(
        ((i, j) for i in range(len(sequences)) for j in range(len(conditions))),
        key=lambda pair: len(sequences[pair[0]]),
    )
    records = []
    offset = 0
    while offset < len(pairs):
        end = offset
        while end < len(pairs) and end - offset < batch_size:
            if (end - offset + 1) * len(sequences[pairs[end][0]]) > max_batch_tokens:
                break
            end += 1
        batch = pairs[offset:end]
        lengths = torch.tensor([len(sequences[i]) for i, _ in batch], device=device)
        width = max(len(sequences[i]) for i, _ in batch)
        ids = torch.tensor(
            [
                list(sequences[i]) + [pad_token_id] * (width - len(sequences[i]))
                for i, _ in batch
            ],
            device=device,
        )
        time = torch.arange(width, device=device)[None, :]
        mask = time < lengths[:, None]
        selected = (
            (mask & time.ne(0)) if positions == "all" else time.eq(lengths[:, None] - 1)
        )
        ci = torch.tensor([j for _, j in batch], device=device)
        with (
            BatchedDirectionSteering(
                model, layer, bank[direction_index[ci]], strength_bank[ci], selected
            ),
            ActivationRecorder(model.layers, [model.n_layers - 1]) as recorder,
        ):
            model.forward(ids, attention_mask=mask.long())
        final = recorder.activations[model.n_layers - 1]
        rows = torch.arange(len(batch), device=device)
        logits = model.unembed(final[rows, lengths - 1]).float()
        log_probs = logits.log_softmax(dim=-1)[:, labels]
        values = (
            torch.cat((log_probs, logits.argmax(dim=-1)[:, None]), dim=1).cpu().numpy()
        )
        for (item, condition), value in zip(batch, values, strict=True):
            records.append(
                {
                    "item": item,
                    "condition": conditions[condition].name,
                    **{
                        f"logp_{c}": float(v)
                        for c, v in zip("ABCD", value[:4], strict=True)
                    },
                    "top_token_id": int(value[4]),
                }
            )
        offset = end
    return (
        pd.DataFrame(records).sort_values(["item", "condition"]).reset_index(drop=True)
    )


def summarize_capability(
    scores: pd.DataFrame,
    items: pd.DataFrame,
    *,
    baseline: str = "baseline",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired accuracy loss, stratified bootstrap CI, NLL, answer mass and flips.

    ``items`` has unique ``item``, ``subject`` and integer ``answer`` (0..3).
    Positive loss means degradation. Intervals are pointwise, conditional on the
    selected subjects, and do not correct for searching across steering settings.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if items.empty or items.item.duplicated().any():
        raise ValueError("Items must be nonempty with unique item IDs")
    if not items.answer.isin(range(4)).all():
        raise ValueError("Answers must be integers from 0 to 3")
    if scores.duplicated(["item", "condition"]).any():
        raise ValueError("Duplicate item-condition scores")
    if baseline not in set(scores.condition):
        raise ValueError("Baseline scores are missing")
    expected = set(items.item)
    if any(set(group.item) != expected for _, group in scores.groupby("condition")):
        raise ValueError("Every condition must cover exactly the same items")
    detailed = scores.merge(items, on="item", validate="many_to_one")
    logp = detailed[[f"logp_{c}" for c in "ABCD"]].to_numpy()
    if not np.isfinite(logp).all():
        raise ValueError("Nonfinite answer log probabilities")
    choice_logmass = np.logaddexp.reduce(logp, axis=1)
    detailed["prediction"] = logp.argmax(axis=1)
    detailed["correct"] = detailed.prediction.eq(detailed.answer)
    detailed["answer_mass"] = np.exp(choice_logmass)
    detailed["nll"] = -logp[
        np.arange(len(detailed)), detailed.answer.to_numpy(dtype=int)
    ]
    detailed["conditional_nll"] = detailed.nll + choice_logmass
    base = detailed[detailed.condition.eq(baseline)].set_index("item")
    detailed["baseline_correct"] = detailed.item.map(base.correct)
    detailed["baseline_prediction"] = detailed.item.map(base.prediction)
    rng = np.random.default_rng(seed)
    result = []
    for condition, group in detailed.groupby("condition", sort=False):
        for subject, subset in [
            ("ALL", group),
            *list(group.groupby("subject", sort=True)),
        ]:
            loss = subset.baseline_correct.astype(float) - subset.correct.astype(float)
            bootstrap = np.zeros(n_bootstrap)
            for _, stratum in subset.assign(loss=loss).groupby("subject", sort=True):
                x = stratum.loss.to_numpy()
                bootstrap += rng.choice(x, (n_bootstrap, len(x)), replace=True).sum(
                    axis=1
                )
            bootstrap /= len(subset)
            lo, hi = np.quantile(bootstrap, [0.025, 0.975])
            result.append(
                dict(
                    condition=condition,
                    subject=subject,
                    n=len(subset),
                    baseline_accuracy=subset.baseline_correct.mean(),
                    accuracy=subset.correct.mean(),
                    loss_pp=100 * loss.mean(),
                    loss_ci_low_pp=100 * lo,
                    loss_ci_high_pp=100 * hi,
                    correct_to_wrong=int(
                        (subset.baseline_correct & ~subset.correct).sum()
                    ),
                    wrong_to_correct=int(
                        (~subset.baseline_correct & subset.correct).sum()
                    ),
                    prediction_flip_rate=subset.prediction.ne(
                        subset.baseline_prediction
                    ).mean(),
                    nll=subset.nll.mean(),
                    conditional_nll=subset.conditional_nll.mean(),
                    answer_mass=subset.answer_mass.mean(),
                )
            )
    return pd.DataFrame(result), detailed
