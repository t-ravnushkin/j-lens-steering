"""Offline steering tests using tiny randomly initialized Gemma weights."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from transformers import Gemma2Config, Gemma2ForCausalLM

from jlens import from_hf
from jlens.hooks import ActivationRecorder
from jlens.sae_steering import (
    BatchedDirectionSteering,
    SteeringCondition,
    encode_choice_prompts,
    evaluate_choice_steering,
    load_gemma_scope_directions,
    summarize_capability,
)


@pytest.fixture
def gemma():
    torch.manual_seed(4)
    config = Gemma2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=32,
        attn_logit_softcapping=10.0,
        final_logit_softcapping=2.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    hf = Gemma2ForCausalLM(config).eval()
    return hf, from_hf(hf, None, compile=False, force_bos=False)


def bank():
    return torch.eye(16)[:2]


@pytest.mark.parametrize("positions", ["all", "last"])
def test_batched_matches_serial_and_true_native_baseline(gemma, positions):
    hf, model = gemma
    sequences = [[1, 5, 7], [1, 8, 9, 10, 11], [1, 6]]
    conditions = [
        SteeringCondition("baseline", 0, 0),
        SteeringCondition("positive", 0, 0.3),
        SteeringCondition("negative", 1, -0.2),
    ]
    options = dict(layer=1, pad_token_id=0, positions=positions)
    batched = evaluate_choice_steering(
        model,
        sequences,
        [3, 4, 5, 6],
        bank(),
        conditions,
        batch_size=7,
        max_batch_tokens=25,
        **options,
    )
    serial = evaluate_choice_steering(
        model, sequences, [3, 4, 5, 6], bank(), conditions, batch_size=1, **options
    )
    cols = [f"logp_{c}" for c in "ABCD"]
    np.testing.assert_allclose(batched[cols], serial[cols], atol=2e-6)
    assert batched.top_token_id.tolist() == serial.top_token_id.tolist()
    base = batched[batched.condition.eq("baseline")]
    with torch.inference_mode():
        # Independent native causal-LM output checks the final norm and softcap.
        for i, seq in enumerate(sequences):
            native = hf(torch.tensor([seq])).logits[0, -1].float().log_softmax(-1)
            np.testing.assert_allclose(
                base.iloc[i][cols].to_numpy(dtype=float),
                native[[3, 4, 5, 6]].numpy(),
                atol=2e-6,
            )
    assert not np.allclose(base[cols], batched[batched.condition.eq("positive")][cols])
    assert all(not layer._forward_hooks for layer in model.layers)


def test_update_norm_positions_and_zero_are_exact(gemma):
    _, model = gemma
    ids = torch.tensor([[1, 7, 8, 0], [1, 4, 5, 6]])
    mask = torch.tensor([[False, True, True, False], [False, True, True, True]])
    with torch.inference_mode(), ActivationRecorder(model.layers, [1]) as rec:
        model.forward(ids, attention_mask=ids.ne(0).long())
    clean = rec.activations[1].clone()
    with (
        torch.inference_mode(),
        BatchedDirectionSteering(model, 1, bank(), torch.tensor([0.2, 0.0]), mask),
        ActivationRecorder(model.layers, [1]) as rec,
    ):
        model.forward(ids, attention_mask=ids.ne(0).long())
    patched = rec.activations[1]
    assert torch.equal(clean[1], patched[1])
    assert torch.equal(clean[0, [0, 3]], patched[0, [0, 3]])
    expected = 0.2 * clean[0, 1:3].norm(dim=-1)
    torch.testing.assert_close((patched - clean)[0, 1:3].norm(dim=-1), expected)
    assert torch.count_nonzero((patched - clean)[0, 1:3, 1:]) == 0


def test_hook_cleans_up_after_exception_and_preserves_tuple():
    class Block(nn.Module):
        def forward(self, x):
            return x, "preserved"

    class Model:
        n_layers, d_model = 1, 2
        layers = [Block()]

    model = Model()
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        BatchedDirectionSteering(
            model,
            0,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0.5]),
            torch.tensor([[False, True]]),
        ),
    ):
        out, extra = model.layers[0](torch.ones(1, 2, 2))
        assert extra == "preserved" and out[0, 1, 0] > 1
        raise RuntimeError("deliberate")
    assert not model.layers[0]._forward_hooks


def test_npz_decoder_orientation_and_normalization(tmp_path):
    path = tmp_path / "params.npz"
    np.savez(path, W_dec=np.array([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]]))
    torch.testing.assert_close(
        load_gemma_scope_directions(path, [1, 0], d_model=2),
        torch.tensor([[0.0, 1.0], [0.6, 0.8]]),
    )
    for ids, width in [([0], 3), ([3], 2), ([2], 2), ([0, 0], 2)]:
        with pytest.raises(ValueError):
            load_gemma_scope_directions(path, ids, d_model=width)


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if len(text) > 1 and text[-2:] in [" A", " B", " C", " D"]:
            return [7] * len(text[:-2]) + [10 + "ABCD".index(text[-1])]
        return [7] * len(text)


def test_bos_boundary_and_length_contract():
    seqs, choices = encode_choice_prompts(Tokenizer(), ["Q:", "long Q:"])
    assert seqs[0] == [1, 7, 7] and choices == [10, 11, 12, 13]
    with pytest.raises(ValueError, match="limit"):
        encode_choice_prompts(Tokenizer(), ["long Q:"], max_seq_len=3)

    class BadTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            return [7] * len(text)

    with pytest.raises(ValueError, match="single context-stable"):
        encode_choice_prompts(BadTokenizer(), ["Q:"])


def test_summary_paired_loss_and_probability_mass():
    items = pd.DataFrame(
        {"item": [0, 1, 2, 3], "subject": ["s", "s", "t", "t"], "answer": [0, 1, 2, 3]}
    )
    rows = []
    for condition, predictions in [("baseline", [0, 1, 0, 3]), ("sae", [1, 1, 2, 0])]:
        for i, predicted in enumerate(predictions):
            probs = np.full(4, 0.05)
            probs[predicted] = 0.5
            rows.append(
                {
                    "item": i,
                    "condition": condition,
                    **dict(
                        zip([f"logp_{c}" for c in "ABCD"], np.log(probs), strict=True)
                    ),
                }
            )
    scores = pd.DataFrame(rows)
    summary, detail = summarize_capability(scores, items, seed=9)
    sae = summary.query("condition == 'sae' and subject == 'ALL'").iloc[0]
    assert sae.accuracy == 0.5 and sae.loss_pp == 25
    assert sae.correct_to_wrong == 2 and sae.wrong_to_correct == 1
    assert sae.loss_ci_low_pp <= 25 <= sae.loss_ci_high_pp
    np.testing.assert_allclose(detail.answer_mass, 0.65)
    np.testing.assert_allclose(detail.conditional_nll, detail.nll + np.log(0.65))
    with pytest.raises(ValueError, match="same items"):
        summarize_capability(scores.iloc[:-1], items)


def test_batch_budget_rejects_oversize(gemma):
    _, model = gemma
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_choice_steering(
            model,
            [[1, 3, 4]],
            [3, 4, 5, 6],
            bank(),
            [SteeringCondition("baseline", 0, 0)],
            layer=1,
            pad_token_id=0,
            max_batch_tokens=2,
        )


def test_notebook_real_run_path_resumes_without_loading_weights(tmp_path, monkeypatch):
    """Exercise orchestration/export/cache branches with a local stand-in, not downloads."""
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks/jacobian_lens/gemma_sae_steering_mmlu.ipynb").read_text()
    )
    sources = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    env = {"REPO_DIR": root, "Path": Path}
    exec(sources[1], env)  # Imports/configuration; checkout/install cell is not run.
    env["RUN_GEMMA"] = True
    env["CACHE_DIR"] = tmp_path / "cache"
    env["CFG"].update(
        subjects=["abstract_algebra"],
        per_subject=2,
        feature_ids=[0],
        random_seeds=[11],
        strengths=[0.0, 0.1],
        cache_chunk_items=1,
        device="cpu",
        dtype="float32",
    )
    env["display"] = lambda *args: None
    monkeypatch.setattr(env["plt"], "show", lambda: None)

    class FakeTokenizer(Tokenizer):
        pad_token_id = 0
        backend_tokenizer = SimpleNamespace(to_str=lambda: "test-tokenizer")

        def decode(self, ids):
            return str(ids)

    class FakeDataset:
        rows = [
            dict(
                question=f"Question {i}?",
                choices=["one", "two", "three", "four"],
                subject="abstract_algebra",
                answer=i,
            )
            for i in range(3)
        ]

        def __getitem__(self, key):
            return (
                [r[key] for r in self.rows] if isinstance(key, str) else self.rows[key]
            )

    class FakeModel(nn.Module):
        n_layers, d_model = 26, 2304
        input_device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, self.d_model)
            self.layers = nn.ModuleList([nn.Identity() for _ in range(self.n_layers)])
            self.head = nn.Linear(self.d_model, 32)

        def forward(self, input_ids, attention_mask=None):
            h = self.embed(input_ids)
            for layer in self.layers:
                h = layer(h)
            return h

        def unembed(self, h):
            return self.head(h)

    loads = []

    def load_model(*args, **kwargs):
        loads.append(True)
        return FakeModel()

    sae_path = tmp_path / "params.npz"
    np.savez(sae_path, W_dec=np.ones((1, 2304), dtype=np.float32))
    env.update(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        load_dataset=lambda *a, **k: FakeDataset(),
        hf_hub_download=lambda *a, **k: str(sae_path),
        from_hf=lambda model, *a, **k: model,
    )
    exec(sources[3], env)  # Prepare real manifests and condition/direction banks.
    exec(sources[4], env)  # Evaluate and save chunks.
    assert len(loads) == 1
    original = env["scores"].copy()
    exec(sources[4], env)  # Resume; no model load or inference is needed.
    assert len(loads) == 1
    pd.testing.assert_frame_equal(original, env["scores"])
    exec(sources[5], env)  # Tables and plots.
    exec(sources[6], env)  # Changed-answer inspection.
    exec(sources[7], env)  # Conclusions.
    assert (env["run_dir"] / "capability.png").is_file()
    assert (env["run_dir"] / "summary.csv").is_file()
    env["CFG"]["positions"] = "last"
    with pytest.raises(RuntimeError, match="Configuration changed"):
        env["require_prepared"]()
