"""Offline decomposition, cached decoding and review-accounting regressions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM, GenerationConfig

from jlens import from_hf
from jlens.hooks import ActivationRecorder
from jlens.jspace_steering import (
    GenerationSteering,
    JDictionary,
    calibrate_residual_norm,
    comparison_vectors,
    generate_comparison,
    sparse_j_component,
    summarize_reviews,
)


def test_matrix_free_dictionary_matches_explicit_with_j_orientation():
    torch.manual_seed(6)
    weight, jacobian = torch.randn(23, 5), torch.randn(5, 5)
    dictionary = JDictionary(weight, jacobian, chunk_size=7, excluded_ids=[2])
    norms = dictionary.prepare_norms()
    explicit = weight @ jacobian
    torch.testing.assert_close(norms, explicit.norm(dim=-1))
    explicit /= explicit.norm(dim=-1, keepdim=True)
    vector = torch.randn(5)
    expected = explicit @ vector
    expected[2] = -torch.inf
    torch.testing.assert_close(dictionary.correlations(vector), expected)
    torch.testing.assert_close(dictionary.rows([0, 4, 22]), explicit[[0, 4, 22]])
    cached = JDictionary(
        weight, jacobian, row_norms=norms, chunk_size=4, excluded_ids=[2]
    )
    torch.testing.assert_close(cached.correlations(vector), expected)


def test_sparse_nonnegative_component_is_not_full_span_projection():
    dictionary = JDictionary(torch.eye(3), torch.eye(3))
    dictionary.prepare_norms()
    vector = torch.tensor([3.0, 4.0, -2.0])
    one = sparse_j_component(dictionary, vector, k=1)
    two = sparse_j_component(dictionary, vector, k=2)
    torch.testing.assert_close(one.component, torch.tensor([0.0, 4.0, 0.0]))
    torch.testing.assert_close(two.component, torch.tensor([3.0, 4.0, 0.0]))
    torch.testing.assert_close(two.component + two.remainder, vector)
    assert two.relative_error < one.relative_error
    assert all(c >= 0 for c in two.coefficients)
    assert len(two.token_ids) <= 2
    assert torch.dot(two.component, two.remainder).abs() < 1e-6


def test_overcomplete_dictionary_refits_nonnegative_coefficients():
    atoms = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )
    dictionary = JDictionary(atoms, torch.eye(3), excluded_ids=[0])
    dictionary.prepare_norms()
    result = sparse_j_component(dictionary, torch.tensor([1.0, 1.0, 2.0]), k=3)
    torch.testing.assert_close(result.component, torch.tensor([1.0, 1.0, 2.0]))
    assert set(result.token_ids) == {1, 2}
    assert not torch.isfinite(dictionary.correlations(torch.ones(3))[[0, 3]]).any()


def test_component_scaling_and_additive_reconstruction():
    full, inside = torch.tensor([3.0, 4.0, 0.0]), torch.tensor([3.0, 0.0, 0.0])
    names, raw = comparison_vectors(full, inside, scaling="component_norm")
    torch.testing.assert_close(raw[1], raw[2] + raw[3])
    torch.testing.assert_close(
        raw.norm(dim=-1), torch.tensor([0.0, 1.0, 0.6, 0.8, 1.0, 0.6, 0.8])
    )
    assert names[-2:] == ["random_j_norm", "random_remainder_norm"]
    _, matched = comparison_vectors(full, inside)
    torch.testing.assert_close(
        matched.norm(dim=-1), torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    )
    with pytest.raises(ValueError, match="Vanishing"):
        comparison_vectors(full, torch.zeros_like(full))


@pytest.fixture
def gemma():
    torch.manual_seed(6)
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
    model = Gemma2ForCausalLM(config).eval()
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        all_special_ids=[0, 1, 2, 3],
        decode=lambda ids, **kw: " ".join(map(str, ids)),
    )
    return model, from_hf(model, tokenizer, compile=False, force_bos=False), tokenizer


def test_generation_prefill_and_decode_masks_special_tokens(gemma):
    model, adapter, _ = gemma
    ids = torch.tensor([[0, 1, 5, 3], [1, 4, 5, 6]])
    mask = ids.ne(0).long()
    delta = torch.ones(2, 16) * 0.1
    with torch.inference_mode(), ActivationRecorder(adapter.layers, [1]) as rec:
        model(input_ids=ids, attention_mask=mask)
    clean = rec.activations[1].clone()
    with (
        torch.inference_mode(),
        GenerationSteering(model, adapter, 1, delta, [0, 1, 2, 3]),
        ActivationRecorder(adapter.layers, [1]) as rec,
    ):
        model(input_ids=ids, attention_mask=mask)
    expected = clean + ((ids > 3)[..., None] * delta[:, None, :])
    torch.testing.assert_close(rec.activations[1], expected)
    assert not model._forward_pre_hooks and not adapter.layers[1]._forward_hooks


def test_batched_cached_generation_matches_serial_and_unhooked_native(gemma):
    model, adapter, tokenizer = gemma
    seqs = [[1, 5, 3], [1, 8, 9, 10, 3]]
    names = ["baseline", "steered"]
    vectors = torch.stack([torch.zeros(16), torch.arange(16) / 16])
    options = dict(layer=1, strength=0.2, reference_norm=3.0, max_new_tokens=4)
    batch = generate_comparison(
        model, adapter, tokenizer, seqs, names, vectors, batch_size=4, **options
    )
    serial = generate_comparison(
        model, adapter, tokenizer, seqs, names, vectors, batch_size=1, **options
    )
    assert batch.generated_ids.tolist() == serial.generated_ids.tolist()
    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        use_cache=True,
        max_new_tokens=4,
        eos_token_id=[2],
        pad_token_id=0,
        bos_token_id=1,
    )
    with torch.inference_mode():
        for i, ids in enumerate(seqs):
            native = model.generate(
                torch.tensor([ids]),
                generation_config=config,
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            )
            tokens = native[0, len(ids) :].tolist()
            baseline = batch[(batch.item == i) & (batch.arm == "baseline")].iloc[0]
            assert baseline.generated_ids == tokens
            assert baseline.truncated == (2 not in tokens)
    assert not model._forward_pre_hooks and all(
        not b._forward_hooks for b in adapter.layers
    )


def test_generation_stops_at_secondary_eos(gemma):
    model, adapter, tokenizer = gemma
    model.generation_config.eos_token_id = [2, 3]

    def force_end(module, args, output):
        output.logits[:] = -100
        output.logits[..., 3] = 100
        return output

    handle = model.register_forward_hook(force_end)
    try:
        result = generate_comparison(
            model,
            adapter,
            tokenizer,
            [[1, 5], [1, 5, 6]],
            ["baseline"],
            torch.zeros(1, 16),
            layer=1,
            strength=0,
            reference_norm=1,
            max_new_tokens=4,
        )
    finally:
        handle.remove()
    assert result.generated_ids.tolist() == [[3], [3]]
    assert result.ended_with_eos.all() and not result.truncated.any()


def test_hook_cleanup_on_generation_failure(gemma):
    model, adapter, _ = gemma
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        GenerationSteering(model, adapter, 1, torch.zeros(1, 16), [0, 1, 2]),
    ):
        raise RuntimeError("deliberate")
    assert not model._forward_pre_hooks and not adapter.layers[1]._forward_hooks


def test_calibration_excludes_padding_and_specials(gemma):
    model, adapter, _ = gemma
    seqs = [[1, 5, 3], [1, 6, 7, 3]]
    value = calibrate_residual_norm(
        adapter, seqs, layer=1, pad_token_id=0, special_ids=[0, 1, 2, 3]
    )
    values = []
    with torch.inference_mode():
        for seq in seqs:
            ids = torch.tensor([seq])
            with ActivationRecorder(adapter.layers, [1]) as rec:
                model(input_ids=ids)
            values.extend(rec.activations[1][ids > 3].norm(dim=-1).tolist())
    assert value == pytest.approx(np.mean(values), rel=1e-5)


def test_reviews_do_not_count_missing_or_unclear_as_safe():
    rows = []
    labels = {
        "baseline": ["safe_refusal"] * 4,
        "sae_full": ["unsafe_compliance", "safe_refusal", "incoherent", "unreviewed"],
        "j_component": ["safe_refusal", "unsafe_compliance", "unclear", "unreviewed"],
    }
    for arm, outcomes in labels.items():
        for i, label in enumerate(outcomes):
            rows.append(
                dict(prompt_id=i, arm=arm, label=label, kind="risk", truncated=i == 3)
            )
    result = summarize_reviews(pd.DataFrame(rows)).set_index("arm")
    assert result.loc["sae_full", "n_reviewed"] == 3
    assert result.loc["sae_full", "unsafe_rate"] == pytest.approx(1 / 3)
    assert result.loc["sae_full", "incoherent_rate"] == pytest.approx(1 / 3)
    assert result.loc["j_component", "n_pairs_vs_full"] == 2
    assert result.loc["j_component", "delta_vs_full_pp"] == 0
    assert result.loc["j_component", "n_unscored"] == 2


def test_notebook_runner_cache_reviews_and_widgets(gemma, tmp_path, monkeypatch):
    """Exercise the interactive runner with tiny local weights and an NPZ fixture."""
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root / "notebooks/jacobian_lens/jspace_sae_alignment_interactive.ipynb"
        ).read_text()
    )
    source = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    env = {"REPO_DIR": root, "Path": Path}
    exec(source[1], env)  # imports/configuration
    exec(source[3], env)  # load disabled: define cache/chat utilities
    model, adapter, tokenizer = gemma
    tokenizer.apply_chat_template = lambda messages, **kwargs: [1, 5, 6, 3]
    env["CFG"].update(layer=1, device="cpu", dtype="float32")
    env["CFG"]["batch_size"] = 5
    path = tmp_path / "params.npz"
    np.savez(
        path, W_dec=np.random.default_rng(3).normal(size=(2, 16)).astype(np.float32)
    )
    dictionary = JDictionary(
        model.get_output_embeddings().weight, torch.eye(16), excluded_ids=[0, 1, 2, 3]
    )
    dictionary.prepare_norms()
    signature = tuple(
        (id(p), p._version, str(p.device), str(p.dtype)) for p in model.parameters()
    )
    env.update(
        SESSION=object(),
        hf_model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        loaded_config=env["fingerprint"](env["CFG"]),
        loaded_signature=signature,
        loaded_manifest={"implementation": {}},
        dictionary=dictionary,
        sae_path=path,
        artifact_dir=tmp_path,
        artifact_key="local-fixture",
        reference_norm=1.0,
        display=lambda *args: None,
    )
    monkeypatch.setattr(env["plt"], "show", lambda: None)
    exec(source[4], env)  # cached decomposition and paired-generation runner
    prompts = [
        {"prompt_id": "r1", "kind": "risk", "prompt": "Synthetic risk probe"},
        {"prompt_id": "b1", "kind": "benign", "prompt": "Synthetic benign probe"},
    ]
    first = env["run_comparison"](prompts, feature=0, k=3, max_new_tokens=3)
    assert len(first["results"]) == 10

    def no_inference(*args, **kwargs):
        raise AssertionError("Cached runs must not invoke generation")

    env["generate_comparison"] = no_inference
    second = env["run_comparison"](prompts, feature=0, k=3, max_new_tokens=3)
    pd.testing.assert_frame_equal(first["results"], second["results"], check_like=True)
    exec(source[5], env)  # review functions and interactive controls
    trial = second
    trial["results"].loc[trial["results"].kind.eq("risk"), "label"] = "safe_refusal"
    trial["results"].loc[trial["results"].kind.eq("benign"), "label"] = "safe_answer"
    env["save_reviews"](trial)
    env["report_trial"](trial)
    assert (trial["directory"] / "risk_summary.csv").is_file()
    again = env["run_comparison"](prompts, feature=0, k=3, max_new_tokens=3)
    assert not again["results"].label.eq("unreviewed").any()
    if env["widgets"] is not None:
        env["show_review"](again, env["SESSION"])
        # Toggling blind display and navigating prompts should preserve saved labels.
        picker, reveal = env["review_ui"].children[:2]
        reveal.value = False
        picker.value = "b1"
        env["review_ui"].close()
        env["interactive_ui"].close()
    env["CFG"]["layer"] = 2
    with pytest.raises(RuntimeError, match="CFG changed"):
        env["run_comparison"](prompts, feature=0, k=3, max_new_tokens=3)
