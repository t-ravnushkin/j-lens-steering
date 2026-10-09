"""Offline Goodfire column orientation, Llama hooks and replication-stage checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pandas as pd
import pytest
import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from jlens import from_hf
from jlens.goodfire_steering import goodfire_decoder_norms, load_goodfire_directions
from jlens.jspace_steering import generate_comparison


def test_goodfire_uses_columns_and_exposes_removed_features(tmp_path):
    path = tmp_path / "sae.pth"
    decoder = torch.tensor(
        [[3.0, 0.0, 0.0, float("nan")], [4.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    torch.save({"decoder_linear.weight": decoder}, path)
    norms = goodfire_decoder_norms(path, d_model=3)
    torch.testing.assert_close(norms[:3], torch.tensor([5.0, 2.0, 0.0]))
    assert torch.isnan(norms[3])
    rows = load_goodfire_directions(path, [1, 0], d_model=3)
    torch.testing.assert_close(rows, torch.tensor([[0.0, 1.0, 0.0], [0.6, 0.8, 0.0]]))
    for ids in ([2], [3], [4], [-1], [0, 0], [], [True]):
        with pytest.raises(ValueError):
            load_goodfire_directions(path, ids, d_model=3)
    with pytest.raises(ValueError, match="d_model"):
        load_goodfire_directions(path, [0], d_model=4)


def test_llama_sdpa_steering_preserves_native_baseline_and_batching():
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = [2, 3]
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        all_special_ids=[0, 1, 2, 3],
        decode=lambda ids, **kw: str(ids),
    )
    adapter = from_hf(model, tokenizer, compile=False, force_bos=False)
    inputs = [[1, 5, 6], [1, 7]]
    vectors = torch.stack([torch.zeros(16), torch.arange(16) / 16])
    options = dict(layer=1, strength=0.5, reference_norm=2, max_new_tokens=5)
    batch = generate_comparison(
        model,
        adapter,
        tokenizer,
        inputs,
        ["baseline", "sae_full"],
        vectors,
        batch_size=4,
        **options,
    )
    serial = generate_comparison(
        model,
        adapter,
        tokenizer,
        inputs,
        ["baseline", "sae_full"],
        vectors,
        batch_size=1,
        **options,
    )
    assert batch.generated_ids.tolist() == serial.generated_ids.tolist()
    native_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        use_cache=True,
        max_new_tokens=5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=[2, 3],
    )
    with torch.inference_mode():
        for i, tokens in enumerate(inputs):
            native = model.generate(
                input_ids=torch.tensor([tokens]),
                attention_mask=torch.ones(1, len(tokens), dtype=torch.long),
                generation_config=native_config,
            )
            assert (
                native[0, len(tokens) :].tolist()
                == batch[(batch.item == i) & batch.arm.eq("baseline")]
                .iloc[0]
                .generated_ids
            )
    assert not model._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in adapter.layers)


def test_llama_replication_notebook_cache_and_empty_judgments(tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(
        root / "notebooks/jacobian_lens/llama_goodfire_rogue_scalpel_replication.ipynb",
        4,
    )
    ns = dict(Path=Path, REPO_DIR=root)
    exec(notebook.cells[4].source, ns)
    ns["STUDY"] = {
        **ns["PROFILES"]["smoke"],
        "feature_seed": 42,
        "prompt_seed": 42,
        "random_seed_start": 42,
        "max_new_tokens": 512,
        "prompt_chunk": 4,
        "condition_chunk": 8,
    }
    source = pd.DataFrame(
        [
            dict(
                Index=i,
                Goal=f"Benign test topic {i}",
                Behavior="test",
                Category=str(i // 10),
            )
            for i in range(100)
        ]
    )
    csv = tmp_path / "data.csv"
    source.to_csv(csv, index=False)
    checkpoint = tmp_path / "standin.pth"
    checkpoint.write_bytes(b"test provenance only")
    loads = []

    def column_norms(path, **kwargs):
        loads.append("norms")
        norms = torch.ones(65536)
        norms[0] = 0
        norms[1] = float("nan")
        return norms

    ns.update(
        MODE="generate",
        PROFILE="smoke",
        CACHE_DIR=tmp_path,
        hf_hub_download=lambda repo, file, **kw: str(
            csv if file.endswith(".csv") else checkpoint
        ),
        goodfire_decoder_norms=column_norms,
        tqdm=lambda seq, **kw: seq,
        display=lambda *a, **kw: None,
    )
    exec(notebook.cells[6].source, ns)
    assert ns["sae_metadata"]["n_excluded"] == 2
    assert not {0, 1} & set(ns["feature_ids"])
    assert set(ns["conditions"].family) == {"baseline", "sae_full", "random"}
    assert len(ns["conditions"]) == 9 and ns["LOAD_MODEL"]
    first_plan = ns["plan_id"]
    exec(notebook.cells[6].source, ns)
    assert ns["plan_id"] == first_plan and loads == ["norms"]
    calls = []

    def generate(model, adapter, tokenizer, sequences, names, vectors, **kwargs):
        calls.append(len(sequences) * len(names))
        # All SAE rows are empty; baseline/random rows are benign text.
        return pd.DataFrame(
            [
                dict(
                    item=i,
                    arm=name,
                    generated_ids=[3] if name.startswith("sae_full") else [4, 3],
                    text="" if name.startswith("sae_full") else "A benign answer.",
                    ended_with_eos=True,
                    truncated=False,
                    n_generated_tokens=1 if name.startswith("sae_full") else 2,
                    n_content_tokens=0 if name.startswith("sae_full") else 1,
                    repeated_trigram_fraction=0,
                    stop_token_id=3,
                    stop_reason="native_stop",
                )
                for i in range(len(sequences))
                for name in names
            ]
        )

    ns.update(
        hf_model=object(),
        adapter=object(),
        tokenizer=object(),
        reference_norm=1,
        condition_vectors=torch.zeros(9, 4),
        generation_ids=[[1, 4]] * 20,
        generate_comparison=generate,
    )
    exec(notebook.cells[10].source, ns)
    assert sum(calls) == 180
    exec(notebook.cells[6].source, ns)
    assert not ns["LOAD_MODEL"] and ns["missing_jobs"] == []
    exec(
        notebook.cells[8].source, ns
    )  # Completed generation must skip all target downloads.
    ns.update(MODE="judge")
    ns["JUDGE"]["device"] = "cpu"
    judged_ids = []
    model_loads = []
    dummy_model = SimpleNamespace()
    dummy_model.to = lambda device: dummy_model
    dummy_model.eval = lambda: dummy_model

    def judge_loader(*args, **kwargs):
        model_loads.append(1)
        assert kwargs["attn_implementation"] == "sdpa"
        return dummy_model

    def judge(model, tokenizer, rows, **kwargs):
        assert len(rows) <= 16
        assert all(r["text"] for r in rows)
        judged_ids.extend(r["response_id"] for r in rows)
        return [
            dict(
                response_id=r["response_id"],
                judge_status="ok",
                unsafe=False,
                coherent=True,
                refusal=False,
                reason="Benign.",
                judge_text="test",
                judge_generated_ids=[4, 3],
            )
            for r in rows
        ]

    ns.update(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=judge_loader),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *a, **kw: SimpleNamespace(
                chat_template="test",
                backend_tokenizer=SimpleNamespace(to_str=lambda: "test"),
            )
        ),
        judge_batch=judge,
    )
    exec(notebook.cells[12].source, ns)
    assert len(judged_ids) == 100 and len(set(judged_ids)) == 100
    exec(notebook.cells[12].source, ns)
    assert model_loads == [1] and len(judged_ids) == 100
    exec(notebook.cells[14].source, ns)
    assert ns["outcomes"].response_id.is_unique and len(ns["outcomes"]) == 180
    assert ns["summary"].coverage.eq(1).all()
    sae = ns["summary"][ns["summary"].family.eq("sae_full")]
    assert sae.incoherent_rate.eq(1).all() and sae.refusal_rate.eq(0).all()
    exec(notebook.cells[17].source, ns)
    audit = pd.read_csv(ns["JUDGE_DIR"] / "human_audit.csv")
    assert not {"judge_unsafe", "family"} & set(audit)
    # Changed judge configuration leaves target plan identity untouched.
    ns["JUDGE"]["batch_size"] = 8
    exec(notebook.cells[6].source, ns)
    assert ns["plan_id"] == first_plan
    ns["CFG"]["batch_size"] = 1
    ns["MODE"] = "generate"
    with pytest.raises(RuntimeError, match="Settings/items changed"):
        exec(notebook.cells[8].source, ns)
    # Corrupted response identities fail the resume check.
    path = ns["block_path"](ns["jobs"][0])
    payload = json.loads(path.read_text())
    payload["records"][0]["response_id"] = "wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance"):
        ns["read_block"](ns["jobs"][0])
