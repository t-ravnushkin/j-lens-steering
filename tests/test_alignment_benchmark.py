"""Offline benchmark sampling, judge parsing, inference and paired statistics."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from jlens.alignment_benchmark import (
    benchmark_summary,
    digest,
    judge_batch,
    judge_input,
    make_conditions,
    paired_component_effects,
    parse_judgment,
    select_benchmark_items,
)


def verdict(**overrides):
    value = dict(verdict="SAFE", coherent=True, refusal=True, reason="Declined.")
    return "Ignored UNSAFE reasoning</think>" + json.dumps({**value, **overrides})


def test_stratified_sampling_is_order_independent_and_excludes_targets():
    source = pd.DataFrame(
        [
            dict(
                Index=i,
                Goal=f"Goal {i}",
                Category=str(i // 4),
                Behavior=f"Behavior {i}",
                Target="Never prefill this",
            )
            for i in range(8)
        ]
    )
    selected = select_benchmark_items(
        {"harmful": source, "benign": source}, per_category=2, seed=7
    )
    reordered = select_benchmark_items(
        {"benign": source.iloc[::-1], "harmful": source.iloc[::-1]},
        per_category=2,
        seed=7,
    )
    pd.testing.assert_frame_equal(selected, reordered)
    assert len(selected) == 8 and selected.prompt_id.is_unique
    assert selected.groupby(["kind", "category"]).size().eq(2).all()
    assert "Never prefill" not in selected.to_json()
    with pytest.raises(ValueError, match="fewer"):
        select_benchmark_items({"harmful": source}, per_category=5, seed=7)


def test_condition_grid_has_one_baseline_and_independent_random_controls():
    conditions = make_conditions([4, 7], [10, 11, 12], [0.5, 1.0])
    assert len(conditions) == 1 + 2 * (2 * 3 + 3)
    assert conditions.condition_id.is_unique
    assert conditions.family.eq("baseline").sum() == 1
    assert conditions.family.eq("random").sum() == 6
    for strengths in ([0], [float("nan")], [1, 1]):
        with pytest.raises(ValueError):
            make_conditions([4], [10], strengths)
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})
    with pytest.raises(ValueError):
        digest({"bad": float("nan")})


@pytest.mark.parametrize(
    "text,truncated,status",
    [
        (verdict(), False, "ok"),
        (verdict(verdict="UNSAFE", refusal=False), False, "ok"),
        (verdict(), True, "judge_truncated"),
        ('{"verdict":"SAFE"}', False, "missing_think_boundary"),
        ("</think>UNSAFE", False, "invalid"),
        (verdict(coherent="true"), False, "invalid"),
        (verdict(verdict="UNSAFE"), False, "contradictory"),
        (
            verdict(verdict="UNSAFE", refusal=False, coherent=False),
            False,
            "contradictory",
        ),
        (
            '</think>{"verdict":[],"coherent":true,"refusal":false,"reason":"x"}',
            False,
            "invalid",
        ),
    ],
)
def test_judge_parser_never_reads_reasoning_as_verdict(text, truncated, status):
    result = parse_judgment(text, truncated=truncated)
    assert result["judge_status"] == status
    if status != "ok":
        assert result["unsafe"] is None


class JudgeTokenizer:
    pad_token_id = 0
    bos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False and not kwargs["truncation"]
        assert kwargs["enable_thinking"] is True
        return [1] + [4] * (10 if "OVERLONG" in messages[0]["content"] else 2)

    def decode(self, ids, **kwargs):
        if ids[0] == 1:
            return "prompt<|im_start|>assistant\n"
        return verdict()


class JudgeModel:
    training = False
    generation_config = SimpleNamespace(eos_token_id=[2, 3])

    def __init__(self):
        self.batches = []

    def get_input_embeddings(self):
        return SimpleNamespace(weight=torch.zeros(1))

    def forward(self):
        pass

    def generate(self, input_ids, attention_mask, generation_config):
        self.batches.append(len(input_ids))
        assert generation_config.do_sample is False
        assert generation_config.eos_token_id == [2, 3]
        tokens = torch.tensor([[8, 3]] * len(input_ids))
        return torch.cat([input_ids, tokens], dim=-1)


def test_judge_batches_and_retains_overlength_as_unscored():
    model = JudgeModel()
    rows = [
        dict(response_id=str(i), prompt="benign request", text="response")
        for i in range(3)
    ]
    rows.append(dict(response_id="long", prompt="OVERLONG", text="response"))
    result = judge_batch(
        model,
        JudgeTokenizer(),
        rows,
        batch_size=2,
        max_input_tokens=5,
        max_new_tokens=4,
        max_batch_tokens=20,
    )
    indexed = {r["response_id"]: r for r in result}
    assert model.batches == [2, 1]
    assert indexed["long"]["judge_status"] == "input_too_long"
    assert indexed["long"]["unsafe"] is None
    assert all(indexed[str(i)]["judge_status"] == "ok" for i in range(3))
    assert indexed["0"]["judge_generated_ids"] == [8, 3]
    assert '"response": "response"' in judge_input("benign request", "response")


def crossed_frame():
    records = []
    for feature in range(3):
        for prompt in range(4):
            for family in ("sae_full", "j_component", "remainder"):
                records.append(
                    dict(
                        response_id=f"{feature}-{prompt}-{family}",
                        condition_id=f"{feature}-{family}",
                        vector_id=feature,
                        prompt_id=str(prompt),
                        category=str(prompt // 2),
                        kind="harmful",
                        family=family,
                        strength=1.0,
                        judge_status="ok",
                        unsafe=family != "j_component",
                        coherent=True,
                        refusal=False,
                        truncated=prompt == 0,
                    )
                )
    return pd.DataFrame(records)


def test_paired_bootstrap_and_missing_judgment_bounds():
    frame = crossed_frame()
    effects = paired_component_effects(frame, n_bootstrap=100, seed=1).set_index(
        "family"
    )
    assert effects.loc["j_component", "delta_vs_full_pp"] == -100
    assert effects.loc["j_component", "ci_high_pp"] == -100
    assert effects.loc["remainder", "delta_vs_full_pp"] == 0
    frame = frame.astype({key: "object" for key in ("unsafe", "coherent", "refusal")})
    frame.loc[0, ["judge_status", "unsafe", "coherent", "refusal"]] = [
        "invalid",
        None,
        None,
        None,
    ]
    effects = paired_component_effects(frame, n_bootstrap=100, seed=1)
    assert effects.n_planned_pairs.eq(12).all()
    assert effects.n_scored_pairs.eq(11).all()
    summary = benchmark_summary(frame).set_index("family")
    full = summary.loc["sae_full"]
    assert full.n == 12 and full.n_scored == 11
    assert full.compliance_rate == 1
    assert full.unresolved_lower == 11 / 12 and full.unresolved_upper == 1
    assert summary.loc["j_component", "safe_truncated"] == 3
    frame.loc[:, "judge_status"] = "not_judged"
    assert np.isnan(benchmark_summary(frame).compliance_rate).all()
    assert paired_component_effects(frame, n_bootstrap=2).n_scored_pairs.eq(0).all()


def test_benchmark_notebook_stage_caches_resume_and_keep_audit_blinded(tmp_path):
    """Run the notebook orchestration with local CSVs and deterministic model stand-ins."""
    from pathlib import Path

    import nbformat

    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(
        root / "notebooks/jacobian_lens/jspace_sae_rogue_scalpel_benchmark.ipynb",
        as_version=4,
    )
    ns = {"Path": Path, "REPO_DIR": root}
    exec(notebook.cells[4].source, ns)
    source = pd.DataFrame(
        [
            dict(
                Index=i,
                Goal=f"Explain a benign topic {i}",
                Behavior="test",
                Category=str(i // 10),
            )
            for i in range(100)
        ]
    )
    csv = tmp_path / "local.csv"
    source.to_csv(csv, index=False)
    ns.update(
        MODE="generate",
        CACHE_DIR=tmp_path,
        N_BOOTSTRAP=10,
        hf_hub_download=lambda *a, **kw: str(csv),
        display=lambda *a: None,
        tqdm=lambda sequence, **kw: sequence,
    )
    exec(notebook.cells[6].source, ns)
    assert ns["LOAD_MODEL"]
    assert len(ns["items"]) == 20
    generation_calls = []

    def fake_generate(model, adapter, tokenizer, sequences, names, vectors, **kwargs):
        generation_calls.append((len(sequences), len(names)))
        return pd.DataFrame(
            [
                dict(
                    item=i,
                    arm=name,
                    generated_ids=[4, 3],
                    text="A benign answer.",
                    ended_with_eos=True,
                    truncated=False,
                    n_generated_tokens=2,
                    n_content_tokens=1,
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
        generation_ids=[[1, 4]] * len(ns["items"]),
        condition_vectors=torch.zeros(len(ns["conditions"]), 4),
        reference_norm=1,
        generate_comparison=fake_generate,
    )
    exec(notebook.cells[11].source, ns)
    assert len(generation_calls) == len(ns["jobs"])
    exec(notebook.cells[6].source, ns)
    assert ns["missing_jobs"] == [] and not ns["LOAD_MODEL"]
    # Analysis before judging preserves the full planned denominator.
    exec(notebook.cells[13].source, ns)
    exec(notebook.cells[15].source, ns)
    assert ns["outcomes"].judge_status.eq("not_judged").all()
    assert ns["summary"].coverage.eq(0).all()

    tokenizer = JudgeTokenizer()
    tokenizer.chat_template = "test native template"
    tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "test tokenizer")
    model = JudgeModel()
    model.to = lambda device: model
    model.eval = lambda: model
    loads = []

    def load_model(*args, **kwargs):
        assert kwargs["attn_implementation"] == "sdpa"
        loads.append(1)
        return model

    ns.update(
        MODE="judge",
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
    )
    ns["JUDGE"]["device"] = "cpu"
    exec(notebook.cells[13].source, ns)
    assert len(loads) == 1
    assert max(model.batches) == 16
    assert all(size <= 16 for size in model.batches)
    assert sum(model.batches) == len(ns["items"]) * len(ns["conditions"])
    exec(notebook.cells[13].source, ns)
    assert len(loads) == 1  # Cached judging must not load weights again.
    exec(notebook.cells[15].source, ns)
    assert ns["summary"].coverage.eq(1).all()
    assert ns["summary"].compliance_rate.eq(0).all()
    assert ns["outcomes"].response_id.is_unique
    exec(notebook.cells[18].source, ns)
    audit_path = ns["JUDGE_DIR"] / "human_audit.csv"
    audit = pd.read_csv(audit_path)
    assert not {"judge_unsafe", "judge_reason", "family", "vector_id"} & set(audit)
    audit["human_unsafe"] = 0
    audit.to_csv(audit_path, index=False)
    exec(notebook.cells[18].source, ns)
    assert pd.read_csv(audit_path).human_unsafe.eq(0).all()
    # Tampered response identities cannot be accepted as a completed block.
    path = ns["block_path"](ns["jobs"][0])
    payload = json.loads(path.read_text())
    payload["records"][0]["response_id"] = "wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance"):
        ns["read_block"](ns["jobs"][0])


@pytest.mark.parametrize(
    "state",
    ["missing", "empty_calibration", "changed", "offline_override", "overlength"],
)
def test_benchmark_loader_preflight_before_weights(state, tmp_path):
    from pathlib import Path

    import nbformat

    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(
        root / "notebooks/jacobian_lens/jspace_sae_rogue_scalpel_benchmark.ipynb",
        as_version=4,
    )
    ns = {"Path": Path, "REPO_DIR": root}
    exec(notebook.cells[4].source, ns)
    ns["CFG"]["device"] = "cpu"
    items = pd.DataFrame([dict(kind="harmful", prompt="Explain a benign test topic.")])
    if state == "empty_calibration":
        items["kind"] = "benign"
    conditions = make_conditions([0], [1], [1.0])
    plan = dict(
        model=ns["CFG"],
        study=ns["STUDY"],
        items=items.to_dict("records"),
        conditions=conditions.to_dict("records"),
    )
    ns.update(
        MODE="generate",
        PLAN=plan,
        plan_id=digest(plan),
        items=items,
        conditions=conditions,
        RUN_DIR=tmp_path,
        jobs=[(0, 1, 0, 1)],
        read_block=lambda job: None,
    )
    downloads = []

    def load_tokenizer(*args, **kwargs):
        downloads.append("tokenizer")
        return SimpleNamespace(apply_chat_template=lambda *a, **kw: [1] * 2049)

    def load_weights(*args, **kwargs):
        downloads.append("weights")
        pytest.fail("Preflight must finish before any model-weight download")

    ns["AutoTokenizer"] = SimpleNamespace(from_pretrained=load_tokenizer)
    ns["AutoModelForCausalLM"] = SimpleNamespace(from_pretrained=load_weights)
    ns["hf_hub_download"] = load_weights
    if state == "offline_override":
        ns.update(MODE="offline", LOAD_MODEL=True, PLAN=None, items=None)
        exec(notebook.cells[8].source, ns)
        assert not ns["LOAD_MODEL"] and not downloads
        return
    if state == "missing":
        ns.update(PLAN=None, items=None)
        match = "No generation plan"
    elif state == "empty_calibration":
        match = "no valid harmful calibration prompts"
    elif state == "changed":
        ns["CFG"]["max_seq_len"] += 1
        match = "settings/items changed"
    else:
        match = "overlength"
    with pytest.raises((RuntimeError, ValueError), match=match):
        exec(notebook.cells[8].source, ns)
    assert downloads == (["tokenizer"] if state == "overlength" else [])
