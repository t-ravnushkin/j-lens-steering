"""Offline archive integrity, external judge parsing, concurrency and resume tests."""

import io
import json
import threading
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import nbformat
import pytest

from jlens import archived_judging as aj
from jlens.alignment_benchmark import digest


@pytest.fixture
def saved_run(tmp_path):
    plan = {
        "protocol": "llama-goodfire-replication-v1",
        "study": {"prompt_chunk": 1, "condition_chunk": 1},
        "items": [
            dict(prompt_id=f"p{i}", prompt="Benign prompt", kind="benign")
            for i in range(2)
        ],
        "conditions": [dict(condition_id="baseline", family="baseline", strength=0)],
    }
    root = tmp_path / "run"
    aj.atomic_json(root / "plan.json", plan)
    row = dict(
        response_id=digest([digest(plan), "p0", "baseline"]),
        prompt_id="p0",
        condition_id="baseline",
        text="Answer",
        n_content_tokens=1,
        truncated=False,
    )
    aj.atomic_json(
        root / "generations" / "p0-1_c0-1.json",
        {
            "plan_id": digest(plan),
            "job": [0, 1, 0, 1],
            "records": [row],
        },
    )
    return root, plan


def test_nested_archive_partial_run_and_resume(saved_run, tmp_path):
    root, plan = saved_run
    archive = tmp_path / "res.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in root.rglob("*.json"):
            zf.write(path, "root/res/old_run/" + str(path.relative_to(root)))
    imported = aj.import_zip(archive, tmp_path / "cache")
    loaded_plan, rows, coverage = aj.load_run(imported[0])
    assert loaded_plan == plan and len(rows) == 1
    assert coverage.generation_coverage.tolist() == [0.5]
    original_time = (imported[0] / "plan.json").stat().st_mtime_ns
    assert aj.import_zip(archive, tmp_path / "cache") == imported
    assert (imported[0] / "plan.json").stat().st_mtime_ns == original_time
    assert rows.iloc[0].prompt == "Benign prompt"


@pytest.mark.parametrize("kind", ["identity", "coverage", "duplicate_block", "plan"])
def test_corrupt_generations_fail(saved_run, kind):
    root, _ = saved_run
    path = root / "generations" / "p0-1_c0-1.json"
    value = json.loads(path.read_text())
    if kind == "identity":
        value["records"][0]["response_id"] = "wrong"
    elif kind == "coverage":
        value["records"] = []
    elif kind == "duplicate_block":
        aj.atomic_json(path.with_name("duplicate.json"), value)
    else:
        value["plan_id"] = "wrong"
    aj.atomic_json(path, value)
    with pytest.raises(ValueError):
        aj.load_run(root)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "nested/../../escape", "a\\b"]
)
def test_zip_paths_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        aj.import_zip(archive, tmp_path / "cache")


def test_zip_symlink_and_size_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "target")
    with pytest.raises(ValueError, match="Unsafe"):
        aj.import_zip(archive, tmp_path / "cache")
    with pytest.raises(ValueError, match="size limit"):
        aj.import_zip(archive, tmp_path / "cache", max_bytes=1)


def request_row(i=0, text="Answer"):
    return dict(
        response_id=str(i),
        prompt="A benign question.",
        text=text,
        n_content_tokens=1 if text else 0,
    )


def verdict(row):
    return dict(
        response_id=row["response_id"],
        judge_status="ok",
        unsafe=False,
        coherent=True,
        refusal=False,
        reason="Benign answer.",
    )


def api_body(finish="stop", text=None):
    return {
        "id": "test-id",
        "model": "test-model",
        "provider": "test-provider",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.01},
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": text
                    or json.dumps(
                        {
                            "verdict": "SAFE",
                            "coherent": True,
                            "refusal": False,
                            "reason": "Benign.",
                        }
                    )
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "finish,status",
    [
        ("stop", "ok"),
        ("length", "judge_truncated"),
        ("content_filter", "judge_blocked_or_incomplete"),
    ],
)
def test_api_blinding_parsing_and_provenance(monkeypatch, finish, status):
    captured = []

    def transport(request, timeout):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps(api_body(finish)).encode())

    monkeypatch.setattr(aj, "urlopen", transport)
    row = {**request_row(), "family": "sae_full", "vector_id": 100}
    result = aj.openrouter_judgment(row, api_key="secret", config={"model": "test"})
    assert result["judge_status"] == status
    assert result["actual_model"] == "test-model" and result["usage"]["cost"] == 0.01
    assert "secret" not in json.dumps(result)
    sent = captured[0]["messages"]
    assert json.loads(sent[1]["content"]) == {
        "request": row["prompt"],
        "response": row["text"],
    }
    if status != "ok":
        assert result["unsafe"] is None


def test_api_retries_429_and_does_not_retry_auth_or_invalid(monkeypatch):
    calls, sleeps = [], []

    def transport(request, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError("test", 429, "limited", {"Retry-After": "3"}, None)
        return io.BytesIO(json.dumps(api_body()).encode())

    monkeypatch.setattr(aj, "urlopen", transport)
    monkeypatch.setattr(aj.time, "sleep", sleeps.append)
    result = aj.openrouter_judgment(request_row(), api_key="secret", config={})
    assert len(calls) == 2 and result["attempts"] == 2 and sleeps == [3]

    def denied(request, timeout):
        raise HTTPError("test", 401, "unauthorized", {}, None)

    monkeypatch.setattr(aj, "urlopen", denied)
    result = aj.openrouter_judgment(request_row(), api_key="secret", config={})
    assert result["judge_status"] == "http_401" and result["attempts"] == 1
    monkeypatch.setattr(
        aj,
        "urlopen",
        lambda *a, **kw: io.BytesIO(
            json.dumps(api_body(text="I cannot judge this.")).encode()
        ),
    )
    result = aj.openrouter_judgment(request_row(), api_key="secret", config={})
    assert result["judge_status"] == "invalid" and result["unsafe"] is None


def test_cache_concurrent_resume_limits_and_input_identity(tmp_path):
    rows = [request_row(i) for i in range(6)] + [request_row(9, "")]
    cache = aj.JudgmentCache(tmp_path, {"model": "test"})
    calls, active, peak = [], 0, 0
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def judge(row):
        nonlocal active, peak
        with lock:
            calls.append(row["response_id"])
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=10)
        time.sleep(0.01)
        with lock:
            active -= 1
        return verdict(row)

    aj.judge_concurrently(rows, cache, judge, concurrency=2, max_requests=4)
    assert len(calls) == 4 and peak == 2
    assert len(cache.pending(rows)) == 2
    assert cache.read(rows[-1])["label_source"] == "empty_rule"
    aj.judge_concurrently(rows, cache, judge, concurrency=2)
    assert len(calls) == 6 and not cache.pending(rows)
    aj.judge_concurrently(rows, cache, judge, concurrency=2)
    assert len(calls) == 6
    assert cache.read({**rows[0], "text": "Changed answer"}) is None
    assert aj.JudgmentCache(tmp_path, {"model": "other"}).read(rows[0]) is None
    cache.save(rows[0], {**verdict(rows[0]), "judge_status": "invalid", "unsafe": None})
    assert not cache.pending(rows)
    assert cache.pending(rows, retry_failed=True) == [rows[0]]


def test_fatal_error_stops_submitting_and_is_cached(tmp_path):
    cache = aj.JudgmentCache(tmp_path, {})
    rows = [request_row(i) for i in range(5)]
    with pytest.raises(RuntimeError, match="http_402"):
        aj.judge_concurrently(
            rows,
            cache,
            lambda row: {
                **verdict(row),
                "judge_status": "http_402",
                "unsafe": None,
            },
            concurrency=1,
        )
    assert len(cache.pending(rows)) == 4


def test_notebook_offline_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(
        root / "notebooks/jacobian_lens/archived_steering_openrouter_judging.ipynb", 4
    )
    ns = {"REPO_DIR": root, "Path": Path, "os": __import__("os")}
    # Run configuration and all experiment cells; no installation, GPU or network.
    for cell in notebook.cells[2:]:
        if cell.cell_type == "code":
            source = cell.source.replace(
                'REPO_DIR / "runs" / "archived_judging_demo"',
                f"Path({str(tmp_path)!r})",
            )
            exec(source, ns)
    assert len(ns["outcomes"]) == 4
    assert ns["summary"].coverage.eq(1).all()
    assert ns["generation_coverage"].n_generated.tolist() == [2, 2, 0]
    assert ns["cache"].pending(ns["records"]) == []
    assert (ns["REPORT_DIR"] / "human_audit.csv").exists()


@pytest.mark.parametrize("backend", ["openrouter", "local_qwen"])
def test_notebook_judge_backends_resume_without_reloading(
    tmp_path, monkeypatch, backend
):
    from types import SimpleNamespace

    import transformers

    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(
        root / "notebooks/jacobian_lens/archived_steering_openrouter_judging.ipynb", 4
    )
    ns = {"REPO_DIR": root, "Path": Path, "os": __import__("os")}
    exec(notebook.cells[2].source, ns)
    exec(
        notebook.cells[4].source.replace(
            'REPO_DIR / "runs" / "archived_judging_demo"', f"Path({str(tmp_path)!r})"
        ),
        ns,
    )
    ns.update(MODE="judge", JUDGE_BACKEND=backend, MOUNT_DRIVE=False)
    calls, loads = [], []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    def api_judge(row, **kwargs):
        assert kwargs["api_key"] == "test-secret"
        calls.append(row["response_id"])
        return verdict(row)

    def local_judge(model, tokenizer, rows, **kwargs):
        calls.extend(row["response_id"] for row in rows)
        assert len(rows) == 2  # Across prompt/block boundaries, not one pass per row.
        return [verdict(row) for row in rows]

    model = SimpleNamespace()
    model.to = lambda device: model
    model.eval = lambda: model

    def load_model(*args, **kwargs):
        loads.append(1)
        return model

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", load_model
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object()
    )
    ns.update(openrouter_judgment=api_judge, judge_batch=local_judge)
    exec(notebook.cells[6].source, ns)
    exec(notebook.cells[7].source, ns)
    exec(notebook.cells[6].source, ns)
    exec(notebook.cells[7].source, ns)
    assert len(calls) == 2
    assert len(loads) == (1 if backend == "local_qwen" else 0)
    assert not ns["cache"].pending(ns["records"])
