"""Tests de la validación del juez y del visor, sin Ollama."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import traces as T  # noqa: E402
import validate_judge as V  # noqa: E402

ROOT = Path(__file__).parent.parent
PUBLISHED = ROOT / "results" / "published"


def test_metrics_math():
    rows = ([{"label": "PASS", "verdict": "PASS"}] * 8 + [{"label": "PASS", "verdict": "FAIL"}] * 2
            + [{"label": "FAIL", "verdict": "FAIL"}] * 3 + [{"label": "FAIL", "verdict": "PASS"}] * 1)
    m = V.metrics(rows)
    assert m["TPR"] == 0.8 and m["TNR"] == 0.75 and m["accuracy"] == 11 / 14
    assert round(m["kappa"], 3) == 0.512   # (0.786 - 0.561) / (1 - 0.561)


def test_always_pass_judge_is_caught_by_tnr():
    # 90 % de acuerdo... pero no caza ningún fallo
    rows = [{"label": "PASS", "verdict": "PASS"}] * 9 + [{"label": "FAIL", "verdict": "PASS"}]
    m = V.metrics(rows)
    assert m["accuracy"] == 0.9 and m["TNR"] == 0.0


def test_invalid_judge_output_counts_against_judge():
    rows = [{"label": "FAIL", "verdict": "INVALID"}, {"label": "PASS", "verdict": "INVALID"}]
    m = V.metrics(rows)
    assert m["TPR"] == 0.0 and m["TNR"] == 0.0


@pytest.fixture
def labeled_env(tmp_path, monkeypatch):
    """Etiqueta automáticamente trazas reales publicadas usando el grader como 'humano'."""
    monkeypatch.setattr(T, "LABELS_PATH", tmp_path / "labels.json")
    monkeypatch.setattr(T, "RUN_DIRS", [PUBLISHED])
    for run in ("run_20261104-140809_qwen2.5-7b.json", "run_20261117-131236_qwen2.5-7b.json",
                "run_20261105-170309_llama3.1-8b.json"):
        for t in T.load_traces(PUBLISHED / run):
            label = "PASS" if t["grader_passed"] else "FAIL"
            T.save_label(t["key"], label, "" if label == "PASS" else "crítica de prueba",
                         "" if label == "PASS" else "modo de prueba")
    return tmp_path


def test_labels_roundtrip_and_split_is_deterministic(labeled_env):
    labeled = T.all_labeled_traces()
    assert len(labeled) >= 30
    assert {V.split_of(t) for t in labeled} == {"dev", "test"}
    assert all(V.split_of(t) == V.split_of(t) for t in labeled)


def test_validate_end_to_end_with_perfect_and_lazy_judges(labeled_env, monkeypatch):
    labels = T.load_labels()

    def perfect(model, messages, format=None, options=None):
        # el juez "perfecto" copia la etiqueta humana del caso que aparece en el prompt
        prompt = messages[0]["content"]
        for t in T.all_labeled_traces():
            if f"RESPUESTA: {t['answer'] or '(vacía)'}\n\nPrimero" in prompt:
                return NS(message=NS(content=json.dumps(
                    {"critica": "x", "veredicto": labels[t["key"]]["label"]})))
        return NS(message=NS(content="{}"))

    monkeypatch.chdir(labeled_env)
    out = V.main(["--threshold", "0.9", "--out-dir", str(labeled_env / "jv")], chat_fn=perfect)
    assert out["test"]["TPR"] == 1.0 and out["test"]["TNR"] == 1.0 and out["reliable"]

    def lazy(model, messages, format=None, options=None):
        return NS(message=NS(content=json.dumps({"critica": "ok", "veredicto": "PASS"})))

    out = V.main(["--out-dir", str(labeled_env / "jv")], chat_fn=lazy)
    assert out["test"]["TNR"] == 0.0 and not out["reliable"]


def test_fewshot_examples_use_dev_critiques(labeled_env):
    dev = [t for t in T.all_labeled_traces() if V.split_of(t) == "dev"]
    ex = V.build_examples(dev)
    assert "VEREDICTO: FAIL — crítica de prueba" in ex


def test_viewer_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "trace_viewer.py"), default_timeout=30).run()
    assert not at.exception
    assert any("Visor de trazas" in t.value for t in at.title)


def test_j2_shows_tool_args_and_j1_does_not():
    trace = {"calls": [{"name": "get_stock", "args": {"sku": "CBL-1042", "warehouse": "total"},
                        "output": '{"found": false}'}]}
    assert "warehouse" not in V.format_calls(trace, with_args=False)
    assert 'get_stock({"sku": "CBL-1042", "warehouse": "total"})' in V.format_calls(trace, with_args=True)


def test_fewshot_is_stratified_by_failure_mode(labeled_env):
    labels = T.load_labels()
    keys = [k for k, v in labels.items() if v["label"] == "FAIL"][:2]
    labels[keys[0]]["failure_mode"] = "modo A"
    labels[keys[1]]["failure_mode"] = "modo B"
    T.LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    dev = T.all_labeled_traces()   # todos como DEV para el test
    ex = V.build_examples(dev)
    assert ex.count("VEREDICTO: FAIL") <= 4 and "VEREDICTO: PASS" in ex


def test_tests_do_not_write_reports_into_the_repo(labeled_env):
    # el informe de prueba de esta suite apareció en results/ del usuario: no debe volver a pasar
    before = set((ROOT / "results").rglob("judge_*.json"))
    V.main(["--out-dir", str(labeled_env / "jv")],
           chat_fn=lambda **k: NS(message=NS(content=json.dumps({"critica": "x", "veredicto": "PASS"}))))
    assert set((ROOT / "results").rglob("judge_*.json")) == before
