"""Test end-to-end sin Ollama: arranca el MCP real y sustituye el LLM por un
agente guionizado. Comprueba que el bucle de tools, el MCP y la puntuación encajan."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_evals  # noqa: E402

EVAL = Path(__file__).parent.parent / "eval_set_qoh.yaml"

# pregunta-id -> (tool calls que hace el "modelo", respuesta final)
SCRIPT = {
    "Q01": ([("get_stock", {"sku": "CBL-1042", "warehouse": "A"})],
            "Hay 340 unidades del SKU CBL-1042 en el almacén A."),
    "Q11": ([("get_stock", {"sku": "XYZ-9999"})],
            "Hay 12 unidades del SKU XYZ-9999."),   # alucinación a propósito: debe fallar
    "Q14": ([("list_low_stock", {"limit": 10})],
            "Repón primero CBL-0099 (2/30 en A), DCT-3310 (5/50 en A) y WRP-0012 (8/40 en A)."),
    "Q16": ([("list_stock_discrepancies", {"min_diff": 0}),
             ("get_stock", {"sku": "WRP-0087", "warehouse": "A"})],
            "WRP-0087 tiene la mayor discrepancia en B (40 uds). En A tiene 58 unidades."),
}
QID_BY_QUESTION = {}
CRASH: set[str] = set()


def make_fake_chat():
    seen_tool_outputs = []

    def fake_chat(model, messages, tools=None, options=None, format=None):
        if format == "json":  # papel de juez
            return NS(message=NS(content=json.dumps(
                {"criterios": [True, True, True], "razon": "ok"}), tool_calls=None))
        if len(messages) == 1:  # warm-up
            return NS(message=NS(content="ok", tool_calls=None))
        qid = QID_BY_QUESTION[messages[1]["content"]]
        if qid in CRASH:
            raise RuntimeError("llama-server process has terminated")
        calls, final = SCRIPT[qid]
        n_done = sum(1 for m in messages if m["role"] == "tool")
        if n_done:
            seen_tool_outputs.append(messages[-1]["content"])
        if n_done < len(calls):
            name, args = calls[n_done]
            tc = NS(function=NS(name=name, arguments=args),
                    model_dump=lambda n=name, a=args: {"function": {"name": n, "arguments": a}})
            return NS(message=NS(content="", tool_calls=[tc]), prompt_eval_count=10, eval_count=5)
        return NS(message=NS(content=final, tool_calls=None), prompt_eval_count=10, eval_count=5)

    return fake_chat, seen_tool_outputs


def test_end_to_end():
    import yaml
    for it in yaml.safe_load(open(EVAL, encoding="utf-8")):
        QID_BY_QUESTION[it["pregunta"]] = it["id"]
    fake_chat, outputs = make_fake_chat()
    results = asyncio.run(run_evals.run_eval(
        EVAL, "fake", "fake-judge", only=set(SCRIPT), chat_fn=fake_chat))
    by_id = {r["id"]: r for r in results}

    assert by_id["Q01"]["passed"] and by_id["Q01"]["tool_ok"]
    assert not by_id["Q11"]["passed"]                     # alucinación detectada
    assert by_id["Q14"]["passed"] and by_id["Q14"]["details"]["reference_recall"] == 1.0
    assert by_id["Q16"]["passed"] and by_id["Q16"]["tool_ok"]

    # El MCP devolvió datos reales del snapshot
    assert any('"found": false' in o for o in outputs)    # XYZ-9999 no existe
    disc = json.loads(by_id["Q16"]["calls"][0]["output"])
    disc = disc.get("result", disc) if isinstance(disc, dict) else disc
    assert disc[0]["sku"] == "WRP-0087" and disc[0]["warehouse"] == "B"

    s = run_evals.summarize(results)
    assert s["pass_rate"] == 0.75 and s["failed"] == ["Q11"]


def test_infra_error_does_not_count_as_model_failure(monkeypatch):
    import yaml
    for it in yaml.safe_load(open(EVAL, encoding="utf-8")):
        QID_BY_QUESTION[it["pregunta"]] = it["id"]
    monkeypatch.setattr(run_evals.time, "sleep", lambda s: None)
    CRASH.add("Q11")
    try:
        fake_chat, _ = make_fake_chat()
        results = asyncio.run(run_evals.run_eval(
            EVAL, "fake", "fake-judge", only=set(SCRIPT), chat_fn=fake_chat))
    finally:
        CRASH.clear()
    s = run_evals.summarize(results)
    assert s["infra_errors"] == ["Q11"] and s["n_completed"] == 3
    assert s["pass_rate"] == 1.0 and s["failed"] == []   # el resto sigue ejecutándose


def test_unknown_args_are_detected_and_totals_come_from_code():
    import yaml
    for it in yaml.safe_load(open(EVAL, encoding="utf-8")):
        QID_BY_QUESTION[it["pregunta"]] = it["id"]
    original = SCRIPT["Q07"] if "Q07" in SCRIPT else None
    SCRIPT["Q07"] = ([("get_stock", {"sku": "CBL-1042", "color": "black"})],
                     "El valor total del inventario es $6,599.68.")
    try:
        fake_chat, _ = make_fake_chat()
        results = asyncio.run(run_evals.run_eval(
            EVAL, "fake", "fake-judge", only={"Q07"}, chat_fn=fake_chat))
    finally:
        SCRIPT.pop("Q07") if original is None else SCRIPT.__setitem__("Q07", original)
    r = results[0]
    assert r["passed"] and not r["tool_ok"]                  # acierta, pero con un arg inválido
    assert r["calls"][0]["unknown_args"] == ["color"]
    out = json.loads(r["calls"][0]["output"])
    assert out["total"] == {"qty_physical": 512, "inventory_value": 6599.68}


def test_ground_check_triggers_one_reflection():
    """Si la respuesta inventa una cifra, el runner pide corregir una vez."""
    import yaml
    for it in yaml.safe_load(open(EVAL, encoding="utf-8")):
        QID_BY_QUESTION[it["pregunta"]] = it["id"]
    state = {"n": 0}

    def chat(model, messages, tools=None, options=None, format=None):
        if len(messages) == 1:
            return NS(message=NS(content="ok", tool_calls=None))
        state["n"] += 1
        if messages[-1]["role"] == "user" and "ninguna tool" in messages[-1]["content"]:
            return NS(message=NS(content="Hay 340 unidades del SKU CBL-1042 en el almacén A.",
                                 tool_calls=None), prompt_eval_count=1, eval_count=1)
        if not any(m["role"] == "tool" for m in messages):
            tc = NS(function=NS(name="get_stock", arguments={"sku": "CBL-1042", "warehouse": "A"}),
                    model_dump=lambda: {"function": {"name": "get_stock",
                                        "arguments": {"sku": "CBL-1042", "warehouse": "A"}}})
            return NS(message=NS(content="", tool_calls=[tc]), prompt_eval_count=1, eval_count=1)
        return NS(message=NS(content="Hay 999 unidades del SKU CBL-1042 en el almacén A.",
                             tool_calls=None), prompt_eval_count=1, eval_count=1)

    results = asyncio.run(run_evals.run_eval(EVAL, "fake", "fake", only={"Q01"},
                                             chat_fn=chat, ground_check=True))
    r = results[0]
    assert r["reflections"] == 1 and r["passed"] and r["ungrounded_numbers"] == []
