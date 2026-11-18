"""Carga de trazas y etiquetas humanas, compartida por el visor y la validación del juez.

Una traza = una pregunta de una ejecución: pregunta, tool calls, respuesta y veredicto del
grader. Las etiquetas humanas (pass/fail + crítica + modo de fallo) se guardan en
labels/human_labels.json con clave "<archivo de la ejecución>::<id de pregunta>".
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
LABELS_PATH = ROOT / "labels" / "human_labels.json"
RUN_DIRS = [ROOT / "results", ROOT / "results" / "published"]


def list_runs() -> list[Path]:
    """Ejecuciones brutas disponibles (no las repuntuadas), la más reciente primero."""
    runs: dict[str, Path] = {}
    for d in RUN_DIRS:
        for p in d.glob("run_*.json"):
            runs.setdefault(p.name, p)
    return sorted(runs.values(), key=lambda p: p.name, reverse=True)


def load_traces(run_path: Path) -> list[dict]:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    out = []
    for r in data["results"]:
        if "infra_error" in r:
            continue
        out.append({
            "key": f"{run_path.name}::{r['id']}",
            "run": run_path.name,
            "model": data.get("model"),
            "prompt_version": data.get("system_prompt_version"),
            "id": r["id"],
            "categoria": r["categoria"],
            "metodo": r["metodo"],
            "pregunta": r["pregunta"],
            "answer": r.get("answer", ""),
            "calls": r.get("calls", []),
            "grader_passed": r.get("passed"),
            "grader_details": r.get("details", {}),
            "tool_ok": r.get("tool_ok"),
            "latency_s": r.get("latency_s"),
            "ungrounded_numbers": r.get("ungrounded_numbers"),
        })
    return out


def dedupe_key(trace: dict) -> tuple[str, str]:
    """Dos trazas con la misma pregunta y la misma respuesta son el mismo caso."""
    return trace["pregunta"].strip(), trace["answer"].strip()


def load_labels() -> dict[str, dict]:
    try:
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save_label(key: str, label: str, critique: str, failure_mode: str) -> None:
    labels = load_labels()
    labels[key] = {"label": label, "critique": critique.strip(),
                   "failure_mode": failure_mode.strip()}
    LABELS_PATH.parent.mkdir(exist_ok=True)
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_label(key: str) -> None:
    labels = load_labels()
    if labels.pop(key, None) is not None:
        LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                               encoding="utf-8")


def all_labeled_traces() -> list[dict]:
    """Trazas con etiqueta humana, cruzando todas las ejecuciones disponibles."""
    labels = load_labels()
    by_run: dict[str, list[str]] = {}
    for key in labels:
        run, _ = key.split("::", 1)
        by_run.setdefault(run, []).append(key)
    runs = {p.name: p for p in list_runs()}
    out = []
    for run, keys in by_run.items():
        if run not in runs:
            continue
        traces = {t["key"]: t for t in load_traces(runs[run])}
        for k in keys:
            if k in traces:
                out.append({**traces[k], **labels[k]})
    return out
