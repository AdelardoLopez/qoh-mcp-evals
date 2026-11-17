"""Repuntúa una ejecución guardada con la versión ACTUAL de los graders, sin volver
a llamar al modelo. Úsalo cuando cambies los graders: solo así es justo comparar
una ejecución antigua con una nueva.

Las preguntas llm_judge conservan el veredicto del juez original (re-evaluarlas
requeriría el modelo), pero se les vuelve a aplicar el recall de SKUs de referencia.

Uso:
    python regrade.py results/run_20261104-151921_qwen2.5-7b.json
    -> results/regraded/run_20261104-151921_qwen2.5-7b_g4.json
"""

import json
import sys
from pathlib import Path

import yaml

import graders
from run_evals import ROOT, summarize


def regrade(run_path: Path) -> Path:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    items = {i["id"]: i for i in yaml.safe_load(
        open(ROOT / data.get("eval_file", "eval_set_qoh.yaml"), encoding="utf-8"))}
    changed = []
    for r in data["results"]:
        if "infra_error" in r:
            continue
        item = items[r["id"]]
        old = r["passed"]
        if item["metodo"] == "llm_judge":
            crit = r["details"].get("criterios", [])
            ref = set(item.get("referencia", []))
            recall = len(ref & graders.extract_skus(r["answer"])) / len(ref) if ref else None
            r["passed"] = bool(crit) and all(crit) and (recall is None or recall >= 2 / 3)
            r["details"]["reference_recall"] = recall
            r["details"]["judge_reused"] = True
        else:
            r["passed"], r["details"] = graders.grade(item, r["answer"])
        r["tool_ok"], r["tool_details"] = graders.grade_tools(item.get("tool_esperada"),
                                                              r.get("calls", []))
        r["ungrounded_numbers"] = graders.ungrounded_numbers(
            r["answer"], item["pregunta"], [c["output"] for c in r.get("calls", [])])
        if old != r["passed"]:
            changed.append(f"{r['id']}: {'PASS' if old else 'FAIL'} -> "
                           f"{'PASS' if r['passed'] else 'FAIL'}")
    data["summary"] = summarize(data["results"])
    data["grader_version"] = graders.GRADER_VERSION
    data["regraded_from"] = run_path.name

    out_dir = ROOT / "results" / "regraded"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_path.stem}_{graders.GRADER_VERSION}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Repuntuado con graders {graders.GRADER_VERSION}: {out}")
    print("Cambios de veredicto: " + (", ".join(changed) if changed else "ninguno"))
    print(f"Acierto: {data['summary']['pass_rate']:.0%}   "
          f"Groundedness: {data['summary']['grounded_rate']:.0%}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python regrade.py results/<run>.json")
    regrade(Path(sys.argv[1]))
