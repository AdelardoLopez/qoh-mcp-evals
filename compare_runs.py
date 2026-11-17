"""Compara dos ejecuciones del eval pregunta a pregunta.

Uso:
    python compare_runs.py                       # las dos últimas de results/
    python compare_runs.py results/a.json results/b.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data, {r["id"]: r for r in data["results"]}


def status(r):
    if r is None:
        return "-"
    if "infra_error" in r:
        return "ERR"
    return "PASS" if r["passed"] else "FAIL"


def main():
    if len(sys.argv) == 3:
        a_path, b_path = sys.argv[1], sys.argv[2]
    else:
        runs = sorted((ROOT / "results").glob("run_*.json"))
        if len(runs) < 2:
            raise SystemExit("Necesitas al menos dos ejecuciones en results/.")
        a_path, b_path = runs[-2], runs[-1]
    a, ra = load(a_path)
    b, rb = load(b_path)

    def label(d):
        ctx = d.get("ollama_options", {}).get("num_ctx", "default")
        gc = " ground-check" if d.get("ground_check") else ""
        return (f"{d['model']} prompt {d.get('system_prompt_version', '?')} num_ctx {ctx}"
                f" grader {d.get('grader_version', '?')}{gc}")

    print(f"A: {Path(a_path).name}  ({label(a)})")
    print(f"B: {Path(b_path).name}  ({label(b)})\n")
    print(f"{'ID':<5}{'A':<6}{'B':<6}cambio")
    for qid in sorted(set(ra) | set(rb)):
        sa, sb = status(ra.get(qid)), status(rb.get(qid))
        change = ""
        if sa == "FAIL" and sb == "PASS":
            change = "arreglado"
        elif sa == "PASS" and sb == "FAIL":
            change = "REGRESION"
        print(f"{qid:<5}{sa:<6}{sb:<6}{change}")

    sa, sb = a["summary"], b["summary"]
    print(f"\n{'':<22}{'A':>8}{'B':>8}")
    for key, fmt in [("pass_rate", "{:.0%}"), ("tool_accuracy", "{:.0%}"),
                     ("avg_latency_s", "{:.1f}s"), ("avg_tool_calls", "{:.2f}"),
                     ("grounded_rate", "{:.0%}")]:
        if key in sa and key in sb:
            print(f"{key:<22}{fmt.format(sa[key]):>8}{fmt.format(sb[key]):>8}")


if __name__ == "__main__":
    main()
