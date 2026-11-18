"""Valida el LLM juez contra tus etiquetas humanas (critique shadowing, Hamel Husain).

1. Toma las trazas que etiquetaste en trace_viewer.py (pass/fail + crítica).
2. Las reparte de forma determinista en DEV (para afinar el juez) y TEST (reservado).
3. Ejecuta el juez binario sobre cada traza y compara con tu etiqueta.
4. Reporta TPR, TNR, acuerdo y kappa de Cohen por separado en DEV y TEST, y los desacuerdos.

Positivo = PASS.
  TPR = de las respuestas que TÚ aprobaste, % que el juez también aprueba.
  TNR = de las respuestas que TÚ suspendiste, % que el juez también suspende (la clave:
        mide si el juez caza los fallos). El acuerdo global engaña si casi todo es PASS.

Uso:
    python validate_judge.py                       # juez zero-shot
    python validate_judge.py --fewshot             # + tus críticas de DEV como ejemplos
    python validate_judge.py --judge-model qwen2.5:14b
    python validate_judge.py --threshold 0.9       # umbral para declarar el juez fiable
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import traces as T

ROOT = Path(__file__).parent

JUDGE_PROMPT_VERSION = "j1"
JUDGE_PROMPT = """Eres un evaluador estricto de un asistente de inventario (almacenes A y B).
Decide si la RESPUESTA es aceptable.

PASS solo si se cumplen TODAS:
1. Responde a lo que se pregunta.
2. Toda cifra y todo SKU de la respuesta aparece en los DATOS DE LAS TOOLS o en la pregunta.
3. Si las tools no tienen el dato, el SKU no existe o la pregunta es ambigua, lo dice o pide
   aclaración en lugar de suponer.
Si falla cualquiera, FAIL. La longitud no suma puntos.
{examples}
PREGUNTA: {pregunta}
DATOS DE LAS TOOLS: {tool_outputs}
RESPUESTA: {respuesta}

Primero escribe la crítica, después el veredicto. Devuelve SOLO JSON:
{{"critica": "<una o dos frases>", "veredicto": "PASS" | "FAIL"}}"""


def split_of(trace: dict) -> str:
    """Reparto determinista ~50/50 por caso (pregunta+respuesta): el mismo caso cae
    siempre en el mismo lado, y los duplicados entre ejecuciones no se cuelan en TEST."""
    h = hashlib.sha256("||".join(T.dedupe_key(trace)).encode()).hexdigest()
    return "dev" if int(h, 16) % 2 == 0 else "test"


def build_examples(dev: list[dict], k: int = 6) -> str:
    """Few-shot con tus críticas de DEV: mitad FAIL y mitad PASS cuando es posible."""
    fails = [t for t in dev if t["label"] == "FAIL"][: k // 2]
    passes = [t for t in dev if t["label"] == "PASS"][: k - len(fails)]
    if not fails and not passes:
        return ""
    lines = ["\nEjemplos juzgados por el experto:"]
    for t in fails + passes:
        lines.append(f"- PREGUNTA: {t['pregunta']}\n  RESPUESTA: {t['answer']}\n"
                     f"  VEREDICTO: {t['label']} — {t['critique'] or 'correcta'}")
    return "\n".join(lines) + "\n"


def judge(chat_fn, model: str, trace: dict, examples: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        examples=examples, pregunta=trace["pregunta"],
        tool_outputs=json.dumps([c["output"] for c in trace["calls"]], ensure_ascii=False)
        if trace["calls"] else "(ninguna tool llamada)",
        respuesta=trace["answer"] or "(vacía)")
    resp = chat_fn(model=model, messages=[{"role": "user", "content": prompt}],
                   format="json", options={"temperature": 0})
    try:
        out = json.loads(resp.message.content)
        verdict = str(out.get("veredicto", "")).strip().upper()
        if verdict not in ("PASS", "FAIL"):
            raise ValueError(verdict)
        return {"verdict": verdict, "critique": out.get("critica", "")}
    except (json.JSONDecodeError, ValueError, AttributeError):
        # Un juez que no sabe responder en formato cuenta como FAIL del juez, no se ignora.
        return {"verdict": "INVALID", "critique": str(getattr(resp.message, "content", ""))[:300]}


def metrics(rows: list[dict]) -> dict:
    tp = sum(r["label"] == "PASS" and r["verdict"] == "PASS" for r in rows)
    fn = sum(r["label"] == "PASS" and r["verdict"] != "PASS" for r in rows)
    tn = sum(r["label"] == "FAIL" and r["verdict"] == "FAIL" for r in rows)
    fp = sum(r["label"] == "FAIL" and r["verdict"] != "FAIL" for r in rows)
    n = len(rows)
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    acc = (tp + tn) / n if n else None
    # kappa de Cohen: acuerdo corregido por el que se daría por azar
    kappa = None
    if n:
        p_yes = ((tp + fn) / n) * ((tp + fp) / n)
        p_no = ((tn + fp) / n) * ((tn + fn) / n)
        pe = p_yes + p_no
        kappa = (acc - pe) / (1 - pe) if pe < 1 else None
    return {"n": n, "human_pass": tp + fn, "human_fail": tn + fp,
            "TPR": tpr, "TNR": tnr, "accuracy": acc, "kappa": kappa,
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def pct(x):
    return "-" if x is None else f"{x:.0%}"


def num(x):
    return "-" if x is None else f"{x:.2f}"


def main(argv=None, chat_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="qwen2.5:7b")
    ap.add_argument("--fewshot", action="store_true",
                    help="usa tus críticas de DEV como ejemplos en el prompt del juez")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="TPR y TNR mínimos en TEST para declarar el juez fiable")
    args = ap.parse_args(argv)
    if chat_fn is None:
        import ollama
        chat_fn = ollama.chat

    labeled = T.all_labeled_traces()
    # un caso = pregunta+respuesta; si lo etiquetaste en varias ejecuciones, cuenta una vez
    unique = {}
    for t in labeled:
        unique.setdefault(T.dedupe_key(t), t)
    cases = list(unique.values())
    if len(cases) < 10:
        raise SystemExit(f"Solo hay {len(cases)} casos etiquetados distintos. Etiqueta al "
                         "menos 30 (con 10+ FAIL) en trace_viewer.py para que las métricas signifiquen algo.")

    dev = [t for t in cases if split_of(t) == "dev"]
    test = [t for t in cases if split_of(t) == "test"]
    examples = build_examples(dev) if args.fewshot else ""

    rows = []
    for t in cases:
        j = judge(chat_fn, args.judge_model, t, examples)
        rows.append({"key": t["key"], "id": t["id"], "split": split_of(t),
                     "pregunta": t["pregunta"], "answer": t["answer"],
                     "label": t["label"], "human_critique": t["critique"],
                     "failure_mode": t["failure_mode"],
                     "verdict": j["verdict"], "judge_critique": j["critique"]})
        print(f"{t['id']:<4} [{split_of(t):<4}] humano {t['label']:<4} juez {j['verdict']:<7}"
              f"{'' if t['label'] == j['verdict'] else '  <- desacuerdo'}", flush=True)

    m_dev = metrics([r for r in rows if r["split"] == "dev"])
    m_test = metrics([r for r in rows if r["split"] == "test"])

    print("\n" + "=" * 62)
    print(f"Juez: {args.judge_model} · prompt {JUDGE_PROMPT_VERSION}"
          f"{' + few-shot DEV' if args.fewshot else ''}")
    print(f"{'':<10}{'n':>4}{'PASS/FAIL humano':>18}{'TPR':>7}{'TNR':>7}{'acuerdo':>9}{'kappa':>7}")
    for name, m in (("DEV", m_dev), ("TEST", m_test)):
        print(f"{name:<10}{m['n']:>4}{m['human_pass']:>10}/{m['human_fail']:<7}"
              f"{pct(m['TPR']):>7}{pct(m['TNR']):>7}{pct(m['accuracy']):>9}{num(m['kappa']):>7}")
    if m_test["human_fail"] < 5:
        print("Aviso: menos de 5 FAIL en TEST; el TNR es poco fiable. Etiqueta más fallos.")
    ok = (m_test["TPR"] or 0) >= args.threshold and (m_test["TNR"] or 0) >= args.threshold
    print(f"\nVeredicto: el juez {'ES' if ok else 'NO es'} fiable "
          f"(umbral {args.threshold:.0%} en TPR y TNR de TEST).")
    disagreements = [r for r in rows if r["label"] != r["verdict"]]
    if disagreements:
        print("Desacuerdos: " + ", ".join(f"{r['id']}({r['split']})" for r in disagreements))

    out = ROOT / "results" / "judge_validation" / (
        f"judge_{datetime.now():%Y%m%d-%H%M%S}_{args.judge_model.replace(':', '-')}"
        f"{'_fewshot' if args.fewshot else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "judge_model": args.judge_model, "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "fewshot": args.fewshot, "threshold": args.threshold, "reliable": ok,
        "dev": m_dev, "test": m_test, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Informe: {out.relative_to(ROOT)}")
    return {"dev": m_dev, "test": m_test, "reliable": ok, "rows": rows}


if __name__ == "__main__":
    main()
