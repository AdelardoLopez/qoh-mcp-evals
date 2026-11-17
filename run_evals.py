"""Ejecuta el eval set contra el MCP QoH usando un modelo local de Ollama.

Flujo por pregunta:
  1. Arranca el servidor MCP (stdio) y lista sus tools.
  2. Bucle de agente: el modelo decide qué tools llamar; se ejecutan vía MCP.
  3. Se puntúa la respuesta final con graders deterministas (o LLM juez).
  4. Se guarda todo (respuesta, tool calls, latencia, tokens) en results/.

Uso:
    ollama pull qwen2.5:7b
    python run_evals.py                          # todas las preguntas
    python run_evals.py --only Q01,Q11,Q16       # subconjunto
    python run_evals.py --model llama3.1:8b      # comparar modelos
    python run_evals.py --judge-model qwen2.5:14b
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import graders

ROOT = Path(__file__).parent

# Versiona este prompt: si lo cambias, cambian los resultados.
# Prompts versionados: se elige con --prompt y la versión queda en cada informe.
PROMPTS = {
    "v2": """Eres un asistente de inventario. Respondes en español sobre el stock
de los almacenes A y B usando EXCLUSIVAMENTE los datos que devuelven tus tools.
Reglas:
- Si una tool indica que un SKU no existe, dilo claramente y no inventes cifras.
- Si la pregunta pide datos que tus tools no tienen (ventas, beneficios, precios en
  marketplaces...), explica que no dispones de esos datos.
- Si la pregunta sobre un producto concreto NO incluye un código de SKU (formato
  AAA-0000), NO elijas tú un SKU: pregunta al usuario qué código quiere consultar.
- No hagas cálculos: usa los totales y valores que ya devuelven las tools.
- Sé conciso: da la cifra o la lista pedida y una frase de contexto como máximo.

Ejemplos:
Usuario: ¿Cuánto hay de la funda trenzada?
Asistente: ¿Qué código de SKU quieres consultar? Por ejemplo, WRP-0087.
Usuario: ¿Qué debería reponer primero?
Asistente: (llama a list_low_stock y responde con los SKUs más urgentes y su stock)""",
    "v3": """Eres un asistente de inventario. Respondes SIEMPRE en español sobre el stock
de los almacenes A y B usando EXCLUSIVAMENTE los datos que devuelven tus tools.
Reglas:
- Si la pregunta incluye un código con formato AAA-0000 (ej. CBL-1042, XYZ-9999),
  llama SIEMPRE a la tool con ese código, aunque no te suene: la tool te dirá si existe.
- Solo si la pregunta sobre un producto concreto NO incluye ningún código, pregunta al
  usuario qué código quiere consultar. No elijas tú un SKU.
- Si una tool indica que un SKU no existe, dilo claramente y no inventes cifras.
- Si la pregunta pide datos que tus tools no tienen (ventas, beneficios, precios en
  marketplaces...), explica que no dispones de esos datos.
- Si la respuesta necesita datos de varias tools o almacenes, llama a todas las que
  hagan falta ANTES de responder. Nunca supongas una cifra que no te haya dado una tool.
- No hagas cálculos: usa los totales y valores que ya devuelven las tools.
- Usa solo los parámetros que define cada tool.
- Sé conciso: da la cifra o la lista pedida y una frase de contexto como máximo.""",
}
SYSTEM_PROMPT_VERSION = "v2"
SYSTEM_PROMPT = PROMPTS[SYSTEM_PROMPT_VERSION]

JUDGE_PROMPT = """Eres un evaluador estricto. Evalúa la RESPUESTA de un asistente de
inventario según cada criterio de la RÚBRICA. La longitud NO suma puntos: penaliza el
relleno. Usa los DATOS DE LAS TOOLS como única verdad; cualquier cifra o SKU que no
aparezca en ellos es inventado.

PREGUNTA: {pregunta}
DATOS DE LAS TOOLS: {tool_outputs}
SKUs DE REFERENCIA (respuesta ideal): {referencia}
RESPUESTA: {respuesta}

RÚBRICA:
{rubrica}

Devuelve SOLO JSON: {{"criterios": [true|false, ...], "razon": "<una frase>"}}
con un booleano por criterio, en el mismo orden."""


# Opciones de inferencia comunes. --cpu añade num_gpu=0 (fuerza CPU).
OLLAMA_OPTIONS = {"temperature": 0}


class InfraError(Exception):
    """Fallo de infraestructura (Ollama caído, sin memoria...). No es un fallo del modelo."""


def call_llm(chat_fn, retries=2, **kwargs):
    """Llama al LLM reintentando ante caídas del servidor (p. ej. llama-server terminado).
    Si agota los reintentos lanza InfraError para no contarlo como fallo del modelo."""
    for attempt in range(retries + 1):
        try:
            return chat_fn(**kwargs)
        except Exception as e:  # ResponseError, ConnectionError...
            if attempt == retries:
                raise InfraError(f"{type(e).__name__}: {e}") from e
            time.sleep(5 * (attempt + 1))


def _attr(obj, *names):
    """El SDK mcp 2.x usa snake_case (input_schema) y el 1.x camelCase (inputSchema)."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def tool_param_names(tools) -> dict[str, set[str]]:
    return {t.name: set((_attr(t, "input_schema", "inputSchema") or {})
                        .get("properties", {}).keys()) for t in tools}


def mcp_tools_to_ollama(tools) -> list[dict]:
    return [{
        "type": "function",
        "function": {"name": t.name, "description": t.description or "",
                     "parameters": _attr(t, "input_schema", "inputSchema")},
    } for t in tools]


def tool_result_text(result) -> str:
    structured = _attr(result, "structured_content", "structuredContent")
    if structured:
        return json.dumps(structured, ensure_ascii=False)
    return "\n".join(getattr(c, "text", str(c)) for c in result.content)


GROUND_CHECK_MSG = ("Tu respuesta contiene cifras que ninguna tool ha devuelto: {nums}. "
                    "No supongas datos: llama a las tools que falten y responde de nuevo "
                    "usando solo sus resultados.")


async def run_agent(session, chat_fn, model, ollama_tools, question, max_steps=6,
                    param_names=None, ground_check=False):
    """Bucle de agente. Devuelve respuesta final, tool calls y métricas."""
    messages = [{"role": "system", "content": PROMPTS[SYSTEM_PROMPT_VERSION]},
                {"role": "user", "content": question}]
    calls, tokens_in, tokens_out = [], 0, 0
    reflections = 0
    t0 = time.perf_counter()
    for _ in range(max_steps):
        resp = call_llm(chat_fn, model=model, messages=messages, tools=ollama_tools,
                        options=OLLAMA_OPTIONS)
        msg = resp.message
        tokens_in += getattr(resp, "prompt_eval_count", 0) or 0
        tokens_out += getattr(resp, "eval_count", 0) or 0
        if not msg.tool_calls:
            answer = msg.content or ""
            ungrounded = graders.ungrounded_numbers(answer, question,
                                                    [c["output"] for c in calls])
            # Reflexión (cap. 6): un único reintento si inventa cifras.
            if ground_check and ungrounded and reflections == 0:
                reflections += 1
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content": GROUND_CHECK_MSG.format(
                    nums=", ".join(f"{n:g}" for n in ungrounded))})
                continue
            return {"answer": answer, "calls": calls,
                    "latency_s": round(time.perf_counter() - t0, 2),
                    "tokens_in": tokens_in, "tokens_out": tokens_out,
                    "hit_max_steps": False, "ungrounded_numbers": ungrounded,
                    "reflections": reflections}
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name, args = tc.function.name, dict(tc.function.arguments or {})
            # Un argumento que la tool no acepta se ignora en silencio en el servidor:
            # el modelo cree que ha filtrado y no es así. Se registra como fallo.
            unknown = sorted(set(args) - param_names.get(name, set(args))) if param_names else []
            try:
                result = await session.call_tool(name, args)
                out = tool_result_text(result)
            except Exception as e:  # tool inexistente o args inválidos: se le dice al modelo
                out = json.dumps({"error": str(e)})
            calls.append({"name": name, "args": args, "unknown_args": unknown,
                          "output": out[:2000]})
            messages.append({"role": "tool", "content": out, "tool_name": name})
    return {"answer": "", "calls": calls, "latency_s": round(time.perf_counter() - t0, 2),
            "tokens_in": tokens_in, "tokens_out": tokens_out, "hit_max_steps": True,
            "ungrounded_numbers": [], "reflections": reflections}


def llm_judge(chat_fn, judge_model, item, run) -> tuple[bool, dict]:
    rubrica = item.get("rubrica", [])
    prompt = JUDGE_PROMPT.format(
        pregunta=item["pregunta"],
        tool_outputs=json.dumps([c["output"] for c in run["calls"]], ensure_ascii=False),
        referencia=item.get("referencia", []),
        respuesta=run["answer"],
        rubrica="\n".join(f"{i+1}. {r}" for i, r in enumerate(rubrica)),
    )
    resp = call_llm(chat_fn, model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    format="json", options=OLLAMA_OPTIONS)
    try:
        verdict = json.loads(resp.message.content)
        crit = [bool(x) for x in verdict.get("criterios", [])]
    except (json.JSONDecodeError, AttributeError):
        return False, {"error": "el juez no devolvió JSON válido",
                       "raw": resp.message.content[:500]}
    # Chequeo determinista complementario: ¿cubre los SKUs de referencia?
    ref = set(item.get("referencia", []))
    ref_recall = len(ref & graders.extract_skus(run["answer"])) / len(ref) if ref else None
    # El juez (mismo modelo pequeño) puede equivocarse: además exige recall >= 2/3
    # de los SKUs de referencia. Un juez sin validar no decide solo.
    passed = (len(crit) == len(rubrica) and all(crit)
              and (ref_recall is None or ref_recall >= 2 / 3))
    return passed, {"criterios": crit, "razon": verdict.get("razon"),
                    "reference_recall": ref_recall}


async def run_eval(eval_path, model, judge_model, only=None, chat_fn=None,
                   server_cmd=None, ground_check=False):
    if chat_fn is None:
        import ollama
        chat_fn = ollama.chat
    items = yaml.safe_load(open(eval_path, encoding="utf-8"))
    if only:
        items = [i for i in items if i["id"] in only]

    params = server_cmd or StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "qoh_mcp_server.py")])
    results = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            ollama_tools = mcp_tools_to_ollama(mcp_tools)
            param_names = tool_param_names(mcp_tools)
            # Warm-up: carga el modelo en memoria para que el cold start no cuente
            # como latencia de Q01, y detecta pronto si Ollama no funciona.
            try:
                call_llm(chat_fn, model=model, options=OLLAMA_OPTIONS,
                         messages=[{"role": "user", "content": "ok"}])
            except InfraError as e:
                raise SystemExit(
                    f"\nOllama no responde con el modelo '{model}': {e}\n"
                    f"Prueba primero:  ollama run {model} \"hola\"\n"
                    "Si falla también, el problema es de Ollama (memoria/GPU), no del eval.")
            for item in items:
                try:
                    run = await run_agent(session, chat_fn, model, ollama_tools,
                                          item["pregunta"], param_names=param_names,
                                          ground_check=ground_check)
                    if item["metodo"] == "llm_judge":
                        passed, details = llm_judge(chat_fn, judge_model, item, run)
                    else:
                        passed, details = graders.grade(item, run["answer"])
                except InfraError as e:
                    results.append({"id": item["id"], "categoria": item["categoria"],
                                    "metodo": item["metodo"], "pregunta": item["pregunta"],
                                    "infra_error": str(e)})
                    print(f"{item['id']:<4} ERROR infraestructura: {e}", flush=True)
                    continue
                tool_ok, tool_details = graders.grade_tools(item.get("tool_esperada"),
                                                            run["calls"])
                results.append({
                    "id": item["id"], "categoria": item["categoria"],
                    "metodo": item["metodo"], "pregunta": item["pregunta"],
                    "passed": passed, "details": details,
                    "tool_ok": tool_ok, "tool_details": tool_details, **run,
                })
                mark = "PASS" if passed else "FAIL"
                print(f"{item['id']:<4} {mark}  tools={'ok' if tool_ok else 'x '}  "
                      f"{run['latency_s']:>5.1f}s  {item['categoria']}", flush=True)
    return results


def summarize(results) -> dict:
    """Las métricas se calculan solo sobre preguntas completadas: un error de
    infraestructura no es un fallo del modelo y se reporta aparte."""
    done = [r for r in results if "infra_error" not in r]

    def rate(rs):
        return round(sum(r["passed"] for r in rs) / len(rs), 3) if rs else 0.0

    def avg(key):
        return round(sum(r[key] for r in done) / len(done), 2) if done else 0.0
    by_cat: dict[str, list] = {}
    for r in done:
        by_cat.setdefault(r["categoria"], []).append(r)
    return {
        "n": len(results),
        "n_completed": len(done),
        "pass_rate": rate(done),
        "tool_accuracy": rate([{"passed": r["tool_ok"]} for r in done]),
        "avg_latency_s": avg("latency_s"),
        "avg_tool_calls": round(sum(len(r["calls"]) for r in done) / len(done), 2) if done else 0.0,
        "total_tokens": sum(r["tokens_in"] + r["tokens_out"] for r in done),
        "max_tokens_in": max((r["tokens_in"] for r in done), default=0),
        "grounded_rate": rate([{"passed": not r.get("ungrounded_numbers")} for r in done]),
        "ungrounded": [r["id"] for r in done if r.get("ungrounded_numbers")],
        "by_categoria": {c: {"n": len(rs), "pass_rate": rate(rs)} for c, rs in by_cat.items()},
        "failed": [r["id"] for r in done if not r["passed"]],
        "infra_errors": [r["id"] for r in results if "infra_error" in r],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "eval_set_qoh.yaml"))
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--judge-model", default=None, help="por defecto, el mismo modelo")
    ap.add_argument("--only", default=None, help="IDs separados por coma, ej. Q01,Q11")
    ap.add_argument("--cpu", action="store_true",
                    help="fuerza la inferencia en CPU (si la GPU hace caer a Ollama)")
    ap.add_argument("--prompt", default="v2", choices=sorted(PROMPTS),
                    help="versión del system prompt")
    ap.add_argument("--ground-check", action="store_true",
                    help="si la respuesta tiene cifras que no devolvió ninguna tool, "
                         "se le pide al modelo que corrija (1 reintento)")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="ventana de contexto de Ollama (ej. 8192). Si el prompt + tools "
                         "la superan, Ollama recorta el principio sin avisar")
    args = ap.parse_args()
    global SYSTEM_PROMPT_VERSION
    SYSTEM_PROMPT_VERSION = args.prompt
    if args.cpu:
        OLLAMA_OPTIONS["num_gpu"] = 0
    if args.num_ctx:
        OLLAMA_OPTIONS["num_ctx"] = args.num_ctx

    only = set(args.only.split(",")) if args.only else None
    judge = args.judge_model or args.model
    results = asyncio.run(run_eval(args.eval, args.model, judge, only,
                                   ground_check=args.ground_check))
    summary = summarize(results)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / f"run_{stamp}_{args.model.replace(':', '-')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "judge_model": judge,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "eval_file": Path(args.eval).name, "ollama_options": OLLAMA_OPTIONS,
        "ground_check": args.ground_check, "grader_version": graders.GRADER_VERSION,
        "summary": summary, "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"Modelo: {args.model}   Juez: {judge}   Prompt: {SYSTEM_PROMPT_VERSION}   "
          f"Opciones: {OLLAMA_OPTIONS}")
    print(f"Completadas: {summary['n_completed']}/{summary['n']}   "
          f"Acierto: {summary['pass_rate']:.0%}   Tools correctas: {summary['tool_accuracy']:.0%}")
    print(f"Latencia media: {summary['avg_latency_s']}s   "
          f"Tool calls por pregunta: {summary['avg_tool_calls']}   "
          f"Groundedness: {summary['grounded_rate']:.0%}")
    if summary["ungrounded"]:
        print(f"Cifras no respaldadas por tools en: {', '.join(summary['ungrounded'])}")
    for c, s in summary["by_categoria"].items():
        print(f"  {c:<18} {s['pass_rate']:.0%}  (n={s['n']})")
    if summary["failed"]:
        print(f"Fallos: {', '.join(summary['failed'])}  -> revisa {out.name}")
    if summary["infra_errors"]:
        print(f"Errores de infraestructura (no puntúan): {', '.join(summary['infra_errors'])}"
              "  -> vuelve a lanzarlas con --only")


if __name__ == "__main__":
    main()
