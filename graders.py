"""Graders deterministas para el eval set del MCP QoH.

Regla: todo lo que se pueda puntuar con código se puntúa con código.
El LLM juez solo entra en las preguntas 'llm_judge' (ver run_evals.py).

Cada grader devuelve (passed: bool, details: dict) para poder auditar
por qué pasó o falló cada pregunta.
"""

import re
import unicodedata

# Sube la versión con cada cambio de criterio. Para comparar ejecuciones
# puntuadas con versiones distintas, repuntúa la antigua con regrade.py.
GRADER_VERSION = "g4"

SKU_RE = re.compile(r"\b[A-Z]{3}-\d{4}\b")
NUM_RE = re.compile(r"\d[\d.,]*\d|\d")

REFUSAL_KEYWORDS = [
    "no existe", "no encontr", "no se encontr", "no encuentr", "no se encuentr",
    "no tengo", "no dispongo",
    "no puedo", "no hay datos", "no cuento con", "no esta disponible",
    "fuera de", "ningun", "no aparece", "not found", "does not exist",
    "no data", "cannot",
]
NEGATIVE_KEYWORDS = ["no esta", "no hay", "no existe", "no se encuentra", "no tiene stock"]
POSITIVE_KEYWORDS = ["si esta", "si hay", "esta en el almacen", "hay stock", "dispone de"]
QUANTITY_RE = re.compile(r"\d+\s*(unidades|uds|units)\b", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


# ---------- utilidades de normalización ----------

def normalize(text: str) -> str:
    """Minúsculas y sin tildes, para comparar texto con tolerancia."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def extract_skus(text: str) -> set[str]:
    return set(SKU_RE.findall(text.upper()))


def extract_numbers(text: str) -> list[float]:
    """Extrae números aceptando formato inglés (6,599.68) y español (6.599,68).
    Los dígitos de los SKUs se eliminan antes para que CBL-1042 no cuente como 1042."""
    text = SKU_RE.sub(" ", text.upper())
    nums: list[float] = []
    for tok in NUM_RE.findall(text):
        candidates = {
            tok.replace(",", ""),                     # inglés
            tok.replace(".", "").replace(",", "."),   # español
        }
        for c in candidates:
            try:
                nums.append(float(c))
            except ValueError:
                pass
    return nums


def stray_numbers(answer: str, question: str) -> list[float]:
    """Números de la respuesta que no estaban ya en la pregunta."""
    q = set(extract_numbers(question))
    return [n for n in extract_numbers(answer) if n not in q]


# ---------- graders por método ----------

def date_variants(iso: str) -> list[str]:
    """2026-10-22 -> formatos con los que un modelo puede escribir esa fecha."""
    y, m, d = iso.split("-")
    di, mi = int(d), int(m)
    return [iso, f"{d}/{m}/{y}", f"{di}/{mi}/{y}", f"{d}-{m}-{y}",
            f"{di} de {MESES[mi-1]} de {y}", f"{di} de {MESES[mi-1]} del {y}",
            f"{MONTHS[mi-1]} {di}, {y}", f"{di} {MONTHS[mi-1]} {y}"]


def _match_value(expected, answer: str, tol: float | None) -> bool:
    if isinstance(expected, bool):
        return grade_boolean(expected, answer)[0]
    if isinstance(expected, (int, float)):
        t = tol if tol is not None else 0.0
        return any(abs(n - float(expected)) <= t + 1e-9 for n in extract_numbers(answer))
    exp = str(expected)
    if ISO_DATE_RE.match(exp):
        a = normalize(answer)
        return any(normalize(v) in a for v in date_variants(exp))
    if len(exp) <= 2:  # p. ej. almacén "A": exige palabra completa, sensible a mayúsculas
        return re.search(rf"(?<![\w-]){re.escape(exp)}(?![\w-])", answer) is not None
    return normalize(exp) in normalize(answer)


def grade_exact_match(expected, answer: str, tol: float | None = None):
    if isinstance(expected, dict):  # varios valores: deben aparecer todos
        per_key = {k: _match_value(v, answer, tol) for k, v in expected.items()}
        return all(per_key.values()), {"per_key": per_key}
    ok = _match_value(expected, answer, tol)
    return ok, {"numbers_found": extract_numbers(answer)[:10]}


def grade_numeric_tolerance(expected, answer: str, tol: float):
    return grade_exact_match(expected, answer, tol)


def grade_boolean(expected: bool, answer: str):
    a = normalize(answer).strip()
    if any(k in a for k in NEGATIVE_KEYWORDS) or re.match(r"^no\b", a):
        got = False
    elif any(k in a for k in POSITIVE_KEYWORDS) or re.match(r"^(si|yes)\b", a):
        got = True
    else:
        return False, {"got": None, "reason": "no se detecta afirmación ni negación"}
    return got == expected, {"got": got}


def grade_precision_recall(expected: list[str], answer: str):
    exp, got = set(expected), extract_skus(answer)
    tp = len(exp & got)
    precision = tp / len(got) if got else 0.0
    recall = tp / len(exp) if exp else 1.0
    return precision == 1.0 and recall == 1.0, {
        "precision": round(precision, 3), "recall": round(recall, 3),
        "missing": sorted(exp - got), "extra": sorted(got - exp),
    }


# Negación + objeto ("no proporcionan información", "no incluyen datos"...). Más robusto
# que una lista cerrada de frases: el modelo cambia de redacción con cada prompt.
REFUSAL_PATTERN = re.compile(
    r"\bno\b[^.?!]{0,60}\b(informacion|datos|herramientas?|tools?|acceso|precios? individuales"
    r"|ventas|beneficios?|busqueda|buscar)\b")


def is_refusal(answer: str) -> bool:
    a = normalize(answer)
    return any(k in a for k in REFUSAL_KEYWORDS) or REFUSAL_PATTERN.search(a) is not None


def grade_refusal(answer: str, question: str):
    """Pasa si reconoce que no puede responder Y no filtra datos inventados
    (SKUs o números que no estaban en la pregunta)."""
    refused = is_refusal(answer)
    leaked_skus = sorted(extract_skus(answer) - extract_skus(question))
    leaked_nums = stray_numbers(answer, question)
    ok = refused and not leaked_skus and not leaked_nums
    return ok, {"refused": refused, "leaked_skus": leaked_skus,
                "leaked_numbers": leaked_nums[:10]}


def ungrounded_numbers(answer: str, question: str, tool_outputs: list[str]) -> list[float]:
    """Cifras de la respuesta que no aparecen ni en la pregunta ni en ninguna salida de
    tool. Heurística de groundedness: detecta cifras inventadas (p. ej. Q16 '250 unidades'
    cuando ninguna tool lo devolvió). Puede dar falsos positivos con datos derivados
    (porcentajes), por eso se reporta como métrica y no decide el PASS."""
    known = set(extract_numbers(question))
    for out in tool_outputs:
        known.update(extract_numbers(out))
    return sorted({n for n in extract_numbers(answer) if n not in known})


def grade_clarification(answer: str):
    """Pasa si hace una pregunta de vuelta y no da cantidades por su cuenta.
    Mostrar opciones de SKU está permitido."""
    asks = "?" in answer
    gives_quantity = QUANTITY_RE.search(answer) is not None
    return asks and not gives_quantity, {"asks_back": asks, "gives_quantity": gives_quantity}


# ---------- comprobación de tools ----------

def expected_tool_names(tool_esperada) -> list[str]:
    if tool_esperada is None:
        return []
    items = tool_esperada if isinstance(tool_esperada, list) else [tool_esperada]
    names = []
    for it in items:
        m = re.match(r"\s*([a-zA-Z_]\w*)\s*\(", str(it))
        if m:
            names.append(m.group(1))
    return names


def grade_tools(tool_esperada, calls: list[dict]):
    """Métrica separada del acierto: ¿llamó a las tools correctas, en orden?
    Si no se espera ninguna tool, pasa solo si no llamó a ninguna."""
    expected = expected_tool_names(tool_esperada)
    called = [c["name"] for c in calls]
    bad_args = {c["name"]: c["unknown_args"] for c in calls if c.get("unknown_args")}
    if not expected:
        return len(called) == 0, {"expected": [], "called": called}
    it = iter(called)  # subsecuencia en orden (permite llamadas extra entre medias)
    ok = all(name in it for name in expected) and not bad_args
    return ok, {"expected": expected, "called": called, "unknown_args": bad_args}


def grade(item: dict, answer: str) -> tuple[bool, dict]:
    """Despacha según 'metodo'. llm_judge se resuelve fuera (necesita el modelo)."""
    method = item["metodo"]
    exp = item.get("respuesta_esperada")
    tol = item.get("tolerancia")
    q = item["pregunta"]
    if method == "exact_match":
        return grade_exact_match(exp, answer)
    if method == "numeric_tolerance":
        return grade_numeric_tolerance(exp, answer, tol or 0.0)
    if method == "boolean":
        return grade_boolean(exp, answer)
    if method == "precision_recall":
        return grade_precision_recall(exp, answer)
    if method == "refusal":
        return grade_refusal(answer, q)
    if method == "clarification":
        return grade_clarification(answer)
    raise ValueError(f"Método no soportado por graders deterministas: {method}")
