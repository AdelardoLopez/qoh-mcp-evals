# QoH MCP · Evaluación de un agente de inventario con LLMs locales

Servidor MCP que expone una herramienta de stock por almacén (QoH, *quantity on hand*) a un
LLM local vía Ollama, y la suite de evaluación con la que se llevó de **71 % a 100 %** en el set de
desarrollo y a **75 % en bruto / 88 % corregido en un held-out nunca usado para iterar**, analizando
errores y cambiando una sola cosa por experimento.

> Los datos son **sintéticos**: un snapshot de 16 filas inspirado en un catálogo de gestión de
> cables. No contiene datos reales de ninguna empresa.

## Resultados

Configuración final: `qwen2.5:7b` · prompt `v3` · `num_ctx 8192` · `--ground-check`.
Métricas puntuadas con los graders actuales (`g5`); entre paréntesis, el resultado bruto que dio
la ejecución con los graders de su momento.

| Conjunto | Preguntas | Acierto | Tools correctas | Groundedness | Latencia media |
|---|---|---|---|---|---|
| Desarrollo (usado para iterar), 2 ejecuciones | 17 | 100 % · 100 % | 94 % | 100 % | 34,6 s · 32,2 s |
| **Held-out (nunca usado para iterar)** | 8 | **88 %** (75 % bruto) | 100 % | 100 % | 45,8 s |

La cifra que vale es la del held-out: el set de desarrollo está inflado por definición, porque el
sistema se ajustó mirando sus fallos. La diferencia entre el 75 % bruto y el 88 % es un falso
negativo del evaluador en H06 que se corrigió de forma general (ver abajo); se publican las dos.

### Cómo se llegó ahí

| # | Cambio (uno por experimento) | Acierto dev bruto | Repuntuado g5 | Qué se aprendió |
|---|---|---|---|---|
| 1 | Baseline: prompt v1 | 71 % | 76 % | 5 fallos: 1 grader, 1 límite del modelo (aritmética), 1 prompt, 2 diseño/descripción de tools |
| 2 | Arreglos de tools + prompt v2 | 76 % | 76 % | El prompt v2 arregló un caso límite y rompió dos casos normales |
| 3 | Solo `num_ctx 8192` | 76 % | 76 % | Sin efecto: hipótesis de contexto truncado **rechazada** con datos |
| 4 | Solo prompt v3 | 82 % | 94 % | Las 2 "regresiones" (Q12, Q15) eran falsos negativos del grader |
| 5 | Solo `--ground-check` (reflexión) | 100 % | 100 % | Elimina la cifra inventada de la pregunta de varios pasos; groundedness 94 % → 100 % por +32 % de latencia |

Comparación de modelos (mismo prompt v3 y mismo eval, g5): `qwen2.5:7b` 94 % / 26 s frente a
`llama3.1:8b` 71 % / 54 s. El modelo más pequeño gana.

### Held-out: fallos y cómo se trataron

Regla: el sistema (prompt, tools, modelo) **no se tocó** tras ver el held-out. Solo se corrigió el
evaluador cuando el fallo era de medición, con un arreglo general y un test de regresión.

- **H04 — fallo real del modelo.** Debía listar los SKUs con más stock virtual que físico; listó todas
  las discrepancias e invirtió el sentido. Trabajo futuro: que la tool devuelva la discrepancia con signo.
- **H06 — falso negativo del evaluador.** Rechazo correcto ("se centran en el inventario, no en el plazo
  de entrega"), que el detector de rechazos por frases no reconocía (4.º caso). Ahora basta una negación
  explícita, y lo que protege es la comprobación de que no se filtran SKUs ni cifras inventadas.
- **H02 — falso positivo de groundedness.** "$9.10" se interpretaba también como 910 (punto de miles) y
  se marcaba como cifra inventada, lo que **disparó una reflexión innecesaria** (latencia extra). Ahora un
  número está respaldado si alguna de sus interpretaciones aparece en los datos. Mismo bug en Q07 del set
  de desarrollo ("$6,599.68").
- Trabajo futuro: normalizar el SKU en la tool (`cbl1042` → `CBL-1042`); hoy lo resuelve el modelo.

## Lecciones

1. **Analiza los errores antes de tocar el prompt.** En el primer análisis, solo 1 de 5 fallos se
   arreglaba con el prompt; el resto eran del grader, de las tools o un límite del modelo.
2. **El evaluador también se evalúa.** 6 de los 14 fallos investigados (43 %) eran bugs de los graders, no del modelo.
   Cada salida real que destapa un bug se convierte en un test de regresión (73 tests).
3. **No pidas cálculos al LLM.** Los totales y valores se calculan en la tool.
4. **Groundedness ≠ acierto.** Una respuesta puede contener la cifra correcta y además otra inventada.
5. **Un cambio por experimento y graders versionados**; si cambia el grader, repuntúa lo antiguo.

## Estructura

| Archivo | Qué hace |
|---|---|
| `qoh_mcp_server.py` | Servidor MCP con 4 tools: `get_stock`, `list_stock_discrepancies`, `list_price_discrepancies`, `list_low_stock` |
| `data/qoh_snapshot_2026-10-22.csv` | Snapshot congelado: la única fuente de verdad del eval |
| `eval_set_qoh.yaml` | Set de desarrollo: 17 preguntas (consulta, agregación, comparación, casos límite, fuera de alcance, varios pasos, abierta) |
| `eval_holdout_qoh.yaml` | Held-out: 8 preguntas nuevas, ejecutadas una sola vez con la configuración congelada |
| `graders.py` | Puntuación determinista: exact match, tolerancia numérica, booleano, precision/recall, rechazo, aclaración, groundedness |
| `run_evals.py` | Bucle de agente (Ollama + MCP), reintentos ante caídas, LLM juez para preguntas abiertas, informe JSON |
| `compare_runs.py` | Compara dos ejecuciones pregunta a pregunta (arreglos y regresiones) |
| `regrade.py` | Repuntúa una ejecución guardada con los graders actuales, sin volver a llamar al modelo |
| `tests/` | Tests del evaluador (respuesta buena pasa, mala falla) y end-to-end sin LLM |
| `results/published/` | Informes JSON brutos de las ejecuciones citadas; `regraded/` = repuntuados con g5 |

## Uso

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b

python -m pytest tests -q                                   # 1) el evaluador funciona
python run_evals.py --prompt v3 --num-ctx 8192 --ground-check   # 2) eval de desarrollo
python run_evals.py --eval eval_holdout_qoh.yaml --prompt v3 --num-ctx 8192 --ground-check
python compare_runs.py                                      # compara las dos últimas ejecuciones
```

Opciones: `--model`, `--judge-model`, `--only Q01,Q11`, `--cpu`, `--prompt v2|v3`,
`--num-ctx`, `--ground-check`.

## Decisiones de diseño

- **Determinista primero**: todas las preguntas salvo la abierta se puntúan con código; el juez se
  contrasta con un recall de SKUs de referencia porque un juez sin validar no decide solo.
- **Dos métricas separadas**: acierto de la respuesta y acierto de las tools (nombre, orden y
  argumentos válidos), para distinguir "razonó mal" de "eligió mal la herramienta".
- **Tests de alucinación**: SKU inexistente, búsqueda imposible, pregunta ambigua y fuera de alcance.
- **Errores de infraestructura aparte**: una caída de Ollama se reintenta y, si persiste, no cuenta
  como fallo del modelo.
- **Prompts y graders versionados** (`--prompt`, `GRADER_VERSION`), registrados en cada informe.
- **Groundedness**: toda cifra de la respuesta debe aparecer en la salida de alguna tool. Con
  `--ground-check`, si el modelo inventa una cifra se le pide que corrija una vez (reflexión).

## Limitaciones

- Eval pequeño (17 + 8 preguntas) y una ejecución por configuración: el modelo no es del todo
  determinista, así que las diferencias de una pregunta (≈6 puntos) no son significativas.
- El groundedness es una heurística numérica: puede marcar datos derivados (porcentajes) como no respaldados.
- El juez es el mismo modelo de 7B que se evalúa.

## Licencia

MIT
