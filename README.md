# QoH MCP · Evaluación de un agente de inventario con LLMs locales

Servidor MCP que expone una herramienta de stock por almacén (QoH, *quantity on hand*) a un
LLM local vía Ollama, y la suite de evaluación con la que se llevó de **71 % a 100 %** en el set de
desarrollo, y **75 % en un held-out nunca usado para iterar**, analizando errores y cambiando una
sola cosa por experimento.

> Los datos son **sintéticos**: un snapshot de 16 filas inspirado en un catálogo de gestión de
> cables. No contiene datos reales de ninguna empresa.

## Resultados

Configuración final: `qwen2.5:7b` · prompt `v3` · `num_ctx 8192` · `--ground-check`.

| Conjunto | Preguntas | Acierto | Tools correctas | Groundedness | Latencia media |
|---|---|---|---|---|---|
| Desarrollo (usado para iterar) | 17 | 100 % | 94 % | 94 % | 34,6 s |
| **Held-out (nunca usado para iterar)** | 8 | **75 %** | 100 % | 88 % | 45,8 s |

La cifra que vale es la del held-out: el set de desarrollo está inflado por definición,
porque el sistema se ajustó mirando sus fallos.

### Cómo se llegó ahí

| # | Cambio (uno por experimento) | Acierto dev | Qué se aprendió |
|---|---|---|---|
| 1 | Baseline: prompt v1 | 71 % | 5 fallos: 1 grader, 1 límite del modelo (aritmética), 1 prompt, 2 diseño/descripción de tools |
| 2 | Arreglos de grader y tools + prompt v2 | 76 % | El prompt v2 arregló un caso límite y rompió dos casos normales |
| 3 | Solo `num_ctx 8192` | 76 % | Sin efecto: hipótesis de contexto truncado **rechazada** con datos |
| 4 | Solo prompt v3 | 82 % → 94 %* | *Tras repuntuar con graders g4: 2 "regresiones" eran falsos negativos del grader |
| 5 | Solo `--ground-check` (reflexión) | 100 % | Elimina la cifra inventada en la pregunta de varios pasos; groundedness 76 % → 94 % por +32 % de latencia |

Comparación de modelos (mismo prompt v3 y mismo eval): `qwen2.5:7b` 82 % / 26 s frente a
`llama3.1:8b` 65 % / 54 s. El modelo más pequeño gana.

### Fallos del held-out (no corregidos, para mantenerlo limpio)

- **H04**: filtrar SKUs por el *sentido* de la discrepancia (virtual > físico), no por su tamaño.
- **H06**: pregunta fuera de alcance (plazos de proveedor); análisis de la traza pendiente.
- **H02**: pasa, pero contiene una cifra no respaldada por las tools.
- Trabajo futuro: normalizar el SKU en la tool (`cbl1042` → `CBL-1042`); hoy lo resuelve el modelo.

## Lecciones

1. **Analiza los errores antes de tocar el prompt.** En el primer análisis, solo 1 de 5 fallos se
   arreglaba con el prompt; el resto eran del grader, de las tools o un límite del modelo.
2. **El evaluador también se evalúa.** La mitad de los fallos aparentes fueron bugs de los graders.
   Cada salida real que destapa un bug se convierte en un test de regresión (69 tests).
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
| `results/published/` | Informes JSON de las ejecuciones citadas en este README |

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
