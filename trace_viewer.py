"""Visor de trazas con anotación humana (análisis de errores al estilo Hamel Husain).

    streamlit run trace_viewer.py

- Muestra cada traza completa en una pantalla: pregunta, tool calls con sus salidas,
  respuesta y veredicto del grader.
- Permite etiquetar como experto: PASS/FAIL, una crítica (por qué) y un modo de fallo libre.
  Los modos de fallo se escriben libremente y se reutilizan: así la taxonomía sale de los
  datos (bottom-up), no de una lista fija.
- Pestaña "Modos de fallo": cuenta los modos anotados (dónde atacar primero).
- Las etiquetas alimentan validate_judge.py, que mide si el LLM juez coincide contigo.
"""

import json
from collections import Counter

import streamlit as st

import traces as T

st.set_page_config(page_title="QoH · trazas", layout="wide")
st.title("Visor de trazas · QoH MCP")

runs = T.list_runs()
if not runs:
    st.warning("No hay ejecuciones en results/ ni en results/published/.")
    st.stop()

# ---------- filtros ----------
with st.sidebar:
    run_path = st.selectbox("Ejecución", runs, format_func=lambda p: p.name)
    all_traces = T.load_traces(run_path)
    labels = T.load_labels()

    cats = sorted({t["categoria"] for t in all_traces})
    sel_cats = st.multiselect("Categoría", cats, default=cats)
    grader = st.radio("Veredicto del grader", ["Todas", "Solo FAIL", "Solo PASS"],
                      horizontal=True)
    only_unlabeled = st.checkbox("Solo sin etiquetar", value=False)
    hide_dupes = st.checkbox("Ocultar casos ya etiquetados en otra ejecución", value=True,
                             help="Misma pregunta y misma respuesta = mismo caso.")

    labeled_cases = set()
    for t_all in T.all_labeled_traces():
        labeled_cases.add(T.dedupe_key(t_all))

    def keep(t):
        if t["categoria"] not in sel_cats:
            return False
        if grader == "Solo FAIL" and t["grader_passed"]:
            return False
        if grader == "Solo PASS" and not t["grader_passed"]:
            return False
        if only_unlabeled and t["key"] in labels:
            return False
        if hide_dupes and t["key"] not in labels and T.dedupe_key(t) in labeled_cases:
            return False
        return True

    visible = [t for t in all_traces if keep(t)]
    n_lab = sum(1 for t in all_traces if t["key"] in labels)
    st.caption(f"{len(visible)} trazas visibles · {n_lab}/{len(all_traces)} etiquetadas "
               f"en esta ejecución · {len(labels)} etiquetas en total")

tab_trace, tab_modes = st.tabs(["Trazas", "Modos de fallo"])

# ---------- una traza ----------
with tab_trace:
    if not visible:
        st.info("Ninguna traza con estos filtros.")
    else:
        if "idx" not in st.session_state:
            st.session_state.idx = 0
        st.session_state.idx = min(st.session_state.idx, len(visible) - 1)
        nav1, nav2, nav3 = st.columns([1, 1, 6])
        if nav1.button("◀ Anterior", disabled=st.session_state.idx == 0):
            st.session_state.idx -= 1
            st.rerun()
        if nav2.button("Siguiente ▶", disabled=st.session_state.idx >= len(visible) - 1):
            st.session_state.idx += 1
            st.rerun()
        t = visible[st.session_state.idx]
        nav3.markdown(f"**{st.session_state.idx + 1} / {len(visible)}** · `{t['id']}` · "
                      f"{t['categoria']} · {t['model']} · prompt {t['prompt_version']}")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Pregunta")
            st.write(t["pregunta"])
            st.subheader("Respuesta del modelo")
            st.info(t["answer"] or "(vacía)")
            st.subheader(f"Tool calls ({len(t['calls'])})")
            if not t["calls"]:
                st.caption("Ninguna.")
            for i, c in enumerate(t["calls"], 1):
                bad = f" ⚠️ args inválidos: {c['unknown_args']}" if c.get("unknown_args") else ""
                with st.expander(f"{i}. {c['name']}({json.dumps(c['args'], ensure_ascii=False)}){bad}"):
                    try:
                        st.code(json.dumps(json.loads(c["output"]), ensure_ascii=False, indent=1),
                                language="json")
                    except (json.JSONDecodeError, TypeError):
                        st.code(c["output"])

        with right:
            st.subheader("Grader automático")
            verdict = "✅ PASS" if t["grader_passed"] else "❌ FAIL"
            st.markdown(f"{verdict} · método `{t['metodo']}` · tools "
                        f"{'ok' if t['tool_ok'] else '✗'} · {t['latency_s']} s")
            if t.get("ungrounded_numbers"):
                st.warning(f"Cifras no respaldadas: {t['ungrounded_numbers']}")
            with st.expander("Detalles del grader"):
                st.json(t["grader_details"])

            st.subheader("Tu etiqueta (experto)")
            prev = labels.get(t["key"], {})
            known_modes = sorted({v["failure_mode"] for v in labels.values()
                                  if v.get("failure_mode")})
            with st.form(key=f"form_{t['key']}"):
                # Sin valor por defecto: que el anotador decida sin ancla (ni del grader).
                label = st.radio("¿La respuesta es aceptable?", ["PASS", "FAIL"],
                                 index={"PASS": 0, "FAIL": 1}.get(prev.get("label")),
                                 horizontal=True)
                critique = st.text_area(
                    "Crítica: por qué (servirá de ejemplo para el juez)",
                    value=prev.get("critique", ""), height=110,
                    placeholder="Ej.: Da el stock de A pero inventa el de B: ninguna tool devolvió 250.")
                mode = st.text_input("Modo de fallo (solo si FAIL; texto libre)",
                                     value=prev.get("failure_mode", ""),
                                     placeholder="Ej.: inventa cifra / sentido de la discrepancia")
                if known_modes:
                    st.caption("Ya usados: " + " · ".join(known_modes))
                save = st.form_submit_button("Guardar y siguiente", type="primary")
            if save:
                if label is None:
                    st.error("Elige PASS o FAIL.")
                elif label == "FAIL" and not critique.strip():
                    st.error("Un FAIL necesita crítica: es lo que hace útil la etiqueta.")
                else:
                    T.save_label(t["key"], label, critique, mode if label == "FAIL" else "")
                    if st.session_state.idx < len(visible) - 1 and not only_unlabeled:
                        st.session_state.idx += 1
                    st.rerun()
            if prev and st.button("Borrar etiqueta"):
                T.delete_label(t["key"])
                st.rerun()

# ---------- conteo de modos de fallo ----------
with tab_modes:
    labeled = T.all_labeled_traces()
    fails = [x for x in labeled if x["label"] == "FAIL"]
    st.metric("Etiquetas humanas", len(labeled),
              f"{len(fails)} FAIL · {len(labeled) - len(fails)} PASS", delta_color="off")
    counts = Counter(x["failure_mode"] or "(sin modo)" for x in fails)
    if counts:
        st.subheader("Modos de fallo, de más a menos frecuente")
        st.bar_chart({m: n for m, n in counts.most_common()}, horizontal=True)
        st.dataframe(
            [{"modo": x["failure_mode"], "id": x["id"], "ejecución": x["run"],
              "crítica": x["critique"]} for x in sorted(fails, key=lambda x: x["failure_mode"])],
            use_container_width=True, hide_index=True)
    else:
        st.caption("Aún no hay fallos etiquetados.")
