"""Comprueba que el held-out set está bien formado y que sus graders distinguen una
respuesta buena de una mala. No ejecuta el modelo: validar el evaluador no contamina
el held-out."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
import graders  # noqa: E402

ITEMS = {i["id"]: i for i in yaml.safe_load(
    open(Path(__file__).parent.parent / "eval_holdout_qoh.yaml", encoding="utf-8"))}

CASES = [
    ("H01", "En el almacén B hay 75 unidades físicas de WRP-0012.", True),
    ("H01", "En el almacén B hay 70 unidades de WRP-0012.", False),       # coge el virtual
    ("H02", "El precio de HSE-0501 en A es $9.10.", True),
    ("H02", "El precio de HSE-0501 es $12.50.", False),                  # precio de B
    ("H03", "Suma 95 unidades físicas (5 en A y 90 en B).", True),
    ("H03", "Suma 108 unidades.", False),                                # suma el virtual
    ("H04", "CBL-0099 (en B) y DCT-3310 (en A).", True),
    ("H04", "WRP-0087, CBL-1042 y DCT-3310.", False),                    # tamaño, no sentido
    ("H05", "Hay 340 unidades del SKU CBL-1042 en el almacén A.", True),
    ("H05", "El SKU cbl1042 no existe.", False),
    ("H06", "No dispongo de datos de proveedores ni de plazos de entrega.", True),
    ("H06", "El plazo habitual es de 15 días.", False),
    ("H06", 'Mis herramientas se centran en el inventario de los almacenes A y B, no en el plazo de entrega de los proveedores. Puedo ayudarte a verificar el stock actual de fundas trenzadas en nuestros almacenes si me proporcionas el código SKU. ¿Podrías darme ese código, por favor?', True),   # real qwen2.5:7b, falso negativo de g4
    ("H06", "Suele tardar una semana.", False),               # inventa sin cifras ni negación
    ("H07", "HSE-0501 (3.40 de diferencia); en el almacén B tiene 31 unidades.", True),
    ("H07", "HSE-0501, con 44 unidades.", False),                        # stock de A
]


@pytest.mark.parametrize("qid,answer,should_pass", CASES)
def test_holdout_grader(qid, answer, should_pass):
    passed, details = graders.grade(ITEMS[qid], answer)
    assert passed == should_pass, f"{qid}: {answer!r} -> {details}"


def test_holdout_ids_do_not_clash_with_dev_set():
    dev = yaml.safe_load(open(Path(__file__).parent.parent / "eval_set_qoh.yaml",
                              encoding="utf-8"))
    assert not set(ITEMS) & {i["id"] for i in dev}
    assert not {i["pregunta"] for i in ITEMS.values()} & {i["pregunta"] for i in dev}
