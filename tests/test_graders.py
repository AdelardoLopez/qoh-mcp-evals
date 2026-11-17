"""Tests del evaluador: un eval que no se ha probado no es fiable.
Cada pregunta tiene una respuesta buena (debe pasar) y una mala (debe fallar)."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
import graders  # noqa: E402

ITEMS = {i["id"]: i for i in yaml.safe_load(
    open(Path(__file__).parent.parent / "eval_set_qoh.yaml", encoding="utf-8"))}

CASES = [
    # id, respuesta, debe_pasar
    ("Q01", "Hay 340 unidades del SKU CBL-1042 en el almacén A.", True),
    ("Q01", "Hay 172 unidades del SKU CBL-1042 en el almacén A.", False),
    ("Q01", "El SKU CBL-1042 tiene stock en el almacén A.", False),  # sin cifra
    ("Q02", "En el almacén B hay 85 unidades de CBL-1043.", True),
    ("Q03", "El precio es $12.89.", True),
    ("Q03", "El precio es 12,89 $.", True),          # formato español
    ("Q03", "El precio es $12.99.", False),
    ("Q04", "Sí, el SKU CBL-1042 está en el almacén A con 340 unidades.", True),
    ("Q04", "No, el SKU CBL-1042 no está en el almacén A.", False),
    ("Q05", "Se actualizó el 2026-10-22.", True),
    ("Q05", "Se actualizó el 2026-10-21.", False),
    # salidas reales de qwen2.5:7b (run 2026-11-04) convertidas en tests de regresión
    ("Q05", "El SKU CBL-1042 se actualizó en el almacén A el 22 de octubre de 2026.", True),
    ("Q05", "Se actualizó el 21 de octubre de 2026.", False),
    ("Q05", "Last updated on October 22, 2026.", True),
    ("Q06", "El stock total es de 512 unidades (340 en A y 172 en B).", True),
    ("Q06", "El stock total es de 340 unidades.", False),
    ("Q07", "El valor total del inventario es de $6,599.68.", True),
    ("Q07", "El valor total del inventario es de 6.599,68 €.", True),
    ("Q07", "El valor total es $4,382.60.", False),
    ("Q07", "El valor total del inventario del SKU CBL-1042 en el almacén A es $4,342.60 "
            "y en el almacén B es $2,193.68.", False),   # no suma y además calcula mal
    ("Q08", "El almacén A tiene más unidades (120 frente a 85).", True),
    ("Q08", "El almacén B tiene más unidades.", False),
    ("Q09", "Los SKUs son CBL-1042, WRP-0087 y DCT-3310.", True),
    ("Q09", "Los SKUs son CBL-1042 y WRP-0087.", False),             # falta uno (recall)
    ("Q09", "CBL-1042, WRP-0087, DCT-3310 y CBL-2210.", False),      # sobra uno (precision)
    ("Q10", "CBL-2210 y HSE-0501 tienen precio distinto.", True),
    ("Q11", "El SKU XYZ-9999 no existe en el inventario.", True),
    ("Q11", "El SKU XYZ-9999 no se encuentra en nuestros registros.", True),  # salida real de qwen2.5:7b
    ("Q11", "Hay 25 unidades del SKU XYZ-9999.", False),             # alucinación
    ("Q11", "El SKU XYZ-9999 no existe, pero quizá tenga 10 unidades.", False),
    ("Q12", "Ningún SKU tiene un precio de 999999999.", True),
    ("Q12", "El SKU HSE-0501 tiene ese precio.", False),
    ("Q12", 'Mis herramientas no proporcionan información sobre los precios individuales de los SKUs. Puedo verificar el stock y las discrepancias de stock, pero no los precios específicos. ¿Podrías proporcionarme un código SKU para que pueda realizar una consulta?', True),   # real qwen2.5:7b v3
    ("Q13", "¿A qué SKU te refieres? Hay varios cables negros: CBL-1042, CBL-1043 y CBL-0099.", True),
    ("Q13", "Hay 340 unidades del cable negro CBL-1042.", False),
    ("Q13", "En el almacén A hay 340 unidades del cable negro y en el almacén B hay 172 unidades.", False),
    ("Q15", "No dispongo de datos de ventas ni de beneficio; solo tengo stock y precio.", True),
    ("Q15", "El que más beneficio da es HSE-0501.", False),
    ("Q15", 'Mis herramientas no proporcionan información sobre ventas o beneficios. Puedo ayudarte a verificar el stock físico y virtual de los productos si lo necesitas. ¿Podrías indicarme un SKU específico o prefieres que liste los productos con stock bajo?', True),   # real qwen2.5:7b v3
    ("Q15", "No tengo datos de beneficio, pero HSE-0501 es el más caro, así que dará más.", False),
    ("Q16", "WRP-0087 tiene la mayor discrepancia en B (40 uds). En A tiene 58 unidades.", True),
    ("Q16", "WRP-0087 tiene la mayor discrepancia en B.", False),     # falta stock en A
    ("Q16", 'El SKU "CBL-1042" del almacén B tiene la mayor discrepancia entre stock físico y '
            "virtual, con una diferencia de 15 unidades. En el almacén A, ese SKU tiene 340 "
            "unidades físicas.", False),
    ("Q17", "HSE-0501: 12.50 en B frente a 9.10 en A, diferencia de 3.40.", True),
    ("Q17", "HSE-0501, con una diferencia de 3.10.", False),
]


@pytest.mark.parametrize("qid,answer,should_pass", CASES)
def test_grader(qid, answer, should_pass):
    passed, details = graders.grade(ITEMS[qid], answer)
    assert passed == should_pass, f"{qid}: {answer!r} -> {details}"


def test_sku_digits_are_not_numbers():
    assert 1042.0 not in graders.extract_numbers("SKU CBL-1042 sin cifra")


def test_tool_check_order_and_none():
    calls = [{"name": "list_stock_discrepancies"}, {"name": "get_stock"}]
    assert graders.grade_tools(ITEMS["Q16"]["tool_esperada"], calls)[0]
    assert not graders.grade_tools(ITEMS["Q16"]["tool_esperada"], calls[::-1])[0]
    assert graders.grade_tools(None, [])[0]
    assert not graders.grade_tools(None, [{"name": "get_stock"}])[0]


def test_tool_check_fails_on_unknown_args():
    calls = [{"name": "list_low_stock", "unknown_args": ["ratio"]}]
    ok, details = graders.grade_tools(ITEMS["Q14"]["tool_esperada"], calls)
    assert not ok and details["unknown_args"] == {"list_low_stock": ["ratio"]}


def test_ungrounded_numbers_detects_invented_stock():
    # Q16 real: la tool solo devolvió discrepancias de B; "250" en A es inventado
    out = '{"result": [{"sku": "WRP-0087", "warehouse": "B", "qty_physical": 200, "qty_virtual": 160, "diff": 40}]}'
    ans = "WRP-0087 tiene una diferencia de 40 unidades. En el almacén A tiene 250 unidades."
    assert graders.ungrounded_numbers(ans, "pregunta", [out]) == [250.0]


def test_grounded_answer_has_no_ungrounded_numbers():
    out = '{"warehouses": [{"qty_physical": 340, "price": 12.89, "updated_at": "2026-10-22"}]}'
    ans = "Hay 340 unidades a $12.89, actualizado el 22 de octubre de 2026."
    assert graders.ungrounded_numbers(ans, "pregunta", [out]) == []


def test_decimal_with_trailing_zero_is_grounded():
    # H02 real: "$9.10" se marcaba como 910 inventado y disparaba una reflexión innecesaria
    out = '{"price": 9.1}'
    assert graders.ungrounded_numbers("Tiene un precio de $9.10.", "q", [out]) == []


def test_thousands_separator_is_grounded():
    # Q07 real: "$6,599.68" se marcaba como 6.59968 inventado
    out = '{"inventory_value": 6599.68}'
    assert graders.ungrounded_numbers("El valor es de $6,599.68.", "q", [out]) == []
