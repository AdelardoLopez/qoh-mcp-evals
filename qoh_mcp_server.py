"""Servidor MCP de demo sobre un snapshot congelado de QoH (datos anonimizados).

Expone las 4 tools que necesita el eval set. Sustituye la carga del CSV por tu
fuente real (MySQL, API interna...) manteniendo los mismos nombres y salidas:
así el eval set sigue siendo válido.

Ejecutar en solitario (para probar con un cliente MCP):
    python qoh_mcp_server.py
"""

import csv
import os
from pathlib import Path

try:  # SDK mcp 2.x
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # SDK mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

DATA_PATH = Path(os.environ.get(
    "QOH_DATA", Path(__file__).parent / "data" / "qoh_snapshot_2026-10-22.csv"
))

mcp = Server("qoh-stock")


def _load() -> list[dict]:
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("qty_physical", "qty_virtual", "reorder_point"):
            r[k] = int(r[k])
        r["price"] = float(r["price"])
    return rows


@mcp.tool()
def get_stock(sku: str, warehouse: str | None = None) -> dict:
    """Stock de UN SKU concreto. Úsala solo cuando la pregunta incluye un código de SKU
    (formato AAA-0000, ej. CBL-1042); si no lo incluye, pide el código al usuario.
    Devuelve por almacén: unidades físicas, virtuales, punto de reposición, precio,
    valor de inventario (unidades x precio) y fecha de actualización. Sin 'warehouse'
    devuelve A y B y un bloque 'total' con unidades y valor ya sumados."""
    sku_norm = sku.strip().upper()
    rows = [r for r in _load() if r["sku"] == sku_norm]
    if warehouse:
        rows = [r for r in rows if r["warehouse"] == warehouse.strip().upper()]
    if not rows:
        return {"found": False, "sku": sku_norm, "error": "SKU no encontrado"}
    warehouses = [
        {**{k: r[k] for k in ("warehouse", "qty_physical", "qty_virtual",
                              "reorder_point", "price", "updated_at")},
         "inventory_value": round(r["qty_physical"] * r["price"], 2)}
        for r in rows
    ]
    out = {"found": True, "sku": sku_norm, "description": rows[0]["description"],
           "warehouses": warehouses}
    if len(warehouses) > 1:  # los cálculos los hace el código, no el LLM
        out["total"] = {
            "qty_physical": sum(w["qty_physical"] for w in warehouses),
            "inventory_value": round(sum(w["inventory_value"] for w in warehouses), 2),
        }
    return out


@mcp.tool()
def list_stock_discrepancies(min_diff: int = 0, warehouse: str | None = None) -> list[dict]:
    """SKUs cuyo stock físico y virtual difieren en MÁS de 'min_diff' unidades, ordenados
    de mayor a menor diferencia. Cada fila es un SKU en un almacén. Usa 'warehouse'
    ("A" o "B") para limitar el resultado a un almacén; el primer elemento es la mayor
    discrepancia."""
    rows = _load()
    if warehouse:
        rows = [r for r in rows if r["warehouse"] == warehouse.strip().upper()]
    out = []
    for r in rows:
        diff = abs(r["qty_physical"] - r["qty_virtual"])
        if diff > min_diff:
            out.append({"sku": r["sku"], "warehouse": r["warehouse"],
                        "qty_physical": r["qty_physical"],
                        "qty_virtual": r["qty_virtual"], "diff": diff})
    return sorted(out, key=lambda x: x["diff"], reverse=True)


@mcp.tool()
def list_price_discrepancies() -> list[dict]:
    """SKUs con precio distinto entre el almacén A y el B, ordenados de mayor a
    menor diferencia absoluta de precio."""
    by_sku: dict[str, dict] = {}
    for r in _load():
        by_sku.setdefault(r["sku"], {})[r["warehouse"]] = r["price"]
    out = []
    for sku, p in by_sku.items():
        if "A" in p and "B" in p and p["A"] != p["B"]:
            out.append({"sku": sku, "price_A": p["A"], "price_B": p["B"],
                        "diff": round(abs(p["A"] - p["B"]), 2)})
    return sorted(out, key=lambda x: x["diff"], reverse=True)


@mcp.tool()
def list_low_stock(limit: int = 10) -> list[dict]:
    """Úsala para preguntas sobre qué reponer, qué productos están bajos o bajo mínimos.
    No necesita ningún dato del usuario. Devuelve las filas (SKU + almacén) con menos
    stock físico respecto a su punto de reposición, de más urgente a menos urgente.
    El campo 'ratio' lo calcula la tool (stock/punto de reposición): menor de 1 significa
    por debajo del mínimo."""
    rows = _load()
    out = [{"sku": r["sku"], "warehouse": r["warehouse"],
            "qty_physical": r["qty_physical"], "reorder_point": r["reorder_point"],
            "ratio": round(r["qty_physical"] / r["reorder_point"], 3)}
           for r in rows if r["reorder_point"] > 0]
    return sorted(out, key=lambda x: x["ratio"])[:limit]


if __name__ == "__main__":
    mcp.run()
