"""Prueba controlada de fallo y recuperación con copias de la fuente (nunca modifica el original).

1. Clave duplicada      -> el control de unicidad detiene el flujo; la versión vigente no cambia.
2. Columna ausente      -> el control de columnas detiene el flujo; la versión vigente no cambia.
3. Variación elevada    -> pasa los controles, el monitoreo alerta y la regla rechaza; la vigente no cambia.
4. Copia corregida      -> se eliminan los duplicados y la ejecución vuelve a publicar.

Uso:  python lab11/prueba_fallo.py [--config RUTA]     (requiere el servidor de MLflow; ~5 minutos)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

import pipeline as pl


def vigente(pub):
    p = pub / "vigente.json"
    return json.loads(p.read_text(encoding="utf-8"))["run_id"] if p.is_file() else None


def ultimo_registro(log_dir, run_antes):
    regs = sorted((p for p in log_dir.glob("*.json") if not p.name.startswith(("restauracion", "sin_mlflow"))),
                  key=os.path.getmtime)
    reg = json.loads(regs[-1].read_text(encoding="utf-8"))
    return reg if reg["run_id"] not in run_antes else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=pl.ROOT / "config.json")
    config = parser.parse_args().config.resolve()
    cfg, base = json.loads(config.read_text(encoding="utf-8")), config.parent
    source = Path(os.environ.get("LAB11_FUENTE") or base / cfg["source"]).resolve()
    pub, log_dir = base / cfg["published_dir"], base / cfg["log_dir"]
    work = base / cfg["output_dir"] / "prueba_fallo"
    work.mkdir(parents=True, exist_ok=True)
    original = pd.read_excel(source, sheet_name=cfg["sheet"], engine="openpyxl")

    no_res = original["País"] != cfg["pais_residentes"]
    meses = pl.periodos(original)
    variado = original.copy()
    escala = no_res & (meses >= pd.Period("2025-07", "M")) & (meses <= pd.Period("2025-12", "M"))
    variado.loc[escala, "Viajero"] *= 1.4
    con_duplicado = pd.concat([original, original.iloc[[100]]], ignore_index=True)
    casos = [("clave_duplicada", con_duplicado, 1),
             ("columna_ausente", original.drop(columns="Tipo de Viajero"), 1),
             ("variacion_elevada", variado, 3),
             ("copia_corregida", con_duplicado.drop_duplicates(subset=pl.KEY), 0)]

    resumen, vistos = [], set()
    for nombre, df, esperado in casos:
        ruta = work / f"{nombre}.xlsx"
        df.to_excel(ruta, sheet_name=cfg["sheet"], index=False)
        antes = vigente(pub)
        code = pl.main(["--config", str(config), "--source", str(ruta)])
        reg = ultimo_registro(log_dir, vistos)
        vistos.add(reg["run_id"])
        despues = vigente(pub)
        ok = code == esperado and ((despues == antes) if esperado else (despues == reg["run_id"]))
        resumen.append({"caso": nombre, "codigo_salida": code, "codigo_esperado": esperado, "run_id": reg["run_id"],
                        "estado": reg["estado"], "errores": reg["errores"][:1],
                        "alertas": [a for a in reg["advertencias"] if a.startswith("monitoreo")],
                        "vigente_antes": antes, "vigente_despues": despues, "resultado_esperado": ok})
        print(f"{'OK ' if ok else 'FALLA'} {nombre}: salida {code} (esperada {esperado}); vigente {antes} -> {despues}")

    (base / "evidencia").mkdir(exist_ok=True)
    (base / "evidencia" / "prueba_fallo.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
    return 0 if all(r["resultado_esperado"] for r in resumen) else 1


if __name__ == "__main__":
    raise SystemExit(main())
