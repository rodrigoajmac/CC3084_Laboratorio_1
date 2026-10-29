"""Avance del laboratorio 11: valida la fuente mensual y crea un candidato aislado."""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
KEY = ["Año", "Mes cod", "Vía", "Frontera", "País", "Región", "Región dos",
       "Regiones OMT", "MCEO", "Agrupación Residencia", "Tipo de Viajero"]
REQUIRED = KEY + ["Viajero"]


def month(value):
    return pd.Period(value, freq="M")


def check(name, failed, warning="", detail=""):
    return {"control": name, "estado": "error" if failed else "advertencia" if warning else "ok",
            "detalle": detail or warning}


def validate(df, cfg, sheet):
    results = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        results.append(check("columnas_y_tipos", True, detail=f"Faltan: {missing}"))
        return results, None

    numeric = ["Año", "Mes cod", "Viajero"]
    bad_types = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    bad_types += [c for c in KEY[2:] if not (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))]
    results.append(check("columnas_y_tipos", bool(bad_types),
                         detail=f"Tipos incompatibles: {bad_types}" if bad_types else "Columnas y tipos válidos"))
    if bad_types:
        return results, None

    years = pd.to_numeric(df["Año"], errors="coerce")
    months = pd.to_numeric(df["Mes cod"], errors="coerce")
    invalid = years.isna() | months.isna() | (years % 1 != 0) | (months % 1 != 0) | ~months.between(1, 12)
    expected = pd.period_range(month(cfg["first_month"]), month(cfg["last_observed_month"]), freq="M")
    observed = set(pd.Period(year=int(y), month=int(m), freq="M") for y, m in
                   zip(years[~invalid], months[~invalid]))
    gaps = [str(p) for p in expected if p not in observed]
    outside = sum(p < expected[0] or p > expected[-1] for p in observed)
    results.append(check("fechas_y_cobertura", bool(invalid.sum() or gaps),
                         warning=f"{outside} meses fuera del período configurado" if outside else "",
                         detail=f"Fechas inválidas: {int(invalid.sum())}; meses faltantes: {gaps[:12]}"))

    # La clave representa un conteo mensual por combinación de categorías.
    duplicates = int(df.duplicated(subset=KEY).sum())
    results.append(check("unicidad", duplicates > 0, detail=f"Claves duplicadas: {duplicates}"))

    missing_critical = int(df[REQUIRED].isna().any(axis=1).sum())
    optional = [c for c in df.columns if c not in REQUIRED]
    optional_rate = float(df[optional].isna().sum().sum() / (len(df) * len(optional))) if optional and len(df) else 0.0
    results.append(check("valores_faltantes", missing_critical > 0,
                         warning=f"Faltantes opcionales: {optional_rate:.2%}" if optional_rate > cfg["missing_optional_warning_rate"] else "",
                         detail=f"Filas con faltantes indispensables: {missing_critical}; tasa opcional: {optional_rate:.2%}"))

    counts = pd.to_numeric(df["Viajero"], errors="coerce")
    invalid_counts = int((~counts.map(math.isfinite) | (counts < 0)).sum())
    results.append(check("valores_y_denominadores", invalid_counts > 0,
                         detail=f"Conteos negativos o no finitos: {invalid_counts}; no se usan denominadores en este método"))

    # La hoja Datos contiene año y mes explícitos; no se agregan presencias diarias.
    method_failed = sheet != "Datos" or "Fecha" in df.columns or not {"Año", "Mes cod"}.issubset(df.columns)
    results.append(check("regla_metodologica", method_failed,
                         detail="Fuente mensual Datos con Año/Mes cod; no se infiere el mes desde fechas diarias"))
    return results, None if any(r["estado"] == "error" for r in results) else counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--source", type=Path, help="Copia alternativa para probar controles")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    source = args.source.resolve() if args.source else (config_path.parent / cfg["source"]).resolve()
    out_root = (config_path.parent / cfg["output_dir"]).resolve()
    run_dir = out_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"inicio_utc": datetime.now(timezone.utc).isoformat(), "fuente": str(source),
              "hoja": cfg["sheet"], "controles": [], "estado": "fallido"}
    try:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        report["sha256_fuente"] = digest.hexdigest()
        df = pd.read_excel(source, sheet_name=cfg["sheet"], engine="openpyxl")
        checks, counts = validate(df, cfg, cfg["sheet"])
        report["controles"] = checks
        if counts is None:
            raise ValueError("Controles críticos fallidos; no se creó candidato")

        frame = df.assign(Viajero=counts, periodo=pd.PeriodIndex.from_fields(year=df["Año"].astype(int),
                                                                              month=df["Mes cod"].astype(int), freq="M"))
        series = frame.groupby("periodo")["Viajero"].sum().sort_index()
        series = series.loc[month(cfg["first_month"]):month(cfg["last_observed_month"])]
        if len(series) != len(pd.period_range(month(cfg["first_month"]), month(cfg["last_observed_month"]))) or series.isna().any():
            raise ValueError("La serie preparada tiene meses faltantes")
        prepared = pd.DataFrame({"periodo": series.index.astype(str), "viajeros_observados": series.to_numpy()})
        prepared.to_csv(run_dir / "serie_preparada.csv", index=False)

        # Pronóstico inicial: valor del mismo mes del año previo. Sin azar ni ajuste de interfaz.
        target = month(cfg["forecast_month"])
        if target <= month(cfg["last_observed_month"]) or target - 12 not in series.index:
            raise ValueError("El mes a estimar debe ser futuro y tener antecedente hace 12 meses")
        estimate = float(series.loc[target - 12])
        candidate = pd.DataFrame([{"destino": "Guatemala", "periodo_estimado": str(target),
                                   "valor": estimate, "unidad": "viajeros/mes",
                                   "metodo": cfg["method"], "fecha_generacion_utc": report["inicio_utc"]}])
        candidate.to_csv(run_dir / "estimaciones_candidatas.csv", index=False)
        report["estado"] = "candidato_generado_sin_publicar"
        report["filas_fuente"] = len(df)
        report["meses_observados"] = len(series)
        report["estimacion_candidata"] = estimate
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        report["fin_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "reporte_calidad.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{report['estado']}: {run_dir}")
        if "error" in report:
            print(report["error"], file=sys.stderr)
    return 0 if report["estado"] == "candidato_generado_sin_publicar" else 1


if __name__ == "__main__":
    raise SystemExit(main())
