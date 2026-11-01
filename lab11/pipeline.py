"""Laboratorio 11: pipeline reproducible del observatorio, desde la fuente hasta la versión aprobada.

Etapas: lectura -> validación (6 controles) -> preparación -> ajuste -> evaluación temporal ->
estimación -> almacenamiento -> monitoreo -> aceptación -> publicación. Cada ejecución queda en MLflow.

Uso (desde la raíz del repositorio):  python lab11/pipeline.py [--config RUTA] [--source RUTA]
Códigos de salida: 0 publicada, 3 rechazada por la regla de aceptación, 1 fallida (nada se publica).
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
KEY = ["Año", "Mes cod", "Vía", "Frontera", "País", "Región", "Región dos",
       "Regiones OMT", "MCEO", "Agrupación Residencia", "Tipo de Viajero"]
REQUIRED = KEY + ["Viajero"]
CONTROLES = ["columnas_y_tipos", "fechas_y_cobertura", "unicidad", "valores_faltantes",
             "valores_y_denominadores", "regla_metodologica"]
EST_COLS = ["destino", "periodo_estimado", "valor", "unidad", "tipo", "run_id",
            "fecha_generacion_utc", "version_procedimiento", "metodo"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def month(value):
    return pd.Period(value, freq="M")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name, failed, warning="", detail=""):
    return {"control": name, "estado": "error" if failed else "advertencia" if warning else "ok",
            "detalle": detail + (f" | Advertencia: {warning}" if warning else "")}


def periodos(df):
    return pd.PeriodIndex.from_fields(year=df["Año"].astype(int), month=df["Mes cod"].astype(int), freq="M")


def serie_objetivo(df, cfg):
    """Serie mensual de la variable: suma de conteos por mes, sin rellenar meses ausentes."""
    filas = df[df["País"] != cfg["pais_residentes"]] if cfg["excluir_residentes"] else df
    return filas.groupby(periodos(filas))["Viajero"].sum().sort_index() * cfg["factor_reparto"]


# ------------------------------------------------------------------ controles de calidad
def validate(df, cfg, sheet):
    """Ejecuta los seis controles. Devuelve la lista de resultados; cualquier 'error' detiene el flujo."""
    results = []

    def omitir_resto():
        hechos = {r["control"] for r in results}
        results.extend({"control": c, "estado": "omitido", "detalle": "No se evaluó: falló un control previo"}
                       for c in CONTROLES if c not in hechos)
        return results

    # 1. Columnas y tipos
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        results.append(check("columnas_y_tipos", True, detail=f"Faltan columnas indispensables: {missing}"))
        return omitir_resto()
    numeric = ["Año", "Mes cod", "Viajero"]
    bad_types = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    bad_types += [c for c in KEY[2:] if not (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))]
    results.append(check("columnas_y_tipos", bool(bad_types),
                         detail=f"Tipos incompatibles: {bad_types}" if bad_types else "Columnas y tipos válidos"))
    if bad_types:
        return omitir_resto()

    # 2. Fechas y cobertura: los meses ausentes no se rellenan con cero; detienen el flujo.
    years, months = df["Año"], df["Mes cod"]
    invalid = years.isna() | months.isna() | (years % 1 != 0) | (months % 1 != 0) | ~months.between(1, 12)
    expected = pd.period_range(month(cfg["first_month"]), month(cfg["last_observed_month"]), freq="M")
    valid = df[~invalid]
    observed = set(periodos(valid))
    target_months = set(serie_objetivo(valid, cfg).index) if len(valid) else set()
    gaps = [str(p) for p in expected if p not in observed]
    # Meses con registros de la fuente pero sin la variable (p. ej., cierre de fronteras de 2020): quedan
    # ausentes, no en cero. Detienen solo si caen en el período que usa el procedimiento.
    gaps_target = [p for p in expected if p in observed and p not in target_months]
    gaps_fit = [str(p) for p in gaps_target if p >= month(cfg["fit_start"])]
    gaps_old = [str(p) for p in gaps_target if p < month(cfg["fit_start"])]
    after = sorted(str(p) for p in observed if p > expected[-1])
    before = sum(p < expected[0] for p in observed)
    aviso = []
    if gaps_old:
        aviso.append(f"meses sin la variable antes del período de ajuste, se dejan ausentes: {gaps_old}")
    if after:
        aviso.append(f"la fuente trae meses posteriores a la cobertura configurada: {after[:6]}")
    if before:
        aviso.append(f"{before} meses anteriores al período configurado se ignoran")
    results.append(check("fechas_y_cobertura", bool(invalid.sum() or gaps or gaps_fit), warning="; ".join(aviso),
                         detail=f"Fechas inválidas: {int(invalid.sum())}; meses sin registros: {gaps[:12]}; "
                                f"meses del período de ajuste sin la variable: {gaps_fit[:12]}"))

    # 3. Unicidad: una fila por combinación de categorías y mes (granularidad mensual de la fuente).
    duplicates = int(df.duplicated(subset=KEY).sum())
    results.append(check("unicidad", duplicates > 0, detail=f"Claves duplicadas: {duplicates}"))

    # 4. Faltantes: combinaciones sin fila = sin viajeros en esa categoría (permitido);
    #    nulos en campos indispensables o meses sin registros impiden estimar.
    missing_critical = int(df[REQUIRED].isna().any(axis=1).sum())
    optional = [c for c in df.columns if c not in REQUIRED]
    optional_rate = float(df[optional].isna().sum().sum() / (len(df) * len(optional))) if optional and len(df) else 0.0
    limit = cfg["quality"]["missing_optional_warning_rate"]
    results.append(check("valores_faltantes", missing_critical > 0,
                         warning=f"faltantes opcionales {optional_rate:.2%} > {limit:.0%}" if optional_rate > limit else "",
                         detail=f"Filas con faltantes indispensables: {missing_critical}; tasa opcional ({optional}): {optional_rate:.2%}"))

    # 5. Valores y denominadores: conteos finitos y no negativos; el WAPE divide entre la suma observada
    #    de los meses evaluados y el MAPE entre cada mes observado.
    counts = pd.to_numeric(df["Viajero"], errors="coerce")
    bad = ~np.isfinite(counts) | (counts < 0)
    invalid_counts = int(bad.sum())
    serie = serie_objetivo(df[~invalid & ~bad], cfg)
    eval_months = [month(o) + i for o in cfg["validation_origins"] for i in range(1, cfg["horizon"] + 1)]
    denominators = serie.reindex(eval_months)
    zeros = [str(p) for p, v in denominators.items() if v == 0]
    results.append(check("valores_y_denominadores", invalid_counts > 0 or float(denominators.fillna(0).sum()) == 0,
                         warning=f"meses evaluados con valor observado cero (se excluyen del MAPE): {zeros}" if zeros else "",
                         detail=f"Conteos negativos o no finitos: {invalid_counts}; "
                                f"denominador del WAPE: {float(denominators.fillna(0).sum()):,.0f}"))

    # 6. Regla metodológica: la serie mensual proviene de la hoja mensual (Año, Mes cod) y no de fechas
    #    diarias; y no mezcla meses con y sin cobertura de residentes (la fuente dejó de contarlos en 2023).
    daily = [c for c in df.columns if str(c).strip().lower() in ("fecha", "día", "dia")]
    residents = valid[valid["País"] == cfg["pais_residentes"]]
    last_resident = str(periodos(residents).max()) if len(residents) else None
    mixes = (not cfg["excluir_residentes"]) and last_resident is not None and month(last_resident) < expected[-1]
    reappear = last_resident is not None and month(last_resident) >= month("2023-01")
    method_failed = sheet != "Datos" or bool(daily) or mixes
    results.append(check("regla_metodologica", method_failed,
                         warning="los residentes reaparecen desde 2023: revise la definición de la variable" if reappear else "",
                         detail=f"Hoja '{sheet}' con Año/Mes cod; columnas diarias: {daily}; último mes con residentes: "
                                f"{last_resident}; residentes excluidos en todo el período: {cfg['excluir_residentes']}"))
    return results


def calidad_resumen(df):
    optional = [c for c in df.columns if c not in REQUIRED]
    return {"filas": int(len(df)),
            "duplicados": int(df.duplicated(subset=[c for c in KEY if c in df.columns]).sum()),
            "tasa_faltantes_opcional": float(df[optional].isna().mean().mean()) if optional else 0.0,
            "meses_cubiertos": int(periodos(df).nunique())}


# ------------------------------------------------------------------ procedimiento
def ajustar(y, cfg):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    m = cfg["model"]
    s = y.copy()
    s.index = s.index.to_timestamp()
    s = s.asfreq("MS")
    return ExponentialSmoothing(s, trend=m["trend"], damped_trend=m["damped_trend"], seasonal=m["seasonal"],
                                seasonal_periods=m["seasonal_periods"], initialization_method="estimated").fit()


def ingenuo_estacional(train, h):
    """y_hat(T+i) = y(T+i-12): repite el mismo mes del año anterior (requiere h <= 12)."""
    last = train.index.max()
    return np.array([float(train[last + i - 12]) for i in range(1, h + 1)])


def estimar(train, cfg, h):
    """Estima los h meses posteriores al último mes de `train` usando solo esos datos.
    Devuelve las estimaciones, los parámetros para reproducirlas y el modelo ajustado (si existe)."""
    assert h <= 12, "El ingenuo estacional y la referencia requieren h <= 12"
    if cfg["method"] == "ingenuo_estacional_12":
        last = train.index.max()
        params = {"formula": "y_hat(t) = y(t - 12)",
                  "valores_base": {str(last + i - 12): float(train[last + i - 12]) for i in range(1, h + 1)}}
        return ingenuo_estacional(train, h), params, None
    if cfg["method"] == "holt_winters_amortiguado":
        fit = ajustar(train, cfg)
        params = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fit.params.items()}
        return fit.forecast(h).to_numpy(), params, fit
    raise ValueError(f"Método desconocido: {cfg['method']}")


def evaluar(y, cfg):
    """Validación temporal: oculta los meses posteriores a cada origen y los estima solo con datos previos.
    La referencia de la regla es el ingenuo estacional; último valor y promedio de 12 meses son contexto."""
    h, rows = cfg["horizon"], []
    for origin in map(month, cfg["validation_origins"]):
        train = y.loc[month(cfg["fit_start"]):origin]
        assert train.index.max() <= origin, "Se usó información posterior al origen"
        pred = estimar(train, cfg, h)[0]
        ref = ingenuo_estacional(train, h)
        for i in range(1, h + 1):
            t = origin + i
            if t not in y.index:          # sin valor observado comparable: no se calcula error
                continue
            rows.append({"origen": str(origin), "periodo": str(t), "horizonte": i, "observado": float(y[t]),
                         "procedimiento": float(pred[i - 1]), "referencia": float(ref[i - 1]),
                         "ctx_ultimo_valor": float(train.iloc[-1]), "ctx_promedio_12": float(train.iloc[-12:].mean()),
                         "ultimo_mes_entrenamiento": str(train.index.max())})
    return pd.DataFrame(rows)


def metricas(obs, pred):
    e = pred - obs
    nz = obs != 0
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mape": float(np.mean(np.abs(e[nz] / obs[nz])) * 100) if nz.any() else float("nan"),
            "wape": float(np.sum(np.abs(e)) / np.sum(obs)), "meses_mape_excluidos": int((~nz).sum())}


def integridad(est, cfg, run_id):
    expected = [str(month(cfg["last_observed_month"]) + i) for i in range(1, cfg["horizon"] + 1)]
    problems = []
    if list(est.columns) != EST_COLS:
        problems.append("columnas distintas de las requeridas")
    if list(est["periodo_estimado"]) != expected:
        problems.append(f"períodos {list(est['periodo_estimado'])} != {expected}")
    if not np.isfinite(est["valor"]).all() or (est["valor"] < 0).any():
        problems.append("valores negativos o no finitos")
    if set(est["run_id"]) != {run_id}:
        problems.append("identificador de ejecución inconsistente")
    return problems


# ------------------------------------------------------------------ publicación y monitoreo
def leer_vigente(pub_dir):
    pointer = pub_dir / "vigente.json"
    if not pointer.is_file():
        return None
    info = json.loads(pointer.read_text(encoding="utf-8"))
    vdir = pub_dir / info["version_dir"]
    meta = json.loads((vdir / "metadatos.json").read_text(encoding="utf-8"))
    return {"info": info, "meta": meta, "estimaciones": pd.read_csv(vdir / "estimaciones.csv")}


def escribir_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def publicar(pub_dir, run_id, est, obs, meta):
    """Publicación atómica: primero la carpeta completa de la versión, después el puntero vigente.json."""
    versions = pub_dir / "versiones"
    tmp, final = versions / f".{run_id}.tmp", versions / run_id
    tmp.mkdir(parents=True, exist_ok=False)
    est.to_csv(tmp / "estimaciones.csv", index=False)
    obs.to_csv(tmp / "observados.csv", index=False)
    escribir_json(tmp / "metadatos.json", meta)
    os.replace(tmp, final)
    escribir_json(pub_dir / "vigente.json.tmp", {"run_id": run_id, "version_dir": f"versiones/{run_id}",
                                                "publicado_utc": now(), "motivo": "aprobada por la regla de aceptación"})
    os.replace(pub_dir / "vigente.json.tmp", pub_dir / "vigente.json")


def monitorear(cfg, y, est, calidad, fuente_max, vigente):
    """Tres verificaciones: actualización, calidad y resultados. Las variaciones solo generan alertas."""
    mon, last = cfg["monitoring"], month(cfg["last_observed_month"])
    out = []

    # Actualización: fuente y resultados deben llegar al último período esperado.
    expected_first = str(last + 1)
    estado = "ok" if fuente_max == last and est["periodo_estimado"].iloc[0] == expected_first else (
        "alerta" if fuente_max > last else "error")
    out.append({"verificacion": "actualizacion", "estado": estado,
                "detalle": f"Último mes de la fuente: {fuente_max}; esperado: {last}; primera estimación: "
                           f"{est['periodo_estimado'].iloc[0]} (esperada {expected_first})"})

    # Calidad: cambios frente a la versión aprobada vigente.
    if vigente is None:
        out.append({"verificacion": "calidad", "estado": "ok", "detalle": "Sin versión aprobada previa para comparar"})
    else:
        prev = vigente["meta"]["calidad"]
        alerts = []
        row_change = abs(calidad["filas"] - prev["filas"]) / max(prev["filas"], 1)
        if row_change > mon["max_row_change_rate"]:
            alerts.append(f"filas cambian {row_change:.1%}")
        if abs(calidad["tasa_faltantes_opcional"] - prev["tasa_faltantes_opcional"]) > mon["max_missing_rate_change"]:
            alerts.append("cambia la tasa de faltantes")
        if calidad["duplicados"] != prev["duplicados"]:
            alerts.append(f"duplicados {prev['duplicados']} -> {calidad['duplicados']}")
        if calidad["meses_cubiertos"] != prev["meses_cubiertos"]:
            alerts.append(f"meses cubiertos {prev['meses_cubiertos']} -> {calidad['meses_cubiertos']}")
        out.append({"verificacion": "calidad", "estado": "alerta" if alerts else "ok",
                    "detalle": "; ".join(alerts) or f"Sin cambios relevantes frente a {vigente['info']['run_id']}"})

    # Resultados: negativos/no finitos detienen; variaciones grandes piden revisión.
    vals = est.set_index("periodo_estimado")["valor"]
    if not np.isfinite(vals).all() or (vals < 0).any():
        out.append({"verificacion": "resultados", "estado": "error", "detalle": "Estimaciones negativas o no finitas"})
    else:
        alerts = []
        for p, v in vals.items():
            ly = y.get(month(p) - 12)
            if ly and abs(v / ly - 1) > mon["max_change_vs_last_year"]:
                alerts.append(f"{p}: {v / ly - 1:+.0%} frente al mismo mes del año anterior")
        if vigente is not None:
            prev = vigente["estimaciones"].set_index("periodo_estimado")["valor"]
            for p in vals.index.intersection(prev.index):
                if abs(vals[p] / prev[p] - 1) > mon["max_change_vs_previous_version"]:
                    alerts.append(f"{p}: {vals[p] / prev[p] - 1:+.0%} frente a la versión vigente")
        out.append({"verificacion": "resultados", "estado": "alerta" if alerts else "ok",
                    "detalle": "; ".join(alerts) or "Estimaciones finitas, no negativas y dentro de los umbrales"})
    return out


# ------------------------------------------------------------------ utilidades de trazabilidad
def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20)
        except Exception:
            return None
    head = run("rev-parse", "HEAD")
    commit = head.stdout.strip() if head and head.returncode == 0 else os.environ.get("GIT_COMMIT", "desconocido")
    diff = run("diff", "--quiet", "HEAD", "--", "lab11/pipeline.py", "lab11/config.json")
    dirty = "desconocido" if diff is None or diff.returncode not in (0, 1) else str(diff.returncode == 1).lower()
    return commit, dirty


def versiones_dependencias():
    from importlib.metadata import version
    return {p: version(p) for p in ("pandas", "numpy", "openpyxl", "statsmodels", "mlflow")} | {
        "python": sys.version.split()[0]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--source", type=Path, help="Ruta alternativa de la fuente (p. ej., una copia de prueba)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):    # la consola de Windows no es UTF-8 por omisión
        stream.reconfigure(encoding="utf-8")

    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    source = (args.source or (Path(os.environ["LAB11_FUENTE"]) if os.environ.get("LAB11_FUENTE") else base / cfg["source"])).resolve()
    pub_dir, log_dir = base / cfg["published_dir"], base / cfg["log_dir"]
    for d in (pub_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])

    commit, dirty = git_info()
    log = {"inicio_utc": now(), "fin_utc": None, "run_id": None, "etapas_completadas": [], "advertencias": [],
           "errores": [], "estado": "fallido", "publicacion": {"resultado": "no_publicada", "motivo": None},
           "git_commit": commit, "codigo_modificado_sin_commit": dirty, "fuente": source.name}
    vigente_antes = leer_vigente(pub_dir)
    log["publicacion"]["vigente_antes"] = vigente_antes["info"]["run_id"] if vigente_antes else None

    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow_tracking_uri"])
    try:
        if uri.startswith("http"):          # falla en segundos, no tras minutos de reintentos
            urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=5)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg["mlflow_experiment"])
        run = mlflow.start_run(run_name=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    except Exception as exc:
        log["errores"].append(f"MLflow no disponible en {uri}: {exc}")
        log["fin_utc"] = now()
        escribir_json(log_dir / f"sin_mlflow_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json", log)
        print(f"fallido: MLflow no disponible en {uri}", file=sys.stderr)
        return 1

    run_id = log["run_id"] = run.info.run_id
    run_dir = base / cfg["output_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "fuente": source.name, "hoja": cfg["sheet"], "controles": []}
    etapa = "lectura"
    code = 1
    try:
        mlflow.set_tags({"git_commit": commit, "codigo_modificado_sin_commit": dirty, "destino": cfg["destino"],
                         "variable": cfg["variable"], "fuente_archivo": source.name,
                         "config_sha256": sha256(config_path), "version_procedimiento": cfg["procedure_version"]})
        mlflow.log_params({"metodo": cfg["method"], "referencia": cfg["reference"], "fit_start": cfg["fit_start"],
                           "first_month": cfg["first_month"], "last_observed_month": cfg["last_observed_month"],
                           "horizon": cfg["horizon"], "validation_origins": ",".join(cfg["validation_origins"]),
                           "factor_reparto": cfg["factor_reparto"], "excluir_residentes": cfg["excluir_residentes"],
                           "random_seed": cfg["random_seed"], **{f"modelo_{k}": v for k, v in cfg["model"].items()},
                           **{f"aceptacion_{k}": v for k, v in cfg["acceptance"].items()}})
        mlflow.log_dict(versiones_dependencias(), "entorno/versiones.json")
        mlflow.log_artifact(str(config_path), "configuracion")

        # Lectura
        report["sha256_fuente"] = fuente_sha = sha256(source)
        mlflow.set_tag("fuente_sha256", fuente_sha)
        df = pd.read_excel(source, sheet_name=cfg["sheet"], engine="openpyxl")
        log["etapas_completadas"].append(etapa)

        # Validación
        etapa = "validacion"
        report["controles"] = checks = validate(df, cfg, cfg["sheet"])
        escribir_json(run_dir / "reporte_calidad.json", report)
        for c in checks:
            mlflow.log_metric(f"control_{c['control']}_ok", int(c["estado"] in ("ok", "advertencia")))
            if c["estado"] == "advertencia":
                log["advertencias"].append(f"{c['control']}: {c['detalle']}")
        failed = [c["control"] for c in checks if c["estado"] == "error"]
        mlflow.set_tag("controles_calidad", "fallidos: " + ",".join(failed) if failed else "aprobados")
        if failed:
            raise RuntimeError(f"Controles críticos fallidos: {failed}. No se evalúa ni se publica.")
        log["etapas_completadas"].append(etapa)

        # Preparación
        etapa = "preparacion"
        calidad = calidad_resumen(df)
        y_all = serie_objetivo(df, cfg)
        fuente_max = periodos(df).max()
        y = y_all.loc[month(cfg["first_month"]):month(cfg["last_observed_month"])]
        obs = pd.DataFrame({"destino": cfg["destino"], "periodo": y.index.astype(str), "valor": y.to_numpy(),
                            "unidad": cfg["unidad"], "tipo": "observado"})
        obs.to_csv(run_dir / "serie_preparada.csv", index=False)
        log["etapas_completadas"].append(etapa)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Ajuste y evaluación temporal (antes de la estimación final)
            etapa = "evaluacion"
            ev = evaluar(y, cfg)
            ev.to_csv(run_dir / "evaluacion.csv", index=False)
            m_proc = metricas(ev["observado"].to_numpy(), ev["procedimiento"].to_numpy())
            m_ref = metricas(ev["observado"].to_numpy(), ev["referencia"].to_numpy())
            m_ctx = {f"ctx_{c}_{k}": v for c in ("ultimo_valor", "promedio_12")
                     for k, v in metricas(ev["observado"].to_numpy(), ev[f"ctx_{c}"].to_numpy()).items() if k in ("mae", "wape")}
            log["etapas_completadas"].append(etapa)

            etapa = "ajuste"
            pred, params, fit = estimar(y.loc[month(cfg["fit_start"]):], cfg, cfg["horizon"])
            if fit is not None:
                fit.save(str(run_dir / "modelo.pkl"))
            log["etapas_completadas"].append(etapa)

            etapa = "estimacion"
            periods = [str(month(cfg["last_observed_month"]) + i) for i in range(1, cfg["horizon"] + 1)]
            est = pd.DataFrame({"destino": cfg["destino"], "periodo_estimado": periods, "valor": pred,
                                "unidad": cfg["unidad"], "tipo": "estimación", "run_id": run_id,
                                "fecha_generacion_utc": log["inicio_utc"],
                                "version_procedimiento": f"{cfg['procedure_version']}+{commit[:7]}",
                                "metodo": cfg["method"]})[EST_COLS]
            log["etapas_completadas"].append(etapa)
        log["advertencias"] += sorted({f"{w.category.__name__}: {w.message}" for w in caught
                                       if not issubclass(w.category, (DeprecationWarning, FutureWarning))})
        mlflow.log_metrics({f"proc_{k}": v for k, v in m_proc.items()} | {f"ref_{k}": v for k, v in m_ref.items()}
                           | m_ctx | {"meses_evaluados": len(ev), "ratio_mae_proc_ref": m_proc["mae"] / m_ref["mae"]})
        mlflow.log_dict(params, "modelo/parametros.json")
        if fit is not None:
            mlflow.log_artifact(str(run_dir / "modelo.pkl"), "modelo")

        etapa = "almacenamiento"
        est.to_csv(run_dir / "estimaciones.csv", index=False)
        for f in ("reporte_calidad.json", "serie_preparada.csv", "evaluacion.csv", "estimaciones.csv"):
            mlflow.log_artifact(str(run_dir / f), "resultados")
        log["etapas_completadas"].append(etapa)

        # Monitoreo y reproducibilidad frente a la versión vigente
        etapa = "monitoreo"
        mon = monitorear(cfg, y, est, calidad, fuente_max, vigente_antes)
        escribir_json(run_dir / "monitoreo.json", mon)
        mlflow.log_artifact(str(run_dir / "monitoreo.json"), "resultados")
        for m in mon:
            mlflow.log_metric(f"monitoreo_{m['verificacion']}_alerta", int(m["estado"] != "ok"))
            if m["estado"] != "ok":
                log["advertencias"].append(f"monitoreo {m['verificacion']} ({m['estado']}): {m['detalle']}")
        repro = None
        if vigente_antes and vigente_antes["meta"].get("fuente_sha256") == fuente_sha and \
                vigente_antes["meta"].get("config_sha256") == sha256(config_path):
            prev = vigente_antes["estimaciones"].set_index("periodo_estimado")["valor"]
            repro = float((est.set_index("periodo_estimado")["valor"] / prev - 1).abs().max())
            mlflow.log_metric("dif_rel_max_vs_vigente", repro)
            mlflow.set_tag("reproducible_vs_vigente", str(repro <= cfg["reproducibility_rel_tolerance"]).lower())
        log["etapas_completadas"].append(etapa)

        # Regla de aceptación (fijada en config.json antes de evaluar)
        etapa = "aceptacion"
        acc = cfg["acceptance"]
        problems = integridad(est, cfg, run_id)
        reglas = {"controles_calidad_sin_errores": not failed,
                  "salidas_integras": not problems,
                  "monitoreo_sin_errores": all(m["estado"] != "error" for m in mon),
                  "wape_menor_o_igual_umbral": m_proc["wape"] <= acc["max_wape"],
                  "mae_no_mayor_que_ratio_x_referencia": m_proc["mae"] <= acc["max_mae_ratio_vs_reference"] * m_ref["mae"]}
        aprobada = all(reglas.values())
        mlflow.log_dict({"reglas": reglas, "problemas_integridad": problems, "umbrales": acc}, "resultados/aceptacion.json")
        mlflow.set_tag("decision", "aprobada" if aprobada else "rechazada")
        log["etapas_completadas"].append(etapa)

        etapa = "publicacion"
        if aprobada:
            meta = {"run_id": run_id, "destino": cfg["destino"], "variable": cfg["variable"], "unidad": cfg["unidad"],
                    "metodo": cfg["method"], "version_procedimiento": est["version_procedimiento"].iloc[0],
                    "git_commit": commit, "fecha_generacion_utc": log["inicio_utc"], "fuente_archivo": source.name,
                    "fuente_sha256": fuente_sha, "config_sha256": sha256(config_path),
                    "periodo_ajuste": [cfg["fit_start"], cfg["last_observed_month"]],
                    "periodos_evaluados": [ev["periodo"].iloc[0], ev["periodo"].iloc[-1]], "meses_evaluados": len(ev),
                    "metricas_procedimiento": m_proc, "metricas_referencia": m_ref, "metricas_contexto": m_ctx,
                    "parametros_procedimiento": params, "reglas_aceptacion": reglas,
                    "calidad": calidad, "monitoreo": mon, "dif_rel_max_vs_vigente_anterior": repro}
            publicar(pub_dir, run_id, est, obs, meta)
            log["publicacion"] = {**log["publicacion"], "resultado": "publicada", "motivo": "aprobada"}
            log["estado"], code = "publicada", 0
        else:
            no_ok = [k for k, v in reglas.items() if not v]
            log["publicacion"]["motivo"] = f"rechazada por la regla de aceptación: {no_ok}; se conserva la versión vigente"
            log["estado"], code = "rechazada", 3
        log["etapas_completadas"].append(etapa)
    except Exception as exc:
        log["errores"].append(f"[{etapa}] {exc}")
        log["publicacion"]["motivo"] = f"falló la etapa {etapa}; se conserva la versión vigente"
        mlflow.set_tags({"etapa_fallida": etapa, "error": str(exc)[:500]})
        if not isinstance(exc, RuntimeError):
            log["errores"].append(traceback.format_exc(limit=3))
    finally:
        vigente_despues = leer_vigente(pub_dir)
        log["publicacion"]["vigente_despues"] = vigente_despues["info"]["run_id"] if vigente_despues else None
        log["fin_utc"] = now()
        escribir_json(log_dir / f"{run_id}.json", log)
        mlflow.log_artifact(str(log_dir / f"{run_id}.json"), "registro")
        mlflow.set_tag("estado", log["estado"])
        mlflow.end_run(status="FINISHED" if code in (0, 3) else "FAILED")
        print(f"{log['estado']}: run {run_id} | vigente: {log['publicacion']['vigente_despues']}")
        for e in log["errores"][:1]:
            print(e, file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
