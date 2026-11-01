"""Pruebas con datos sintéticos de los controles, la validación temporal y la publicación.

Ejecutar:  python -m unittest discover -s lab11/tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as pl  # noqa: E402

CFG = json.loads((Path(pl.ROOT) / "config.json").read_text(encoding="utf-8")) | {
    "first_month": "2021-01", "fit_start": "2022-01", "last_observed_month": "2026-06",
    "method": "ingenuo_estacional_12"}


def sintetico():
    """Dos categorías de no residentes por mes y residentes hasta diciembre de 2022, como la fuente real."""
    rows = []
    for p in pd.period_range("2021-01", "2026-06", freq="M"):
        base = {"Año": p.year, "Mes cod": p.month, "Mes": p.strftime("%B"), "Vía": "Aérea", "Frontera": "Aeropuerto",
                "Región": "R", "Región dos": "R2", "Regiones OMT": "OMT", "MCEO": "M", "Tipo de Viajero": "Turista"}
        rows.append(base | {"País": "El Salvador", "Agrupación Residencia": "El Salvador", "Viajero": 1000.0 + p.month})
        rows.append(base | {"País": "México", "Agrupación Residencia": "México", "Viajero": 500.0})
        if p.year <= 2022:
            rows.append(base | {"País": "Guatemala", "Agrupación Residencia": "Guatemala", "Viajero": 800.0})
    return pd.DataFrame(rows)


def estados(df, cfg=CFG, sheet="Datos"):
    return {r["control"]: r["estado"] for r in pl.validate(df, cfg, sheet)}


class Controles(unittest.TestCase):
    def test_datos_validos_pasan_los_seis(self):
        self.assertEqual(set(estados(sintetico()).values()), {"ok"})

    def test_columna_ausente_detiene(self):
        e = estados(sintetico().drop(columns="Viajero"))
        self.assertEqual(e["columnas_y_tipos"], "error")
        self.assertEqual(e["unicidad"], "omitido")

    def test_tipo_incompatible_detiene(self):
        df = sintetico().astype({"Viajero": str})
        self.assertEqual(estados(df)["columnas_y_tipos"], "error")

    def test_mes_faltante_en_ajuste_detiene_y_no_se_rellena(self):
        df = sintetico()
        df = df[~((df["Año"] == 2024) & (df["Mes cod"] == 5))]
        self.assertEqual(estados(df)["fechas_y_cobertura"], "error")

    def test_mes_sin_variable_antes_del_ajuste_solo_advierte(self):
        df = sintetico()
        df = df[~((df["Año"] == 2021) & (df["Mes cod"] == 4) & (df["País"] != "Guatemala"))]
        self.assertEqual(estados(df)["fechas_y_cobertura"], "advertencia")
        self.assertNotIn(pd.Period("2021-04", "M"), pl.serie_objetivo(df, CFG).index)  # ausente, no cero

    def test_fecha_invalida_detiene(self):
        df = sintetico()
        df.loc[0, "Mes cod"] = 13
        self.assertEqual(estados(df)["fechas_y_cobertura"], "error")

    def test_clave_duplicada_detiene(self):
        df = sintetico()
        self.assertEqual(estados(pd.concat([df, df.iloc[[5]]]))["unicidad"], "error")

    def test_faltante_indispensable_detiene_y_opcional_advierte(self):
        df = sintetico()
        df.loc[3, "País"] = None
        self.assertEqual(estados(df)["valores_faltantes"], "error")
        df = sintetico()
        df.loc[: len(df) // 5, "Mes"] = None
        self.assertEqual(estados(df)["valores_faltantes"], "advertencia")

    def test_conteo_negativo_o_infinito_detiene(self):
        for bad in (-1.0, np.inf):
            df = sintetico()
            df.loc[2, "Viajero"] = bad
            self.assertEqual(estados(df)["valores_y_denominadores"], "error")

    def test_denominador_cero_detiene(self):
        df = sintetico()
        df.loc[df["Año"] >= 2025, "Viajero"] = 0.0
        self.assertEqual(estados(df)["valores_y_denominadores"], "error")

    def test_regla_metodologica(self):
        self.assertEqual(estados(sintetico(), sheet="Diario")["regla_metodologica"], "error")
        self.assertEqual(estados(sintetico().assign(Fecha="2024-01-01"))["regla_metodologica"], "error")
        # Incluir residentes mezclaría meses con y sin su cobertura (cambio de 2023).
        self.assertEqual(estados(sintetico(), CFG | {"excluir_residentes": False})["regla_metodologica"], "error")


class Procedimiento(unittest.TestCase):
    def setUp(self):
        self.y = pl.serie_objetivo(sintetico(), CFG)

    def test_evaluacion_no_usa_informacion_futura(self):
        ev = pl.evaluar(self.y, CFG)
        self.assertTrue((pd.PeriodIndex(ev["ultimo_mes_entrenamiento"], freq="M") <
                         pd.PeriodIndex(ev["periodo"], freq="M")).all())
        self.assertEqual(len(ev), 12)

    def test_cambiar_un_mes_futuro_no_altera_su_estimacion(self):
        y2 = self.y.copy()
        y2[pd.Period("2025-08", "M")] *= 10          # mes oculto en el origen 2025-06
        a, b = pl.evaluar(self.y, CFG), pl.evaluar(y2, CFG)
        fila = a["periodo"] == "2025-08"
        self.assertEqual(a.loc[fila, "procedimiento"].item(), b.loc[fila, "procedimiento"].item())

    def test_ingenuo_estacional(self):
        pred, params, fit = pl.estimar(self.y, CFG, 6)
        self.assertEqual(list(pred), [float(self.y[pd.Period("2025-07", "M") + i]) for i in range(6)])
        self.assertIsNone(fit)

    def test_mape_excluye_ceros(self):
        m = pl.metricas(np.array([0.0, 100.0]), np.array([10.0, 110.0]))
        self.assertEqual(m["meses_mape_excluidos"], 1)
        self.assertAlmostEqual(m["mape"], 10.0)


class Publicacion(unittest.TestCase):
    def test_publicacion_atomica_y_lectura(self):
        with tempfile.TemporaryDirectory() as tmp:
            pub = Path(tmp)
            self.assertIsNone(pl.leer_vigente(pub))
            est = pd.DataFrame({c: ["x"] for c in pl.EST_COLS}).assign(valor=[1.0], periodo_estimado=["2026-07"])
            pl.publicar(pub, "r1", est, pd.DataFrame({"periodo": ["2026-06"]}), {"calidad": {}})
            self.assertEqual(pl.leer_vigente(pub)["info"]["run_id"], "r1")
            self.assertEqual([d.name for d in (pub / "versiones").iterdir()], ["r1"])

    def test_integridad_detecta_negativos(self):
        periods = [str(pd.Period("2026-06", "M") + i) for i in range(1, 7)]
        est = pd.DataFrame({"destino": "G", "periodo_estimado": periods, "valor": [1, 2, 3, 4, 5, -1.0], "unidad": "u",
                            "tipo": "estimación", "run_id": "r", "fecha_generacion_utc": "f",
                            "version_procedimiento": "v", "metodo": "m"})
        self.assertIn("valores negativos o no finitos", pl.integridad(est, CFG, "r"))

    def test_monitoreo_alerta_por_variacion(self):
        y = pl.serie_objetivo(sintetico(), CFG)
        periods = [str(pd.Period("2026-06", "M") + i) for i in range(1, 7)]
        est = pd.DataFrame({"periodo_estimado": periods, "valor": [float(y[pd.Period(p, "M") - 12]) * 2 for p in periods]})
        mon = {m["verificacion"]: m["estado"] for m in
               pl.monitorear(CFG, y, est, {}, pd.Period("2026-06", "M"), None)}
        self.assertEqual(mon, {"actualizacion": "ok", "calidad": "ok", "resultados": "alerta"})


if __name__ == "__main__":
    unittest.main()
