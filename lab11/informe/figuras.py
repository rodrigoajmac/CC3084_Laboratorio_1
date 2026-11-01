"""Figuras del informe a partir de los artefactos registrados en MLflow y de la versión vigente.

Uso:  python lab11/informe/figuras.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
ART = LAB / "mlflow_data" / "artifacts" / "1"
HW, NAIVE = "f169bf4b71154f5eabe1055410b83948", "e528864e11e44070bb8f440901a60084"
AZUL, NARANJA, VIOLETA, GRIS = "#1F4E79", "#D9661F", "#6A4C93", "#8A949E"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10})


def periodos(s):
    return pd.PeriodIndex(s, freq="M").to_timestamp()


def main():
    vig = json.loads((LAB / "publicado" / "vigente.json").read_text(encoding="utf-8"))
    vdir = LAB / "publicado" / vig["version_dir"]
    obs = pd.read_csv(vdir / "observados.csv")
    est = pd.read_csv(vdir / "estimaciones.csv")
    ev_n = pd.read_csv(ART / NAIVE / "artifacts" / "resultados" / "evaluacion.csv")
    ev_h = pd.read_csv(ART / HW / "artifacts" / "resultados" / "evaluacion.csv")

    o = obs[obs["periodo"] >= "2023-07"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(periodos(o["periodo"]), o["valor"] / 1e3, color=AZUL, lw=1.8, marker="o", ms=3, label="Observado")
    for origen in ev_n["origen"].unique():
        n, h = ev_n[ev_n["origen"] == origen], ev_h[ev_h["origen"] == origen]
        ax.plot(periodos(h["periodo"]), h["procedimiento"] / 1e3, color=VIOLETA, ls=":", lw=1.6, marker="s", ms=3,
                label="Holt-Winters 1.0 (rechazado)" if origen == "2025-06" else None)
        ax.plot(periodos(n["periodo"]), n["procedimiento"] / 1e3, color=NARANJA, ls="--", lw=1.6, marker="D", ms=3,
                label="Ingenuo estacional 2.0 (aprobado)" if origen == "2025-06" else None)
        ax.axvline(pd.Period(origen, "M").to_timestamp(), color=GRIS, ls="--", lw=0.8)
    ax.set_ylabel("miles de visitantes por mes")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(LAB / "informe" / "validacion.png", dpi=200)

    o = obs[obs["periodo"] >= "2023-07"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(periodos(o["periodo"]), o["valor"] / 1e3, color=AZUL, lw=1.8, marker="o", ms=3, label="Observado")
    x = list(periodos(o["periodo"].tail(1))) + list(periodos(est["periodo_estimado"]))
    y = [o["valor"].iloc[-1] / 1e3] + list(est["valor"] / 1e3)
    ax.plot(x, y, color=NARANJA, ls="--", lw=1.6, marker="D", ms=4, markevery=list(range(1, len(x))),
            label=f"Estimación vigente ({vig['run_id'][:8]})")
    ax.set_ylabel("miles de visitantes por mes")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(LAB / "informe" / "estimaciones.png", dpi=200)


if __name__ == "__main__":
    main()
