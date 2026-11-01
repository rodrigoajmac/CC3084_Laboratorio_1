"""Consulta las ejecuciones del experimento en MLflow y las exporta a lab11/evidencia/ejecuciones_mlflow.csv.

Uso:  python lab11/consultar_mlflow.py
"""

import json
import os
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent
COLS = {"run_id": "run_id", "start_time": "inicio", "status": "estado_mlflow", "tags.estado": "estado",
        "tags.decision": "decision", "tags.etapa_fallida": "etapa_fallida", "tags.error": "error",
        "tags.git_commit": "git_commit", "tags.version_procedimiento": "version_procedimiento",
        "params.metodo": "metodo", "tags.fuente_archivo": "fuente", "tags.fuente_sha256": "fuente_sha256",
        "tags.controles_calidad": "controles", "metrics.proc_mae": "mae", "metrics.proc_wape": "wape",
        "metrics.ref_mae": "mae_referencia", "metrics.ratio_mae_proc_ref": "ratio_mae",
        "metrics.dif_rel_max_vs_vigente": "dif_rel_vs_vigente"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow_tracking_uri"]))
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow_experiment"]], order_by=["start_time ASC"])
    tabla = runs.reindex(columns=list(COLS)).rename(columns=COLS)
    (ROOT / "evidencia").mkdir(exist_ok=True)
    tabla.to_csv(ROOT / "evidencia" / "ejecuciones_mlflow.csv", index=False)
    print(tabla[["run_id", "estado", "metodo", "mae", "wape", "etapa_fallida"]].to_string(index=False))


if __name__ == "__main__":
    main()
