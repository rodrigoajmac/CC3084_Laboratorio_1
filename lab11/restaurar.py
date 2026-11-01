"""Lista las versiones aprobadas o restaura una de ellas como la versión vigente del tablero.

Uso:  python lab11/restaurar.py --listar
      python lab11/restaurar.py RUN_ID
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def escribir_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def versiones(pub):
    out = []
    for d in sorted((pub / "versiones").glob("*")):
        if d.name.startswith(".") or not (d / "metadatos.json").is_file():
            continue                     # carpetas temporales de una publicación interrumpida
        meta = json.loads((d / "metadatos.json").read_text(encoding="utf-8"))
        out.append(meta)
    return sorted(out, key=lambda m: m["fecha_generacion_utc"])


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--listar", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent
    pub, logs = base / cfg["published_dir"], base / cfg["log_dir"]
    pointer = pub / "vigente.json"
    vigente = json.loads(pointer.read_text(encoding="utf-8"))["run_id"] if pointer.is_file() else None

    if args.listar or not args.run_id:
        for m in versiones(pub):
            marca = "*" if m["run_id"] == vigente else " "
            print(f"{marca} {m['run_id']}  {m['fecha_generacion_utc']}  {m['metodo']:<24} "
                  f"WAPE {m['metricas_procedimiento']['wape']:.1%}  fuente {m['fuente_archivo']}")
        print("(* = vigente)" if vigente else "No hay versión vigente.")
        return 0

    meta = next((m for m in versiones(pub) if m["run_id"] == args.run_id), None)
    if meta is None or not all(meta["reglas_aceptacion"].values()):
        print(f"{args.run_id} no es una versión aprobada disponible en {pub / 'versiones'}", file=sys.stderr)
        return 1
    escribir_json(pub / "vigente.json.tmp", {"run_id": args.run_id, "version_dir": f"versiones/{args.run_id}",
                                             "publicado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                             "motivo": f"restaurada manualmente; reemplaza a {vigente}"})
    os.replace(pub / "vigente.json.tmp", pointer)
    logs.mkdir(parents=True, exist_ok=True)
    escribir_json(logs / f"restauracion_{args.run_id}.json",
                  {"accion": "restauracion", "vigente_antes": vigente, "vigente_despues": args.run_id,
                   "fecha_utc": json.loads(pointer.read_text(encoding="utf-8"))["publicado_utc"]})
    print(f"Vigente: {args.run_id} (antes: {vigente})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
