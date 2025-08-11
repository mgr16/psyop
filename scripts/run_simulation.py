#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_simulation.py – Envoltura para ejecutar simulaciones con perfiles.
Ubicación oficial: scripts/run_simulation.py
"""

import argparse, json, os, sys, tempfile
from pathlib import Path

DEFAULT_CONFIG = "config_example.json"

def load_cfg(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        project_root = Path(__file__).resolve().parent.parent
        alt = project_root / path
        if alt.exists():
            p = alt
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cfg(cfg: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def apply_overrides(cfg: dict, mode: str) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg.setdefault("solver", {})
    cfg.setdefault("evolution", {})
    cfg.setdefault("output", {})
    cfg["solver"].setdefault("sommerfeld", True)

    if mode == "quick":
        cfg["mesh"]["lc"] = max(2.0, float(cfg["mesh"].get("lc", 1.0)) * 2.0)
        cfg["evolution"]["t_end"] = min(5.0, float(cfg["evolution"].get("t_end", 10.0)))
        cfg["evolution"]["output_every"] = max(2, int(cfg["evolution"].get("output_every", 10)//2))
        cfg["output"]["qnm_analysis"] = False
        cfg["output"]["save_series"] = False
        cfg["solver"]["sponge"] = {"enabled": False}

    elif mode == "medium":
        cfg["mesh"]["lc"] = float(cfg["mesh"].get("lc", 1.0)) * 1.5
        cfg["evolution"]["t_end"] = float(cfg["evolution"].get("t_end", 30.0)) * 0.5
        cfg["evolution"]["output_every"] = max(5, int(cfg["evolution"].get("output_every", 10)))
        cfg["solver"].setdefault("sponge", {"enabled": True, "r_start": 0.8*cfg["mesh"]["R"], "strength": 0.1})

    elif mode == "full":
        pass

    else:
        raise ValueError(f"Modo no soportado: {mode}")
    return cfg

def run_main_with_config(config_path: str) -> int:
    # Intentar importar main desde la raíz y simular CLI; si falla, invocar como proceso
    project_root = Path(__file__).resolve().parent.parent
    try:
        sys.path.insert(0, str(project_root))
        import main as main_mod
        # Simular argumentos de línea de comandos
        argv_backup = sys.argv[:]
        try:
            sys.argv = ["main.py", "--config", config_path]
            rc = main_mod.main()
            return int(rc) if isinstance(rc, int) else 0
        except Exception:
            return os.system(f"{sys.executable} {project_root / 'main.py'} --config {config_path}")
        finally:
            sys.argv = argv_backup
    except Exception:
        return os.system(f"{sys.executable} {project_root / 'main.py'} --config {config_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Ruta a config base (JSON)")
    p.add_argument("--mode", choices=["quick","medium","full"], default="quick", help="Perfil de ejecución")
    p.add_argument("--outdir", default=None, help="(Opcional) Directorio base de resultados")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    if args.outdir:
        cfg.setdefault("output", {})["dir"] = args.outdir

    cfg_mode = apply_overrides(cfg, args.mode)

    with tempfile.TemporaryDirectory() as td:
        tmp_cfg = Path(td)/"config.tmp.json"
        save_cfg(cfg_mode, str(tmp_cfg))
        print(f"=== Ejecutando modo {args.mode.upper()} con config: {tmp_cfg} ===")
        rc = run_main_with_config(str(tmp_cfg))
        print(f"=== Finalizado con código {rc} ===")
        return 0 if rc == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
