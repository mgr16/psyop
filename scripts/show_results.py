#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_results.py – Explorador de resultados y QNM heurístico.
Ubicación oficial: scripts/show_results.py
"""

import os
from pathlib import Path
import numpy as np

try:
    # Importar desde paquete reorganizado si existe
    from psyop.analysis.qnm import compute_qnm  # type: ignore
    from psyop.analysis.qnm import estimate_peak  # type: ignore
    HAS_QNM = True
except Exception:
    try:
        # Compatibilidad con layout antiguo
        from quasinormal_modes import compute_qnm, estimate_peak  # type: ignore
        HAS_QNM = True
    except Exception:
        HAS_QNM = False

DEFAULT_RESULTS_DIR = "results"

def find_latest_run(results_dir: str) -> Path | None:
    base = Path(results_dir)
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def list_files(run_dir: Path):
    print("\n" + "="*60)
    print(f"📂 Resultados en: {run_dir}")
    print("="*60)
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(run_dir)
            size = p.stat().st_size
            print(f" - {rel}  ({size} bytes)")


def _guess_timeseries(run_dir: Path):
    candidates = list(run_dir.glob("**/*.txt")) + list(run_dir.glob("**/*.csv")) + list(run_dir.glob("**/*.npy"))
    for f in candidates:
        try:
            if f.suffix == ".npy":
                arr = np.load(f)
                y = np.asarray(arr).reshape(-1)
                dt = 1.0
                t = np.arange(len(y))*dt
                return t, y
            else:
                data = np.loadtxt(f)
                if data.ndim == 1:
                    y = data.reshape(-1)
                    dt = 1.0
                    t = np.arange(len(y))*dt
                    return t, y
                elif data.shape[1] >= 2:
                    t = data[:,0]
                    y = data[:,1]
                    return t, y
        except Exception:
            continue
    return None


def maybe_qnm(run_dir: Path):
    if not HAS_QNM:
        print("\n(QNM) quasinormal_modes no disponible.")
        return
    guess = _guess_timeseries(run_dir)
    if guess is None:
        print("\n(QNM) No se detectó serie temporal adecuada (t, y).")
        return
    t, y = guess
    if len(t) < 16:
        print("\n(QNM) Serie demasiado corta.")
        return
    dt = float(np.mean(np.diff(t)))
    freqs, spec = compute_qnm(y, dt, window="hann", pad_factor=4)
    # estimate_peak puede no existir en algunas versiones; fallback simple
    try:
        from psyop.analysis.qnm import estimate_peak as _est  # type: ignore
        f_peak, s_peak = _est(freqs, spec)
    except Exception:
        idx = int(np.argmax(np.abs(spec)))
        f_peak, s_peak = float(freqs[idx]), float(spec[idx])
    print("\n" + "-"*60)
    print(f"(QNM) dt≈{dt:.4g}, N={len(y)}, pico ~ {f_peak:.6g} (ampl={s_peak:.3g})")
    print("-"*60)


def main():
    results_dir = os.environ.get("RESULTS_DIR", DEFAULT_RESULTS_DIR)
    run = find_latest_run(results_dir)
    if run is None:
        print(f"⚠️  No se encontraron runs en '{results_dir}'. Ejecuta tu simulación primero.")
        return 1
    list_files(run)
    maybe_qnm(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
