#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Punto de entrada principal del proyecto PSYOP.
# Usa el paquete interno `psyop` y deja la raíz limpia.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json, os, time, argparse
import numpy as np

# Importaciones condicionales de frameworks
HAS_DOLFINX = False
HAS_FENICS = False
try:
    import dolfinx.fem as fem
    import dolfinx.io
    import ufl
    from mpi4py import MPI
    HAS_DOLFINX = True
    print("✓ DOLFINx disponible")
except Exception:
    pass
try:
    import fenics as fe
    if not HAS_DOLFINX:
        import ufl  # type: ignore
    HAS_FENICS = True
    print("✓ FEniCS legacy disponible")
except Exception:
    pass


def create_example_config(filename: str = "config_example.json"):
    cfg = {
        "mesh": {"R": 30.0, "lc": 1.5, "mesh_type": "gmsh"},
        "solver": {
            "degree": 1,
            "cfl": 0.3,
            "potential_type": "quadratic",
            "potential_params": {"m_squared": 1.0},
            "ko_eps": 0.0
        },
        "metric": {"type": "flat", "M": 1.0},
        "initial_conditions": {"type": "gaussian", "A": 0.01, "r0": 10.0, "w": 3.0, "v0": 1.0},
        "evolution": {"t_end": 50.0, "output_every": 10, "verbose": True},
        "output": {"dir": "results", "qnm_analysis": True}
    }
    with open(filename, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"✓ Configuración de ejemplo creada: {filename}")


def validate_config(cfg: dict):
    required = ["mesh", "metric", "solver", "initial_conditions", "evolution", "output"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Falta clave en config: {k}")
    if "R" not in cfg["mesh"]:
        raise ValueError("mesh.R requerido")
    if "cfl" not in cfg["solver"]:
        raise ValueError("solver.cfl requerido")


def main():
    parser = argparse.ArgumentParser(description="Simulación de campo escalar con BC Sommerfeld")
    parser.add_argument("--config", type=str, default=None, help="Archivo de configuración JSON")
    parser.add_argument("--output", type=str, default="results", help="Directorio de salida")
    parser.add_argument("--create-config", action="store_true", help="Crear configuración de ejemplo")
    parser.add_argument("--test", action="store_true", help="Test básico sin FEM")
    args = parser.parse_args()

    if args.create_config:
        create_example_config()
        return 0

    # Entorno
    if args.test or not (HAS_DOLFINX or HAS_FENICS):
        print("=== MODO TEST/CONFIGURACIÓN ===")
        print(f"✓ NumPy disponible: {np.__version__}")
        print(f"✓ DOLFINx: {'Sí' if HAS_DOLFINX else 'No'}")
        print(f"✓ FEniCS: {'Sí' if HAS_FENICS else 'No'}")
        if not (HAS_DOLFINX or HAS_FENICS):
            print("\nInstala dolfinx o fenics para ejecutar simulaciones")
        return 0

    # Importar componentes del paquete solo cuando hay entorno FEM
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import make_background
    from psyop.utils.utils import compute_dt_cfl

    # Config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    else:
        with open('config_example.json', 'r') as f:
            cfg = json.load(f)
    if isinstance(cfg, dict) and isinstance(cfg.get("output"), dict):
        out = cfg["output"]
        if "dir" not in out and "results_dir" in out:
            out["dir"] = out["results_dir"]
    validate_config(cfg)

    # Mesh y métrica
    comm = MPI.COMM_WORLD if HAS_DOLFINX else None
    mesh, cell_tags, facet_tags = build_ball_mesh(R=cfg["mesh"]["R"], lc=cfg["mesh"]["lc"], comm=comm)
    bg = make_background(cfg["metric"])
    alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f = bg.build(mesh)
    c_max = getattr(bg, 'max_characteristic_speed', lambda m: 1.0)(mesh)

    # Output
    dt = compute_dt_cfl(mesh, cfl=cfg["solver"]["cfl"], c_max=c_max)
    outdir = os.path.join(cfg.get("output", {}).get("dir", args.output), time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'config.json'), 'w') as g:
        json.dump(cfg, g, indent=2)

    # Solver
    solver = FirstOrderKGSolver(
        mesh=mesh,
        domain_radius=cfg["mesh"]["R"],
        degree=cfg["solver"]["degree"],
        potential_type=cfg["solver"]["potential_type"],
        potential_params=cfg["solver"].get("potential_params", {}),
        cfl_factor=cfg["solver"]["cfl"],
        cfg=cfg
    )
    solver.set_background(alpha=alpha_f, beta=beta_f, gammaInv=gammaInv_f, sqrtg=sqrtg_f, K=K_f)
    if facet_tags is not None:
        outer_tag = get_outer_tag(facet_tags, default=2)
        solver.enable_sommerfeld(facet_tags, outer_tag)

    ic = cfg["initial_conditions"]
    if ic.get("type") == "gaussian":
        phi0 = GaussianBump(mesh=mesh, V=solver.V_scalar if hasattr(solver, 'V_scalar') else None,
                            A=ic["A"], r0=ic["r0"], w=ic["w"], v0=ic["v0"]) 
        solver.set_initial_conditions(phi0.get_function())

    # Evolución básica con salida
    t, step = 0.0, 0
    t_end = cfg["evolution"]["t_end"]
    output_every = cfg["evolution"]["output_every"]

    if HAS_DOLFINX:
        with dolfinx.io.XDMFFile(mesh.comm, os.path.join(outdir, "phi_evolution.xdmf"), "w") as xdmf:
            xdmf.write_mesh(mesh)
            while t < t_end:
                solver.ssp_rk3_step(dt)
                t += dt
                step += 1
                if step % output_every == 0:
                    phi, Pi = solver.get_fields()
                    xdmf.write_function(phi, t)
    else:
        phi_file = fe.File(os.path.join(outdir, "phi_evolution.pvd"))
        while t < t_end:
            solver.ssp_rk3_step(dt)
            t += dt
            step += 1
            if step % output_every == 0:
                phi, Pi = solver.get_fields()
                phi_file << (phi, t)

    print(f"✓ Simulación completada: t_final={t:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
