#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo utils.py

Este módulo proporciona funciones utilitarias para la simulación, tales como:
  - Creación de mallas personalizadas.
  - Guardado de soluciones en formato PVD.
  - Cálculo de normas y energías.
  - Funciones de logging para registrar mensajes y diagnósticos.
  - Lectura de archivos de configuración (por ejemplo, en formato JSON).

Estas funciones ayudan a mantener el código principal limpio y a reutilizar código en otros proyectos.
"""

import fenics as fe
import numpy as np
import os
import json
import datetime

from fem_backend import is_dolfinx

def create_mesh(domain_min=(-20, -20, -20), domain_max=(20, 20, 20), nx=15, ny=15, nz=15):
    """
    Crea una malla BoxMesh para un dominio rectangular definido por los puntos
    mínimo y máximo.

    Parámetros:
        domain_min (tuple): Coordenadas mínimas del dominio, e.g., (-20, -20, -20).
        domain_max (tuple): Coordenadas máximas del dominio, e.g., (20, 20, 20).
        nx, ny, nz (int): Número de divisiones en cada dirección.

    Retorna:
        mesh (Mesh): Malla creada con FEniCS.
    """
    return fe.BoxMesh(fe.Point(*domain_min), fe.Point(*domain_max), nx, ny, nz)

def save_solution(solution, filename, time=None):
    """
    Guarda una solución de FEniCS en un archivo .pvd.

    Parámetros:
        solution (Function): Solución a guardar.
        filename (str): Ruta base del archivo (sin extensión).
        time (float, opcional): Tiempo actual de la solución para etiquetar la salida.
    """
    # Asegurarse de que el directorio de destino existe
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    file = fe.File(f"{filename}.pvd")
    if time is not None:
        # Permite etiquetar la solución con el tiempo
        file << (solution, time)
    else:
        file << solution

def compute_norm(solution, norm_type='L2'):
    """
    Calcula la norma de una solución de FEniCS.

    Parámetros:
        solution (Function): Función de FEniCS a evaluar.
        norm_type (str): Tipo de norma a calcular ('L2' o 'H1').

    Retorna:
        norm_value (float): Valor numérico de la norma.
    """
    if norm_type == 'L2':
        return fe.norm(solution, 'L2')
    elif norm_type == 'H1':
        return fe.norm(solution, 'H1')
    else:
        raise ValueError("norm_type debe ser 'L2' o 'H1'.")

def compute_energy(phi, Vh, potential_params):
    """
    Calcula una estimación de la energía del campo escalar, considerando un
    término cinético y un potencial dado.

    Se utiliza una forma de energía del tipo:
        E = ∫ [ 0.5 |∇φ|² + V(φ) ] dx
    donde el potencial se puede definir, por ejemplo, como:
        V(φ) = lam*(φ**4/4 - v0²*φ**2/2)
    
    Parámetros:
        phi (Function): Campo escalar.
        Vh (FunctionSpace): Espacio de funciones asociado a phi.
        potential_params (dict): Diccionario que debe contener:
            - 'lam': coeficiente del potencial.
            - 'v0' : valor base del campo.
    
    Retorna:
        energy (float): Energía total calculada en el dominio.
    """
    lam = potential_params.get('lam', 0.1)
    v0  = potential_params.get('v0', 1.0)
    # Definir un potencial de ejemplo
    V = lam * (phi**4 / 4.0 - v0**2 * phi**2 / 2.0)
    grad_phi = fe.grad(phi)
    energy = fe.assemble(0.5 * fe.dot(grad_phi, grad_phi) * fe.dx + V * fe.dx)
    return energy

def log_message(message, logfile="simulation.log"):
    """
    Escribe un mensaje de log con timestamp en un archivo de log y lo muestra en consola.

    Parámetros:
        message (str): Mensaje a registrar.
        logfile (str): Ruta del archivo de log (por defecto "simulation.log").
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    # Escribir en el archivo de log
    with open(logfile, "a") as f:
        f.write(full_message + "\n")
    # Imprimir el mensaje en consola
    print(full_message)

def load_config(filename):
    """
    Carga parámetros de configuración desde un archivo JSON.

    Parámetros:
        filename (str): Ruta del archivo JSON de configuración.

    Retorna:
        config (dict): Diccionario con los parámetros de configuración.
    """
    with open(filename, "r") as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    # Bloque de pruebas para verificar el funcionamiento de utils.py

    # Prueba de creación de malla
    mesh = create_mesh()
    print("Malla creada:", mesh)
    
    # Crear un espacio de funciones y una función de prueba
    V = fe.FunctionSpace(mesh, 'CG', 1)
    u = fe.Function(V)
    u.interpolate(fe.Constant(1.0))
    print("Norma L2 de u:", compute_norm(u, norm_type='L2'))
    
    # Calcular energía usando parámetros de ejemplo
    energy = compute_energy(u, V, {'lam': 0.1, 'v0': 1.0})
    print("Energía estimada:", energy)
    
    # Registrar un mensaje de log
    log_message("Prueba de utils.py completada.")
    
    # Guardar una configuración de ejemplo (opcional)
    config_example = {"param1": 1, "param2": 2}
    config_filename = "example_config.json"
    with open(config_filename, "w") as f:
        json.dump(config_example, f, indent=4)
    log_message(f"Archivo de configuración '{config_filename}' guardado.")

def _approx_hmin(mesh, sample: int = 4000) -> float:
    """
    Estima h_min muestreando longitudes de aristas locales. Suficiente para CFL.
    Compatible con FEniCS legacy y DOLFINx.
    """
    try:
        if is_dolfinx() and hasattr(mesh, 'topology'):
            # DOLFINx - método robusto por aristas
            topo = mesh.topology
            topo.create_connectivity(1, 0)  # edges->vertices
            e2v = topo.connectivity(1, 0)
            X = mesh.geometry.x
            h_min = np.inf
            
            # Muestra al azar algunas aristas (o todas si son pocas)
            edges = np.arange(topo.index_map(1).size_local, dtype=np.int32)
            if edges.size > sample:
                rng = np.random.default_rng(12345)
                edges = rng.choice(edges, size=sample, replace=False)
                
            for e in edges:
                vs = e2v.links(e)
                if len(vs) >= 2:
                    h = np.linalg.norm(X[vs[1]] - X[vs[0]])
                    if h < h_min and h > 0:
                        h_min = h
            return float(h_min) if h_min != np.inf else 1.0
            
        else:
            # FEniCS legacy - estimación mejorada
            if hasattr(mesh, 'coordinates'):
                coords = mesh.coordinates()
            elif hasattr(mesh, 'geometry'):
                coords = mesh.geometry.x
            else:
                return 1.0  # fallback seguro
                
            # Calcular estimación basada en volumen y número de celdas
            bbox = np.max(coords, axis=0) - np.min(coords, axis=0)
            volume = np.prod(bbox)
            
            if hasattr(mesh, 'num_cells'):
                num_cells = mesh.num_cells()
            else:
                num_cells = 1000  # estimación conservadora
                
            # h ≈ (volume/num_cells)^(1/dim)
            dim = len(bbox)
            h_est = (volume / max(num_cells, 1))**(1.0/dim)
            return float(max(h_est, 1e-6))  # evitar h muy pequeño
            
    except Exception:
        # Fallback ultra-seguro
        return 1.0

def compute_dt_cfl(mesh, cfl=0.3, c_max=1.0):
    """
    dt = cfl * h_min / c_max
    - DOLFINx: estimación rápida de h_min por bounding box local
    - FEniCS: usa hmin(mesh)
    """
    if is_dolfinx():
        x = mesh.geometry.x
        dofmap = mesh.geometry.dofmap.array.reshape(-1, mesh.geometry.cmap.dim)
        take = min(dofmap.shape[0], 2000)
        ids = np.linspace(0, dofmap.shape[0] - 1, num=take, dtype=np.int32)
        hs = []
        for cid in ids:
            pts = x[dofmap[cid]]
            h = np.linalg.norm(pts.max(0) - pts.min(0))
            hs.append(h)
        hmin = float(np.min(hs)) if hs else 1.0
    else:
        import fenics as fe
        # FEniCS legacy: usar fe.hmin si existe; si no, usar mesh.hmin()
        try:
            hmin_local = float(getattr(fe, "hmin")(mesh))
        except Exception:
            try:
                hmin_local = float(mesh.hmin())
            except Exception:
                # Fallback por bounding box
                if hasattr(mesh, "coordinates"):
                    coords = mesh.coordinates()
                    bbox = coords.max(0) - coords.min(0)
                    hmin_local = float(np.min(bbox) / 10.0) if coords.size > 0 else 1.0
                else:
                    hmin_local = 1.0
        # Intentar reducción MPI si disponible
        try:
            comm = mesh.mpi_comm()
            hmin = fe.MPI.min(comm, hmin_local)
        except Exception:
            hmin = hmin_local
    return float(cfl) * float(hmin) / max(float(c_max), 1e-12)
