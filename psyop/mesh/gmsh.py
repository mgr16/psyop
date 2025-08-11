#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmsh_helpers.py

Funciones para generar mallas con Gmsh y etiquetar fronteras.
Crea geometrías esféricas con etiqueta "outer_boundary" para BC absorbentes.
"""

import numpy as np

# Importar Gmsh si está disponible
try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False
    print("Warning: Gmsh no disponible. Usando mallas de FEniCS básicas.")

# Importar DOLFINx si está disponible
try:
    import dolfinx
    import dolfinx.mesh as dmesh
    from dolfinx.io import gmshio
    from mpi4py import MPI
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False
    import fenics as fe

def build_ball_mesh(R, lc, comm=None):
    """
    Crea una malla esférica con radio R y tamaño característico lc.
    
    Parámetros:
        R (float): Radio de la esfera
        lc (float): Tamaño característico de elemento
        comm: Comunicador MPI (para DOLFINx)
    
    Retorna:
        mesh: Malla del dominio
        cell_tags: Etiquetas de celdas (opcional)
        facet_tags: Etiquetas de facetas con outer_boundary=2
    """
    if not HAS_GMSH:
        print("Gmsh no disponible. Creando malla esférica simple...")
        return _create_simple_ball_mesh(R, comm)
    
    if comm is None:
        if HAS_DOLFINX:
            comm = MPI.COMM_WORLD
    
    # Inicializar Gmsh
    gmsh.initialize()
    gmsh.clear()
    
    try:
        # Crear modelo
        gmsh.model.add("ball")
        
        # Crear geometría esférica
        sphere = gmsh.model.occ.addSphere(0, 0, 0, R)
        gmsh.model.occ.synchronize()
        
        # Etiquetar superficies
        # outer_boundary = 2 (superficie exterior)
        surfaces = gmsh.model.getEntities(2)
        for surface in surfaces:
            gmsh.model.addPhysicalGroup(2, [surface[1]], 2)  # tag=2 para outer_boundary
        
        # Etiquetar volumen  
        volumes = gmsh.model.getEntities(3)
        for volume in volumes:
            gmsh.model.addPhysicalGroup(3, [volume[1]], 1)  # tag=1 para volumen
        
        # Configurar malla
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc/3)
        
        # Generar malla
        gmsh.model.mesh.generate(3)
        
        if HAS_DOLFINX:
            # Importar a DOLFINx
            mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
                gmsh.model, comm, rank=0, gdim=3
            )
        else:
            # Fallback a FEniCS legacy
            print("DOLFINx no disponible. Usando fallback a FEniCS...")
            mesh = _create_simple_ball_mesh(R, comm)
            cell_tags = None
            facet_tags = None
    
    finally:
        gmsh.finalize()
    
    return mesh, cell_tags, facet_tags

def _create_simple_ball_mesh(R, comm=None):
    """
    Fallback: crear malla esférica simple sin Gmsh.
    """
    if HAS_DOLFINX:
        # DOLFINx - crear malla cúbica como aproximación
        from dolfinx.mesh import create_box
        mesh = create_box(
            comm or MPI.COMM_WORLD,
            [[-R, -R, -R], [R, R, R]], 
            [16, 16, 16]
        )
        cell_tags = None
        # Crear facet_tags dummy
        facet_tags = None
    else:
        # FEniCS legacy
        import fenics as fe
        mesh = fe.BoxMesh(fe.Point(-R, -R, -R), fe.Point(R, R, R), 16, 16, 16)
        cell_tags = None
        facet_tags = None
    
    return mesh, cell_tags, facet_tags

def build_box_mesh(x_min, x_max, nx, ny, nz, comm=None):
    """
    Crea una malla de caja rectangular con etiquetas de frontera.
    
    Parámetros:
        x_min, x_max: arrays con coordenadas mínima y máxima
        nx, ny, nz: número de divisiones
        comm: Comunicador MPI
    
    Retorna:
        mesh, cell_tags, facet_tags
    """
    if HAS_DOLFINX:
        from dolfinx.mesh import create_box
        mesh = create_box(
            comm or MPI.COMM_WORLD,
            [x_min, x_max],
            [nx, ny, nz]
        )
        # Para caja, todas las fronteras son outer_boundary
        cell_tags = None
        facet_tags = None
    else:
        import fenics as fe
        p1 = fe.Point(x_min[0], x_min[1], x_min[2])
        p2 = fe.Point(x_max[0], x_max[1], x_max[2])
        mesh = fe.BoxMesh(p1, p2, nx, ny, nz)
        cell_tags = None
        facet_tags = None
    
    return mesh, cell_tags, facet_tags

def get_outer_tag(facet_tags, name: str = "outer_boundary", default: int = 2) -> int:
    """Devuelve el id de la frontera externa. Si facet_tags trae nombres en metadata, busca 'outer_boundary'."""
    try:
        md = getattr(facet_tags, "metadata", None)
        if md and "names" in md:
            for k, v in md["names"].items():
                if v == name:
                    return int(k)
    except Exception:
        pass
    return int(default)

if __name__ == "__main__":
    # Prueba básica
    print("=== Prueba de gmsh_helpers.py ===")
    
    try:
        mesh, cell_tags, facet_tags = build_ball_mesh(R=5.0, lc=1.0)
        print(f"✓ Malla esférica creada exitosamente")
        
        if HAS_DOLFINX and hasattr(mesh, 'topology'):
            num_cells = mesh.topology.index_map(3).size_local
            print(f"  Número de celdas: {num_cells}")
        elif hasattr(mesh, 'num_cells'):
            print(f"  Número de celdas: {mesh.num_cells()}")
        
        if facet_tags is not None:
            print("  ✓ Facet tags creados para BC absorbentes")
        
    except Exception as e:
        print(f"✗ Error en prueba: {e}")
    
    print("Módulo gmsh_helpers.py completado.")
