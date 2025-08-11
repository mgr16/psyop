#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
initial_conditions.py

Condiciones iniciales para campos escalares.
Compatible con FEniCS legacy y DOLFINx.
"""

import numpy as np

# Importaciones condicionales
try:
    import dolfinx.fem as fem
    import ufl
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False
    import fenics as fe

class GaussianBump:
    """
    Condición inicial tipo bump gaussiano para campo escalar.
    φ(r) = v0 * (1 + A * exp(-((r - r0)²)/w²))
    """
    
    def __init__(self, mesh, V, A=1e-3, r0=8.0, w=2.0, v0=1.0):
        """
        Parámetros:
            mesh: Malla del dominio
            V: Espacio de funciones
            A: Amplitud de la perturbación
            r0: Centro radial de la perturbación
            w: Ancho de la perturbación
            v0: Valor de vacío del campo
        """
        self.mesh = mesh
        self.V = V
        self.A = float(A)
        self.r0 = float(r0)
        self.w = float(w)
        self.v0 = float(v0)
        
        if HAS_DOLFINX:
            self.phi = fem.Function(V, name="phi_initial")
            self._set_dolfinx_values()
        else:
            self.phi = fe.Function(V, name="phi_initial")
            self._set_fenics_values()
    
    def _gaussian_expr(self, x):
        """Evaluación de la expresión gaussiana."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        r = np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)
        perturbation = self.A * np.exp(-((r - self.r0)**2) / (self.w**2))
        return self.v0 * (1.0 + perturbation)
    
    def _set_dolfinx_values(self):
        """Configurar valores para DOLFINx."""
        # Obtener coordenadas de los DOFs
        if hasattr(self.V, 'tabulate_dof_coordinates'):
            dof_coords = self.V.tabulate_dof_coordinates()
        else:
            # Fallback para versiones más nuevas
            dof_coords = self.mesh.geometry.x
        
        # Evaluar la función gaussiana
        values = self._gaussian_expr(dof_coords)
        
        # Asignar valores
        self.phi.x.array[:] = values.astype(np.float64)
    
    def _set_fenics_values(self):
        """Configurar valores para FEniCS legacy."""
        class GaussianExpression(fe.UserExpression):
            def __init__(self, A, r0, w, v0, **kwargs):
                super().__init__(**kwargs)
                self.A = A
                self.r0 = r0
                self.w = w
                self.v0 = v0
            
            def eval(self, value, x):
                r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
                perturbation = self.A * np.exp(-((r - self.r0)**2) / (self.w**2))
                value[0] = self.v0 * (1.0 + perturbation)
            
            def value_shape(self):
                return ()
        
        gaussian_expr = GaussianExpression(
            A=self.A, r0=self.r0, w=self.w, v0=self.v0, degree=3
        )
        self.phi.interpolate(gaussian_expr)
    
    def get_function(self):
        """Retorna la función inicializada."""
        return self.phi

class PlaneWave:
    """
    Condición inicial de onda plana para pruebas.
    φ(x) = A * sin(k·x) donde k·x = kx*x + ky*y + kz*z
    """
    
    def __init__(self, mesh, V, A=0.1, k=[1, 0, 0], v0=1.0):
        """
        Parámetros:
            A: Amplitud
            k: Vector de onda [kx, ky, kz]
            v0: Valor base del campo
        """
        self.mesh = mesh
        self.V = V
        self.A = float(A)
        self.k = np.array(k, dtype=float)
        self.v0 = float(v0)
        
        if HAS_DOLFINX:
            self.phi = fem.Function(V, name="phi_wave")
            self._set_dolfinx_wave()
        else:
            self.phi = fe.Function(V, name="phi_wave")
            self._set_fenics_wave()
    
    def _wave_expr(self, x):
        """Evaluación de la onda plana."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # k·x
        k_dot_x = np.dot(x, self.k)
        return self.v0 + self.A * np.sin(k_dot_x)
    
    def _set_dolfinx_wave(self):
        """Configurar onda plana para DOLFINx."""
        if hasattr(self.V, 'tabulate_dof_coordinates'):
            dof_coords = self.V.tabulate_dof_coordinates()
        else:
            dof_coords = self.mesh.geometry.x
        
        values = self._wave_expr(dof_coords)
        self.phi.x.array[:] = values.astype(np.float64)
    
    def _set_fenics_wave(self):
        """Configurar onda plana para FEniCS legacy."""
        class WaveExpression(fe.UserExpression):
            def __init__(self, A, k, v0, **kwargs):
                super().__init__(**kwargs)
                self.A = A
                self.k = k
                self.v0 = v0
            
            def eval(self, value, x):
                k_dot_x = self.k[0]*x[0] + self.k[1]*x[1] + self.k[2]*x[2]
                value[0] = self.v0 + self.A * np.sin(k_dot_x)
            
            def value_shape(self):
                return ()
        
        wave_expr = WaveExpression(A=self.A, k=self.k, v0=self.v0, degree=3)
        self.phi.interpolate(wave_expr)
    
    def get_function(self):
        """Retorna la función inicializada."""
        return self.phi

def create_zero_field(V):
    """Crea un campo escalar con valor cero."""
    if HAS_DOLFINX:
        phi = fem.Function(V, name="zero_field")
        phi.x.array[:] = 0.0
    else:
        phi = fe.Function(V, name="zero_field")
        phi.interpolate(fe.Constant(0.0))
    
    return phi

if __name__ == "__main__":
    # Prueba básica
    print("=== Prueba de initial_conditions.py ===")
    
    try:
        if HAS_DOLFINX:
            from dolfinx.mesh import create_box
            from mpi4py import MPI
            mesh = create_box(MPI.COMM_WORLD, [[-5, -5, -5], [5, 5, 5]], [8, 8, 8])
            V = fem.FunctionSpace(mesh, ("Lagrange", 1))
        else:
            import fenics as fe
            mesh = fe.BoxMesh(fe.Point(-5, -5, -5), fe.Point(5, 5, 5), 8, 8, 8)
            V = fe.FunctionSpace(mesh, 'CG', 1)
        
        # Prueba Gaussian Bump
        gaussian = GaussianBump(mesh, V, A=0.1, r0=2.0, w=1.0)
        phi_gauss = gaussian.get_function()
        print("✓ Gaussian Bump creado exitosamente")
        
        # Prueba Plane Wave
        wave = PlaneWave(mesh, V, A=0.1, k=[1, 1, 0])
        phi_wave = wave.get_function()
        print("✓ Plane Wave creado exitosamente")
        
        # Prueba campo cero
        phi_zero = create_zero_field(V)
        print("✓ Campo cero creado exitosamente")
        
        print("Módulo initial_conditions.py completado exitosamente.")
        
    except Exception as e:
        print(f"✗ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
