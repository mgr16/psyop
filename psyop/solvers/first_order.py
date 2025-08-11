#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solver_first_order.py

Solver de primer orden para la ecuación de Klein-Gordon usando formulación (φ, Π).
Implementa SSP-RK3 con condiciones de frontera Sommerfeld.
Compatible con FEniCS legacy y DOLFINx.
"""

import numpy as np
import time

# Importaciones condicionales
try:
    import dolfinx.fem as fem
    import dolfinx.fem.petsc
    import ufl
    from petsc4py import PETSc
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False
    try:
        import fenics as fe
        import ufl
    except ImportError:
        raise ImportError("Se requiere FEniCS legacy o DOLFINx para ejecutar el solver")

# Importar módulos del proyecto
try:
    from .utils import compute_dt_cfl
    from .potential import get_potential
    from .initial_conditions import GaussianBump, create_zero_field
except ImportError:
    # Importación absoluta como fallback
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import compute_dt_cfl
    from potential import get_potential
    from initial_conditions import GaussianBump, create_zero_field

class FirstOrderKGSolver:
    """
    Solver de primer orden para Klein-Gordon usando formulación (φ, Π).
    
    Sistema a resolver:
    ∂φ/∂t = Π
    ∂Π/∂t = ∇²φ - V'(φ)
    
    Con condiciones de frontera Sommerfeld: ∂φ/∂n + (1/r)φ = 0
    """
    
    def __init__(self, mesh, degree=1, potential_type="higgs", potential_params=None,
                 cfl_factor=0.5, domain_radius=10.0, **kwargs):
        """
        Inicializa el solver.
        
        Args:
            mesh: Malla del dominio
            degree: Grado de los elementos finitos
            potential_type: Tipo de potencial a usar
            potential_params: Parámetros del potencial
            cfl_factor: Factor CFL para time stepping adaptativo
            domain_radius: Radio del dominio (para BC Sommerfeld)
        """
        self.mesh = mesh
        self.degree = degree
        self.cfl_factor = cfl_factor
        self.domain_radius = domain_radius
        
        # Configurar espacios de funciones
        self._setup_function_spaces()
        
        # Configurar potencial
        if potential_params is None:
            potential_params = {}
        self.potential = get_potential(potential_type, **potential_params)
        
        # Configurar formas variacionales
        self._setup_variational_forms()
        
        # Configurar condiciones de frontera Sommerfeld
        self._setup_sommerfeld_bc()
        
        # Configurar solver de matriz de masa
        self._setup_mass_matrix_solver()
        
        # Variables de estado
        self.current_time = 0.0
        self.current_dt = None
        # Registrar config completa si fue pasada (para flags como solver.sommerfeld)
        self.cfg = kwargs.get("cfg", {})
        # Flags de BC y métricas
        self.has_sommerfeld = False
        self.alpha_f = None
        self.beta_f  = None
        self.gammaInv_f = None
        self.sqrtg_f   = None
        self.K_f       = None
        # Función derivada temporal (du/dt)
        if HAS_DOLFINX:
            self.du = fem.Function(self.V, name="du")
        else:
            self.du = fe.Function(self.V, name="du")
        print(f"✓ FirstOrderKGSolver inicializado (grado={degree}, CFL={cfl_factor})")
    
    def _setup_function_spaces(self):
        """Configura los espacios de funciones."""
        if HAS_DOLFINX:
            # Espacio vectorial para (φ, Π)
            element = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), self.degree, dim=2)
            self.V = fem.FunctionSpace(self.mesh, element)
            
            # Espacio escalar auxiliar
            self.V_scalar = fem.FunctionSpace(self.mesh, ("Lagrange", self.degree))
            
            # Funciones de prueba
            v = ufl.TestFunction(self.V)
            self.test_phi, self.test_Pi = ufl.split(v)
            
            # Funciones de estado
            self.u = fem.Function(self.V, name="u")
            self.u_new = fem.Function(self.V, name="u_new")
            self.u1 = fem.Function(self.V, name="u1")
            self.u2 = fem.Function(self.V, name="u2")
            
            # Componentes actuales (dependen de self.u)
            self.phi_c, self.Pi_c = ufl.split(self.u)
            
        else:
            # Espacios escalares separados para φ y Π
            self.V_scalar = fe.FunctionSpace(self.mesh, 'CG', self.degree)
            
            # Elemento vectorial para (φ, Π)
            element = fe.VectorElement('CG', self.mesh.ufl_cell(), self.degree, dim=2)
            self.V = fe.FunctionSpace(self.mesh, element)
            
            # Funciones de prueba y trial
            u = fe.TrialFunction(self.V)
            v = fe.TestFunction(self.V)
            
            # Extraer componentes
            self.phi, self.Pi = fe.split(u)
            self.test_phi, self.test_Pi = fe.split(v)
            
            # Función de solución
            self.u = fe.Function(self.V, name="u")
            self.u_new = fe.Function(self.V, name="u_new")
            
            # Funciones auxiliares para SSP-RK3
            self.u1 = fe.Function(self.V, name="u1")
            self.u2 = fe.Function(self.V, name="u2")
    
    def _setup_variational_forms(self):
        """Configura las formas variacionales con métrica."""
        # Forma de masa con √γ
        if HAS_DOLFINX:
            dx = ufl.Measure("dx", domain=self.mesh)
        else:
            dx = ufl.dx
        
        # Sistema de ecuaciones con métrica:
        # M * du/dt = F(u)
        # donde u = [φ, Π] y F(u) = [αΠ + β·∇φ, α∇·(γ⁻¹∇φ) + αKΠ - αV'(φ)]
        
        # Matriz de masa ponderada: ∫ √γ (test_φ * φ + test_Pi * Pi) dx
        u_trial = ufl.TrialFunction(self.V)
        v_test = ufl.TestFunction(self.V)
        phi_trial, Pi_trial = ufl.split(u_trial)
        test_phi_mass, test_Pi_mass = ufl.split(v_test)
        
        sqrtg = self._SQRTG()
        self.mass_form = (test_phi_mass * phi_trial + test_Pi_mass * Pi_trial) * sqrtg * dx
        
        # Forma del lado derecho (RHS) con métrica
        self.rhs_form = self._rhs_phi_form() + self._rhs_Pi_form()
    
    def _rhs_phi_form(self):
        """RHS para φ: ∂φ/∂t = αΠ + β·∇φ"""
        if HAS_DOLFINX:
            dx = ufl.Measure("dx", domain=self.mesh)
        else:
            dx = ufl.dx
            
        alpha = self._ALPHA()
        beta  = self._BETA()
        sqrtg = self._SQRTG()

        term = (alpha * self.Pi_c) * self.test_phi * sqrtg * dx
        if beta is not None:
            term += ufl.dot(beta, ufl.grad(self.phi_c)) * self.test_phi * sqrtg * dx
        return term

    def _rhs_Pi_form(self):
        """RHS para Π: ∂Π/∂t = α∇·(γ⁻¹∇φ) + αKΠ - αV'(φ)"""
        if HAS_DOLFINX:
            dx = ufl.Measure("dx", domain=self.mesh)
        else:
            dx = ufl.dx
            
        alpha = self._ALPHA()
        beta  = self._BETA()
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()
        K = self._K()

        # Término de potencial
        Vp = self.potential.derivative(self.phi_c)

        # Difusión: - ∫ √γ * α * (γ^{ij} ∂_i φ ∂_j test_Pi) dx
        diffusion = - alpha * ufl.inner(ufl.dot(gammaInv, ufl.grad(self.phi_c)), ufl.grad(self.test_Pi)) * sqrtg * dx

        # Transporte + curvatura extrínseca + potencial
        transport = (alpha*K*self.Pi_c - alpha*Vp) * self.test_Pi * sqrtg * dx
        if beta is not None:
            transport += ufl.dot(beta, ufl.grad(self.Pi_c)) * self.test_Pi * sqrtg * dx

        # Aporte de borde de Sommerfeld (si está habilitado)
        boundary_term = self._sommerfeld_boundary_term()

        return diffusion + transport + boundary_term
    
    def _setup_sommerfeld_bc(self):
        """
        Configura condiciones de frontera Sommerfeld.
        En la frontera: ∂φ/∂n + (1/r)φ = 0
        """
        try:
            if HAS_DOLFINX:
                self._setup_sommerfeld_dolfinx()
            else:
                self._setup_sommerfeld_fenics()
            self.has_sommerfeld = True
            print("✓ Condiciones de frontera Sommerfeld configuradas")
        except Exception as e:
            print(f"⚠ Error configurando Sommerfeld BC: {e}")
            self.has_sommerfeld = False
    
    def _setup_sommerfeld_dolfinx(self):
        """Configura Sommerfeld BC para DOLFINx."""
        # Identificar facetas de frontera externa
        mesh = self.mesh
        facet_dim = mesh.topology.dim - 1
        
        # Crear función para identificar frontera externa (etiqueta 2)
        def boundary_marker(x):
            # Asume que la frontera externa está marcada con tag=2
            return np.isclose(np.linalg.norm(x, axis=0), self.domain_radius, atol=0.1)
        
        # Obtener facetas de frontera
        boundary_facets = fem.locate_entities_boundary(mesh, facet_dim, boundary_marker)
        
        # Almacenar información de frontera
        self.boundary_facets = boundary_facets
        
        # Crear medida de integración en la frontera
        facet_tags = np.zeros(mesh.topology.connectivity(facet_dim, 0).num_nodes, dtype=np.int32)
        facet_tags[boundary_facets] = 2  # Tag para frontera externa
        
        # Crear MeshTags
        self.facet_tags = fem.meshtags(mesh, facet_dim, boundary_facets, facet_tags[boundary_facets])
        self.ds_outer = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags, subdomain_id=2)
    
    def _setup_sommerfeld_fenics(self):
        """Configura Sommerfeld BC para FEniCS legacy."""
        # Para FEniCS legacy, la frontera externa debería estar marcada
        # Si el mesh viene de gmsh_helpers.py, la frontera externa tiene tag=2
        mesh = self.mesh
        
        # Crear SubDomain para frontera externa
        class OuterBoundary(fe.SubDomain):
            def __init__(self, radius, tol=1e-1):
                super().__init__()
                self.radius = radius
                self.tol = tol
            
            def inside(self, x, on_boundary):
                r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
                return on_boundary and abs(r - self.radius) < self.tol
        
        # Marcar frontera externa
        outer_boundary = OuterBoundary(self.domain_radius)
        boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)
        outer_boundary.mark(boundary_markers, 2)
        
        # Crear medida de integración
        self.ds_outer = fe.Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=2)
    
    def _add_sommerfeld_terms(self, phi, Pi, test_phi, test_Pi):
        """
        Añade términos de Sommerfeld a las formas variacionales.
        
        Condición: ∂φ/∂n + (1/r)φ = 0 en la frontera
        Se implementa débilmente en la ecuación de evolución de Π.
        """
        if not self.has_sommerfeld:
            return ufl.Constant(self.mesh, 0.0) if HAS_DOLFINX else fe.Constant(0.0)
        
        # Coordenadas en la frontera
        if HAS_DOLFINX:
            x = ufl.SpatialCoordinate(self.mesh)
        else:
            x = ufl.SpatialCoordinate(self.mesh)
        
        # Radio en la frontera
        r = ufl.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        
        # Vector normal exterior
        n = ufl.FacetNormal(self.mesh)
        
        # Término de Sommerfeld: -∫ (∂φ/∂n + φ/r) * test_Pi * ds
        # Usando la condición ∂φ/∂n = -φ/r
        sommerfeld_term = -(phi / r) * test_Pi * self.ds_outer
        
        return sommerfeld_term
    
    def _setup_mass_matrix_solver(self):
        """Configura el solver de la matriz de masa con métrica."""
        if HAS_DOLFINX:
            # DOLFINx
            self.mass_matrix = fem.petsc.assemble_matrix(fem.form(self.mass_form))
            self.mass_matrix.assemble()
            
            # Configurar solver
            self.mass_solver = PETSc.KSP().create(self.mesh.comm)
            self.mass_solver.setOperators(self.mass_matrix)
            self.mass_solver.setType(PETSc.KSP.Type.CG)
            self.mass_solver.getPC().setType(PETSc.PC.Type.HYPRE)
            self.mass_solver.setTolerances(rtol=1e-12, atol=1e-15)
            # Vectores PETSc para resolver A w = b
            self.rhs_vec = self.mass_matrix.createVecRight()
            self.sol_vec = self.mass_matrix.createVecLeft()
        else:
            # FEniCS legacy: usar la forma de masa con métrica
            try:
                # Intentar ensamblar la forma de masa con métrica
                self.mass_matrix = fe.assemble(self.mass_form)
            except:
                # Fallback a forma simple si hay problemas
                u_test = fe.TestFunction(self.V)
                u_trial = fe.TrialFunction(self.V)
                sqrtg = self._SQRTG()
                mass_simple = fe.inner(u_trial, u_test) * sqrtg * fe.dx
                self.mass_matrix = fe.assemble(mass_simple)
            
            # Configurar solver simple
            self.mass_solver = fe.LUSolver(self.mass_matrix)
    
    def set_initial_conditions(self, phi_init=None, Pi_init=None):
        """
        Establece condiciones iniciales.
        
        Args:
            phi_init: Función inicial para φ (si None, usa Gaussian bump)
            Pi_init: Función inicial para Π (si None, usa cero)
        """
        if HAS_DOLFINX:
            V_scalar = self.V_scalar
            if phi_init is None:
                phi_init = GaussianBump(self.mesh, V_scalar).get_function()
            if Pi_init is None:
                Pi_init = create_zero_field(V_scalar)
            try:
                self.u.x.array[0::2] = phi_init.x.array[:]
                self.u.x.array[1::2] = Pi_init.x.array[:]
            except Exception:
                # Fallback genérico: si falla el intercalado, copia por longitud común
                n = min(self.u.x.array.size//2, phi_init.x.array.size, Pi_init.x.array.size)
                self.u.x.array[0:2*n:2] = phi_init.x.array[:n]
                self.u.x.array[1:2*n:2] = Pi_init.x.array[:n]
        else:
            if phi_init is None:
                phi_init = GaussianBump(self.mesh, self.V_scalar).get_function()
            if Pi_init is None:
                Pi_init = create_zero_field(self.V_scalar)
            fe.assign(self.u.sub(0), phi_init)
            fe.assign(self.u.sub(1), Pi_init)
        print("✓ Condiciones iniciales establecidas")
    
    def _assemble_rhs_and_solve_du(self):
        """Ensamblar RHS(u) y resolver M du = RHS, dejando du en self.du."""
        rhs_form_with_bc = self.rhs_form  # _sommerfeld_boundary_term ya incluido
        if HAS_DOLFINX:
            self.rhs_vec.zeroEntries()
            fem.petsc.assemble_vector(self.rhs_vec, fem.form(rhs_form_with_bc))
            self.rhs_vec.assemble()
            self.mass_solver.solve(self.rhs_vec, self.sol_vec)
            # Copiar solución a self.du
            self.du.x.array[:] = self.sol_vec.getArray(readonly=True)
        else:
            b = fe.assemble(rhs_form_with_bc)
            self.mass_solver.solve(self.du.vector(), b)
    
    def ssp_rk3_step(self, dt):
        """
        Realiza un paso de integración SSP-RK3.
        
        u^(1) = u^n + dt * L(u^n)
        u^(2) = (3/4) * u^n + (1/4) * u^(1) + (1/4) * dt * L(u^(1))
        u^(n+1) = (1/3) * u^n + (2/3) * u^(2) + (2/3) * dt * L(u^(2))
        """
        # Etapa 1
        self._assemble_rhs_and_solve_du()
        if HAS_DOLFINX:
            self.u1.x.array[:] = self.u.x.array[:] + dt * self.du.x.array[:]
        else:
            self.u1.vector()[:] = self.u.vector()[:] + dt * self.du.vector()[:]
        # Etapa 2
        if HAS_DOLFINX:
            self.u.x.array[:] = self.u1.x.array[:]
        else:
            self.u.assign(self.u1)
        self._assemble_rhs_and_solve_du()
        if HAS_DOLFINX:
            self.u2.x.array[:] = (0.75 * self.u1.x.array[:] + 0.25 * self.u.x.array[:] + 0.25 * dt * self.du.x.array[:])
        else:
            self.u2.vector()[:] = (0.75 * self.u1.vector()[:] + 0.25 * self.u.vector()[:] + 0.25 * dt * self.du.vector()[:])
        # Etapa 3
        if HAS_DOLFINX:
            self.u.x.array[:] = self.u2.x.array[:]
        else:
            self.u.assign(self.u2)
        self._assemble_rhs_and_solve_du()
        if HAS_DOLFINX:
            self.u_new.x.array[:] = (1.0/3.0 * self.u1.x.array[:] + 2.0/3.0 * self.u2.x.array[:] + 2.0/3.0 * dt * self.du.x.array[:])
            self.u.x.array[:] = self.u_new.x.array[:]
        else:
            self.u_new.vector()[:] = (1.0/3.0 * self.u1.vector()[:] + 2.0/3.0 * self.u2.vector()[:] + 2.0/3.0 * dt * self.du.vector()[:])
            self.u.assign(self.u_new)
        self.current_time += dt
    
    def energy(self):
        """E = ∫ sqrtg [ 1/2 (γ^{ij}∂_i φ ∂_j φ) + 1/2 Π^2 + V(φ) ] dx"""
        sqrtg = self._SQRTG()
        gammaInv = self._GAMMAINV()
        gradphi = ufl.dot(gammaInv, ufl.grad(self.phi_c))
        # V(φ) usando el potencial configurado
        try:
            Vphi = self.potential.evaluate(self.phi_c)
        except Exception:
            Vphi = 0.5 * self.phi_c * self.phi_c
        energy_density = (0.5 * ufl.inner(gradphi, ufl.grad(self.phi_c)) + 0.5 * self.Pi_c * self.Pi_c + Vphi) * sqrtg
        
        if HAS_DOLFINX:
            return float(fem.assemble_scalar(fem.form(energy_density * ufl.dx)))
        else:
            return float(fe.assemble(energy_density * ufl.dx))

    def boundary_flux(self):
        """Flujo aproximado en Γ_out: F ≈ ∫ Π (∂n φ) ds"""
        if not getattr(self, 'has_sommerfeld', False):
            return 0.0
        n = ufl.FacetNormal(self.mesh)
        flux_density = self.Pi_c * ufl.dot(ufl.grad(self.phi_c), n)
        if HAS_DOLFINX:
            formF = fem.form(flux_density * self._ds(self._outer_tag))
            return float(fem.assemble_scalar(formF))
        else:
            return float(fe.assemble(flux_density * self._ds(self._outer_tag)))

if __name__ == "__main__":
    # Prueba básica
    print("=== Prueba de solver_first_order.py ===")
    
    try:
        # Crear mesh simple para prueba
        if HAS_DOLFINX:
            from dolfinx.mesh import create_box
            from mpi4py import MPI
            mesh = create_box(MPI.COMM_WORLD, [[-2, -2, -2], [2, 2, 2]], [4, 4, 4])
        else:
            import fenics as fe
            mesh = fe.BoxMesh(fe.Point(-2, -2, -2), fe.Point(2, 2, 2), 4, 4, 4)
        
        # Crear solver
        solver = FirstOrderKGSolver(
            mesh=mesh,
            degree=1,
            potential_type="quadratic",
            potential_params={"m_squared": 1.0},
            cfl_factor=0.3,
            domain_radius=2.0
        )
        
        # Establecer condiciones iniciales
        solver.set_initial_conditions()
        
        # Evolución corta
        solver.evolve(t_final=0.1, verbose=True)
        
        # Obtener campos finales
        phi, Pi = solver.get_fields()
        
        print("✓ Solver de primer orden completado exitosamente")
        
    except Exception as e:
        print(f"✗ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
