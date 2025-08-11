#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
potential.py

Potenciales para campos escalares.
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

class HiggsPotential:
    """
    Potencial de Higgs: V(φ) = ½ m² φ² + ¼ λ φ⁴
    
    La derivada es: V'(φ) = m² φ + λ φ³
    """
    
    def __init__(self, m_squared=1.0, lambda_coupling=0.1):
        """
        Parámetros:
            m_squared: Parámetro de masa al cuadrado
            lambda_coupling: Constante de auto-acoplamiento
        """
        self.m_squared = float(m_squared)
        self.lambda_coupling = float(lambda_coupling)
    
    def evaluate(self, phi):
        """
        Evalúa V(φ).
        
        Args:
            phi: Función o variable UFL
        
        Returns:
            Expresión UFL del potencial
        """
        if HAS_DOLFINX:
            return 0.5 * self.m_squared * phi**2 + 0.25 * self.lambda_coupling * phi**4
        else:
            return 0.5 * self.m_squared * phi**2 + 0.25 * self.lambda_coupling * phi**4
    
    def derivative(self, phi):
        """
        Evalúa V'(φ).
        
        Args:
            phi: Función o variable UFL
        
        Returns:
            Expresión UFL de la derivada del potencial
        """
        if HAS_DOLFINX:
            return self.m_squared * phi + self.lambda_coupling * phi**3
        else:
            return self.m_squared * phi + self.lambda_coupling * phi**3
    
    def evaluate_numpy(self, phi_values):
        """
        Evalúa V(φ) para arrays numpy.
        
        Args:
            phi_values: Array numpy de valores del campo
        
        Returns:
            Array numpy con los valores del potencial
        """
        phi = np.asarray(phi_values)
        return 0.5 * self.m_squared * phi**2 + 0.25 * self.lambda_coupling * phi**4
    
    def derivative_numpy(self, phi_values):
        """
        Evalúa V'(φ) para arrays numpy.
        
        Args:
            phi_values: Array numpy de valores del campo
        
        Returns:
            Array numpy con los valores de la derivada
        """
        phi = np.asarray(phi_values)
        return self.m_squared * phi + self.lambda_coupling * phi**3

class QuadraticPotential:
    """
    Potencial cuadrático simple: V(φ) = ½ m² φ²
    
    La derivada es: V'(φ) = m² φ
    """
    
    def __init__(self, m_squared=1.0):
        """
        Parámetros:
            m_squared: Parámetro de masa al cuadrado
        """
        self.m_squared = float(m_squared)
    
    def evaluate(self, phi):
        """Evalúa V(φ)."""
        return 0.5 * self.m_squared * phi**2
    
    def derivative(self, phi):
        """Evalúa V'(φ)."""
        return self.m_squared * phi
    
    def evaluate_numpy(self, phi_values):
        """Evalúa V(φ) para arrays numpy."""
        phi = np.asarray(phi_values)
        return 0.5 * self.m_squared * phi**2
    
    def derivative_numpy(self, phi_values):
        """Evalúa V'(φ) para arrays numpy."""
        phi = np.asarray(phi_values)
        return self.m_squared * phi

class MexicanHatPotential:
    """
    Potencial sombrero mexicano: V(φ) = ¼ λ (φ² - v²)²
    donde v es el valor de vacío.
    
    La derivada es: V'(φ) = λ φ (φ² - v²)
    """
    
    def __init__(self, lambda_coupling=0.1, vacuum_value=1.0):
        """
        Parámetros:
            lambda_coupling: Constante de auto-acoplamiento
            vacuum_value: Valor de vacío v
        """
        self.lambda_coupling = float(lambda_coupling)
        self.vacuum_value = float(vacuum_value)
        self.v_squared = self.vacuum_value**2
    
    def evaluate(self, phi):
        """Evalúa V(φ)."""
        return 0.25 * self.lambda_coupling * (phi**2 - self.v_squared)**2
    
    def derivative(self, phi):
        """Evalúa V'(φ)."""
        return self.lambda_coupling * phi * (phi**2 - self.v_squared)
    
    def evaluate_numpy(self, phi_values):
        """Evalúa V(φ) para arrays numpy."""
        phi = np.asarray(phi_values)
        return 0.25 * self.lambda_coupling * (phi**2 - self.v_squared)**2
    
    def derivative_numpy(self, phi_values):
        """Evalúa V'(φ) para arrays numpy."""
        phi = np.asarray(phi_values)
        return self.lambda_coupling * phi * (phi**2 - self.v_squared)

class ZeroPotential:
    """
    Potencial nulo: V(φ) = 0
    
    Útil para evolución libre (sin potencial).
    """
    
    def __init__(self):
        pass
    
    def evaluate(self, phi):
        """Evalúa V(φ) = 0."""
        if HAS_DOLFINX:
            return ufl.Constant(phi.function_space.mesh, 0.0)
        else:
            return fe.Constant(0.0)
    
    def derivative(self, phi):
        """Evalúa V'(φ) = 0."""
        if HAS_DOLFINX:
            return ufl.Constant(phi.function_space.mesh, 0.0)
        else:
            return fe.Constant(0.0)
    
    def evaluate_numpy(self, phi_values):
        """Evalúa V(φ) = 0 para arrays numpy."""
        phi = np.asarray(phi_values)
        return np.zeros_like(phi)
    
    def derivative_numpy(self, phi_values):
        """Evalúa V'(φ) = 0 para arrays numpy."""
        phi = np.asarray(phi_values)
        return np.zeros_like(phi)

def get_potential(potential_type="higgs", **kwargs):
    """
    Factory function para crear potenciales.
    
    Args:
        potential_type: Tipo de potencial ("higgs", "quadratic", "mexican_hat", "zero")
        **kwargs: Parámetros específicos del potencial
    
    Returns:
        Instancia del potencial
    """
    potential_map = {
        "higgs": HiggsPotential,
        "quadratic": QuadraticPotential,
        "mexican_hat": MexicanHatPotential,
        "zero": ZeroPotential
    }
    
    if potential_type not in potential_map:
        raise ValueError(f"Tipo de potencial '{potential_type}' no reconocido. "
                        f"Opciones disponibles: {list(potential_map.keys())}")
    
    potential_class = potential_map[potential_type]
    
    # Filtrar kwargs para solo pasar parámetros válidos
    import inspect
    sig = inspect.signature(potential_class.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    return potential_class(**valid_kwargs)

if __name__ == "__main__":
    # Pruebas básicas
    print("=== Prueba de potential.py ===")
    
    try:
        # Prueba potencial de Higgs
        higgs = HiggsPotential(m_squared=1.0, lambda_coupling=0.1)
        print("✓ Potencial de Higgs creado")
        
        # Prueba con valores numpy
        phi_test = np.array([0.0, 1.0, 2.0])
        V_vals = higgs.evaluate_numpy(phi_test)
        dV_vals = higgs.derivative_numpy(phi_test)
        print(f"  φ = {phi_test}")
        print(f"  V(φ) = {V_vals}")
        print(f"  V'(φ) = {dV_vals}")
        
        # Prueba potencial cuadrático
        quad = QuadraticPotential(m_squared=2.0)
        V_quad = quad.evaluate_numpy(phi_test)
        print(f"  V_cuadrático(φ) = {V_quad}")
        
        # Prueba potencial sombrero mexicano
        mexican = MexicanHatPotential(lambda_coupling=0.1, vacuum_value=1.0)
        V_mexican = mexican.evaluate_numpy(phi_test)
        print(f"  V_mexican(φ) = {V_mexican}")
        
        # Prueba factory function
        pot_factory = get_potential("higgs", m_squared=1.5, lambda_coupling=0.2)
        print("✓ Factory function funcional")
        
        # Prueba potencial cero
        zero_pot = ZeroPotential()
        V_zero = zero_pot.evaluate_numpy(phi_test)
        print(f"  V_cero(φ) = {V_zero}")
        
        print("Módulo potential.py completado exitosamente.")
        
    except Exception as e:
        print(f"✗ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
