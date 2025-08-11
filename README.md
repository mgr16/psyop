# PSYOP - Simulación de Campos Escalares en Relatividad General

## Descripción

PSYOP es un simulador de campos escalares evolucionando en fondos de agujeros negros usando elementos finitos. El proyecto ha sido completamente renovado con una arquitectura modular avanzada y métodos numéricos de alto orden.

## 🚀 Mejoras Implementadas (Versión 2.0)

### **Mejora 1: Formulación de Primer Orden con SSP-RK3**
- **Sistema de primer orden**: (φ, Π) donde Π = ∂φ/∂t
- **Integración temporal SSP-RK3**: Strong Stability Preserving Runge-Kutta de orden 3
- **CFL adaptativo**: Paso de tiempo automático basado en el tamaño de malla
- **Solver de matriz de masa**: Inversión eficiente usando PETSc/HYPRE

### **Mejora 2: Condiciones de Frontera Sommerfeld (característica)**
- **Condición física característica**: usa la velocidad de salida `c_out = α − β·n`
- **Implementación débil**: término de borde sobre facetas `tag=2` vía `ds(tag)`
- **Absorción de ondas**: reduce reflexiones sin Robin ad-hoc

### **Mejora 3: Arquitectura Modular Avanzada**
- **Compatibilidad dual**: FEniCS legacy y DOLFINx
- **Generación de mallas**: Gmsh con etiquetas de frontera automáticas
- **Potenciales generalizados**: Higgs, cuadrático, sombrero mexicano
- **Condiciones iniciales flexibles**: Gaussian bump, ondas planas, etc.

## 📁 Estructura del Proyecto

```
PSYOP/
├── main.py                    # Script principal renovado
├── solver_first_order.py      # Solver (φ, Π) con SSP-RK3
├── gmsh_helpers.py           # Generación de mallas con Gmsh
├── initial_conditions.py     # Condiciones iniciales DOLFINx-compatible
├── potential.py              # Potenciales generalizados
├── utils.py                  # Utilidades CFL y mesh analysis
├── metrics.py                # Métricas de agujeros negros
├── quasinormal_modes.py      # Análisis de modos quasi-normales
├── field.py                  # Módulo legacy de campos
├── solver.py                 # Solver legacy
├── solver_dissipation.py     # Solver con disipación
├── test_standalone_logic.py  # Pruebas sin FEniCS
├── test_complete_system.py   # Pruebas completas
└── README.md                 # Esta documentación
```

## 🛠 Instalación

### Opción 1: FEniCS Legacy (Recomendado para estabilidad)
```bash
# Crear entorno conda
conda create -n psyop python=3.9
conda activate psyop

# Instalar FEniCS
conda install -c conda-forge fenics

# Dependencias adicionales
conda install -c conda-forge gmsh numpy matplotlib scipy
```

### Opción 2: DOLFINx (Experimental, última versión)
```bash
# Crear entorno conda
conda create -n psyop-dolfinx python=3.10
conda activate psyop-dolfinx

# Instalar DOLFINx
conda install -c conda-forge dolfinx

# Dependencias adicionales
conda install -c conda-forge gmsh numpy matplotlib scipy petsc4py
```

### Opción 3: Configuración Dual (Recomendado para investigación)
```bash
# Crear entorno conda con ambos frameworks
conda create -n psyop-dual python=3.10
conda activate psyop-dual

# Instalar ambos frameworks
conda install -c conda-forge fenics dolfinx

# Dependencias adicionales
conda install -c conda-forge gmsh numpy matplotlib scipy petsc4py mpi4py

# Verificar instalación dual
python test_dual_frameworks.py
```

**Ventajas de la configuración dual:**
- ✅ Máxima compatibilidad y flexibilidad
- ✅ Migración gradual FEniCS → DOLFINx
- ✅ Validación cruzada de resultados
- ✅ Acceso a todas las características
- ✅ Framework detection automático

### Verificación de la instalación
```bash
# Probar lógica sin FEniCS
python test_standalone_logic.py

# Probar sistema completo (requiere FEniCS/DOLFINx)
python test_complete_system.py
```

## 🏃‍♂️ Uso Rápido

### Simulación básica
```bash
python main.py
```

### Configuración personalizada
```python
# En main.py, modificar sim_params:
sim_params = {
    "mesh": {
        "mesh_type": "gmsh",    # "gmsh" o "builtin"
        "radius": 15.0,         # Radio del dominio
        "resolution": 1.0       # Resolución (menor = más fino)
    },
    
    "solver": {
        "degree": 1,            # Grado de elementos finitos
        "potential_type": "higgs",  # "higgs", "quadratic", "mexican_hat"
        "potential_params": {
            "m_squared": 1.0,
            "lambda_coupling": 0.1
        },
        "cfl_factor": 0.3       # Factor CFL para estabilidad
    },
    
    "evolution": {
        "t_final": 20.0,        # Tiempo final
        "dt": None,             # None = adaptativo
        "verbose": True
    }
}
```

## 🧮 Física y Métodos Numéricos

### Ecuaciones Fundamentales

**Sistema de primer orden:**
```
∂φ/∂t = Π
∂Π/∂t = ∇²φ - V'(φ)
```

**Condición de salida Sommerfeld (característica):**
```
c_out = α − β·n
```
Se implementa como flujo saliente en el término de borde del RHS.

**Potencial de Higgs:**
```
V(φ) = ½m²φ² + ¼λφ⁴
V'(φ) = m²φ + λφ³
```

### Esquema de Integración SSP-RK3

```
u⁽¹⁾ = uⁿ + dt · L(uⁿ)
u⁽²⁾ = ¾uⁿ + ¼u⁽¹⁾ + ¼dt · L(u⁽¹⁾)
uⁿ⁺¹ = ⅓uⁿ + ⅔u⁽²⁾ + ⅔dt · L(u⁽²⁾)
```

### Condición CFL Adaptativa

```
dt = CFL_factor × h_min / c_max
```
donde `h_min` es el tamaño mínimo de celda y `c_max = 1` (velocidad de la luz).

## 📊 Características Avanzadas

### Generación de Mallas
- **Gmsh**: Mallas esféricas con etiquetas de frontera automáticas
- **Fallback**: Mallas cúbicas de FEniCS si Gmsh no está disponible
- **Etiquetas**: Frontera externa marcada con `tag=2` para condiciones Sommerfeld

### Análisis de Modos Quasi-Normales
- **Muestreo temporal**: Registro del campo en puntos específicos
- **FFT**: Análisis espectral para identificar frecuencias características
- **Visualización**: Gráficos automáticos del espectro de frecuencias

### Compatibilidad Multi-Framework
- **Detección automática**: El código detecta si DOLFINx o FEniCS legacy está disponible
- **API unificada**: Misma interfaz para ambos frameworks
- **Importaciones condicionales**: Sin errores si un framework no está instalado

## 🔧 Desarrollo y Extensiones

### Añadir un nuevo potencial
```python
# En potential.py
class CustomPotential:
    def __init__(self, param1=1.0):
        self.param1 = param1
    
    def evaluate(self, phi):
        return self.param1 * phi**6  # Ejemplo
    
    def derivative(self, phi):
        return 6 * self.param1 * phi**5
```

### Añadir condiciones iniciales
```python
# En initial_conditions.py
class CustomInitialCondition:
    def __init__(self, mesh, V, **params):
        # Implementar lógica personalizada
        pass
```

### Modificar el solver
El solver principal está en `solver_first_order.py`. Métodos clave:
- `ssp_rk3_step()`: Integración temporal
- `_setup_sommerfeld_bc()`: Condiciones de frontera
- `_compute_rhs()`: Evaluación del lado derecho

## 📈 Resultados y Validación

### Salidas del programa
- **Campos finales**: φ y Π guardados en formato VTK/XDMF
- **Series temporales**: Evolución del campo en puntos específicos
- **Espectro QNM**: Análisis de frecuencias características
- **Métricas de convergencia**: Normas y estadísticas

### Archivos generados
```
results/
├── phi_final.pvd          # Campo φ final (FEniCS legacy)
├── Pi_final.pvd           # Campo Π final (FEniCS legacy)  
├── phi_final.xdmf         # Campo φ final (DOLFINx)
├── Pi_final.xdmf          # Campo Π final (DOLFINx)
├── time_series.txt        # Series temporales
└── qnm_spectrum.png       # Espectro de modos quasi-normales
```

## ⚡ Rendimiento

### Optimizaciones implementadas
- **Matriz de masa precalculada**: Factorización reutilizada
- **Solver PETSc**: Algoritmos paralelos eficientes
- **CFL adaptativo**: Pasos de tiempo óptimos automáticamente
- **Evaluación vectorizada**: Potenciales evaluados en arrays NumPy

### Benchmarks típicos
- **Mesh 10³ elementos**: ~1-5 segundos por unidad de tiempo físico
- **Mesh 20³ elementos**: ~10-30 segundos por unidad de tiempo físico
- **Escalabilidad**: Excelente con número de cores (PETSc paralelo)

## 🐛 Solución de Problemas

### Error común: "Import could not be resolved"
**Causa**: FEniCS/DOLFINx no instalado
**Solución**: 
```bash
conda install -c conda-forge fenics
# o
conda install -c conda-forge dolfinx
```

### Error: "Gmsh not available"
**Causa**: Gmsh no instalado
**Solución**: El programa usa mallas de fallback automáticamente. Para instalar Gmsh:
```bash
conda install -c conda-forge gmsh
```

### Error de convergencia en el solver
**Causa**: Paso de tiempo demasiado grande o malla muy gruesa
**Solución**: Reducir `cfl_factor` o aumentar resolución de malla

### Memoria insuficiente
**Causa**: Malla demasiado fina
**Solución**: Aumentar `resolution` en parámetros de malla

## 📚 Referencias Técnicas

### Métodos numéricos
- **SSP-RK3**: Gottlieb et al. (2001) "Strong Stability-Preserving High-Order Time Discretization Methods"
- **Elementos Finitos**: Brenner & Scott "The Mathematical Theory of Finite Element Methods"
- **Condiciones Sommerfeld**: Engquist & Majda (1977) "Absorbing boundary conditions for numerical simulation of waves"

### Física
- **Klein-Gordon**: Relativistic quantum mechanics and field theory textbooks
- **Modos Quasi-Normales**: Berti et al. (2009) "Eigenvalues and eigenfunctions of spin-weighted spheroidal harmonics"
- **Agujeros Negros**: Wald "General Relativity", Misner-Thorne-Wheeler "Gravitation"


---

**Versión**: 2.0 (Renovación completa)  
**Compatibilidad**: FEniCS legacy 2019.1+ / DOLFINx 0.6+  
**Python**: 3.8+  
**Licencia**: Proyecto de investigación académica
