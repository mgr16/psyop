from typing import Tuple
import ufl
from fem_backend import is_dolfinx, Constant


class BackgroundCoeffs:
    def build(self, mesh) -> Tuple:
        raise NotImplementedError

    def max_characteristic_speed(self, mesh) -> float:
        return 1.0


class FlatBackgroundCoeffs(BackgroundCoeffs):
    def build(self, mesh):
        dim = mesh.topology.dim if is_dolfinx() else mesh.geometric_dimension()
        alpha_f = Constant(mesh, 1.0)
        beta_f = None  # vector shift = 0
        gammaInv_f = ufl.Identity(dim)
        sqrtg_f = Constant(mesh, 1.0)
        K_f = Constant(mesh, 0.0)
        return alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f


class SchwarzschildIsotropicCoeffs(BackgroundCoeffs):
    def __init__(self, M: float = 1.0):
        self.M = float(M)

    def build(self, mesh):
        # Placeholder físico razonable (mejora futura: α(r), γ⁻¹(r))
        return FlatBackgroundCoeffs().build(mesh)

    def max_characteristic_speed(self, mesh) -> float:
        # En práctica ≤ 1; con placeholder dejamos 1.0
        return 1.0


def make_background(metric_cfg: dict) -> BackgroundCoeffs:
    mtype = metric_cfg.get("type", "flat").lower()
    if mtype == "flat":
        return FlatBackgroundCoeffs()
    if mtype == "schwarzschild":
        return SchwarzschildIsotropicCoeffs(M=metric_cfg.get("M", 1.0))
    raise ValueError(f"Métrica no soportada: {mtype}")
