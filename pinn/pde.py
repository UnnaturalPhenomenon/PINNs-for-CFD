import torch
from pinn.autodiff import gradients, laplacian, advection

def continuity_residual(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Continuity: div(u,v) = 0"""
        r_cont = gradients(u, x) + gradients(v, y)
        return r_cont

def momentum_u_residual(u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, x: torch.Tensor, y: torch.Tensor, re: float) -> torch.Tensor:
        """Momentum u: advection(u) + grad(p) - (1/Re)*laplacian(u) = 0"""
        adv_u = advection(u, u, v, x, y)
        lap_u = laplacian(u, x, y)
        r_u = adv_u + gradients(p, x) - (1.0 / re) * lap_u
        return r_u

def momentum_v_residual(u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, x: torch.Tensor, y: torch.Tensor, re: float) -> torch.Tensor:
        """Momentum v: advection(v) + grad(p) - (1/Re)*laplacian(v) = 0"""
        adv_v = advection(v, u, v, x, y)
        lap_v = laplacian(v, x, y)
        r_v = adv_v + gradients(p, y) - (1.0 / re) * lap_v
        return r_v

def energy_residual(t: torch.Tensor, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, re: float, pr: float) -> torch.Tensor:
        """Energy: advection(T) - (1/(Re*Pr))*laplacian(T) = 0"""
        adv_t = advection(t, u, v, x, y)
        lap_t = laplacian(t, x, y)
        r_t = adv_t - (1.0 / (re * pr)) * lap_t
        return r_t