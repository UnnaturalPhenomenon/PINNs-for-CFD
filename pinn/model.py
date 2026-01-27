"""PINN structure for 2D forced convection heat transfer.

This is a minimal PyTorch skeleton that defines:
- A shared MLP for (u, v, p, T)
- Navier-Stokes + energy residuals
- Boundary condition losses

You can adapt the geometry, sampling, and training loop to your case.
"""

from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn

from pinn.autodiff import gradients
from pinn.pde import (
    continuity_residual, momentum_u_residual, momentum_v_residual, energy_residual)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 6):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()] #input과 hidden layer 묶기
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()] # hidden layer 묶기
        layers.append(nn.Linear(width, out_dim)) # output layer 묶기
        self.model = nn.Sequential(*layers) # 모델 정렬

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) # 연산


class PINNForcedConvection(nn.Module):
    """Physics-informed network for 2D steady forced convection.

    Governing equations (nondimensional):
      continuity: u_x + v_y = 0
      momentum: u u_x + v u_y + p_x - (1/Re) (u_xx + u_yy) = 0
                u v_x + v v_y + p_y - (1/Re) (v_xx + v_yy) = 0
      energy: u T_x + v T_y - (1/(Re*Pr)) (T_xx + T_yy) = 0
    """

    def __init__(
        self,
        re: float = 100.0,
        pr: float = 0.71,
        u_in: float = 1.0,
        t_in: float = 0.0,
        t_wall: float = 1.0,
        width: int = 64,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.re = re
        self.pr = pr
        self.u_in = u_in
        self.t_in = t_in
        self.t_wall = t_wall
        self.net = MLP(in_dim=2, out_dim=4, width=width, depth=depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        xy = torch.cat([x, y], dim=1)
        out = self.net(xy)
        u, v, p, t = out.split(1, dim=1)
        return u, v, p, t

    def residuals(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        x.requires_grad_(True)
        y.requires_grad_(True)

        u, v, p, t = self.forward(x, y)

        # Continuity: div(u,v) = 0
        r_cont = continuity_residual(u, v, x, y)

        # Momentum u: advection(u) + grad(p) - (1/Re)*laplacian(u) = 0
        r_u = momentum_u_residual(u, v, p, x, y, self.re)

        # Momentum v: advection(v) + grad(p) - (1/Re)*laplacian(v) = 0
        r_v = momentum_v_residual(u, v, p, x, y, self.re)

        # Energy: advection(T) - (1/(Re*Pr))*laplacian(T) = 0
        r_t = energy_residual(t, u, v, x, y, self.re, self.pr)

        return {"continuity": r_cont, "momentum_u": r_u, "momentum_v": r_v, "energy": r_t}

    def boundary_loss(self, bc_points: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.net.model[0].weight.device)

        # Inlet: u = U_in, v = 0, T = T_in
        x_in, y_in = bc_points["inlet"]
        u_in, v_in, _, t_in = self.forward(x_in, y_in)
        loss = loss + ((u_in - self.u_in) ** 2).mean()
        loss = loss + (v_in**2).mean()
        loss = loss + ((t_in - self.t_in) ** 2).mean()

        # Walls: no-slip, T = T_wall
        x_w, y_w = bc_points["walls"]
        u_w, v_w, _, t_w = self.forward(x_w, y_w)
        loss = loss + (u_w**2).mean()
        loss = loss + (v_w**2).mean()
        loss = loss + ((t_w - self.t_wall) ** 2).mean()

        # Outlet: zero normal gradients (Neumann), p = 0
        x_out, y_out = bc_points["outlet"]
        x_out.requires_grad_(True)
        y_out.requires_grad_(True)
        u_out, v_out, p_out, t_out = self.forward(x_out, y_out)
        u_x_out = gradients(u_out, x_out)
        v_x_out = gradients(v_out, x_out)
        t_x_out = gradients(t_out, x_out)
        loss = loss + (u_x_out**2).mean()
        loss = loss + (v_x_out**2).mean()
        loss = loss + (t_x_out**2).mean()
        loss = loss + (p_out**2).mean()

        return loss

    def pde_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        residuals = self.residuals(x, y)
        loss = torch.tensor(0.0, device=self.net.model[0].weight.device)
        for value in residuals.values():
            loss = loss + (value**2).mean()
        return loss




