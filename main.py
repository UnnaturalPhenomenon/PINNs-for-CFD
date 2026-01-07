"""PINN structure for 2D forced convection heat transfer.

This is a minimal PyTorch skeleton that defines:
- A shared MLP for (u, v, p, T)
- Navier-Stokes + energy residuals
- Boundary condition losses

You can adapt the geometry, sampling, and training loop to your case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx with autograd."""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def laplacian(y: torch.Tensor, x: torch.Tensor, y_var: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian of y with respect to (x, y_var)."""
    dy_dx = gradients(y, x)
    dy_dy = gradients(y, y_var)
    d2y_dx2 = gradients(dy_dx, x)
    d2y_dy2 = gradients(dy_dy, y_var)
    return d2y_dx2 + d2y_dy2


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 6):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class DomainBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


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
        bounds: DomainBounds,
        re: float = 100.0,
        pr: float = 0.71,
        u_in: float = 1.0,
        t_in: float = 0.0,
        t_wall: float = 1.0,
        width: int = 64,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.bounds = bounds
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

        u_x = gradients(u, x)
        u_y = gradients(u, y)
        v_x = gradients(v, x)
        v_y = gradients(v, y)
        p_x = gradients(p, x)
        p_y = gradients(p, y)
        t_x = gradients(t, x)
        t_y = gradients(t, y)

        u_xx = gradients(u_x, x)
        u_yy = gradients(u_y, y)
        v_xx = gradients(v_x, x)
        v_yy = gradients(v_y, y)
        t_xx = gradients(t_x, x)
        t_yy = gradients(t_y, y)

        r_cont = u_x + v_y
        r_u = u * u_x + v * u_y + p_x - (1.0 / self.re) * (u_xx + u_yy)
        r_v = u * v_x + v * v_y + p_y - (1.0 / self.re) * (v_xx + v_yy)
        r_t = u * t_x + v * t_y - (1.0 / (self.re * self.pr)) * (t_xx + t_yy)

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

    def sample_interior(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, 1, device=device) * (self.bounds.x_max - self.bounds.x_min) + self.bounds.x_min
        y = torch.rand(n, 1, device=device) * (self.bounds.y_max - self.bounds.y_min) + self.bounds.y_min
        return x, y

    def sample_boundaries(
        self, n: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        x_min, x_max = self.bounds.x_min, self.bounds.x_max
        y_min, y_max = self.bounds.y_min, self.bounds.y_max

        # Inlet: x = x_min
        y_in = torch.rand(n, 1, device=device) * (y_max - y_min) + y_min
        x_in = torch.full_like(y_in, x_min)

        # Outlet: x = x_max
        y_out = torch.rand(n, 1, device=device) * (y_max - y_min) + y_min
        x_out = torch.full_like(y_out, x_max)

        # Walls: y = y_min and y = y_max
        x_w = torch.rand(2 * n, 1, device=device) * (x_max - x_min) + x_min
        y_w_bottom = torch.full((n, 1), y_min, device=device)
        y_w_top = torch.full((n, 1), y_max, device=device)
        y_w = torch.cat([y_w_bottom, y_w_top], dim=0)

        return {
            "inlet": (x_in, y_in),
            "outlet": (x_out, y_out),
            "walls": (x_w, y_w),
        }


def example_training_step(
    model: PINNForcedConvection,
    optimizer: torch.optim.Optimizer,
    n_interior: int = 2000,
    n_boundary: int = 400,
) -> float:
    device = next(model.parameters()).device
    x_int, y_int = model.sample_interior(n_interior, device)
    bc_points = model.sample_boundaries(n_boundary, device)

    optimizer.zero_grad(set_to_none=True)
    loss_pde = model.pde_loss(x_int, y_int)
    loss_bc = model.boundary_loss(bc_points)
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()

    return float(loss.item())

