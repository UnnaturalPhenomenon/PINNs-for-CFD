import torch
import torch.nn as nn
from typing import Tuple, Dict

class DomainExpansion(nn.Module):
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def ran_num(self, n_max: torch.Tensor, n_min: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor]:
        return torch.rand(n_max, 1, device=device) * (n_max - n_min) + n_min

    def sample_interior(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        return x, y

    def sample_boundaries(
        self, n: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:

        # Inlet: x = x_min
        y_in = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        x_in = torch.full_like(y_in, self.x_min)

        # Outlet: x = x_max
        y_out = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        x_out = torch.full_like(y_out, self.x_max)

        # Walls: y = y_min and y = y_max
        x_w = torch.rand(2 * n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y_w_bottom = torch.full((n, 1), self.y_min, device=device)
        y_w_top = torch.full((n, 1), self.y_max, device=device)
        y_w = torch.cat([y_w_bottom, y_w_top], dim=0)

        return {
            "inlet": (x_in, y_in),
            "outlet": (x_out, y_out),
            "walls": (x_w, y_w),
        }