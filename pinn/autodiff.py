import torch

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


def laplacian(f: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian of a scalar field f."""
    f_x = gradients(f, x)
    f_y = gradients(f, y)
    f_xx = gradients(f_x, x)
    f_yy = gradients(f_y, y)
    return f_xx + f_yy

def advection(f: torch.Tensor, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the advection term (u*grad(f)) for a scalar field f."""
    f_x = gradients(f, x)
    f_y = gradients(f, y)
    return u * f_x + v * f_y
