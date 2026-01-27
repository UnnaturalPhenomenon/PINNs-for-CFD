import torch
from pinn.model import PINNForcedConvection
from pinn.domain import DomainExpansion

def example_training_step(
    model: PINNForcedConvection,
    optimizer: torch.optim.Optimizer,
    domain: DomainExpansion,
    n_interior: int = 2000,
    n_boundary: int = 400,
) -> float:
    device = next(model.parameters()).device
    x_int, y_int = domain.sample_interior(n_interior, device)
    bc_points = domain.sample_boundaries(n_boundary, device)

    optimizer.zero_grad(set_to_none=True)
    loss_pde = model.pde_loss(x_int, y_int) 
    loss_bc = model.boundary_loss(bc_points)
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()

    return float(loss.item()), float(loss_pde.item()), float(loss_bc.item())