import torch
# import matplotlib.pyplot as plt
from main import DomainBounds, PINNForcedConvection, example_training_step
from pinn.domain import DomainExpansion

if __name__ == "__main__":
    bounds = DomainBounds(x_min=0.0, x_max=5.0, y_min=0.0, y_max=1.0)
    model = PINNForcedConvection(bounds=bounds, re=100.0, pr=0.71)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    domain = DomainExpansion(x_min=0.0, x_max=5.0, y_min=0.0, y_max=1.0)

    for step in range(1000):
        loss_value = example_training_step(model, optim, domain)
        print(f"step={step:03d} loss={loss_value:.6f}")
        