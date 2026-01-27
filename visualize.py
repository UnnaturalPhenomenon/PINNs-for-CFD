"""Visualize trained PINN model predictions with multiple plot types."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn.model import PINNForcedConvection
from pinn.domain import DomainBounds

image_path = './images/'

def load_model(bounds: DomainBounds, model_path: str = "model_weights.pth") -> PINNForcedConvection:
    """Load trained model weights from file."""
    model = PINNForcedConvection(bounds=bounds, re=100.0, pr=0.71)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(model: PINNForcedConvection, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Predict u, v, p, T at given (x, y) coordinates."""
    with torch.no_grad():
        u, v, p, t = model.forward(x, y)
    return {
        "u": u.detach().numpy(),
        "v": v.detach().numpy(),
        "p": p.detach().numpy(),
        "T": t.detach().numpy(),
    }


def plot_heatmaps(model: PINNForcedConvection, bounds: DomainBounds, grid_size: int = 50):
    """Plot predicted field variables as 4-panel heatmaps."""
    x_vals = np.linspace(bounds.x_min, bounds.x_max, grid_size)
    y_vals = np.linspace(bounds.y_min, bounds.y_max, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    x_flat = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32)
    y_flat = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32)
    
    results = predict(model, x_flat, y_flat)
    U = results['u'].reshape(X.shape)
    V = results['v'].reshape(X.shape)
    P = results['p'].reshape(X.shape)
    T = results['T'].reshape(X.shape)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    c1 = axes[0, 0].contourf(X, Y, U, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('Velocity u (x-direction)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(c1, ax=axes[0, 0])
    
    c2 = axes[0, 1].contourf(X, Y, V, levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('Velocity v (y-direction)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(c2, ax=axes[0, 1])
    
    c3 = axes[1, 0].contourf(X, Y, P, levels=20, cmap='viridis')
    axes[1, 0].set_title('Pressure p', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(c3, ax=axes[1, 0])
    
    c4 = axes[1, 1].contourf(X, Y, T, levels=20, cmap='hot')
    axes[1, 1].set_title('Temperature T', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(c4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(image_path + "heatmaps.png", dpi=150, bbox_inches='tight')
    print("✓ Heatmaps saved as heatmaps.png")
    plt.show()


def plot_velocity_vectors(model: PINNForcedConvection, bounds: DomainBounds, grid_size: int = 20):
    """Plot velocity field as vector arrows overlaid on speed magnitude."""
    x_vals = np.linspace(bounds.x_min, bounds.x_max, grid_size)
    y_vals = np.linspace(bounds.y_min, bounds.y_max, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    x_flat = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32)
    y_flat = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32)
    
    results = predict(model, x_flat, y_flat)
    U = results['u'].reshape(X.shape)
    V = results['v'].reshape(X.shape)
    Speed = np.sqrt(U**2 + V**2)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Speed magnitude as background
    contourf = ax.contourf(X, Y, Speed, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contourf, ax=ax, label='Speed magnitude')
    
    # Velocity vectors
    quiver = ax.quiver(X, Y, U, V, scale=50, scale_units='inches', alpha=0.7, color='white', width=0.003)
    ax.quiverkey(quiver, X=0.9, Y=1.05, U=1.0, label='Velocity', labelpos='E', coordinates='axes')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Velocity Field (Speed + Direction)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(image_path + "velocity_vectors.png", dpi=150, bbox_inches='tight')
    print("✓ Velocity vectors saved as velocity_vectors.png")
    plt.show()


def plot_profiles(model: PINNForcedConvection, bounds: DomainBounds):
    """Plot 1D profiles along centerline (y=0.5) at different x positions."""
    x_positions = [0.5, 1.5, 2.5, 3.5, 4.5]
    y_profile = np.linspace(bounds.y_min, bounds.y_max, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, x_pos in enumerate(x_positions[:-1]):
        x_profile = np.full_like(y_profile, x_pos)
        x_tensor = torch.tensor(x_profile.reshape(-1, 1), dtype=torch.float32)
        y_tensor = torch.tensor(y_profile.reshape(-1, 1), dtype=torch.float32)
        
        results = predict(model, x_tensor, y_tensor)
        u_profile = results['u'].squeeze()
        t_profile = results['T'].squeeze()
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        line1 = ax.plot(y_profile, u_profile, 'b-', linewidth=2, label='Velocity u')
        ax.set_xlabel('y', fontsize=11)
        ax.set_ylabel('Velocity u', color='b', fontsize=11)
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3)
        
        line2 = ax2.plot(y_profile, t_profile, 'r-', linewidth=2, label='Temperature T')
        ax2.set_ylabel('Temperature T', color='r', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title(f'Profiles at x = {x_pos:.1f}', fontsize=12, fontweight='bold')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    # Remove last empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(image_path + "profiles.png", dpi=150, bbox_inches='tight')
    print("✓ Profiles saved as profiles.png")
    plt.show()


def plot_centerline(model: PINNForcedConvection, bounds: DomainBounds):
    """Plot variables along domain centerline (y=0.5)."""
    x_vals = np.linspace(bounds.x_min, bounds.x_max, 200)
    y_vals = np.full_like(x_vals, 0.5)
    
    x_tensor = torch.tensor(x_vals.reshape(-1, 1), dtype=torch.float32)
    y_tensor = torch.tensor(y_vals.reshape(-1, 1), dtype=torch.float32)
    
    results = predict(model, x_tensor, y_tensor)
    u_vals = results['u'].squeeze()
    v_vals = results['v'].squeeze()
    p_vals = results['p'].squeeze()
    t_vals = results['T'].squeeze()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(x_vals, u_vals, 'b-', linewidth=2)
    axes[0, 0].set_ylabel('Velocity u', fontsize=11)
    axes[0, 0].set_title('u along centerline (y=0.5)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(x_vals, v_vals, 'g-', linewidth=2)
    axes[0, 1].set_ylabel('Velocity v', fontsize=11)
    axes[0, 1].set_title('v along centerline (y=0.5)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(x_vals, p_vals, 'purple', linewidth=2)
    axes[1, 0].set_ylabel('Pressure p', fontsize=11)
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_title('p along centerline (y=0.5)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(x_vals, t_vals, 'r-', linewidth=2)
    axes[1, 1].set_ylabel('Temperature T', fontsize=11)
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_title('T along centerline (y=0.5)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(image_path + 'centerline.png', dpi=150, bbox_inches='tight')
    print("✓ Centerline plots saved as centerline.png")
    plt.show()


if __name__ == "__main__":
    print("Loading trained model...")

    bounds = DomainBounds(x_min=0.0, x_max=5.0, y_min=0.0, y_max=1.0)
    model = load_model(bounds, "model_weights.pth")
    
    print("Model loaded successfully!\n")
    
    print("=" * 50)
    print("Generating visualizations...")
    print("=" * 50)
    
    print("\n1. Generating heatmaps (u, v, p, T)...")
    plot_heatmaps(model, bounds, grid_size=60)
    
    print("\n2. Generating velocity field (vectors)...")
    plot_velocity_vectors(model, bounds, grid_size=25)
    
    print("\n3. Generating 1D profiles at different x positions...")
    plot_profiles(model, bounds)
    
    print("\n4. Generating centerline plots...")
    plot_centerline(model, bounds)
    
    print("\n" + "=" * 50)
    print("All visualizations complete!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  • heatmaps.png          - 4-panel contour plots")
    print("  • velocity_vectors.png  - Velocity field with streamlines")
    print("  • profiles.png          - 1D profiles at x=0.5,1.5,2.5,3.5")
    print("  • centerline.png        - Variables along y=0.5")
