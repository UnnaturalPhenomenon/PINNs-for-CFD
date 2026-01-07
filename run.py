import torch
import matplotlib.pyplot as plt
from main import DomainBounds, PINNForcedConvection, example_training_step

# 학습 스텝 수 설정
STEPS = 1000

if __name__ == "__main__":
    bounds = DomainBounds(x_min=0.0, x_max=5.0, y_min=0.0, y_max=1.0)
    model = PINNForcedConvection(bounds=bounds, re=100.0, pr=0.71)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []
    
    # 실시간 plot 설정
    plt.figure(figsize=(10, 6))
    plt.ion()  # interactive mode 활성화

    print(f"Starting training with {STEPS} steps...")
    for step in range(STEPS):
        loss_value = example_training_step(model, optim)
        loss_history.append(loss_value)
        
        if step % 100 == 0 or step == STEPS - 1:
            print(f"step={step:04d} loss={loss_value:.6f}")
        
        # 실시간 plot (100 스텝마다 업데이트)
        if step % 100 == 0:
            plt.clf()
            plt.plot(loss_history, 'b-', linewidth=2)
            plt.xlabel('Step', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training Loss - Step {step}/{STEPS}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.pause(0.01)
    
    # 최종 plot 저장
    plt.savefig("loss_history.png", dpi=150)
    print("Loss history plot saved as loss_history.png")
    
    # 모델 저장
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model saved as model_weights.pth")
    print("Training completed!")
    
    plt.show()