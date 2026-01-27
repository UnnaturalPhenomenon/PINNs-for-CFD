import torch
# import matplotlib.pyplot as plt
import argparse
import os
import logging
from pinn.model import PINNForcedConvection
from pinn.domain import DomainBounds, DomainExpansion
from pinn.train import example_training_step
from pinn.log_config import setup_logging

setup_logging('main_copy')

logger = logging.getLogger('main_copy')

def run_main_copy_process(args):
    logger.info("main copy 실험 진행...")
    try:
        logger.info(f"학습 파라미터; steps: {args.steps}, lr: {args.lr}, layers: {args.layers}, neurons: {args.neurons}")
        loss_value, loss_pde, loss_bc = train(args)
    except Exception as e:
        logger.error(f"main copy 실험 중 오류 발생: {e}")
    logger.info("main copy 실험을 종료합니다.")
    logger.info(f"total steps: {args.steps}| final loss: {loss_value:.6f} | loss_pde: {loss_pde:.6f} | loss_bc: {loss_bc:.6f}")
def train(args) -> int:
    # 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    #도메인 및 모델 초기화
    bounds = DomainBounds(x_min=0.0, x_max=5.0, y_min=0.0, y_max=1.0)
    model = PINNForcedConvection(re=args.re, pr=args.pr, depth=args.layers, width=args.neurons)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    domain = DomainExpansion(bounds=bounds)

    for step in range(args.steps):
        loss_value, loss_pde, loss_bc = example_training_step(model, optim, domain)  

        if step % args.log_interval == 0 or step == args.steps - 1:
            print(f"step={step:04d} loss={loss_value:.6f} loss_pde={loss_pde:.6f} loss_bc={loss_bc:.6f}")
    
        # 초기 loss logging
        if step == 0:
            logger.info(f"Initial Loss: {loss_value:.6f} | PDE Loss: {loss_pde:.6f} | Boundary Loss: {loss_bc:.6f}")
    
    # 모델 실행 결과 저장
    model_path = os.path.join(args.output_dir, "model_weights.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")
    print("Training completed!")
    return loss_value, loss_pde, loss_bc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN for Forced Convection problem.")

    # 학습 관련 인자
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--log-interval', type=int, default=100, help='Interval for printing loss.')
    parser.add_argument('--layers', type=int, default=6, help='Number of hidden layers in the neural network.')
    parser.add_argument('--neurons', type=int, default=64, help='Number of neurons per hidden layer.')

     # 물리 파라미터 인자
    parser.add_argument('--re', type=float, default=100.0, help='Reynolds number.')
    parser.add_argument('--pr', type=float, default=0.71, help='Prandtl number.')

    # 저장 인자
    parser.add_argument('--output-dir', type=str, default='./results/main_copy', help='Directory to save logs')

    args = parser.parse_args()
    run_main_copy_process(args)

        