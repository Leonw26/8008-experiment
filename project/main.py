import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EXP_NAME,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOSS_ALPHA,
    DEFAULT_LOSS_STRATEGY,
    DEFAULT_PENALTY_COEF,
    DEFAULT_SERVICE_LEVEL_TARGET,
    DEFAULT_SERVICE_PENALTY_WEIGHT,
    DEFAULT_SOLVER_MAX_ITER,
    DEFAULT_SOLVER_POP_SIZE,
    HISTORY_WINDOW_WEEKS,
)
from data.dataset import get_dataloader
from model.lstm import DemandPredictor
from solver.abca import ABCASolver
from environment.inventory import InventoryEnvironment
from surrogate.model import SurrogateModel
from train.loop import train_predict_and_optimize

def main():
    """
    Predict-and-Optimize 框架全局入口文件
    负责初始化所有模块，并支持 PAO 训练和测试集成本评估。
    """
    parser = argparse.ArgumentParser(description="End-to-End Predict-and-Optimize Training")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help=f'epochs数量 (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--report_to', type=str, default='none', help='是否上报指标到特定平台，如 "wandb"')
    parser.add_argument('--exp_name', type=str, default=DEFAULT_EXP_NAME, help=f'实验/项目名称，当 report_to=wandb 时作为 wandb project name (default: {DEFAULT_EXP_NAME})')
    parser.add_argument('--penalty_coef', type=float, default=DEFAULT_PENALTY_COEF, help=f'缺货额外惩罚系数，基于售价的倍数；设为 0.0 时对齐 baseline 成本口径 (default: {DEFAULT_PENALTY_COEF})')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help=f'预测模型学习率 (default: {DEFAULT_LEARNING_RATE})')
    parser.add_argument(
        '--loss_strategy',
        type=str,
        default=DEFAULT_LOSS_STRATEGY,
        choices=['weighted_sum', 'balanced_sum'],
        help='loss方案: weighted_sum 或 balanced_sum'
    )
    parser.add_argument('--loss_alpha', type=float, default=DEFAULT_LOSS_ALPHA, help=f'预测损失的权重系数 (default: {DEFAULT_LOSS_ALPHA})')
    parser.add_argument('--service_level_target', type=float, default=DEFAULT_SERVICE_LEVEL_TARGET, help=f'PAO 训练时的目标服务率阈值，低于该值会触发惩罚 (default: {DEFAULT_SERVICE_LEVEL_TARGET})')
    parser.add_argument('--service_penalty_weight', type=float, default=DEFAULT_SERVICE_PENALTY_WEIGHT, help=f'PAO 训练时服务率惩罚项的权重 (default: {DEFAULT_SERVICE_PENALTY_WEIGHT})')
    parser.add_argument('--grad_clip_norm', type=float, default=DEFAULT_GRAD_CLIP_NORM, help=f'参数梯度裁剪阈值 (default: {DEFAULT_GRAD_CLIP_NORM})')
    parser.add_argument('--use_segmentation', action='store_true', help='是否启用 segmentation 特征（按 ADI/CV2 分类后嵌入模型）')
    parser.add_argument('--subset_nrows', type=int, default=3049, help='用于快速调试，只加载前 N 个 SKU 数据，默认 3049 (约十分之一)')
    args = parser.parse_args()

    print("Initializing Project Components...")
    
    # 检测运行设备: 强制降级使用 CPU，消除 PAO 架构中频繁的 CPU-GPU 张量拷贝通信开销
    device = torch.device("cpu")
    print(f"DEBUG: Using device: {device} (Forced for M2 Optimization)")
    
    # 1. 初始化 DataLoader
    # 使用固定历史窗口长度，保持数据预处理和模型定义一致。
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(base_dir), 'dataset')
    dataloader = get_dataloader(
        data_path=dataset_path,
        batch_size=DEFAULT_BATCH_SIZE,
        penalty_coef=args.penalty_coef,
        seq_len=HISTORY_WINDOW_WEEKS,
        mode='train',
        subset_nrows=args.subset_nrows
    )
    eval_dataloader = get_dataloader(
        data_path=dataset_path,
        batch_size=DEFAULT_BATCH_SIZE,
        penalty_coef=args.penalty_coef,
        seq_len=HISTORY_WINDOW_WEEKS,
        mode='test',
        subset_nrows=args.subset_nrows
    )
    print(f"DEBUG: Train DataLoader length: {len(dataloader)}")
    print(f"DEBUG: Eval DataLoader length: {len(eval_dataloader)}")
    
    # 2. 初始化预测模型 (A同学负责)
    # 每个时间步只输入当天销量，因此 input_size=1，序列长度由 DataLoader 控制。
    predictor = DemandPredictor(
        input_size=1,
        hidden_size=64,
        output_size=1,
        use_category_embedding=args.use_segmentation
    ).to(device)
    
    # 3. 初始化求解器与环境 (B同学负责)
    solver = ABCASolver(max_iter=DEFAULT_SOLVER_MAX_ITER, pop_size=DEFAULT_SOLVER_POP_SIZE)
    env = InventoryEnvironment()
    
    # 4. 初始化代理模型 (C同学负责)
    surrogate = SurrogateModel()
    
    print(f"Starting End-to-End Predict-and-Optimize Training Loop for {args.epochs} epochs...")
    train_predict_and_optimize(
        dataloader=dataloader, 
        predictor=predictor, 
        solver=solver, 
        env=env, 
        surrogate=surrogate, 
        epochs=args.epochs,
        device=device,
        report_to=args.report_to,
        exp_name=args.exp_name,
        learning_rate=args.learning_rate,
        loss_strategy=args.loss_strategy,
        loss_alpha=args.loss_alpha,
        service_level_target=args.service_level_target,
        service_penalty_weight=args.service_penalty_weight,
        grad_clip_norm=args.grad_clip_norm,
        eval_dataloader=eval_dataloader
    )

if __name__ == "__main__":
    main()
