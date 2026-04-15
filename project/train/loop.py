import torch
import numpy as np
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from constants import DEFAULT_GRAD_CLIP_NORM, DEFAULT_LEARNING_RATE, DEFAULT_LOSS_ALPHA, DEFAULT_LOSS_STRATEGY
from model.lstm import DemandPredictor
from solver.abca import ABCASolver
from environment.inventory import InventoryEnvironment
from surrogate.model import SurrogateModel, SurrogateAutogradFunction
from interfaces import SKUCostParams, GlobalConstraints, PredictorOutput

def build_total_loss(cost_loss: torch.Tensor, pred_loss: torch.Tensor, loss_strategy: str, loss_alpha: float):
    """
    根据选择的策略构造总损失，并返回可单独记录的各项 loss。

    方案说明:
        weighted_sum: 直接优化 cost_loss + alpha * pred_loss
        balanced_sum: 先按当前 batch 的量级归一化，再做加权求和
    """
    if loss_strategy == "weighted_sum":
        total_loss = cost_loss + loss_alpha * pred_loss
        aux_metrics = {
            "scaled_cost_loss": cost_loss.detach(),
            "scaled_pred_loss": (loss_alpha * pred_loss).detach()
        }
        return total_loss, aux_metrics

    if loss_strategy == "balanced_sum":
        cost_scale = cost_loss.detach().abs().clamp_min(1.0)
        pred_scale = pred_loss.detach().abs().clamp_min(1.0)
        normalized_cost_loss = cost_loss / cost_scale
        normalized_pred_loss = pred_loss / pred_scale
        total_loss = normalized_cost_loss + loss_alpha * normalized_pred_loss
        aux_metrics = {
            "scaled_cost_loss": normalized_cost_loss.detach(),
            "scaled_pred_loss": (loss_alpha * normalized_pred_loss).detach()
        }
        return total_loss, aux_metrics

    raise ValueError(f"Unsupported loss_strategy: {loss_strategy}")

def compute_prediction_losses(y_pred_flat: torch.Tensor, true_demand_tensor: torch.Tensor):
    """
    计算更稳健的预测损失，并保留原始 MSE 作为对照监控指标。

    说明:
        pred_loss: 训练时实际使用的 log1p + Huber 损失，降低长尾样本对梯度的冲击
        raw_mse_loss: 原始需求空间上的 MSE，仅用于观测模型是否仍受极端值影响
    """
    safe_true_demand = true_demand_tensor.clamp_min(0.0)
    pred_log = torch.log1p(y_pred_flat)
    true_log = torch.log1p(safe_true_demand)
    pred_loss = F.huber_loss(pred_log, true_log, delta=1.0)
    raw_mse_loss = F.mse_loss(y_pred_flat, safe_true_demand)
    return pred_loss, raw_mse_loss

def build_wandb_config(
    dataloader,
    predictor: DemandPredictor,
    solver: ABCASolver,
    env: InventoryEnvironment,
    surrogate: SurrogateModel,
    epochs: int,
    device: torch.device,
    exp_name: str,
    learning_rate: float,
    loss_strategy: str,
    loss_alpha: float,
    grad_clip_norm: float
):
    """
    汇总关键实验配置，便于在 wandb 中复现实验设置。
    """
    dataset = getattr(dataloader, "dataset", None)
    surrogate_regressor = getattr(surrogate, "model", None)
    surrogate_params = surrogate_regressor.get_params() if hasattr(surrogate_regressor, "get_params") else {}

    config = {
        "exp_name": exp_name,
        "epochs": epochs,
        "batch_size": getattr(dataloader, "batch_size", None),
        "device": str(device),
        "learning_rate": learning_rate,
        "loss_strategy": loss_strategy,
        "loss_alpha": loss_alpha,
        "grad_clip_norm": grad_clip_norm,
        "dataset_mode": getattr(dataset, "mode", None),
        "seq_len": getattr(dataset, "seq_len", None),
        "penalty_coef": getattr(dataset, "penalty_coef", None),
        "predictor_model": predictor.__class__.__name__,
        "predictor_hidden_size": getattr(predictor, "hidden_size", None),
        "predictor_num_layers": getattr(predictor, "num_layers", None),
        "predictor_use_category_embedding": getattr(predictor, "use_category_embedding", None),
        "solver_model": solver.__class__.__name__,
        "solver_max_iter": getattr(solver, "max_iter", None),
        "solver_pop_size": getattr(solver, "pop_size", None),
        "solver_limit": getattr(solver, "limit", None),
        "environment_model": env.__class__.__name__,
        "surrogate_model": surrogate_regressor.__class__.__name__ if surrogate_regressor is not None else surrogate.__class__.__name__,
        "surrogate_max_iter": surrogate_params.get("max_iter"),
        "surrogate_learning_rate": surrogate_params.get("learning_rate"),
        "surrogate_max_depth": surrogate_params.get("max_depth"),
        "surrogate_max_leaf_nodes": surrogate_params.get("max_leaf_nodes"),
        "surrogate_min_samples_leaf": surrogate_params.get("min_samples_leaf"),
        "surrogate_l2_regularization": surrogate_params.get("l2_regularization"),
        "surrogate_random_state": surrogate_params.get("random_state")
    }
    return {key: value for key, value in config.items() if value is not None}

def train_predict_and_optimize(
    dataloader, 
    predictor: DemandPredictor, 
    solver: ABCASolver, 
    env: InventoryEnvironment, 
    surrogate: SurrogateModel,
    epochs: int = 10,
    device: torch.device = torch.device('cpu'),
    report_to: str = "wandb",
    exp_name: str = "8008",
    learning_rate: float = DEFAULT_LEARNING_RATE,
    loss_strategy: str = DEFAULT_LOSS_STRATEGY,
    loss_alpha: float = DEFAULT_LOSS_ALPHA,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM
):
    """
    [Step 6] 端到端 Predict-and-Optimize 训练循环 (C同学负责)
    负责串联 A(Predictor)、B(Solver+Env)、C(Surrogate) 的大动脉
    """
    use_wandb = (report_to.lower() == "wandb")
    if use_wandb:
        import wandb
        config = build_wandb_config(
            dataloader=dataloader,
            predictor=predictor,
            solver=solver,
            env=env,
            surrogate=surrogate,
            epochs=epochs,
            device=device,
            exp_name=exp_name,
            learning_rate=learning_rate,
            loss_strategy=loss_strategy,
            loss_alpha=loss_alpha,
            grad_clip_norm=grad_clip_norm
        )
        wandb.init(
            project=exp_name,
            name=f"run_epo{epochs}_bs{dataloader.batch_size}_{loss_strategy}_lr{learning_rate:.0e}",
            config=config
        )
        
    # 降低学习率，防止在剧烈波动的梯度中“翻车”
    optimizer = Adam(predictor.parameters(), lr=learning_rate)
    
    # 历史缓冲池，用于训练 Surrogate Model
    history_y_pred = []
    history_context = []
    history_true_cost = []
    
    # 记录上次训练代理模型时的损失表面，避免频繁重建
    surrogate_update_freq = 200
    ema_total_loss = None
    ema_cost_loss = None
    ema_pred_loss = None
    
    for epoch in range(epochs):
        for batch_idx, (features, category_idx, true_demand, cost_params_dict) in enumerate(dataloader):
            
            # 将特征数据移动到计算设备 (CUDA/MPS/CPU)
            features = features.to(device)
            category_idx = category_idx.to(device)
            
            # ==========================================================
            # [Step 2] 前向预测 (A)
            # ==========================================================
            optimizer.zero_grad()
            y_pred_tensor = predictor(features, category_idx)  # 预测未来的需求 y_it
            
            # 包装为接口类，注意从设备转移到 CPU
            y_pred_np = y_pred_tensor.detach().cpu().numpy().flatten()
            predictor_out = PredictorOutput(y_pred=y_pred_np)
            true_demand_np = true_demand.numpy().flatten()
            
            # 模拟：将 DataLoader 中取出的字典转为强类型的 DataClass
            # 实际场景中 DataLoader 可能直接返回字典，需要在此处转换
            cost_params_list = []
            
            # 兼容 DataLoader collate_fn 传出的两种格式: 字典(DummyDataset) 或 字典列表(RealDataset)
            if isinstance(cost_params_dict, list) and isinstance(cost_params_dict[0], SKUCostParams):
                cost_params_list = cost_params_dict
            else:
                for i in range(len(y_pred_np)):
                    # 为了应对 DataLoader 批量合并出来的 tensor/list，进行安全提取
                    def safe_extract(key, default_val):
                        val = cost_params_dict.get(key, [default_val] * len(y_pred_np))
                        if isinstance(val, list):
                            return val[i]
                        if isinstance(val, torch.Tensor):
                            return val[i].item()
                        return val
                    
                    cp = SKUCostParams(
                        item_id=safe_extract('item_id', f"item_{i}"),
                        store_id=safe_extract('store_id', f"store_{i}"),
                        c_h=float(safe_extract('c_h', 1.0)),
                        c_u=float(safe_extract('c_u', 5.0)),
                        c_f=float(safe_extract('c_f', 10.0)),
                        v_i=float(safe_extract('v_i', 10.0)),
                        p_i=float(safe_extract('p_i', 20.0))
                    )
                    cost_params_list.append(cp)
            
            # 提取 context (c_h, c_u, c_f, p_i, v_i) 用于训练代理模型
            context_list = []
            for cp in cost_params_list:
                context_list.append([cp.c_h, cp.c_u, cp.c_f, cp.p_i, cp.v_i])
            context_np = np.array(context_list, dtype=np.float32)
            
            # ==========================================================
            # [Step 3] 求解决策 (B)
            # ==========================================================
            # 定义全局约束
            global_constraints = GlobalConstraints(V_max=10000.0, B_total=50000.0)
            
            # ABCA 求解器给出最优订货量 (返回 SolverOutput 接口类)
            solver_out = solver.solve(predictor_out, cost_params_list, global_constraints)
            
            # ==========================================================
            # [Step 4] 真实成本评估 (B)
            # ==========================================================
            # 在真实的环境中，评估该决策 Q_it 会产生多少实际的 Cost
            env_out = env.evaluate_cost(solver_out, true_demand_np, cost_params_list)
            true_costs_np = env_out.true_costs

            
            # 收集数据以训练代理模型
            history_y_pred.extend(y_pred_np)
            history_context.extend(context_np)
            history_true_cost.extend(true_costs_np)
            
            # ==========================================================
            # [Step 5 & 6] 代理模型拟合与反向传播 (C)
            # ==========================================================
            # 每隔 200 个 batch 才更新一次 Surrogate Model (防止目标函数频繁漂移导致神经网络梯度崩溃)
            if len(history_y_pred) >= 1000 and batch_idx % surrogate_update_freq == 0:
                surrogate.train_surrogate(
                    np.array(history_y_pred), 
                    np.array(history_context), 
                    np.array(history_true_cost)
                )
                
                # 大幅扩大滑动窗口，让代理模型有足够长期的记忆，防止灾难性遗忘
                history_y_pred = history_y_pred[-20000:]
                history_context = history_context[-20000:]
                history_true_cost = history_true_cost[-20000:]
                
            if surrogate.is_trained:
                # 关键步骤：使用自定义的 Autograd Function 桥接 PyTorch 和 LightGBM
                # 这样一来，Cost 就能被求导并回传给 y_pred_tensor
                context_tensor = torch.tensor(context_np, dtype=torch.float32, device=device)
                cost_tensor = SurrogateAutogradFunction.apply(y_pred_tensor, context_tensor, surrogate)
                
                # 计算基于端到端决策的业务成本损失
                cost_loss = cost_tensor.mean()
                
                # 使用对长尾需求更稳健的 log1p + Huber 作为预测误差损失。
                true_demand_tensor = true_demand.to(device).view(-1)
                y_pred_flat = y_pred_tensor.view(-1)
                pred_loss, raw_mse_loss = compute_prediction_losses(y_pred_flat, true_demand_tensor)
                
                # 将业务成本损失和预测损失拆开构造，便于切换训练方案和单独监控。
                total_loss, _ = build_total_loss(
                    cost_loss=cost_loss,
                    pred_loss=pred_loss,
                    loss_strategy=loss_strategy,
                    loss_alpha=loss_alpha
                )
                
                # 反向传播 (更新 A同学 的神经网络)
                total_loss.backward()
                clip_grad_norm_(predictor.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

                if ema_total_loss is None:
                    ema_total_loss = total_loss.item()
                    ema_cost_loss = cost_loss.item()
                    ema_pred_loss = pred_loss.item()
                else:
                    ema_total_loss = 0.95 * ema_total_loss + 0.05 * total_loss.item()
                    ema_cost_loss = 0.95 * ema_cost_loss + 0.05 * cost_loss.item()
                    ema_pred_loss = 0.95 * ema_pred_loss + 0.05 * pred_loss.item()
                
                if batch_idx % 50 == 0:
                    # 提取并打印 surrogate_grad 的绝对值均值 (Mean Abs Grad)
                    mean_grad = getattr(surrogate, 'last_mean_abs_grad', 0.0)
                    print(
                        f"Epoch {epoch} | Batch {batch_idx} | Total Loss: {total_loss.item():.4f} "
                        f"| Cost Loss: {cost_loss.item():.4f} | Pred Loss: {pred_loss.item():.4f} "
                        f"| Raw MSE: {raw_mse_loss.item():.4f} | EMA Total: {ema_total_loss:.4f} "
                        f"| Mean Abs Grad: {mean_grad:.4f} | Strategy: {loss_strategy}"
                    )
                    
                if use_wandb:
                    mean_grad = getattr(surrogate, 'last_mean_abs_grad', 0.0)
                    wandb.log({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_loss": total_loss.item(),
                        "cost_loss": cost_loss.item(),
                        "pred_loss": pred_loss.item(),
                        "mse_loss": raw_mse_loss.item(),
                        "mean_true_cost": true_costs_np.mean(),
                        "mean_y_pred": y_pred_np.mean(),
                        "mean_abs_grad": mean_grad
                    })
            else:
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Collecting data to warmup Surrogate...")
                
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "mean_true_cost": true_costs_np.mean(),
                        "mean_y_pred": y_pred_np.mean()
                    })
                    
            
    if use_wandb:
        wandb.finish()
        
    print("DEBUG: Training loop completed successfully.")
