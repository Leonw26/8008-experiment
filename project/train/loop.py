import torch
import numpy as np
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from constants import (
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOSS_ALPHA,
    DEFAULT_LOSS_STRATEGY,
    DEFAULT_SERVICE_LEVEL_TARGET,
    DEFAULT_SERVICE_PENALTY_WEIGHT,
)
from data.category import get_category_name
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

def compute_service_penalty(
    y_pred_flat: torch.Tensor,
    true_demand_tensor: torch.Tensor,
    service_level_target: float,
    service_penalty_weight: float
):
    """
    使用可导的代理服务率惩罚低服务水平预测，抑制 PAO 收缩到零订货。

    这里使用直通估计器近似离散订货量：
    前向传播按整数订货量计算服务率，反向传播仍把梯度传回连续预测值，
    从而让训练阶段的惩罚与求解器的离散决策口径尽量一致。
    """
    safe_true_demand = true_demand_tensor.clamp_min(0.0)
    clipped_pred = y_pred_flat.clamp_min(0.0)
    discrete_order_proxy = clipped_pred + (torch.round(clipped_pred) - clipped_pred).detach()
    fulfilled_proxy = torch.minimum(discrete_order_proxy, safe_true_demand)
    total_demand = safe_true_demand.sum().clamp_min(1e-6)
    proxy_service = fulfilled_proxy.sum() / total_demand
    service_gap = F.relu(torch.tensor(service_level_target, device=y_pred_flat.device) - proxy_service)
    service_penalty = service_penalty_weight * service_gap.pow(2)
    return service_penalty, proxy_service.detach()

def build_global_constraints():
    """
    集中定义每个 batch 共享的运筹约束，便于训练与评估保持一致。
    """
    return GlobalConstraints(V_max=10000.0, B_total=50000.0)

def normalize_cost_params(cost_params_dict, batch_size: int):
    """
    将 DataLoader 返回的成本参数统一转换为 `List[SKUCostParams]`。
    """
    if isinstance(cost_params_dict, list) and cost_params_dict and isinstance(cost_params_dict[0], SKUCostParams):
        return cost_params_dict

    cost_params_list = []
    for i in range(batch_size):
        def safe_extract(key, default_val):
            val = cost_params_dict.get(key, [default_val] * batch_size)
            if isinstance(val, list):
                return val[i]
            if isinstance(val, torch.Tensor):
                return val[i].item()
            return val

        cost_params_list.append(
            SKUCostParams(
                item_id=safe_extract('item_id', f"item_{i}"),
                store_id=safe_extract('store_id', f"store_{i}"),
                c_h=float(safe_extract('c_h', 1.0)),
                c_u=float(safe_extract('c_u', 5.0)),
                c_f=float(safe_extract('c_f', 10.0)),
                v_i=float(safe_extract('v_i', 10.0)),
                p_i=float(safe_extract('p_i', 20.0))
            )
        )
    return cost_params_list

def build_context_array(cost_params_list):
    """
    从成本参数提取代理模型需要的上下文特征。
    """
    context_list = []
    for cp in cost_params_list:
        context_list.append([cp.c_h, cp.c_u, cp.c_f, cp.p_i, cp.v_i])
    return np.array(context_list, dtype=np.float32)

def init_cost_bucket():
    """
    初始化一组可累加的成本统计容器。
    """
    return {
        "total_cost": 0.0,
        "holding_cost": 0.0,
        "shortage_cost": 0.0,
        "order_cost": 0.0,
        "fulfilled_demand": 0.0,
        "true_demand": 0.0,
        "samples": 0,
    }

def update_cost_bucket(bucket, env_out, true_demand_np, mask=None):
    """
    将单个 batch 的成本结果累加到整体或分段桶中。
    """
    if mask is None:
        mask = slice(None)

    bucket["total_cost"] += float(np.sum(env_out.true_costs[mask]))
    bucket["holding_cost"] += float(np.sum(env_out.holding_costs[mask]))
    bucket["shortage_cost"] += float(np.sum(env_out.shortage_costs[mask]))
    bucket["order_cost"] += float(np.sum(env_out.order_costs[mask]))
    bucket["fulfilled_demand"] += float(np.sum(env_out.fulfilled_demand[mask]))
    bucket["true_demand"] += float(np.sum(true_demand_np[mask]))
    bucket["samples"] += int(np.sum(mask)) if not isinstance(mask, slice) else int(len(true_demand_np))

def finalize_cost_bucket(bucket):
    """
    将累加桶转换为最终展示指标，统一输出整体与分段结果。
    """
    true_demand = bucket["true_demand"]
    service_rate = bucket["fulfilled_demand"] / true_demand if true_demand > 0 else 1.0
    mean_cost = bucket["total_cost"] / max(bucket["samples"], 1)
    return {
        "samples": bucket["samples"],
        "cost": bucket["total_cost"],
        "mean_cost": mean_cost,
        "holding": bucket["holding_cost"],
        "shortage": bucket["shortage_cost"],
        "order": bucket["order_cost"],
        "service": service_rate,
    }

def print_evaluation_report(report: dict, split_name: str):
    """
    以可读格式打印整体与 segmentation 评估结果。
    """
    overall = report["overall"]
    print(f"\n========== {split_name.upper()} COST EVALUATION ==========")
    print(f"cost: {overall['cost']:.2f}")
    print(f"mean_cost: {overall['mean_cost']:.4f}")
    print(f"holding: {overall['holding']:.2f}")
    print(f"shortage: {overall['shortage']:.2f}")
    print(f"order: {overall['order']:.2f}")
    print(f"service: {overall['service']:.2%}")
    print(f"samples: {overall['samples']}")

    print("\n--- SEGMENTATION BREAKDOWN ---")
    for segment_name, metrics in report["segments"].items():
        print(
            f"{segment_name}: cost={metrics['cost']:.2f}, mean_cost={metrics['mean_cost']:.4f}, "
            f"holding={metrics['holding']:.2f}, shortage={metrics['shortage']:.2f}, "
            f"order={metrics['order']:.2f}, service={metrics['service']:.2%}, samples={metrics['samples']}"
        )

def evaluate_model(
    dataloader,
    predictor: DemandPredictor,
    solver: ABCASolver,
    env: InventoryEnvironment,
    device: torch.device,
    split_name: str = "test"
):
    """
    在统一成本口径下评估模型，输出整体 cost 与按 segment 的成本分解。
    """
    predictor.eval()
    overall_bucket = init_cost_bucket()
    segment_buckets = {}

    with torch.no_grad():
        for features, category_idx, true_demand, cost_params_batch in dataloader:
            features = features.to(device)
            category_idx_device = category_idx.to(device)
            y_pred_tensor = predictor(features, category_idx_device)

            y_pred_np = y_pred_tensor.detach().cpu().numpy().flatten()
            true_demand_np = true_demand.numpy().flatten()
            category_idx_np = category_idx.numpy().flatten()
            cost_params_list = normalize_cost_params(cost_params_batch, batch_size=len(y_pred_np))

            predictor_out = PredictorOutput(y_pred=y_pred_np)
            solver_out = solver.solve(predictor_out, cost_params_list, build_global_constraints())
            env_out = env.evaluate_cost(solver_out, true_demand_np, cost_params_list)

            update_cost_bucket(overall_bucket, env_out, true_demand_np)

            for segment_idx in np.unique(category_idx_np):
                segment_name = get_category_name(segment_idx)
                segment_buckets.setdefault(segment_name, init_cost_bucket())
                mask = (category_idx_np == segment_idx)
                update_cost_bucket(segment_buckets[segment_name], env_out, true_demand_np, mask=mask)

    predictor.train()
    report = {
        "overall": finalize_cost_bucket(overall_bucket),
        "segments": {name: finalize_cost_bucket(bucket) for name, bucket in segment_buckets.items()}
    }
    print_evaluation_report(report, split_name=split_name)
    return report

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
    service_level_target: float,
    service_penalty_weight: float,
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
        "service_level_target": service_level_target,
        "service_penalty_weight": service_penalty_weight,
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
    service_level_target: float = DEFAULT_SERVICE_LEVEL_TARGET,
    service_penalty_weight: float = DEFAULT_SERVICE_PENALTY_WEIGHT,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    eval_dataloader=None
):
    """
    执行 PAO 训练，并在测试集上统一输出 cost 与 segmentation 评估。
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
            service_level_target=service_level_target,
            service_penalty_weight=service_penalty_weight,
            grad_clip_norm=grad_clip_norm
        )
        wandb.init(
            project=exp_name,
            name=f"pao_epo{epochs}_bs{dataloader.batch_size}_{loss_strategy}_lr{learning_rate:.0e}",
            config=config
        )

    optimizer = Adam(predictor.parameters(), lr=learning_rate)
    history_y_pred = []
    history_context = []
    history_true_cost = []
    surrogate_update_freq = 200
    ema_total_loss = None
    ema_cost_loss = None
    ema_pred_loss = None

    for epoch in range(epochs):
        predictor.train()
        for batch_idx, (features, category_idx, true_demand, cost_params_batch) in enumerate(dataloader):
            features = features.to(device)
            category_idx = category_idx.to(device)
            true_demand_tensor = true_demand.to(device).view(-1)

            optimizer.zero_grad()
            y_pred_tensor = predictor(features, category_idx)
            y_pred_flat = y_pred_tensor.view(-1)
            y_pred_np = y_pred_flat.detach().cpu().numpy()
            true_demand_np = true_demand.numpy().flatten()
            cost_params_list = normalize_cost_params(cost_params_batch, batch_size=len(y_pred_np))
            context_np = build_context_array(cost_params_list)

            predictor_out = PredictorOutput(y_pred=y_pred_np)
            solver_out = solver.solve(predictor_out, cost_params_list, build_global_constraints())
            env_out = env.evaluate_cost(solver_out, true_demand_np, cost_params_list)
            true_costs_np = env_out.true_costs

            pred_loss, raw_mse_loss = compute_prediction_losses(y_pred_flat, true_demand_tensor)
            service_penalty, proxy_service = compute_service_penalty(
                y_pred_flat=y_pred_flat,
                true_demand_tensor=true_demand_tensor,
                service_level_target=service_level_target,
                service_penalty_weight=service_penalty_weight
            )
            cost_loss_value = float(np.mean(true_costs_np))
            effective_mode = "pao"

            history_y_pred.extend(y_pred_np)
            history_context.extend(context_np)
            history_true_cost.extend(true_costs_np)

            if len(history_y_pred) >= 1000 and batch_idx % surrogate_update_freq == 0:
                surrogate.train_surrogate(
                    np.array(history_y_pred),
                    np.array(history_context),
                    np.array(history_true_cost)
                )
                history_y_pred = history_y_pred[-20000:]
                history_context = history_context[-20000:]
                history_true_cost = history_true_cost[-20000:]

            if surrogate.is_trained:
                context_tensor = torch.tensor(context_np, dtype=torch.float32, device=device)
                cost_tensor = SurrogateAutogradFunction.apply(y_pred_tensor, context_tensor, surrogate)
                cost_loss = cost_tensor.mean()
                total_loss, _ = build_total_loss(
                    cost_loss=cost_loss,
                    pred_loss=pred_loss,
                    loss_strategy=loss_strategy,
                    loss_alpha=loss_alpha
                )
                total_loss = total_loss + service_penalty
                cost_loss_value = float(cost_loss.item())
            else:
                total_loss = pred_loss + service_penalty
                effective_mode = "pao_warmup"

            total_loss.backward()
            clip_grad_norm_(predictor.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            if ema_total_loss is None:
                ema_total_loss = total_loss.item()
                ema_cost_loss = cost_loss_value
                ema_pred_loss = pred_loss.item()
            else:
                ema_total_loss = 0.95 * ema_total_loss + 0.05 * total_loss.item()
                ema_cost_loss = 0.95 * ema_cost_loss + 0.05 * cost_loss_value
                ema_pred_loss = 0.95 * ema_pred_loss + 0.05 * pred_loss.item()

            if batch_idx % 50 == 0:
                mean_grad = getattr(surrogate, 'last_mean_abs_grad', 0.0)
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | Mode: {effective_mode} "
                    f"| Total Loss: {total_loss.item():.4f} | Cost Loss: {cost_loss_value:.4f} "
                    f"| Pred Loss: {pred_loss.item():.4f} | Raw MSE: {raw_mse_loss.item():.4f} "
                    f"| Service Penalty: {service_penalty.item():.4f} | Proxy Service: {proxy_service.item():.2%} "
                    f"| EMA Total: {ema_total_loss:.4f} | Mean True Cost: {true_costs_np.mean():.4f} "
                    f"| Mean Abs Grad: {mean_grad:.4f}"
                )

            if use_wandb:
                mean_grad = getattr(surrogate, 'last_mean_abs_grad', 0.0)
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_loss": total_loss.item(),
                    "cost_loss": cost_loss_value,
                    "pred_loss": pred_loss.item(),
                    "service_penalty": service_penalty.item(),
                    "proxy_service": proxy_service.item(),
                    "mse_loss": raw_mse_loss.item(),
                    "mean_true_cost": true_costs_np.mean(),
                    "mean_y_pred": y_pred_np.mean(),
                    "mean_abs_grad": mean_grad
                })

    evaluation_report = None
    if eval_dataloader is not None:
        evaluation_report = evaluate_model(
            dataloader=eval_dataloader,
            predictor=predictor,
            solver=solver,
            env=env,
            device=device,
            split_name="test"
        )

        if use_wandb:
            overall = evaluation_report["overall"]
            wandb.log({
                "test_cost": overall["cost"],
                "test_mean_cost": overall["mean_cost"],
                "test_holding": overall["holding"],
                "test_shortage": overall["shortage"],
                "test_order": overall["order"],
                "test_service": overall["service"]
            })

    if use_wandb:
        wandb.finish()

    print("DEBUG: Training loop completed successfully.")
    return evaluation_report
