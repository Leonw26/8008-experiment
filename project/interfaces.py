from dataclasses import dataclass
import numpy as np

@dataclass
class SKUCostParams:
    """
    单个 SKU 的业务与成本参数 (由 A 同学在 dataset 中组装，B 同学在 solver/env 中使用)
    对应 Walmart M5 真实数据的经济学转换
    """
    item_id: str          # 商品唯一标识，例如 "HOBBIES_1_001"
    store_id: str         # 门店标识，例如 "CA_1"
    
    c_h: float            # 单位持有成本 (Holding Cost)。例如: 基于采购价 p_i 乘以 20% 年化持有率分摊到周
    c_u: float            # 单位缺货成本 (Underage/Shortage Cost)。例如: 销售价格 sell_price 减去采购价 p_i 的毛利
    c_f: float            # 固定订货成本 (Fixed Ordering Cost)。每次下单(Q>0)触发的固定物流/管理费，如 10.0
    
    v_i: float            # 单位商品体积 (Volume)。用于计算仓储空间占用，由类别(cat_id)推算
    p_i: float            # 采购价格 (Procurement Price)。用于计算预算消耗

@dataclass
class GlobalConstraints:
    """
    全局运筹约束 (由 main 或 train loop 定义，B 同学在 solver 中使用)
    控制多 SKU 联合决策时的物理与资金边界
    """
    V_max: float          # 仓库最大可用容积 (Storage Capacity Constraint)
    B_total: float        # 单次补货的全局总预算 (Budget Constraint)

@dataclass
class PredictorOutput:
    """
    A 同学预测模型的输出 (A -> B 的接口)
    """
    y_pred: np.ndarray    # 预测需求量。形状: (batch_size, )。数据类型: float32

@dataclass
class SolverOutput:
    """
    B 同学运筹求解器的输出 (B -> Env 的接口)
    """
    Q_it: np.ndarray      # 最优订货量决策。形状: (batch_size, )。数据类型: int32 (离散值)

@dataclass
class EnvironmentOutput:
    """
    B 同学业务环境的评估输出 (B -> C 的接口)
    """
    true_costs: np.ndarray      # 在真实需求下的总成本。形状: (batch_size, )。数据类型: float32
    holding_costs: np.ndarray   # 持有成本分量。形状: (batch_size, )。数据类型: float32
    shortage_costs: np.ndarray  # 缺货成本分量。形状: (batch_size, )。数据类型: float32
    order_costs: np.ndarray     # 固定订货成本分量。形状: (batch_size, )。数据类型: float32
    fulfilled_demand: np.ndarray # 满足的需求量。形状: (batch_size, )。数据类型: float32
