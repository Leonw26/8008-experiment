import numpy as np
from typing import List
from interfaces import SKUCostParams, SolverOutput, EnvironmentOutput

class InventoryEnvironment:
    """
    [Step 4] 真实库存环境 (B同学负责)
    负责计算在真实的未知需求下，当前决策导致的真实业务成本 (Cost Function)
    """
    def __init__(self):
        pass
        
    def evaluate_cost(self, 
                      solver_out: SolverOutput, 
                      true_demand: np.ndarray, 
                      cost_params: List[SKUCostParams],
                      I_prev: np.ndarray = None) -> EnvironmentOutput:
        """
        计算真实的库存运营成本 (Holding Cost + Shortage Cost + Ordering Cost)
        
        参数:
            solver_out (SolverOutput): ABCA 求解器给出的订货量接口类
            true_demand (np.ndarray): 真实发生的需求量 D_it, 形状 (batch_size, )
            cost_params (List[SKUCostParams]): 各 SKU 的成本参数接口类
            I_prev (np.ndarray): 期初库存量, 默认为 0
            
        返回:
            EnvironmentOutput: 包含每个 SKU 真实总成本的接口类
        """
        Q_it = solver_out.Q_it
        n_items = len(Q_it)
        
        if I_prev is None:
            I_prev = np.zeros(n_items, dtype=np.float32)
            
        # 将 cost_params 转换为 numpy 数组以进行向量化计算
        c_h_arr = np.array([cp.c_h for cp in cost_params], dtype=np.float32)
        c_u_arr = np.array([cp.c_u for cp in cost_params], dtype=np.float32)
        c_f_arr = np.array([cp.c_f for cp in cost_params], dtype=np.float32)
        
        # 向量化计算
        # 真实期末库存 = 期初 + 订货 - 需求
        I_new = I_prev + Q_it - true_demand
        
        # 1. 积压量和缺货量
        overage = np.maximum(0, I_new)
        shortage = np.maximum(0, -I_new)
        
        # 2. 持有成本和缺货成本
        holding_cost = c_h_arr * overage
        shortage_cost = c_u_arr * shortage
        
        # 3. 订货成本: 只有当订货量大于0时才发生
        order_cost = np.where(Q_it > 0, c_f_arr, 0.0)
        
        # 4. 服务水平对应的实际满足需求量 (修正负数 bug：若可用库存为负，则无法满足任何新需求)
        fulfilled_demand = np.minimum(true_demand, np.maximum(0.0, I_prev + Q_it))

        # 5. 总成本
        costs = holding_cost + shortage_cost + order_cost
        
        return EnvironmentOutput(
            true_costs=costs.astype(np.float32),
            holding_costs=holding_cost.astype(np.float32),
            shortage_costs=shortage_cost.astype(np.float32),
            order_costs=order_cost.astype(np.float32),
            fulfilled_demand=fulfilled_demand.astype(np.float32),
            I_curr=I_new.astype(np.float32)
        )
