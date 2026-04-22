import numpy as np
from typing import List
from interfaces import SKUCostParams, GlobalConstraints, PredictorOutput, SolverOutput
from numba import njit

@njit(fastmath=True, cache=True)
def numba_evaluate_cost(Q, demand, I_prev, c_h, c_u, c_f, v_i, p_i, V_max, B_total):
    """
    评估某个解的成本 (适应度函数，越小越好)
    包括：持有成本(h)、缺货成本(u)、固定订货成本(f) 以及 违反全局约束的惩罚项
    """
    I_new = I_prev + Q - demand
    holding_cost = np.sum(c_h * np.maximum(0.0, I_new))
    shortage_cost = np.sum(c_u * np.maximum(0.0, -I_new))
    order_cost = np.sum(c_f * (Q > 0))
    total_cost = holding_cost + shortage_cost + order_cost
    
    total_volume = np.sum(Q * v_i)
    total_budget = np.sum(Q * p_i)
    
    penalty = 0.0
    if total_volume > V_max:
        penalty += 1000.0 * (total_volume - V_max)
    if total_budget > B_total:
        penalty += 1000.0 * (total_budget - B_total)
        
    return total_cost + penalty

@njit(fastmath=True, cache=True)
def numba_discretize_solution(raw_solution):
    """将连续订货量稳定地离散化为非负整数。"""
    clipped = np.maximum(0.0, raw_solution)
    integer_part = np.floor(clipped)
    fractional_part = clipped - integer_part
    round_up_mask = np.random.rand(clipped.shape[0]) < fractional_part
    return (integer_part + round_up_mask).astype(np.int32)

@njit(fastmath=True, cache=True)
def numba_abca_solve(y_pred, I_prev, demand, c_h, c_u, c_f, v_i, p_i, V_max, B_total, 
                     pop_size, max_iter, limit, seed_solutions):
    """Numba 加速的人工蜂群主循环"""
    n_items = len(y_pred)
    population = np.zeros((pop_size, n_items), dtype=np.int32)
    fitness = np.zeros(pop_size)
    trials = np.zeros(pop_size, dtype=np.int32)
    
    safe_y_pred = y_pred.copy()
    for i in range(n_items):
        if np.isnan(safe_y_pred[i]):
            safe_y_pred[i] = 0.0
            
    num_seeds = seed_solutions.shape[0]
    for i in range(pop_size):
        if i < num_seeds:
            population[i] = seed_solutions[i]
        else:
            sampled_solution = np.zeros(n_items, dtype=np.float64)
            for j in range(n_items):
                safe_scale = max(0.5, safe_y_pred[j] * 0.35)
                sampled_solution[j] = np.random.normal(safe_y_pred[j], safe_scale)
            population[i] = numba_discretize_solution(sampled_solution)

        fitness[i] = numba_evaluate_cost(population[i], demand, I_prev, c_h, c_u, c_f, v_i, p_i, V_max, B_total)
        
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_cost = fitness[best_idx]
    
    for _ in range(max_iter):
        # 2. 雇佣蜂阶段 (Employed Bees)
        for i in range(pop_size):
            partner_idx = np.random.randint(0, pop_size - 1)
            if partner_idx >= i:
                partner_idx += 1
                
            j = np.random.randint(0, n_items)
            phi = np.random.uniform(-1.0, 1.0)
            
            new_solution = population[i].copy()
            new_solution[j] = new_solution[j] + int(phi * (population[i][j] - population[partner_idx][j]))
            new_solution[j] = max(0, new_solution[j])
            
            new_cost = numba_evaluate_cost(new_solution, demand, I_prev, c_h, c_u, c_f, v_i, p_i, V_max, B_total)
            
            if new_cost < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_cost
                trials[i] = 0
            else:
                trials[i] += 1
                
        # 3. 观察蜂阶段 (Onlooker Bees)
        fit_values = 1.0 / (1.0 + fitness)
        probs = fit_values / np.sum(fit_values)
        
        t = 0
        i = 0
        while t < pop_size:
            if np.random.rand() < probs[i]:
                t += 1
                partner_idx = np.random.randint(0, pop_size - 1)
                if partner_idx >= i:
                    partner_idx += 1
                    
                j = np.random.randint(0, n_items)
                phi = np.random.uniform(-1.0, 1.0)
                
                new_solution = population[i].copy()
                new_solution[j] = new_solution[j] + int(phi * (population[i][j] - population[partner_idx][j]))
                new_solution[j] = max(0, new_solution[j])
                
                new_cost = numba_evaluate_cost(new_solution, demand, I_prev, c_h, c_u, c_f, v_i, p_i, V_max, B_total)
                
                if new_cost < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_cost
                    trials[i] = 0
                else:
                    trials[i] += 1
            
            i = (i + 1) % pop_size
            
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_cost:
            best_cost = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()
            
        # 4. 侦查蜂阶段 (Scout Bees)
        for i in range(pop_size):
            if trials[i] >= limit:
                sampled_solution = np.zeros(n_items, dtype=np.float64)
                for j in range(n_items):
                    safe_scale = max(0.75, safe_y_pred[j] * 0.5)
                    sampled_solution[j] = np.random.normal(safe_y_pred[j], safe_scale)
                population[i] = numba_discretize_solution(sampled_solution)
                fitness[i] = numba_evaluate_cost(population[i], demand, I_prev, c_h, c_u, c_f, v_i, p_i, V_max, B_total)
                trials[i] = 0

    return best_solution

class ABCASolver:
    """
    [Step 3] 人工蜂群运筹求解器 (B同学负责)
    基于预测的需求量 y_pred 和 全局约束，搜索最优的订单量 Q_it
    """
    def __init__(self, max_iter: int = 10, pop_size: int = 10, limit: int = 5):
        # 显著调小默认参数，防止在端到端训练的每个 Batch 中耗时过长卡住
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.limit = limit  # 侦查蜂重置阈值
        
    def _evaluate_cost(self, Q: np.ndarray, y_pred: np.ndarray, I_prev: np.ndarray, global_constraints: GlobalConstraints, D_true: np.ndarray = None) -> float:
        """
        评估某个解的成本 (适应度函数，越小越好)
        包括：持有成本(h)、缺货成本(u)、固定订货成本(f) 以及 违反全局约束的惩罚项
        """
        # 业务成本计算 (向量化提速)
        # 预测的期末库存 = I_prev + Q - y_pred
        # 如果提供了 D_true，则使用真实需求计算库存流转 (和 baseline 对齐)
        demand = D_true if D_true is not None else y_pred
        I_new = I_prev + Q - demand
        
        holding_cost = np.sum(self._c_h * np.maximum(0, I_new))
        shortage_cost = np.sum(self._c_u * np.maximum(0, -I_new))
        order_cost = np.sum(self._c_f * (Q > 0))
        total_cost = holding_cost + shortage_cost + order_cost
        
        # 资源消耗计算
        total_volume = np.sum(Q * self._v_i)
        total_budget = np.sum(Q * self._p_i)
        
        # 惩罚项 (罚函数法处理约束)
        penalty = 0.0
        if total_volume > global_constraints.V_max:
            penalty += 1000 * (total_volume - global_constraints.V_max)
        if total_budget > global_constraints.B_total:
            penalty += 1000 * (total_budget - global_constraints.B_total)
            
        return total_cost + penalty

    def _discretize_solution(self, raw_solution: np.ndarray) -> np.ndarray:
        """
        将连续订货量稳定地离散化为非负整数。

        这里使用随机舍入而不是直接 `astype(np.int32)` 截断，
        避免 0 < y_pred < 1 时所有候选解都被压成 0，导致搜索塌缩。
        """
        clipped = np.clip(raw_solution, a_min=0.0, a_max=None)
        integer_part = np.floor(clipped).astype(np.int32)
        fractional_part = clipped - integer_part
        round_up_mask = np.random.rand(*clipped.shape) < fractional_part
        return integer_part + round_up_mask.astype(np.int32)

    def _build_seed_population(self, y_pred: np.ndarray) -> List[np.ndarray]:
        """
        为初始种群构造几个具有业务意义的起点，降低零订货塌缩风险。
        """
        safe_y_pred = np.nan_to_num(y_pred, nan=0.0)
        positive_mask = safe_y_pred > 0

        zero_solution = np.zeros_like(safe_y_pred, dtype=np.int32)
        floor_solution = np.floor(np.clip(safe_y_pred, a_min=0.0, a_max=None)).astype(np.int32)
        ceil_solution = np.ceil(np.clip(safe_y_pred, a_min=0.0, a_max=None)).astype(np.int32)
        unit_probe_solution = positive_mask.astype(np.int32)

        return [
            zero_solution,
            floor_solution,
            ceil_solution,
            unit_probe_solution,
        ]

    def solve(self, 
              predictor_out: PredictorOutput, 
              cost_params: List[SKUCostParams], 
              global_constraints: GlobalConstraints,
              I_prev: np.ndarray = None,
              D_true: np.ndarray = None) -> SolverOutput:
        """
        求解最优订货量 Q_it
        
        参数:
            predictor_out (PredictorOutput): A 同学预测输出的数据类
            cost_params (List[SKUCostParams]): 包含各 SKU 成本参数的列表, 与 y_pred 对应
            global_constraints (GlobalConstraints): 全局约束，如 V_max 和 B_total
            I_prev (np.ndarray): 期初库存量, 默认为 0
            D_true (np.ndarray): 真实需求量 (用于对齐 baseline，评估决策效果)
            
        返回:
            SolverOutput: 包含求解出的最优离散订货量的接口类
        """
        y_pred = predictor_out.y_pred
        n_items = len(y_pred)
        
        if I_prev is None:
            I_prev = np.zeros(n_items, dtype=np.float32)
            
        # 预提取参数为 Numpy 数组，大幅加速后续评估
        self._c_h = np.array([p.c_h for p in cost_params])
        self._c_u = np.array([p.c_u for p in cost_params])
        self._c_f = np.array([p.c_f for p in cost_params])
        self._v_i = np.array([p.v_i for p in cost_params])
        self._p_i = np.array([p.p_i for p in cost_params])
        
        # --- ABCA 核心逻辑 ---
        
        # 1. 初始化种群
        # 初始解以预测值为中心随机扰动
        population = np.zeros((self.pop_size, n_items), dtype=np.int32)
        fitness = np.zeros(self.pop_size)
        trials = np.zeros(self.pop_size, dtype=np.int32)
        
        safe_y_pred = np.nan_to_num(y_pred, nan=0.0)
        seed_solutions = self._build_seed_population(safe_y_pred)
        seed_solutions_array = np.array(seed_solutions, dtype=np.int32)
        
        demand = D_true if D_true is not None else y_pred
        
        best_solution = numba_abca_solve(
            y_pred, I_prev, demand, 
            self._c_h, self._c_u, self._c_f, self._v_i, self._p_i, 
            float(global_constraints.V_max), float(global_constraints.B_total),
            self.pop_size, self.max_iter, self.limit, seed_solutions_array
        )

        Q_it = best_solution
        
        return SolverOutput(Q_it=Q_it)
