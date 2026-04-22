import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import (
    DEFAULT_FALLBACK_SELL_PRICE,
    DEFAULT_PENALTY_COEF,
    HISTORY_WINDOW_WEEKS,
    PROFIT_MARGIN_RATE,
)
from interfaces import SKUCostParams
from data.category import compute_adi, compute_cv2, classify_type

class M5InventoryDataset(Dataset):
    """
    M5 沃尔玛库存数据集加载器
    负责读取特征并返回: (features, true_demand, cost_params)
    """
    def __init__(self, data_path: str, mode: str = 'train', seq_len: int = HISTORY_WINDOW_WEEKS, penalty_coef: float = DEFAULT_PENALTY_COEF, horizon: int = 13, subset_nrows: int = None):
        """
        Args:
            data_path: dataset 文件夹路径 (例如: '../dataset')
            mode: 'train' 或 'test'
            seq_len: 历史窗口长度 (默认使用过去 28 天的销量作为特征)
            penalty_coef: 缺货额外惩罚系数 (基于售价的倍数)，默认 0.0 以对齐 baseline 成本口径
            horizon: 预测与模拟的未来天数 (时间循环的长度)
            subset_nrows: 用于快速调试，只加载前 N 行数据
        """
        self.data_path = data_path
        self.mode = mode
        self.seq_len = seq_len
        self.penalty_coef = penalty_coef
        self.horizon = horizon
        self.subset_nrows = subset_nrows
        
        # 1. 加载核心销售数据
        sales_path = os.path.join(data_path, 'sales_train_evaluation.csv')
        price_path = os.path.join(data_path, 'sell_prices.csv')
        calendar_path = os.path.join(data_path, 'calendar.csv')
        
        if not os.path.exists(sales_path):
            raise FileNotFoundError(f"未找到销售数据文件: {sales_path}")
            
        print(f"Loading M5 dataset... {'(Subset: ' + str(subset_nrows) + ' rows)' if subset_nrows else 'This might take a while.'}")
        self.sales_df = pd.read_csv(sales_path, nrows=subset_nrows)
        self.prices_df = pd.read_csv(price_path)
        calendar = pd.read_csv(calendar_path)
        
        # 按照 baseline 逻辑，将日级别数据聚合为周级别数据
        print("Aggregating daily data into weekly data to align with baseline...")
        id_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        d_cols = [col for col in self.sales_df.columns if col.startswith('d_')]
        d_to_wk = calendar.set_index('d')['wm_yr_wk'].to_dict()
        
        sales_weekly_wide = self.sales_df[id_cols + d_cols].set_index(id_cols)
        # 映射列名为周
        sales_weekly_wide.columns = [d_to_wk[c] for c in sales_weekly_wide.columns]
        # 按周求和
        sales_weekly_wide = sales_weekly_wide.T.groupby(level=0).sum().T.reset_index()
        self.sales_df = sales_weekly_wide
        
        # 获取所有表示周数的列名 (形如 11101, 11102...)
        # 为了统一处理，将这些周列重命名为 w_1, w_2, ...
        wk_cols = sorted([c for c in self.sales_df.columns if isinstance(c, int) or (isinstance(c, str) and c.isdigit())])
        wk_rename_map = {wk: f'w_{i+1}' for i, wk in enumerate(wk_cols)}
        self.sales_df.rename(columns=wk_rename_map, inplace=True)
        self.d_cols = [f'w_{i+1}' for i in range(len(wk_cols))]
        
        # 定义特征和目标的时间窗口
        # M5 一共有 282 周，baseline 测试集为最后 13 周，验证集为倒数 26 到 13 周
        # 简单起见，训练集我们取倒数第 26 周作为目标起始周，测试集取倒数第 13 周作为目标起始周
        total_weeks = len(self.d_cols)
        if mode == 'train':
            self.target_wk_idx = total_weeks - 26 # 倒数第26周
        else:
            self.target_wk_idx = total_weeks - 13 # 倒数第13周 (测试集起点)
            
        # 计算该 SKU 在目标周的平均价格，用于模拟业务参数
        # (真实情况下，应该按照 item_id, store_id, wm_yr_wk 去 sell_prices 查，这里做简化映射)
        print("Pre-calculating average prices per SKU-Store...")
        self.avg_prices = self.prices_df.groupby(['item_id', 'store_id'])['sell_price'].mean().to_dict()

        print("Classifying data types based on ADI and CV2...")
        # Calculate ADI and CV2 for each SKU based on historical sales
        def calculate_category(row_sales):
            adi = compute_adi(row_sales)
            cv2 = compute_cv2(row_sales)
            return classify_type(adi, cv2)
            
        # We only use historical sales columns for this calculation
        sales_only = self.sales_df[self.d_cols].values
        # Using numpy array calculation for speed instead of pandas apply
        categories = []
        for i in range(sales_only.shape[0]):
            categories.append(calculate_category(sales_only[i]))

        # 使用 copy() 来避免 PerformanceWarning (DataFrame is highly fragmented)
        self.sales_df = self.sales_df.copy()
        self.sales_df['category_idx'] = categories

        print(f"Dataset initialized with {len(self.sales_df)} SKUs.")

    def __len__(self) -> int:
        return len(self.sales_df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SKUCostParams]:
        """
        获取单条(单个 SKU 或一个 Batch SKU)的数据
        返回: (features, category_idx, true_demand, cost_params)
        """
        row = self.sales_df.iloc[idx]
        
        item_id = row['item_id']
        store_id = row['store_id']
        cat_id = row['cat_id']
        category_idx = torch.tensor(row['category_idx'], dtype=torch.long)
        
        # 1. 提取输入特征 (过去 seq_len 周的销量)
        feature_start_idx = self.target_wk_idx - self.seq_len
        feature_end_idx = self.target_wk_idx
        feature_cols = [f'w_{i}' for i in range(feature_start_idx + 1, feature_end_idx + 1)]
        
        # 将过去 seq_len 周销量重塑为真正的时间序列: (seq_len,) -> (seq_len, 1)
        features_np = row[feature_cols].values.astype(np.float32)
        features = torch.tensor(features_np, dtype=torch.float32).unsqueeze(-1)
        
        # 2. 提取真实需求 (目标周开始的 horizon 周的销量)
        target_cols = [f'w_{i}' for i in range(self.target_wk_idx + 1, self.target_wk_idx + 1 + self.horizon)]
        true_demand_np = row[target_cols].values.astype(np.float32)
        true_demand = torch.tensor(true_demand_np, dtype=torch.float32)
        
        # 3. 构造业务成本参数 (优先对齐 baseline notebook 的统一成本口径)
        # 获取该 SKU 在该门店的历史平均价格 (作为售价)
        sell_price = self.avg_prices.get((item_id, store_id), DEFAULT_FALLBACK_SELL_PRICE)
        
        # 进货价(成本) = 售价 * (1 - 利润率)
        p_i = sell_price * (1 - PROFIT_MARGIN_RATE)
        
        # 持仓成本 = (售价 * 年库存成本率) / 52周，对齐参考 notebook 中的 c_h 定义
        c_h = sell_price * 0.20 / 52
        
        # 缺货成本(c_u): 统一采用 sell_price * 0.35，不再用 penalty
        c_u = sell_price * 0.35
        
        # 固定订货成本
        c_f = 5.0
        
        # 假设体积 v_i 根据类别简单映射
        v_i = 1.0
        if 'HOBBIES' in cat_id:
            v_i = 0.5
        elif 'HOUSEHOLD' in cat_id:
            v_i = 2.0
        elif 'FOODS' in cat_id:
            v_i = 0.8
            
        cost_params = SKUCostParams(
            item_id=item_id,
            store_id=store_id,
            c_h=float(c_h),
            c_u=float(c_u),
            c_f=float(c_f),
            v_i=float(v_i),
            p_i=float(p_i)
        )
        
        return features, category_idx, true_demand, cost_params

def get_dataloader(
    data_path: str,
    batch_size: int = 32,
    mode: str = 'train',
    penalty_coef: float = DEFAULT_PENALTY_COEF,
    seq_len: int = HISTORY_WINDOW_WEEKS,
    shuffle: bool = None,
    subset_nrows: int = None
) -> DataLoader:
    """
    获取 DataLoader
    为了避免在单测中加载整个真实数据集太慢，如果在 data_path 找不到真实文件，
    我们将回退到一个简单的 DummyDataset。
    """
    sales_path = os.path.join(data_path, 'sales_train_evaluation.csv')
    # 自定义 collate_fn 因为 DataLoader 默认无法直接把 List[SKUCostParams] 给 stack 起来
    def custom_collate(batch):
        features = torch.stack([item[0] for item in batch])
        category_idx = torch.stack([item[1] for item in batch])
        demands = torch.stack([item[2] for item in batch])
        # 将多个 dataclass 直接放在列表中
        cost_params_list = [item[3] for item in batch]
        return features, category_idx, demands, cost_params_list

    # 删除 DummyDataset，确保真实数据不存在时直接报错
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"【严重错误】未找到真实数据集文件: {sales_path}。请确保已下载并放置了正确的 M5 数据集 (sales_train_evaluation.csv 等)。")

    dataset = M5InventoryDataset(data_path, mode=mode, seq_len=seq_len, penalty_coef=penalty_coef, subset_nrows=subset_nrows)
    effective_shuffle = (mode == 'train') if shuffle is None else shuffle
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=effective_shuffle, collate_fn=custom_collate)
