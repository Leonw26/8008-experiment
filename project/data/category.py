import numpy as np
import pandas as pd

CATEGORY_LABELS = {
    0: "Smooth",
    1: "Erratic",
    2: "Intermittent",
    3: "Lumpy",
}

def compute_adi(x):
    """
    计算平均需求间隔 (Average Demand Interval)
    """
    non_zero = (x > 0).sum()
    return len(x) / non_zero if non_zero > 0 else np.nan

def compute_cv2(x):
    """
    计算需求变异系数的平方 (Squared Coefficient of Variation)
    """
    mean = x.mean()
    std = x.std()
    return (std / mean) ** 2 if mean > 0 else np.nan

def classify_type(adi, cv2):
    """
    根据 ADI 和 CV2 对时间序列进行分类：
    0: smooth (平稳需求)
    1: erratic (不稳定需求)
    2: intermittent (间歇性需求)
    3: lumpy (块状需求)
    """
    if pd.isna(adi) or pd.isna(cv2):
        return 3 # default to lumpy
    if adi < 1.32 and cv2 < 0.49:
        return 0 # "smooth"
    elif adi < 1.32 and cv2 >= 0.49:
        return 1 # "erratic"
    elif adi >= 1.32 and cv2 < 0.49:
        return 2 # "intermittent"
    else:
        return 3 # "lumpy"

def get_category_name(category_idx):
    """
    将类别索引转换为可读的分段名称，便于做 segmentation 成本报表。
    """
    return CATEGORY_LABELS.get(int(category_idx), "Unknown")
