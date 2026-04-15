# End-to-End Predict-and-Optimize (端到端决策学习框架)

本项目实现了一个基于“预测-优化”流水线的端到端库存决策学习框架。通过神经网络进行需求预测，ABCA 启发式算法进行运筹求解，并利用代理模型把不可导的业务成本转换为可传导的梯度，让训练目标从单纯缩小预测误差转向尽量降低最终库存运营成本。

## 本次修改的最终目的

这次修改的重点不是继续单独追求更小的 `loss`，而是统一用业务成本回答以下问题：

1. `PAO` 在统一成本口径下能否真正降低最终 `cost`。
2. 加入 `segmentation` 后，整体成本和各分段成本是否发生变化。
3. 即使某个模型训练损失更小，它在 `holding / shortage / order / service` 上是否真的更优。

## 启动训练

可以通过命令行运行训练脚本：

```bash
python project/main.py [参数...]
```

### 命令行参数说明

| 参数                 |  类型   |     默认值     | 说明                                                                           |
| :------------------- | :-----: | :------------: | :----------------------------------------------------------------------------- |
| `--epochs`           |  `int`  |      `5`       | 训练的 Epoch 数量。建议先用 `1` 或 `2` 做连通性测试。                          |
| `--report_to`        |  `str`  |     `none`     | 是否上报到 `wandb`。                                                           |
| `--exp_name`         |  `str`  |     `8008`     | 实验名。                                                                       |
| `--penalty_coef`     | `float` |     `0.0`      | 缺货额外惩罚系数。设为 `0.0` 时，对齐参考 notebook 的基础成本口径。            |
| `--loss_strategy`    |  `str`  | `balanced_sum` | PAO 模式下总损失的组合策略。                                                   |
| `--loss_alpha`       | `float` |     `0.5`      | PAO 模式下预测损失在总损失中的权重。                                           |
| `--grad_clip_norm`   | `float` |     `1.0`      | 梯度裁剪阈值。                                                                 |
| `--use_segmentation` |  flag   |    `False`     | 是否启用 ADI/CV2 分段特征。加上该参数后，模型会把 segment embedding 作为输入。 |

## 推荐实验方式

当前仓库仅保留 `PAO` 训练入口，建议至少跑下面 2 组实验：

```bash
# 1) PAO，不加 segmentation
python project/main.py --epochs 3

# 2) PAO，加 segmentation
python project/main.py --epochs 3 --use_segmentation
```

如果你想严格对齐别人给的参考设定，建议继续保持：

```bash
python project/main.py --epochs 3 --penalty_coef 0.0
```

## 评估输出怎么看

训练结束后，程序会自动在测试集上输出统一的成本评估结果：

- `cost`: 总成本，最核心指标，越低越好。
- `mean_cost`: 样本平均成本，便于不同批量或不同设置横向比较。
- `holding`: 持有成本，反映库存积压。
- `shortage`: 缺货成本，反映供给不足。
- `order`: 固定订货成本，反映下单频率的代价。
- `service`: 服务水平，表示满足需求的比例，越高越好。

同时还会输出 `SEGMENTATION BREAKDOWN`，按 `Smooth / Erratic / Intermittent / Lumpy` 分段展示各自的成本与服务水平。这样就能直接观察：

- 为什么某次 `segmentation` 后整体 `cost` 反而上升。
- 是哪一类 SKU 拉高了 `holding` 或 `shortage`。
- `PAO` 的收益是否只集中在某几个 segment 上。

## 报告撰写建议

写结果表时，优先把下面几项放到正文：

- `test cost`
- `test service`
- `holding / shortage / order`
- `segmentation breakdown`

写结论时，建议用“成本优先”的口径，而不是只看预测误差，例如：

- `PAO` 的训练损失不一定最低，但如果 `test cost` 更低，它在业务上就更优。
- `segmentation` 如果让 `shortage` 明显下降，即使 `holding` 略升，也可能是合理取舍。
- 若 `segmentation` 后总成本上升，需要进一步看是哪一个 segment 的补货策略变差。
