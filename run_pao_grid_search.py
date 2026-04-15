#!/usr/bin/env python3
"""
对 PAO 做一轮小范围网格调参，并将结果保存为可查询文件。
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class GridConfig:
    """
    描述一组需要执行的 PAO 超参数组合。
    """

    learning_rate: float
    service_penalty_weight: float
    service_level_target: float
    epochs: int = 5
    use_segmentation: bool = False

    def build_args(self) -> List[str]:
        """
        将配置转换为 `project/main.py` 可直接消费的命令行参数。
        """
        args = [
            "--epochs", str(self.epochs),
            "--learning_rate", str(self.learning_rate),
            "--service_penalty_weight", str(self.service_penalty_weight),
            "--service_level_target", str(self.service_level_target),
        ]
        if self.use_segmentation:
            args.append("--use_segmentation")
        return args

    def slug(self) -> str:
        """
        生成稳定且适合文件名的配置标识。
        """
        seg_part = "seg" if self.use_segmentation else "base"
        lr_part = str(self.learning_rate).replace(".", "p")
        weight_part = str(self.service_penalty_weight).replace(".", "p")
        target_part = str(self.service_level_target).replace(".", "p")
        return f"{seg_part}_lr{lr_part}_w{weight_part}_t{target_part}"


def build_grid() -> List[GridConfig]:
    """
    定义本轮小网格，优先探索能抑制塌缩且不过度推高成本的区间。
    """
    learning_rates = [5e-4, 1e-3]
    service_penalty_weights = [50.0, 100.0]
    service_level_targets = [0.80, 0.90]
    grid: List[GridConfig] = []
    for learning_rate in learning_rates:
        for service_penalty_weight in service_penalty_weights:
            for service_level_target in service_level_targets:
                grid.append(
                    GridConfig(
                        learning_rate=learning_rate,
                        service_penalty_weight=service_penalty_weight,
                        service_level_target=service_level_target,
                    )
                )
    return grid


def parse_metrics(stdout: str) -> Dict[str, float]:
    """
    从训练输出中提取测试集整体指标。
    """
    patterns = {
        "cost": r"^cost:\s*([0-9.+-eE]+)",
        "mean_cost": r"^mean_cost:\s*([0-9.+-eE]+)",
        "holding": r"^holding:\s*([0-9.+-eE]+)",
        "shortage": r"^shortage:\s*([0-9.+-eE]+)",
        "order": r"^order:\s*([0-9.+-eE]+)",
        "service": r"^service:\s*([0-9.+-eE]+)%",
        "samples": r"^samples:\s*([0-9]+)",
    }
    metrics: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"未在输出中找到指标: {key}")
        raw_value = match.group(1)
        metrics[key] = int(raw_value) if key == "samples" else float(raw_value)
    return metrics


def score_result(result: Dict[str, object]) -> float:
    """
    给单次实验打分，优先排除塌缩，再兼顾成本和服务率。
    """
    if result["return_code"] != 0:
        return float("inf")
    service = float(result["service"])
    cost = float(result["cost"])
    collapse_penalty = 1_000_000.0 if service <= 0.0 else 0.0
    low_service_penalty = max(0.0, 10.0 - service) * 10_000.0
    return collapse_penalty + low_service_penalty + cost


def run_experiment(repo_root: Path, output_dir: Path, config: GridConfig) -> Dict[str, object]:
    """
    执行单组 PAO 配置，保存原始日志并返回解析后的结构化结果。
    """
    command = ["uv", "run", "python", "project/main.py", *config.build_args()]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    log_path = output_dir / "logs" / f"{config.slug()}.log"
    log_path.write_text(completed.stdout + "\n\n[stderr]\n" + completed.stderr, encoding="utf-8")

    result: Dict[str, object] = {
        **asdict(config),
        "name": config.slug(),
        "command": " ".join(command),
        "return_code": completed.returncode,
        "log_path": str(log_path),
    }
    if completed.returncode == 0:
        result.update(parse_metrics(completed.stdout))
    else:
        result["error"] = completed.stderr.strip() or "命令执行失败，请查看日志。"
    result["score"] = score_result(result)
    return result


def write_csv(results: List[Dict[str, object]], csv_path: Path) -> None:
    """
    将全部实验结果写入 CSV，方便后续筛选与排序。
    """
    fieldnames = [
        "rank",
        "name",
        "learning_rate",
        "service_penalty_weight",
        "service_level_target",
        "epochs",
        "use_segmentation",
        "return_code",
        "cost",
        "mean_cost",
        "holding",
        "shortage",
        "order",
        "service",
        "samples",
        "score",
        "log_path",
        "command",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = {key: result.get(key, "") for key in fieldnames}
            row["rank"] = rank
            writer.writerow(row)


def render_console_report(results: List[Dict[str, object]]) -> str:
    """
    渲染控制台摘要，优先展示前几名可用配置。
    """
    lines = []
    for result in results[:5]:
        if result["return_code"] != 0:
            lines.append(
                f"{result['name']}: failed (rc={result['return_code']})"
            )
            continue
        lines.append(
            f"{result['name']}: cost={result['cost']:.2f}, "
            f"service={result['service']:.2f}%, order={result['order']:.2f}, "
            f"score={result['score']:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    """
    运行整轮小网格实验，并将排序后的结果保存到时间戳目录。
    """
    repo_root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = repo_root / "results" / f"pao_grid_{timestamp}"
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    results: List[Dict[str, object]] = []
    for index, config in enumerate(grid, start=1):
        print(
            f"[{index}/{len(grid)}] Running {config.slug()} "
            f"(lr={config.learning_rate}, weight={config.service_penalty_weight}, "
            f"target={config.service_level_target})"
        )
        result = run_experiment(repo_root=repo_root, output_dir=output_dir, config=config)
        results.append(result)
        print(
            f"[{index}/{len(grid)}] Finished {config.slug()} "
            f"(rc={result['return_code']}, cost={result.get('cost', 'n/a')}, "
            f"service={result.get('service', 'n/a')})"
        )

    ranked_results = sorted(
        results,
        key=lambda item: (
            item["return_code"] != 0,
            float(item["score"]),
            float(item.get("cost", float("inf"))),
            -float(item.get("service", 0.0)),
        ),
    )

    (output_dir / "summary.json").write_text(
        json.dumps(ranked_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(ranked_results, output_dir / "ranking.csv")

    best_result = ranked_results[0] if ranked_results else {}
    (output_dir / "best_config.json").write_text(
        json.dumps(best_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTop Candidates")
    print(render_console_report(ranked_results))


if __name__ == "__main__":
    main()
