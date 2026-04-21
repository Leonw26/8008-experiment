#!/usr/bin/env python3
"""
批量运行 PAO 实验，并按精简格式打印测试集成本结果。
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def build_experiments() -> List[Dict[str, object]]:
    """
    定义本次需要执行的 PAO 实验矩阵。
    """
    common_args = [
        "--epochs", "5",
        "--learning_rate", "0.0005",
        "--service_penalty_weight", "50",
        "--service_level_target", "0.9",
    ]
    return [
        {
            "name": "baseline",
            "label": "BASELINE",
            "use_segmentation": False,
            "args": [*common_args],
        },
        {
            "name": "segmented",
            "label": "SEGMENTED",
            "use_segmentation": True,
            "args": [*common_args, "--use_segmentation"],
        },
    ]


def parse_metrics(stdout: str) -> Dict[str, object]:
    """
    从训练输出中解析整体指标和 segmentation breakdown。
    """
    overall_patterns = {
        "cost": r"^cost:\s*([0-9.+-eE]+)",
        "mean_cost": r"^mean_cost:\s*([0-9.+-eE]+)",
        "holding": r"^holding:\s*([0-9.+-eE]+)",
        "shortage": r"^shortage:\s*([0-9.+-eE]+)",
        "order": r"^order:\s*([0-9.+-eE]+)",
        "service": r"^service:\s*([0-9.+-eE]+)%",
        "samples": r"^samples:\s*([0-9]+)",
    }
    metrics: Dict[str, object] = {}
    for key, pattern in overall_patterns.items():
        match = re.search(pattern, stdout, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"未在输出中找到指标: {key}")
        value = match.group(1)
        metrics[key] = int(value) if key == "samples" else float(value)

    segments: Dict[str, Dict[str, float]] = {}
    segment_pattern = re.compile(
        r"^(Smooth|Erratic|Intermittent|Lumpy):\s*"
        r"cost=([0-9.+-eE]+),\s*"
        r"mean_cost=([0-9.+-eE]+),\s*"
        r"holding=([0-9.+-eE]+),\s*"
        r"shortage=([0-9.+-eE]+),\s*"
        r"order=([0-9.+-eE]+),\s*"
        r"service=([0-9.+-eE]+)%,\s*"
        r"samples=([0-9]+)$",
        flags=re.MULTILINE,
    )
    for match in segment_pattern.finditer(stdout):
        segments[match.group(1)] = {
            "cost": float(match.group(2)),
            "mean_cost": float(match.group(3)),
            "holding": float(match.group(4)),
            "shortage": float(match.group(5)),
            "order": float(match.group(6)),
            "service": float(match.group(7)),
            "samples": int(match.group(8)),
        }

    metrics["segments"] = segments
    return metrics


def run_experiment(repo_root: Path, output_dir: Path, experiment: Dict[str, object]) -> Dict[str, object]:
    """
    使用 `uv run python` 执行单组实验，并返回解析后的结果。
    """
    command = ["uv", "run", "python", "project/main.py", *experiment["args"]]
    
    # 使用 Popen 实时读取输出，以便在控制台显示进度
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    if process.stdout:
        for line in process.stdout:
            print(line, end="")  # 实时打印进度到控制台
            output_lines.append(line)

    process.wait()
    full_output = "".join(output_lines)

    log_path = output_dir / "logs" / f"{experiment['name']}.log"
    log_path.write_text(full_output, encoding="utf-8")

    result: Dict[str, object] = {
        "name": experiment["name"],
        "label": experiment["label"],
        "use_segmentation": experiment["use_segmentation"],
        "command": " ".join(command),
        "return_code": process.returncode,
        "log_path": str(log_path),
    }

    if process.returncode == 0:
        try:
            result.update(parse_metrics(full_output))
        except Exception as e:
            result["error"] = f"解析指标失败: {e}"
    else:
        result["error"] = "命令执行失败，请查看日志。"

    return result


def render_simple_result(result: Dict[str, object]) -> str:
    """
    将单个实验结果渲染成截图风格的精简文本。
    """
    if result["return_code"] != 0:
        return (
            f"--- {result['label']} ---\n"
            f"error: command failed ({result['return_code']})\n"
            f"log: {result['log_path']}"
        )
    return (
        f"--- {result['label']} ---\n"
        f"cost: {result['cost']:.2f}\n"
        f"holding: {result['holding']:.2f}\n"
        f"shortage: {result['shortage']:.2f}\n"
        f"service: {result['service']:.2f}%"
    )


def render_console_report(results: List[Dict[str, object]]) -> str:
    """
    将全部实验结果渲染成控制台输出文本。
    """
    return "\n\n".join(render_simple_result(result) for result in results)


def main() -> None:
    """
    创建输出目录，运行 PAO 实验，并打印精简结果。
    """
    repo_root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = repo_root / "results" / f"pao_only_{timestamp}"
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    results = []
    experiment_plan = build_experiments()
    for experiment in experiment_plan:
        print(f"Running experiment: {experiment['name']}")
        result = run_experiment(repo_root=repo_root, output_dir=output_dir, experiment=experiment)
        results.append(result)
        print(f"Finished: {experiment['name']} (return_code={result['return_code']})")

    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "experiment_plan.json").write_text(
        json.dumps(experiment_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + render_console_report(results))


if __name__ == "__main__":
    main()
