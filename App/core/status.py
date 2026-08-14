"""Status gates for registered artifacts.

The UI must never turn file existence into an acceptance claim.  This module
keeps the four visible states and their reasons in one testable place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STATUSES = {"validated", "development_only", "not_available", "invalid"}


@dataclass(frozen=True)
class StatusResult:
    status: str
    reason: str
    gate_passed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "status_reason": self.reason, "gate_passed": self.gate_passed}


def evaluate_status(module: str, summary: dict[str, Any], required_exists: bool = True, html_valid: bool = True) -> StatusResult:
    if not summary or "_error" in summary:
        return StatusResult("invalid", "summary.json 无法读取")
    if not required_exists:
        return StatusResult("not_available", "注册表要求的文件缺失")
    if not html_valid:
        return StatusResult("invalid", "注册的 HTML 资源损坏或无法加载")

    if module == "hmi":
        gate = summary.get("quality_gate", {}).get("passed", False)
        scientific = str(summary.get("scientific_status", ""))
        timesteps = int(summary.get("total_timesteps", 0) or 0)
        if scientific == "demo_only" or not bool(gate) or timesteps < 10000:
            return StatusResult("development_only", "HMI 为 demo_only 或质量门禁未通过", False)
        return StatusResult("validated", "HMI 质量门禁通过", True)
    if module == "dt":
        passed = bool(summary.get("metrics", {}).get("validation_pass", False))
        return StatusResult("validated" if passed else "development_only", "留出观测空间验证通过" if passed else "DT 可运行但验证门禁未通过", passed)
    return StatusResult("validated", "注册产物和代表性指标可读取", True)


def badge_text(status: str) -> str:
    return {
        "validated": "已验证",
        "development_only": "开发/演示限定",
        "not_available": "未接入",
        "invalid": "无效",
    }.get(status, "未知状态")
