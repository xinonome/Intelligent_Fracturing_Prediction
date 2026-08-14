"""Adapter for the first-part expert/learning summaries."""

from __future__ import annotations

from typing import Any


class FSLLoader:
    def __init__(self, registry_loader) -> None:
        self.registry = registry_loader
        self.module = registry_loader.module("fsl")
        self.summary = registry_loader.summary("fsl")

    def metrics(self) -> dict[str, Any]:
        window = self.summary.get("window_results", {}).get("4", {})
        transfer = self.registry.module("fsl").get("supporting", {}).get("transfer", {})
        return {
            "binary_macro_f1": window.get("test_binary_macro_f1"),
            "macro_f1": window.get("test_two_stage_grouped_macro_f1"),
            "accuracy": window.get("test_two_stage_grouped_accuracy"),
            "direct_baseline": window.get("test_probability_fusion_macro_f1"),
            "transfer_before": transfer.get("transfer_query_before_finetune", {}).get("accuracy"),
            "transfer_after": transfer.get("transfer_query_after_finetune", {}).get("accuracy"),
            "test_scope": "段级划分；两阶段分类",
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "working_type": "unknown",
            "normal_probability": None,
            "abnormal_probability": None,
            "abnormal_type": "none",
            "rule_hits": [],
            "metrics": self.metrics(),
            "status": self.module.get("status", "not_available"),
        }
