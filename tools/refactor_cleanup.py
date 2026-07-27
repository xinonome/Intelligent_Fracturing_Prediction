"""Generate the cleanup manifest and remove explicitly retired project assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DELETE_DIRS = [
    "__pycache__", ".pytest_cache", "active_run_data", "archive", "pkn_time_output", "Reports", "App/runs",
    "Data/public_beijing_air", "Data/knowledge_graph", "Data/processed",
    "FSL-Expert/runs", "FSL-Expert/time_series_forecasting", "FSL-Expert/report_figures",
    "FSL-Expert/knowledge_graph/full_book_qwen_output_dryrun",
    "FSL-Expert/knowledge_graph/full_book_qwen_output_dryrun_start14",
    "FSL-Expert/knowledge_graph/full_book_qwen_output/page_cache",
    "DT-Crack/full_dataset", "DT-Crack/陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271",
    "DT-Crack/active_run_data", "DT-Crack/examples", "DT-Crack/time_series_forecasting",
    "DT-Crack/dataset_picture_cleaned", "DT-Crack/image_dataset_working",
    "DT-Crack/full_dataset_working", "DT-Crack/image_dataset_working", "DT-Crack/picture",
    "DT-Crack/runs", "DT-Crack/full_dataset_output", "DT-Crack/image_dataset_output",
    "DT-Crack/src", "DT-Crack/scripts", "DT-Crack/data_processing", "DT-Crack/online_correction",
    "DT-Crack/physics_learning", "DT-Crack/inversion_demo", "DT-Crack/docs",
    "HMI-KE/runs",
]

DELETE_FILES = [
    "prepare_code_release.ps1", "prepare_fsl_github_release.ps1", "App/run_app_pyside.bat", "App/run_app_pyside.ps1",
    "Data/multimodal/陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271.zip",
    "Data/multimodal/陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271/hybrid_auto/陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271_layout.pdf",
    "Data/multimodal/陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271/hybrid_auto/陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271_middle.json",
    "DT-Crack/main.py", "DT-Crack/run_dt_pipeline.py", "DT-Crack/example.py", "DT-Crack/pkn4.py",
    "DT-Crack/fracture_animation.py", "DT-Crack/generate_architecture_diagrams.py",
    "DT-Crack/HANDOVER.md", "DT-Crack/chinese_font.py", "DT-Crack/复现.md", "DT-Crack/dt_crack_api.py",
    "FSL-Expert/第三部分_基于GNN的压裂工况标签识别阶段总结.md",
    "FSL-Expert/accuracy_90_metric_options.md", "FSL-Expert/analyze_frac_gnn_predictions.py",
    "FSL-Expert/analyze_frac_gnn_thresholds.py", "FSL-Expert/analyze_working_type_transitions.py",
    "FSL-Expert/augment_gnn_report_summary.py", "FSL-Expert/build_gnn_ppt_addendum.py",
    "FSL-Expert/build_gnn_report_section.py", "FSL-Expert/build_gnn_report_section_rich.py",
    "FSL-Expert/generate_report_figures.py", "FSL-Expert/GNN_压裂数据集训练与划分说明.md",
    "FSL-Expert/plot_frac_gnn_results.py", "FSL-Expert/plot_seed1_multiclass_metrics.py",
    "FSL-Expert/plot_segment_timeseries_with_labels.py", "FSL-Expert/run_fsl_pipeline.py",
    "FSL-Expert/search_split_for_normal_f1.py", "FSL-Expert/split_search_interim_summary.md",
    "FSL-Expert/summarize_accuracy_metric_options.py", "FSL-Expert/train_frac_lgbm.py",
    "FSL-Expert/train_hierarchical_working_type_lgbm.py",
    "FSL-Expert/tune_multiclass_normal_f1_grid.py", "FSL-Expert/tune_multiclass_normal_threshold_best.py",
    "FSL-Expert/tune_normal_f1_threshold.py", "FSL-Expert/utf8_test.py", "FSL-Expert/write_ppt_script.py",
    "FSL-Expert/knowledge_graph/便签数据1(已自动还原).xlsx",
    "FSL-Expert/knowledge_graph/项目介绍and系统架构.pptx",
    "FSL-Expert/knowledge_graph/build_graphml_vs_qwen_comparison.py",
    "FSL-Expert/knowledge_graph/build_kg_comparison_report.py", "FSL-Expert/knowledge_graph/build_kg.py",
    "FSL-Expert/knowledge_graph/export_excel_windows.py", "FSL-Expert/knowledge_graph/export_triples.py",
    "FSL-Expert/knowledge_graph/full_book_qwen_usage.md", "FSL-Expert/knowledge_graph/graphml_qwen_demo_usage.md",
    "FSL-Expert/knowledge_graph/graphml_vs_qwen_full_book_comparison.md",
    "FSL-Expert/knowledge_graph/graphml_vs_qwen_full_book_speech.md",
    "FSL-Expert/knowledge_graph/knowledge_graph_comparison_speech.md",
    "FSL-Expert/knowledge_graph/knowledge_graph_version_comparison.md",
    "FSL-Expert/docs/grouped_working_type_model_comparison.md",
    "FSL-Expert/rule_fusion/一种基于多规则融合的压裂施工工况智能标注方法.docx",
    "FSL-Expert/rule_fusion/专利补充材料-泵序优化.doc",
    "HMI-KE/run_hmi_demo.py", "HMI-KE/run_realtime_monitoring_agent_demo.py",
    "HMI-KE/run_seconds_policy_agent_demo.py", "DT-Crack/run_demo.ps1",
    "HMI-KE/simulator/generate_fsl_real_scenario_visualization.py",
    "HMI-KE/simulator/generate_scenario_visualization.py",
    "HMI-KE/docs/泵序约束与奖励函数说明.md", "HMI-KE/docs/第三部分_技术路线与执行计划.md",
    "HMI-KE/docs/秒点策略智能体框架计划.md",
]


def checked(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise RuntimeError(f"Cleanup path escaped project root: {path}")
    return path


def directory_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest() -> dict:
    entries = []
    for relative in DELETE_DIRS:
        path = checked(relative)
        count, size = directory_size(path)
        entries.append({"action": "delete_directory", "path": relative, "exists": path.exists(), "files": count, "size_bytes": size})
    for relative in DELETE_FILES:
        path = checked(relative)
        entries.append({
            "action": "delete_file", "path": relative, "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256(path) if path.exists() else None,
        })
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "policy": "explicit allowlist cleanup; Data/raw_frac and Data/3Dfrac are protected",
        "representative_artifacts": ["artifacts/fsl", "artifacts/dt", "artifacts/hmi"],
        "entries": entries,
        "planned_delete_bytes": sum(entry["size_bytes"] for entry in entries),
    }


def apply_cleanup(manifest: dict) -> None:
    protected = [(ROOT / "Data" / "raw_frac").resolve(), (ROOT / "Data" / "3Dfrac").resolve()]
    for entry in manifest["entries"]:
        path = checked(entry["path"])
        if any(path == item or path in item.parents for item in protected):
            raise RuntimeError(f"Attempted to delete protected data path: {path}")
        if not path.exists():
            continue
        if entry["action"] == "delete_directory":
            shutil.rmtree(path)
        else:
            path.unlink()
    for cache in ROOT.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache)
    for log in ROOT.rglob("*.log"):
        if ROOT in log.resolve().parents:
            log.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    manifest_path = ROOT / "cleanup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.apply:
        apply_cleanup(manifest)
    print(json.dumps({"manifest": str(manifest_path), "apply": args.apply, "planned_delete_gb": manifest["planned_delete_bytes"] / 1024**3}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
