from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = PROJECT_ROOT / "DT-Crack" / "陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271" / "hybrid_auto"
BOOK_JSON = BOOK_DIR / "陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271_content_list_v2.json"
OUTPUT_DIR = PROJECT_ROOT / "智能标注" / "rule_fusion"

KEYWORDS = [
    "砂堵",
    "堵塞",
    "砂桥",
    "泵压",
    "压力",
    "排量",
    "砂比",
    "加砂",
    "停泵",
    "放喷",
    "酸液",
    "隔离液",
    "沉砂",
]


def iter_text(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "content" and isinstance(value, str):
                yield value
            else:
                yield from iter_text(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_text(item)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def short_snippet(text: str, keyword: str, width: int = 90) -> str:
    index = text.find(keyword)
    if index < 0:
        return text[: width * 2]
    start = max(0, index - width)
    end = min(len(text), index + width)
    return text[start:end]


def extract_candidates(max_page: int = 63) -> list[dict]:
    data = json.loads(BOOK_JSON.read_text(encoding="utf-8"))
    candidates = []
    for page_no, page in enumerate(data[:max_page], start=1):
        text = normalize_text("".join(iter_text(page)))
        if not text:
            continue
        score = sum(text.count(keyword) for keyword in KEYWORDS)
        matched = [keyword for keyword in KEYWORDS if keyword in text]
        if not matched:
            continue
        first_keyword = min(matched, key=lambda keyword: text.find(keyword))
        candidates.append(
            {
                "page_no": page_no,
                "keyword_score": score,
                "matched_keywords": "、".join(matched),
                "snippet": short_snippet(text, first_keyword),
            }
        )
    return candidates


def fused_rules() -> list[dict]:
    """Hand-curated rules fused from Smart_Annotation3.py and pages <= 63 of the book."""

    return [
        {
            "rule_id": "SP-R01",
            "rule_name": "压力快速上升强触发砂堵",
            "source": "Smart_Annotation3.py + 书籍砂堵案例",
            "book_pages": "15, 18, 21",
            "condition": "在加砂或试挤过程中，施工压力/泵压快速升高，达到限压或接近限压，并伴随超压停泵。",
            "data_expression": "delta_pressure >= 10 MPa 且 pressure_slope >= 1.5 MPa/min，可作为强砂堵风险触发。",
            "label": "砂堵",
            "action": "停止继续升砂比或升排量，优先低排量试挤；试挤无效时进入解堵流程。",
            "confidence": "high",
        },
        {
            "rule_id": "SP-R02",
            "rule_name": "压力持续上升基础触发砂堵风险",
            "source": "Smart_Annotation3.py + 书籍经验教训",
            "book_pages": "16, 22",
            "condition": "加砂阶段压力呈逐步上升趋势，尤其在较高砂比或转换液性后压力明显抬升。",
            "data_expression": "delta_pressure >= 8 MPa 且 pressure_slope >= 1.0 MPa/min 且持续时间 >= 30 s。",
            "label": "砂堵风险",
            "action": "增加隔离液量，实时控制砂比，待压力平稳后再逐步提升砂比。",
            "confidence": "high",
        },
        {
            "rule_id": "SP-R03",
            "rule_name": "压力上升并伴随排量下降",
            "source": "Smart_Annotation3.py",
            "book_pages": "15-22",
            "condition": "压力上升后，排量出现连续递减，说明井筒或近井带过流能力下降。",
            "data_expression": "pressure_trigger=true 且 flow_has_decrease=true。",
            "label": "砂堵(降排量)",
            "action": "降低排量，观察压力是否回落；必要时低排量试挤或放喷解堵。",
            "confidence": "medium-high",
        },
        {
            "rule_id": "SP-R04",
            "rule_name": "压力上升并伴随砂比下降",
            "source": "Smart_Annotation3.py",
            "book_pages": "16, 22",
            "condition": "压力上升后，砂比出现连续下降或现场主动降砂比仍无法稳定压力。",
            "data_expression": "pressure_trigger=true 且 sand_ratio_has_decrease=true。",
            "label": "砂堵(降砂比)",
            "action": "停止继续提高砂比，降低砂比或转入低砂比施工，压力稳定后再恢复。",
            "confidence": "medium-high",
        },
        {
            "rule_id": "SP-R05",
            "rule_name": "压力上升并伴随排量、砂比同时下降",
            "source": "Smart_Annotation3.py",
            "book_pages": "15-22",
            "condition": "压力上升后，排量和砂比均连续下降，表明现场已经在降参但压力仍异常。",
            "data_expression": "pressure_trigger=true 且 flow_has_decrease=true 且 sand_ratio_has_decrease=true。",
            "label": "砂堵(降排量降砂比)",
            "action": "判为较高风险砂堵，进入低排量试挤、酸液或放喷等处置流程。",
            "confidence": "high",
        },
        {
            "rule_id": "SP-R06",
            "rule_name": "提排量导致的压力上升不直接标砂堵",
            "source": "Smart_Annotation3.py",
            "book_pages": "14-22",
            "condition": "压力上升同时排量连续升高，且砂比未下降，可能是提排量引起的正常压力响应。",
            "data_expression": "flow_has_increase=true 且 flow_rise >= 1.0 m3/min 且 sand_ratio_has_decrease=false。",
            "label": "不标砂堵/需观察",
            "action": "作为排除规则，避免把正常提排量误判为砂堵。",
            "confidence": "medium",
        },
        {
            "rule_id": "SP-R07",
            "rule_name": "近 120 秒砂比为 0 时排除砂堵",
            "source": "Smart_Annotation3.py",
            "book_pages": "15-22",
            "condition": "压力异常发生前短时间内砂比为 0，缺少支撑剂进入条件，砂堵可能性降低。",
            "data_expression": "recent_120s_has_zero_sand_ratio=true。",
            "label": "不标砂堵/井筒效应或压力波动",
            "action": "作为排除规则；可转为压力异常波动或井筒效应待复核。",
            "confidence": "medium",
        },
        {
            "rule_id": "SP-R08",
            "rule_name": "设备故障停泵后沉砂诱发砂堵",
            "source": "书籍案例经验",
            "book_pages": "17",
            "condition": "压裂设备故障导致停泵，支撑剂沉降，恢复施工后压力异常升高。",
            "data_expression": "pump_stop_or_equipment_fault=true 且 resume_pump_pressure_rise=true。",
            "label": "砂堵风险",
            "action": "施工前加强设备维护；停泵后恢复施工需低排量试挤并重点监测压力。",
            "confidence": "high",
        },
        {
            "rule_id": "SP-R09",
            "rule_name": "高砂比或液性转换前隔离液不足诱发砂堵",
            "source": "书籍经验教训",
            "book_pages": "16, 22",
            "condition": "高砂比加砂或减阻水/胶液转换时隔离液不足，压力出现大幅上升。",
            "data_expression": "high_sand_ratio_or_fluid_switch=true 且 pressure_rise=true。",
            "label": "砂堵风险",
            "action": "转换液性前顶替足够隔离液；高砂比阶段压力不稳时降低砂比。",
            "confidence": "high",
        },
        {
            "rule_id": "SP-R10",
            "rule_name": "试挤无效后采用放喷/酸液解堵",
            "source": "书籍处理过程",
            "book_pages": "15, 18, 20, 21",
            "condition": "砂堵停泵后低排量试挤或酸液试挤无明显压降，重复试挤压力仍快速上升。",
            "data_expression": "low_rate_squeeze_failed=true 或 acid_squeeze_no_pressure_drop=true。",
            "label": "处置规则",
            "action": "采用控制放喷，利用地层压力反向冲刷炮眼；必要时配合酸液解堵。",
            "confidence": "medium-high",
        },
    ]


def write_outputs(candidates: list[dict], rules: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "book_sand_plug_candidates_page_1_63.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["page_no", "keyword_score", "matched_keywords", "snippet"])
        writer.writeheader()
        writer.writerows(candidates)

    (OUTPUT_DIR / "fused_sand_plug_rules.json").write_text(
        json.dumps({"rules": rules, "candidate_count": len(candidates)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (OUTPUT_DIR / "fused_sand_plug_rules.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rule_id",
                "rule_name",
                "source",
                "book_pages",
                "condition",
                "data_expression",
                "label",
                "action",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rules)

    lines = [
        "# 砂堵规则融合说明",
        "",
        "## 数据来源",
        "",
        "- 既有规则：`智能标注/Smart_Annotation3.py`。",
        "- 书籍来源：`陪陵页岩气田试气压裂作业井复杂情况与故障案例分析271` 解析结果。",
        "- 使用范围：仅使用解析页序 1-63 页及以前内容，主要覆盖第 1 章压裂施工作业砂堵案例。",
        "",
        "## 融合原则",
        "",
        "- 书籍案例提供现场经验、成因和处置逻辑。",
        "- 既有智能标注脚本提供可落地的数据阈值和窗口判断方法。",
        "- 融合后规则既保留现场语义，也给出可计算表达式，方便后续接入自动标注或智能体决策。",
        "",
        "## 融合规则",
        "",
    ]
    for rule in rules:
        lines.extend(
            [
                f"### {rule['rule_id']} {rule['rule_name']}",
                "",
                f"- 来源：{rule['source']}；书籍页码：{rule['book_pages']}",
                f"- 触发条件：{rule['condition']}",
                f"- 数据表达：`{rule['data_expression']}`",
                f"- 输出标签：{rule['label']}",
                f"- 建议动作：{rule['action']}",
                f"- 置信度：{rule['confidence']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 备注",
            "",
            "当前规则主要服务于砂堵识别与处置建议。由于书籍为案例型文本，规则仍需要结合现场专家复核，特别是阈值、时间窗口和不同井段适用性。",
        ]
    )
    (OUTPUT_DIR / "fused_sand_plug_rules.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    candidates = extract_candidates(max_page=63)
    rules = fused_rules()
    write_outputs(candidates, rules)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "candidate_count": len(candidates), "rule_count": len(rules)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
