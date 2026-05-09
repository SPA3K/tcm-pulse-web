"""TCM Pulse Type Classifier.

Maps extracted physiological features to Traditional Chinese Medicine
pulse types (脉象) with confidence scores.

Usage:
    python -m src.classification.pulse_classifier --features data/features/ecg_features.json
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PulseCandidate:
    """A candidate TCM pulse type classification."""
    type_cn: str          # 中文名
    type_en: str          # English name
    category: str         # Classification dimension
    confidence: float     # 0.0 - 1.0
    evidence: str         # Human-readable reasoning
    clinical_meaning: str # 中医临床意义


def classify_pulse(features: dict) -> list[PulseCandidate]:
    """Classify extracted features into TCM pulse types.

    Returns candidates sorted by confidence (highest first).
    """
    candidates = []

    hr = features.get("mean_hr_bpm", 72)
    sdnn = features.get("sdnn_ms", 50)
    rmssd = features.get("rmssd_ms", 30)
    rr_cv = features.get("rr_cv", 0.05)
    r_amp = features.get("mean_r_amplitude_mv", 1.0)
    premature = features.get("premature_beat_count", 0)
    pauses = features.get("pause_count", 0)
    has_regular_pauses = features.get("has_regular_pauses", False)
    lf_hf = features.get("lf_hf_ratio")
    sd1 = features.get("poincare_sd1_ms", 20)
    total_beats = features.get("total_beats", 30)

    # ============================================================
    # 一、频率类 (Rate-based) — 高置信度
    # ============================================================

    if hr < 60:
        candidates.append(PulseCandidate(
            type_cn="迟脉", type_en="Slow",
            category="rate",
            confidence=0.95,
            evidence=f"心率 {hr:.0f} bpm < 60 bpm",
            clinical_meaning="寒证、阳虚 — 阳气不足，不能鼓动血脉",
        ))

    if hr > 90:
        conf = 0.95 if hr > 100 else 0.85
        candidates.append(PulseCandidate(
            type_cn="数脉", type_en="Rapid",
            category="rate",
            confidence=conf,
            evidence=f"心率 {hr:.0f} bpm > 90 bpm",
            clinical_meaning="热证、阴虚 — 邪热亢盛或阴虚内热",
        ))

    if hr > 120:
        candidates.append(PulseCandidate(
            type_cn="疾脉", type_en="Racing",
            category="rate",
            confidence=0.90,
            evidence=f"心率 {hr:.0f} bpm > 120 bpm",
            clinical_meaning="阳亢极、阴竭阳脱 — 元气将脱的危候",
        ))

    # ============================================================
    # 二、节律类 (Rhythm-based) — 高置信度
    # ============================================================

    premature_ratio = premature / total_beats if total_beats > 0 else 0

    if premature_ratio > 0.1 and hr > 90:
        candidates.append(PulseCandidate(
            type_cn="促脉", type_en="Skipping-rapid",
            category="rhythm",
            confidence=0.85,
            evidence=f"心率 {hr:.0f} bpm + {premature} 次早搏 (占比 {premature_ratio:.0%})",
            clinical_meaning="阳盛实热、气滞血瘀 — 热邪壅盛或瘀血阻络",
        ))

    if premature_ratio > 0.1 and hr < 65:
        candidates.append(PulseCandidate(
            type_cn="结脉", type_en="Knotted",
            category="rhythm",
            confidence=0.85,
            evidence=f"心率 {hr:.0f} bpm + {premature} 次停搏 (占比 {premature_ratio:.0%})",
            clinical_meaning="阴盛气结、寒痰血瘀 — 阴寒凝结或气血瘀滞",
        ))

    if has_regular_pauses:
        candidates.append(PulseCandidate(
            type_cn="代脉", type_en="Intermittent",
            category="rhythm",
            confidence=0.88,
            evidence=f"检测到规律性停搏模式（等间距漏拍）",
            clinical_meaning="脏气衰微、风证痛证 — 元气亏损，脉气不续",
        ))

    # ============================================================
    # 三、张力类 (Tension-based) — 中置信度
    # ============================================================

    if rmssd < 20:
        conf = 0.70 if lf_hf and lf_hf > 2.0 else 0.60
        candidates.append(PulseCandidate(
            type_cn="弦脉", type_en="Wiry",
            category="tension",
            confidence=conf,
            evidence=f"RMSSD={rmssd:.1f}ms (低副交感)"
                     + (f", LF/HF={lf_hf:.1f}" if lf_hf else ""),
            clinical_meaning="肝胆病、痛证、痰饮 — 肝气郁结，脉管拘急",
        ))

    if lf_hf and lf_hf > 3.0 and hr > 80:
        candidates.append(PulseCandidate(
            type_cn="紧脉", type_en="Tight",
            category="tension",
            confidence=0.55,
            evidence=f"LF/HF={lf_hf:.1f} (极高交感) + HR={hr:.0f}",
            clinical_meaning="寒证、痛证 — 寒邪收引，脉道绷紧",
        ))

    if rmssd > 50 and 55 <= hr <= 78:
        candidates.append(PulseCandidate(
            type_cn="缓脉", type_en="Moderate",
            category="tension",
            confidence=0.60,
            evidence=f"RMSSD={rmssd:.1f}ms (高副交感) + HR={hr:.0f} (从容)",
            clinical_meaning="正常脉或脾胃虚弱 — 脉来和缓，一息四至",
        ))

    # ============================================================
    # 四、振幅类 (Amplitude-based) — 中置信度
    # ============================================================

    if r_amp > 1.5:
        candidates.append(PulseCandidate(
            type_cn="洪脉", type_en="Flooding",
            category="amplitude",
            confidence=0.55,
            evidence=f"R波振幅 {r_amp:.2f}mV > 1.5mV (高振幅)",
            clinical_meaning="气分热盛 — 热邪充斥脉道，气盛血涌",
        ))

    if r_amp < 0.5:
        candidates.append(PulseCandidate(
            type_cn="细脉", type_en="Thready",
            category="amplitude",
            confidence=0.55,
            evidence=f"R波振幅 {r_amp:.2f}mV < 0.5mV (低振幅)",
            clinical_meaning="气血两虚、湿邪 — 气血不足，脉道不充",
        ))

    if r_amp < 0.3 and hr < 65:
        candidates.append(PulseCandidate(
            type_cn="弱脉", type_en="Weak",
            category="amplitude",
            confidence=0.50,
            evidence=f"R波振幅 {r_amp:.2f}mV (极低) + HR={hr:.0f} (偏慢)",
            clinical_meaning="阳气虚衰 — 气虚无力鼓动血脉",
        ))

    # ============================================================
    # 五、流畅度类 (Smoothness-based) — 中低置信度
    # ============================================================

    if rr_cv < 0.03 and premature == 0:
        candidates.append(PulseCandidate(
            type_cn="滑脉", type_en="Slippery",
            category="smoothness",
            confidence=0.50,
            evidence=f"RR变异系数 {rr_cv:.4f} < 0.03 (极规整，无早搏)",
            clinical_meaning="痰湿、食积、妊娠 — 气血充盛，脉道流利",
        ))

    if rr_cv > 0.10:
        candidates.append(PulseCandidate(
            type_cn="涩脉", type_en="Choppy",
            category="smoothness",
            confidence=0.45,
            evidence=f"RR变异系数 {rr_cv:.4f} > 0.10 (不规整)",
            clinical_meaning="气滞血瘀、精伤血少 — 血行不畅，脉气往来艰涩",
        ))

    # Sort by confidence
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def format_diagnosis_report(candidates: list[PulseCandidate], features: dict) -> str:
    """Format classification results into a readable diagnosis report."""
    lines = []
    lines.append("═══ 中医脉诊分析报告 / TCM Pulse Analysis ═══")
    lines.append(f"📱 数据时长: {features.get('duration_sec', '?')}秒 | "
                 f"心搏数: {features.get('n_beats', '?')}")
    lines.append("")

    if not candidates:
        lines.append("未检测到明显脉象特征（可能为平脉/正常脉）")
        lines.append("")
        lines.append("平脉特征: 不浮不沉，不快不慢，从容和缓，节律整齐")
        return "\n".join(lines)

    # Primary pulse
    primary = candidates[0]
    lines.append("━━━ 主脉 (Primary) ━━━")
    lines.append(f"【{primary.type_cn}】{primary.type_en}")
    lines.append(f"  置信度: {primary.confidence:.0%}")
    lines.append(f"  依据: {primary.evidence}")
    lines.append(f"  临床意义: {primary.clinical_meaning}")
    lines.append("")

    # Secondary pulses (兼脉)
    secondary = [c for c in candidates[1:] if c.confidence >= 0.45]
    if secondary:
        lines.append("━━━ 兼脉 (Secondary) ━━━")
        for c in secondary[:3]:
            lines.append(f"【{c.type_cn}】{c.type_en} — {c.confidence:.0%}")
            lines.append(f"  {c.evidence}")
            lines.append(f"  {c.clinical_meaning}")
        lines.append("")

    # Compound pulse interpretation
    type_set = {c.type_cn for c in candidates[:3] if c.confidence >= 0.50}
    compound = _interpret_compound_pulse(type_set)
    if compound:
        lines.append("━━━ 相兼脉分析 ━━━")
        lines.append(compound)
        lines.append("")

    # Key metrics
    lines.append("━━━ 关键指标 ━━━")
    lines.append(f"  心率: {features.get('mean_hr_bpm', '?')} bpm")
    lines.append(f"  HRV SDNN: {features.get('sdnn_ms', '?')} ms")
    lines.append(f"  HRV RMSSD: {features.get('rmssd_ms', '?')} ms")
    lines.append(f"  R波振幅: {features.get('mean_r_amplitude_mv', '?')} mV")
    if features.get("lf_hf_ratio"):
        lines.append(f"  LF/HF: {features['lf_hf_ratio']}")
    lines.append("")

    # Limitations
    lines.append("━━━ 局限性说明 ━━━")
    lines.append("⚠️ 以下脉象维度无法从Apple Watch判定:")
    lines.append("  • 浮/沉 (需不同力度按压对比)")
    lines.append("  • 脉宽 (需空间分辨率)")
    lines.append("  • 脉管硬度 (需触觉传感)")
    lines.append("")
    lines.append("建议结合中医师问诊，或补充MAX30102 PPG传感器获取更完整脉象信息。")
    lines.append("═══════════════════════════════════════════════")

    return "\n".join(lines)


def _interpret_compound_pulse(types: set[str]) -> str:
    """Interpret compound pulse patterns (相兼脉)."""
    compounds = {
        frozenset({"弦脉", "数脉"}): "弦数脉 — 肝火上炎或肝胆湿热",
        frozenset({"弦脉", "细脉"}): "弦细脉 — 肝肾阴虚或血虚肝郁",
        frozenset({"弦脉", "滑脉"}): "弦滑脉 — 肝气郁结兼痰湿",
        frozenset({"数脉", "细脉"}): "细数脉 — 阴虚内热",
        frozenset({"迟脉", "弱脉"}): "迟弱脉 — 阳气虚衰",
        frozenset({"数脉", "洪脉"}): "洪数脉 — 气分热盛（阳明经证）",
        frozenset({"滑脉", "数脉"}): "滑数脉 — 痰热或湿热",
    }

    for pattern, interpretation in compounds.items():
        if pattern.issubset(types):
            return interpretation

    return ""


def main():
    parser = argparse.ArgumentParser(description="Classify TCM pulse types from features")
    parser.add_argument("--features", required=True, help="Path to features JSON file")
    parser.add_argument("--output", help="Output report file (default: stdout)")
    args = parser.parse_args()

    with open(args.features, "r", encoding="utf-8") as f:
        features_list = json.load(f)

    if isinstance(features_list, dict):
        features_list = [features_list]

    for i, features in enumerate(features_list):
        if "error" in features:
            continue

        print(f"\n{'='*50}")
        print(f"ECG Session {i+1}: {features.get('source_file', 'unknown')}")
        print(f"{'='*50}")

        candidates = classify_pulse(features)
        report = format_diagnosis_report(candidates, features)
        print(report)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
                # Also save structured results
                f.write("\n\n--- Structured Results (JSON) ---\n")
                results = [{
                    "type_cn": c.type_cn,
                    "type_en": c.type_en,
                    "category": c.category,
                    "confidence": c.confidence,
                    "evidence": c.evidence,
                    "clinical_meaning": c.clinical_meaning,
                } for c in candidates]
                f.write(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
