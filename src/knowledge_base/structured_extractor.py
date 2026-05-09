"""Extract structured TCM pulse type information from Whisper transcripts.

Uses Claude API to identify and structure pulse type descriptions,
clinical significance, and measurable features from raw transcript text.

Usage:
    python -m src.knowledge_base.structured_extractor --input knowledge-base/raw_transcripts/ --output knowledge-base/structured/
"""

import argparse
import json
from pathlib import Path

try:
    import anthropic
except ImportError:
    raise ImportError("anthropic is required: pip install anthropic")


EXTRACTION_PROMPT = """你是中医脉诊专家。请从以下中医教学视频转写文本中，提取所有提到的脉象类型的结构化信息。

对于每种提到的脉象，请提取以下信息（如果文本中有的话）：

1. chinese_name: 脉象中文名
2. description_quotes: 原文中对该脉象的描述（直接引用原文）
3. finger_feeling: 指下感觉描述（医生手指的触感）
4. clinical_significance: 主病/临床意义（该脉象提示什么病证）
5. differentiation: 与其他脉象的鉴别要点
6. measurable_features: 可量化的物理特征描述（频率、振幅、节律等）
7. teaching_tips: 教学中提到的学习要点或记忆方法
8. timestamp_hint: 大约在文本的哪个位置提到（前段/中段/后段）

请以JSON数组格式输出，每个脉象一个对象。如果某个字段在文本中没有相关信息，设为null。
只提取文本中确实提到的内容，不要补充你自己的知识。

转写文本：
---
{transcript}
---

请输出合法JSON数组（不要加markdown代码块标记）："""


def extract_from_transcript(transcript_text: str, source_file: str = "") -> list[dict]:
    """Use Claude to extract structured pulse info from transcript text."""
    client = anthropic.Anthropic()

    # Truncate very long transcripts to fit context
    max_chars = 15000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "\n\n[...文本过长，已截断...]"

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(transcript=transcript_text),
        }],
    )

    response_text = message.content[0].text.strip()

    # Try to parse JSON from response
    try:
        # Handle potential markdown code block wrapping
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0]
        extracted = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find JSON array in response
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                extracted = json.loads(response_text[start:end])
            except json.JSONDecodeError:
                print(f"  Failed to parse extraction result for {source_file}")
                return []
        else:
            return []

    # Add source metadata
    for item in extracted:
        item["source_file"] = source_file

    return extracted


def merge_pulse_data(all_extractions: list[dict], output_dir: Path):
    """Merge extractions from multiple sources into per-pulse-type files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by pulse type
    by_type: dict[str, list] = {}
    for item in all_extractions:
        name = item.get("chinese_name", "unknown")
        if name not in by_type:
            by_type[name] = []
        by_type[name].append(item)

    # Write per-type files
    pulse_types_dir = output_dir / "pulse_types"
    pulse_types_dir.mkdir(exist_ok=True)

    for i, (name, entries) in enumerate(sorted(by_type.items()), 1):
        output_file = pulse_types_dir / f"{i:02d}_{name}.json"
        merged = {
            "chinese_name": name,
            "sources_count": len(entries),
            "descriptions": [e.get("description_quotes") for e in entries if e.get("description_quotes")],
            "finger_feelings": [e.get("finger_feeling") for e in entries if e.get("finger_feeling")],
            "clinical_significance": list(set(
                sig for e in entries
                for sig in (e.get("clinical_significance") or [])
                if isinstance(e.get("clinical_significance"), list)
            )),
            "differentiations": [e.get("differentiation") for e in entries if e.get("differentiation")],
            "measurable_features": [e.get("measurable_features") for e in entries if e.get("measurable_features")],
            "teaching_tips": [e.get("teaching_tips") for e in entries if e.get("teaching_tips")],
            "sources": [e.get("source_file") for e in entries],
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(all_extractions)} extractions into {len(by_type)} pulse types")
    return by_type


def batch_extract(
    input_dir: str | Path,
    output_dir: str | Path,
) -> list[dict]:
    """Process all transcripts in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_files = sorted(input_dir.glob("*.json"))
    transcript_files = [f for f in transcript_files if not f.name.startswith("_")]

    if not transcript_files:
        print(f"No transcript JSON files in {input_dir}")
        return []

    print(f"Processing {len(transcript_files)} transcripts")

    all_extractions = []

    for i, tf in enumerate(transcript_files, 1):
        print(f"\n[{i}/{len(transcript_files)}] {tf.name}")

        # Load transcript
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)

        full_text = data.get("full_text", "")
        if not full_text:
            segments = data.get("segments", [])
            full_text = "\n".join(s.get("text", "") for s in segments)

        if len(full_text) < 100:
            print("  Too short, skipping")
            continue

        # Extract
        extractions = extract_from_transcript(full_text, tf.name)
        print(f"  Found {len(extractions)} pulse types mentioned")

        for e in extractions:
            print(f"    - {e.get('chinese_name', '?')}")

        all_extractions.extend(extractions)

        # Save individual extraction
        extract_file = output_dir / f"extract_{tf.stem}.json"
        with open(extract_file, "w", encoding="utf-8") as f:
            json.dump(extractions, f, indent=2, ensure_ascii=False)

    # Merge all into per-type files
    if all_extractions:
        merge_pulse_data(all_extractions, output_dir)

    return all_extractions


def main():
    parser = argparse.ArgumentParser(description="Extract structured TCM data from transcripts")
    parser.add_argument("--input", required=True, help="Transcript directory or single JSON file")
    parser.add_argument("--output", default="knowledge-base/structured", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        full_text = data.get("full_text", "")
        results = extract_from_transcript(full_text, input_path.name)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        batch_extract(input_path, args.output)


if __name__ == "__main__":
    main()
