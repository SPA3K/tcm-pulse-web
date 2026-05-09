"""Batch transcribe audio files using faster-whisper with TCM term correction.

Usage:
    python -m src.knowledge_base.whisper_transcriber --input knowledge-base/raw_audio/ --output knowledge-base/raw_transcripts/
    python -m src.knowledge_base.whisper_transcriber --input audio.wav --model large-v3
"""

import argparse
import json
from pathlib import Path
from datetime import timedelta

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("faster-whisper is required: pip install faster-whisper")


# Common Whisper misrecognitions of TCM terminology
TCM_CORRECTIONS = {
    # 脉象名称
    "浮买": "浮脉", "沉买": "沉脉", "弦买": "弦脉", "滑买": "滑脉",
    "涩买": "涩脉", "洪买": "洪脉", "细买": "细脉", "迟买": "迟脉",
    "数买": "数脉", "虚买": "虚脉", "实买": "实脉", "紧买": "紧脉",
    "缓买": "缓脉", "弱买": "弱脉", "微买": "微脉", "促买": "促脉",
    "结买": "结脉", "代买": "代脉", "长买": "长脉", "短买": "短脉",
    "散买": "散脉", "濡买": "濡脉", "革买": "革脉", "芤买": "芤脉",
    "伏买": "伏脉", "牢买": "牢脉", "动买": "动脉", "疾买": "疾脉",
    # 常见术语
    "辩证": "辨证", "辩症": "辨证",
    "寸关迟": "寸关尺", "寸官尺": "寸关尺",
    "气血": "气血", "阴阳": "阴阳",
    "濒湖买学": "濒湖脉学", "频湖脉学": "濒湖脉学",
    "买经": "脉经", "买象": "脉象", "买诊": "脉诊",
    "把买": "把脉", "摸买": "摸脉",
    "号买": "号脉",
    # 脏腑
    "肝胆": "肝胆", "心包": "心包",
    "三焦": "三焦", "膀光": "膀胱",
}


def correct_tcm_terms(text: str) -> str:
    """Apply TCM terminology corrections to Whisper output."""
    for wrong, right in TCM_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def transcribe_file(
    audio_path: str | Path,
    model: WhisperModel,
    language: str = "zh",
    apply_corrections: bool = True,
) -> dict:
    """Transcribe a single audio file."""
    audio_path = Path(audio_path)

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )

    result = {
        "source_file": audio_path.name,
        "source_path": str(audio_path),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_sec": round(info.duration, 1),
        "segments": [],
        "full_text": "",
    }

    full_text_parts = []

    for segment in segments:
        text = segment.text.strip()
        if apply_corrections:
            text = correct_tcm_terms(text)

        seg_data = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text,
        }

        if segment.words:
            seg_data["words"] = [
                {"word": correct_tcm_terms(w.word) if apply_corrections else w.word,
                 "start": round(w.start, 2),
                 "end": round(w.end, 2),
                 "probability": round(w.probability, 3)}
                for w in segment.words
            ]

        result["segments"].append(seg_data)
        full_text_parts.append(text)

    result["full_text"] = "\n".join(full_text_parts)
    return result


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def batch_transcribe(
    input_dir: str | Path,
    output_dir: str | Path,
    model_size: str = "large-v3",
    device: str = "auto",
    language: str = "zh",
) -> list[dict]:
    """Batch transcribe all audio files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    audio_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    )

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return []

    print(f"Found {len(audio_files)} audio files")
    print(f"Loading model: {model_size} (device: {device})")

    # Determine compute type based on device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Using device: {device}, compute_type: {compute_type}")

    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")

        # Skip if already transcribed
        output_file = output_dir / f"{audio_file.stem}.json"
        if output_file.exists():
            print(f"  Already transcribed, skipping")
            with open(output_file, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        try:
            result = transcribe_file(audio_file, model, language=language)

            # Save individual transcript
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            duration_str = format_timestamp(result["duration_sec"])
            n_segments = len(result["segments"])
            print(f"  Duration: {duration_str} | Segments: {n_segments}")
            print(f"  Preview: {result['full_text'][:100]}...")

            results.append(result)

        except Exception as e:
            print(f"  Error: {e}")
            results.append({"source_file": audio_file.name, "error": str(e)})

    # Save batch manifest
    manifest = {
        "total_files": len(audio_files),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "total_duration_sec": sum(r.get("duration_sec", 0) for r in results),
        "files": [r.get("source_file", "") for r in results],
    }
    with open(output_dir / "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Transcription complete:")
    print(f"  Success: {manifest['successful']}/{manifest['total_files']}")
    print(f"  Total duration: {format_timestamp(manifest['total_duration_sec'])}")
    print(f"  Output: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Transcribe TCM teaching audio with Whisper")
    parser.add_argument("--input", required=True, help="Audio file or directory")
    parser.add_argument("--output", default="knowledge-base/raw_transcripts", help="Output directory")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--language", default="zh")
    parser.add_argument("--no-correct", action="store_true", help="Skip TCM term correction")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        print(f"Loading model: {args.model}")
        device = args.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(args.model, device=device, compute_type=compute_type)

        result = transcribe_file(input_path, model, args.language, not args.no_correct)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{input_path.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_file}")
    else:
        # Batch directory
        batch_transcribe(input_path, args.output, args.model, args.device, args.language)


if __name__ == "__main__":
    main()
