"""Batch transcribe short TCM audio files (<15 min) for CPU feasibility."""
import json, time
from pathlib import Path
from faster_whisper import WhisperModel

TCM_CORRECTIONS = {
    '浮买': '浮脉', '沉买': '沉脉', '弦买': '弦脉', '滑买': '滑脉',
    '涩买': '涩脉', '洪买': '洪脉', '细买': '细脉', '迟买': '迟脉',
    '数买': '数脉', '虚买': '虚脉', '实买': '实脉', '紧买': '紧脉',
    '缓买': '缓脉', '弱买': '弱脉', '微买': '微脉', '促买': '促脉',
    '结买': '结脉', '代买': '代脉', '买象': '脉象', '买诊': '脉诊',
    '把买': '把脉', '号买': '号脉', '辩证': '辨证',
    '麥': '脉', '買': '脉', '腐麥': '浮脉', '沈麥': '沉脉',
    '血麥': '弦脉', '紅麥': '洪脉', '緩麥': '缓脉', '沉麥': '沉脉',
    '乾麥': '弦脉', '寸關尺': '寸关尺', '脈搏': '脉搏',
    '脈象': '脉象', '脈診': '脉诊', '脈': '脉',
}

def correct(text):
    for wrong, right in TCM_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text

audio_dir = Path('/mnt/d/tcm-video-cache')
output_dir = Path('/mnt/d/tcm-pulse-diagnosis/knowledge-base/raw_transcripts')
output_dir.mkdir(parents=True, exist_ok=True)

# Only process files < 150MB (~15 min of WAV audio at 16bit 48kHz)
MAX_SIZE_MB = 150
wavs = sorted(
    [f for f in audio_dir.glob('*.wav') if f.stat().st_size / 1024 / 1024 < MAX_SIZE_MB],
    key=lambda f: f.stat().st_size
)

print("Loading model...")
model = WhisperModel('small', device='cpu', compute_type='int8')
print(f"Found {len(wavs)} short audio files (< {MAX_SIZE_MB}MB)\n")

total_start = time.time()
done = 0
skipped = 0

for f in wavs:
    out_file = output_dir / f'{f.stem}.json'
    if out_file.exists():
        print(f"[SKIP] {f.name}")
        skipped += 1
        done += 1
        continue

    size_mb = f.stat().st_size / 1024 / 1024
    print(f"[{done+1}/{len(wavs)}] {f.name} ({size_mb:.0f}MB)")

    start = time.time()
    try:
        segments, info = model.transcribe(str(f), language='zh', beam_size=5, vad_filter=True)
        result = {'source_file': f.name, 'duration_sec': round(info.duration, 1), 'segments': [], 'full_text': ''}
        texts = []
        for seg in segments:
            text = correct(seg.text.strip())
            result['segments'].append({'start': round(seg.start, 2), 'end': round(seg.end, 2), 'text': text})
            texts.append(text)
        result['full_text'] = '\n'.join(texts)

        with open(out_file, 'w', encoding='utf-8') as fp:
            json.dump(result, fp, indent=2, ensure_ascii=False)

        elapsed = time.time() - start
        ratio = elapsed / info.duration if info.duration > 0 else 0
        print(f"  OK: {elapsed:.0f}s (x{ratio:.1f}) | {info.duration:.0f}s audio | {len(texts)} segs")
        print(f"  Preview: {result['full_text'][:100]}...")
    except Exception as e:
        print(f"  ERROR: {e}")

    done += 1

total_elapsed = time.time() - total_start
print(f"\nDone: {done} processed, {skipped} skipped in {total_elapsed/60:.1f} min")
print(f"Skipped long files (>{MAX_SIZE_MB}MB): "
      f"{len(list(audio_dir.glob('*.wav'))) - len(wavs)} files (use GPU for these)")
