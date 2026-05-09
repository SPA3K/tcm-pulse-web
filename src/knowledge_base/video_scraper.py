"""Scrape TCM pulse diagnosis teaching videos from Bilibili and YouTube.

Downloads audio for Whisper transcription.

Usage:
    python -m src.knowledge_base.video_scraper --config config/scraping_targets.yaml
    python -m src.knowledge_base.video_scraper --search "中医脉诊教学" --platform bilibili --max 20
    python -m src.knowledge_base.video_scraper --url "https://www.bilibili.com/video/BVxxxxxxxx"
"""

import argparse
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoMeta:
    video_id: str
    title: str
    platform: str
    url: str
    duration_sec: float
    uploader: str


def download_audio(url: str, output_dir: str | Path, cookies_browser: str = "chrome") -> Path | None:
    """Download audio from a video URL using yt-dlp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(output_dir / "%(id)s_%(title).50s.%(ext)s"),
        "--cookies-from-browser", cookies_browser,
        "--no-playlist",
        "--write-info-json",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  yt-dlp error: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  Download timed out for: {url}")
        return None
    except FileNotFoundError:
        print("  Error: yt-dlp not installed. Run: pip install yt-dlp")
        return None

    # Find the downloaded file
    wav_files = list(output_dir.glob("*.wav"))
    if wav_files:
        return max(wav_files, key=lambda f: f.stat().st_mtime)
    return None


def search_and_download(
    query: str,
    platform: str = "bilibili",
    max_results: int = 20,
    output_dir: str | Path = "knowledge-base/raw_audio",
    cookies_browser: str = "chrome",
) -> list[dict]:
    """Search for videos and download audio."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if platform == "bilibili":
        search_url = f"https://search.bilibili.com/all?keyword={query}"
    elif platform == "youtube":
        search_url = f"ytsearch{max_results}:{query}"
    else:
        raise ValueError(f"Unsupported platform: {platform}")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(output_dir / "%(id)s_%(title).50s.%(ext)s"),
        "--write-info-json",
        "--max-downloads", str(max_results),
        "--no-playlist",
    ]

    if platform == "bilibili":
        cmd.extend(["--cookies-from-browser", cookies_browser])

    cmd.append(search_url)

    print(f"Searching {platform} for: {query}")
    print(f"Max downloads: {max_results}")
    print(f"Output: {output_dir}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0 and "max-downloads" not in result.stderr:
            print(f"Warning: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        print("Search/download timed out (30min limit)")
    except FileNotFoundError:
        print("Error: yt-dlp not installed. Run: pip install yt-dlp")
        return []

    # Collect metadata from .info.json files
    downloaded = []
    for info_file in output_dir.glob("*.info.json"):
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            downloaded.append({
                "video_id": info.get("id", ""),
                "title": info.get("title", ""),
                "platform": platform,
                "url": info.get("webpage_url", ""),
                "duration_sec": info.get("duration", 0),
                "uploader": info.get("uploader", ""),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"Downloaded {len(downloaded)} videos")
    return downloaded


def batch_download_from_config(config_path: str | Path, output_dir: str | Path = "knowledge-base/raw_audio"):
    """Download videos listed in a YAML config file."""
    import yaml

    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    results = []

    # Download individual URLs
    for entry in config.get("urls", []):
        url = entry if isinstance(entry, str) else entry.get("url", "")
        if url:
            print(f"\nDownloading: {url}")
            audio_path = download_audio(url, output_dir)
            if audio_path:
                results.append({"url": url, "audio": str(audio_path)})

    # Search and download
    for search in config.get("searches", []):
        query = search.get("query", "")
        platform = search.get("platform", "bilibili")
        max_results = search.get("max", 10)
        if query:
            found = search_and_download(query, platform, max_results, output_dir)
            results.extend(found)

    # Save download manifest
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nTotal downloaded: {len(results)}")
    print(f"Manifest: {manifest_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Download TCM teaching video audio")
    parser.add_argument("--config", help="YAML config file with URLs and search queries")
    parser.add_argument("--url", help="Single video URL to download")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--platform", default="bilibili", choices=["bilibili", "youtube"])
    parser.add_argument("--max", type=int, default=20, help="Max search results")
    parser.add_argument("--output", default="knowledge-base/raw_audio", help="Output directory")
    args = parser.parse_args()

    if args.config:
        batch_download_from_config(args.config, args.output)
    elif args.url:
        result = download_audio(args.url, args.output)
        if result:
            print(f"Downloaded: {result}")
    elif args.search:
        search_and_download(args.search, args.platform, args.max, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
