"""Parse Apple Health HealthKit export XML and ECG CSV files.

Usage:
    python -m src.data_ingestion.healthkit_parser --input export.xml --output data/parsed/
    python -m src.data_ingestion.healthkit_parser --input export.xml --ecg-dir electrocardiograms/ --output data/parsed/
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import argparse

import numpy as np
import pandas as pd


@dataclass
class HealthKitData:
    heart_rate: pd.DataFrame = field(default_factory=pd.DataFrame)
    hrv: pd.DataFrame = field(default_factory=pd.DataFrame)
    resting_hr: pd.DataFrame = field(default_factory=pd.DataFrame)
    respiratory_rate: pd.DataFrame = field(default_factory=pd.DataFrame)
    ecg_sessions: list = field(default_factory=list)


def parse_export_xml(xml_path: str | Path) -> HealthKitData:
    """Parse HealthKit export.xml into structured DataFrames."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"HealthKit export not found: {xml_path}")

    tree = ET.iterparse(str(xml_path), events=("end",))

    heart_rates = []
    hrv_samples = []
    resting_hrs = []
    resp_rates = []

    type_handlers = {
        "HKQuantityTypeIdentifierHeartRate": heart_rates,
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": hrv_samples,
        "HKQuantityTypeIdentifierRestingHeartRate": resting_hrs,
        "HKQuantityTypeIdentifierRespiratoryRate": resp_rates,
    }

    for event, elem in tree:
        if elem.tag != "Record":
            elem.clear()
            continue

        record_type = elem.get("type", "")
        target_list = type_handlers.get(record_type)

        if target_list is not None:
            target_list.append({
                "timestamp": elem.get("startDate"),
                "value": float(elem.get("value", 0)),
                "source": elem.get("sourceName", ""),
                "device": elem.get("device", ""),
            })

        elem.clear()

    data = HealthKitData()

    if heart_rates:
        data.heart_rate = pd.DataFrame(heart_rates)
        data.heart_rate["timestamp"] = pd.to_datetime(data.heart_rate["timestamp"])
        data.heart_rate = data.heart_rate.sort_values("timestamp").reset_index(drop=True)
        data.heart_rate.rename(columns={"value": "bpm"}, inplace=True)

    if hrv_samples:
        data.hrv = pd.DataFrame(hrv_samples)
        data.hrv["timestamp"] = pd.to_datetime(data.hrv["timestamp"])
        data.hrv = data.hrv.sort_values("timestamp").reset_index(drop=True)
        data.hrv.rename(columns={"value": "sdnn_ms"}, inplace=True)

    if resting_hrs:
        data.resting_hr = pd.DataFrame(resting_hrs)
        data.resting_hr["timestamp"] = pd.to_datetime(data.resting_hr["timestamp"])
        data.resting_hr.rename(columns={"value": "bpm"}, inplace=True)

    if resp_rates:
        data.respiratory_rate = pd.DataFrame(resp_rates)
        data.respiratory_rate["timestamp"] = pd.to_datetime(data.respiratory_rate["timestamp"])
        data.respiratory_rate.rename(columns={"value": "breaths_per_min"}, inplace=True)

    return data


def parse_ecg_csv(csv_path: str | Path) -> dict:
    """Parse a single Apple Watch ECG CSV export.

    Apple Watch ECG CSV format:
    - First ~13 lines are metadata (name, date, classification, etc.)
    - Remaining lines are voltage readings in microvolts at 512 Hz
    """
    csv_path = Path(csv_path)
    metadata = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse metadata header
    data_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line and "," in line and not line.replace(",", "").replace("-", "").replace(".", "").isdigit():
            parts = line.split(",", 1)
            if len(parts) == 2:
                metadata[parts[0].strip()] = parts[1].strip()
            data_start = i + 1
        elif line.replace(",", "").replace("-", "").replace(".", "").replace(" ", "").isdigit():
            data_start = i
            break

    # Parse voltage data
    voltages_uv = []
    for line in lines[data_start:]:
        line = line.strip()
        if line:
            try:
                voltages_uv.append(float(line.split(",")[0]))
            except ValueError:
                continue

    voltage_mv = np.array(voltages_uv) / 1000.0  # Convert µV to mV
    sampling_rate = 512  # Apple Watch ECG is always 512 Hz
    duration_sec = len(voltage_mv) / sampling_rate

    return {
        "metadata": metadata,
        "voltage_mv": voltage_mv,
        "sampling_rate": sampling_rate,
        "duration_sec": duration_sec,
        "n_samples": len(voltage_mv),
    }


def parse_ecg_directory(ecg_dir: str | Path) -> list[dict]:
    """Parse all ECG CSV files in a directory."""
    ecg_dir = Path(ecg_dir)
    if not ecg_dir.exists():
        return []

    sessions = []
    for csv_file in sorted(ecg_dir.glob("*.csv")):
        try:
            session = parse_ecg_csv(csv_file)
            session["file"] = str(csv_file.name)
            sessions.append(session)
        except Exception as e:
            print(f"Warning: Failed to parse {csv_file.name}: {e}")

    return sessions


def summarize(data: HealthKitData) -> dict:
    """Generate summary statistics from parsed data."""
    summary = {}

    if not data.heart_rate.empty:
        summary["heart_rate"] = {
            "count": len(data.heart_rate),
            "date_range": [
                str(data.heart_rate["timestamp"].min()),
                str(data.heart_rate["timestamp"].max()),
            ],
            "mean_bpm": round(data.heart_rate["bpm"].mean(), 1),
            "min_bpm": round(data.heart_rate["bpm"].min(), 1),
            "max_bpm": round(data.heart_rate["bpm"].max(), 1),
        }

    if not data.hrv.empty:
        summary["hrv"] = {
            "count": len(data.hrv),
            "mean_sdnn_ms": round(data.hrv["sdnn_ms"].mean(), 1),
            "min_sdnn_ms": round(data.hrv["sdnn_ms"].min(), 1),
            "max_sdnn_ms": round(data.hrv["sdnn_ms"].max(), 1),
        }

    if data.ecg_sessions:
        summary["ecg"] = {
            "session_count": len(data.ecg_sessions),
            "total_duration_sec": sum(s["duration_sec"] for s in data.ecg_sessions),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Parse Apple Health HealthKit export")
    parser.add_argument("--input", required=True, help="Path to export.xml")
    parser.add_argument("--ecg-dir", help="Path to electrocardiograms/ directory")
    parser.add_argument("--output", default="data/parsed", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing HealthKit export: {args.input}")
    data = parse_export_xml(args.input)

    if args.ecg_dir:
        print(f"Parsing ECG directory: {args.ecg_dir}")
        data.ecg_sessions = parse_ecg_directory(args.ecg_dir)

    # Save parsed data
    if not data.heart_rate.empty:
        data.heart_rate.to_csv(output_dir / "heart_rate.csv", index=False)
        print(f"  Heart rate: {len(data.heart_rate)} samples")

    if not data.hrv.empty:
        data.hrv.to_csv(output_dir / "hrv.csv", index=False)
        print(f"  HRV: {len(data.hrv)} samples")

    if data.ecg_sessions:
        # Save ECG voltage arrays as numpy files
        ecg_output = output_dir / "ecg"
        ecg_output.mkdir(exist_ok=True)
        for i, session in enumerate(data.ecg_sessions):
            np.save(ecg_output / f"ecg_{i:03d}.npy", session["voltage_mv"])
        print(f"  ECG: {len(data.ecg_sessions)} sessions")

    # Save summary
    summary = summarize(data)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary: {json.dumps(summary, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
