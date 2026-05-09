"""Extract TCM-relevant features from Apple Watch ECG waveform data.

Uses neurokit2 for ECG signal processing. Produces a feature vector
suitable for TCM pulse classification.

Usage:
    python -m src.feature_extraction.ecg_features --input data/parsed/ecg/ --output data/features/
"""

import argparse
from pathlib import Path
import json

import numpy as np

try:
    import neurokit2 as nk
except ImportError:
    raise ImportError("neurokit2 is required: pip install neurokit2")


SAMPLING_RATE = 512  # Apple Watch ECG fixed at 512 Hz


def assess_signal_quality(voltage_mv: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """Check ECG signal quality before feature extraction."""
    quality = {"usable": True, "issues": []}

    if len(voltage_mv) < fs * 5:
        quality["usable"] = False
        quality["issues"].append(f"Too short: {len(voltage_mv)/fs:.1f}s (need ≥5s)")

    # Check for flatline
    if np.std(voltage_mv) < 0.01:
        quality["usable"] = False
        quality["issues"].append("Flatline signal (no variation)")

    # Check for saturation/clipping
    max_val = np.max(np.abs(voltage_mv))
    if max_val > 3.0:
        quality["issues"].append(f"Possible clipping: max amplitude {max_val:.2f}mV")

    # Check for excessive noise (high-frequency power)
    if len(voltage_mv) > fs:
        fft = np.fft.rfft(voltage_mv[:fs])
        freqs = np.fft.rfftfreq(fs, 1/fs)
        high_freq_power = np.sum(np.abs(fft[freqs > 40])**2)
        total_power = np.sum(np.abs(fft)**2)
        noise_ratio = high_freq_power / total_power if total_power > 0 else 0
        if noise_ratio > 0.5:
            quality["issues"].append(f"High noise ratio: {noise_ratio:.2f}")
            quality["usable"] = False

    quality["duration_sec"] = len(voltage_mv) / fs
    quality["amplitude_range_mv"] = float(np.ptp(voltage_mv))

    return quality


def extract_ecg_features(voltage_mv: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """Extract TCM-relevant features from ECG voltage data.

    Returns a dict of features mapped to TCM pulse parameters:
    - Rate features → 迟/数/疾 classification
    - Rhythm features → 促/结/代 classification
    - Amplitude features → 洪/细/弱 classification
    - HRV features → 弦/缓 classification
    - Smoothness features → 滑/涩 classification
    """
    # Clean the signal
    ecg_cleaned = nk.ecg_clean(voltage_mv, sampling_rate=fs)

    # Detect R-peaks
    _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
    rpeaks = rpeaks_info["ECG_R_Peaks"]

    if len(rpeaks) < 4:
        return {"error": "Too few R-peaks detected (need ≥4)", "n_peaks": len(rpeaks)}

    # === RR Intervals ===
    rr_samples = np.diff(rpeaks)
    rr_ms = rr_samples / fs * 1000  # Convert to milliseconds
    rr_sec = rr_ms / 1000

    # === Rate Features (频率特征) ===
    mean_hr = 60.0 / np.mean(rr_sec)
    min_hr = 60.0 / np.max(rr_sec)
    max_hr = 60.0 / np.min(rr_sec)

    # === Rhythm Features (节律特征) ===
    rr_diff = np.diff(rr_ms)
    rr_cv = np.std(rr_ms) / np.mean(rr_ms)  # Coefficient of variation

    # Detect premature beats (RR < 80% of mean)
    mean_rr = np.mean(rr_ms)
    premature_beats = np.sum(rr_ms < mean_rr * 0.8)

    # Detect pauses (RR > 150% of mean)
    pauses = np.sum(rr_ms > mean_rr * 1.5)

    # Detect regular pauses (代脉: fixed-interval dropped beats)
    regular_pause = _detect_regular_pauses(rr_ms)

    # === Amplitude Features (振幅特征) ===
    r_amplitudes = ecg_cleaned[rpeaks]
    mean_r_amplitude = float(np.mean(r_amplitudes))
    r_amplitude_std = float(np.std(r_amplitudes))

    # === HRV Time-Domain (心率变异性时域) ===
    sdnn = float(np.std(rr_ms))  # Standard deviation of NN intervals
    rmssd = float(np.sqrt(np.mean(rr_diff**2)))  # Root mean square of successive differences
    pnn50 = float(np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100)  # % of successive RR > 50ms

    # === HRV Frequency-Domain (心率变异性频域) ===
    lf_hf_ratio = _compute_lf_hf_ratio(rr_ms)

    # === Smoothness Features (流畅度特征) ===
    # Poincare plot SD1 (short-term variability)
    sd1 = float(np.std(rr_diff) / np.sqrt(2))
    # Successive RR transition smoothness
    transition_roughness = float(np.mean(np.abs(np.diff(rr_diff))))

    # === Assemble Feature Vector ===
    features = {
        # Rate (频率) → 迟/数/疾
        "mean_hr_bpm": round(mean_hr, 1),
        "min_hr_bpm": round(min_hr, 1),
        "max_hr_bpm": round(max_hr, 1),

        # Rhythm (节律) → 促/结/代
        "rr_cv": round(rr_cv, 4),
        "premature_beat_count": int(premature_beats),
        "pause_count": int(pauses),
        "has_regular_pauses": regular_pause,
        "total_beats": len(rpeaks),

        # Amplitude (振幅) → 洪/细/弱
        "mean_r_amplitude_mv": round(mean_r_amplitude, 3),
        "r_amplitude_std_mv": round(r_amplitude_std, 3),
        "r_amplitude_cv": round(r_amplitude_std / abs(mean_r_amplitude), 4) if mean_r_amplitude != 0 else 0,

        # HRV Time-domain → 弦/缓
        "sdnn_ms": round(sdnn, 1),
        "rmssd_ms": round(rmssd, 1),
        "pnn50_pct": round(pnn50, 1),

        # HRV Frequency-domain → 弦/紧/缓
        "lf_hf_ratio": round(lf_hf_ratio, 2) if lf_hf_ratio is not None else None,

        # Smoothness (流畅度) → 滑/涩
        "poincare_sd1_ms": round(sd1, 1),
        "transition_roughness_ms": round(transition_roughness, 1),

        # Meta
        "duration_sec": round(len(voltage_mv) / fs, 1),
        "n_beats": len(rpeaks),
        "sampling_rate": fs,
    }

    return features


def _detect_regular_pauses(rr_ms: np.ndarray) -> bool:
    """Detect 代脉 pattern: regular pauses at fixed intervals.

    Looks for RR intervals significantly longer than mean that occur
    at roughly equal spacing.
    """
    mean_rr = np.mean(rr_ms)
    pause_indices = np.where(rr_ms > mean_rr * 1.5)[0]

    if len(pause_indices) < 2:
        return False

    # Check if pauses are evenly spaced
    pause_spacing = np.diff(pause_indices)
    if len(pause_spacing) < 1:
        return False

    spacing_cv = np.std(pause_spacing) / np.mean(pause_spacing) if np.mean(pause_spacing) > 0 else 1
    return spacing_cv < 0.2  # Regular if CV < 20%


def _compute_lf_hf_ratio(rr_ms: np.ndarray) -> float | None:
    """Compute LF/HF ratio from RR intervals using Welch's method.

    LF (0.04-0.15 Hz): sympathetic + parasympathetic
    HF (0.15-0.40 Hz): parasympathetic
    LF/HF > 2.0: sympathetic dominant (弦脉 candidate)
    LF/HF < 0.5: parasympathetic dominant (缓脉 candidate)
    """
    from scipy import signal as sig

    if len(rr_ms) < 8:
        return None

    # Interpolate RR intervals to uniform time series (4 Hz)
    rr_sec = rr_ms / 1000
    cumulative_time = np.cumsum(rr_sec)
    interp_rate = 4.0  # Hz
    time_uniform = np.arange(cumulative_time[0], cumulative_time[-1], 1/interp_rate)

    rr_interpolated = np.interp(time_uniform, cumulative_time, rr_ms)
    rr_interpolated -= np.mean(rr_interpolated)  # Detrend

    # Welch PSD
    nperseg = min(len(rr_interpolated), int(interp_rate * 60))  # Max 60s window
    if nperseg < 16:
        return None

    freqs, psd = sig.welch(rr_interpolated, fs=interp_rate, nperseg=nperseg)

    # Integrate LF and HF bands
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])

    if hf_power < 1e-10:
        return None

    return lf_power / hf_power


def extract_from_hrv_samples(sdnn_values: list[float]) -> dict:
    """Extract tension features from HealthKit HRV (SDNN) samples.

    Supplements ECG-derived features when ECG is not available.
    """
    if not sdnn_values:
        return {}

    arr = np.array(sdnn_values)
    return {
        "hrv_mean_sdnn_ms": round(float(np.mean(arr)), 1),
        "hrv_std_sdnn_ms": round(float(np.std(arr)), 1),
        "hrv_min_sdnn_ms": round(float(np.min(arr)), 1),
        "hrv_max_sdnn_ms": round(float(np.max(arr)), 1),
        "hrv_trend": "increasing" if arr[-1] > arr[0] else "decreasing" if arr[-1] < arr[0] else "stable",
    }


def main():
    parser = argparse.ArgumentParser(description="Extract TCM features from ECG data")
    parser.add_argument("--input", required=True, help="Path to ECG .npy files directory or single file")
    parser.add_argument("--output", default="data/features", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        ecg_files = [input_path]
    else:
        ecg_files = sorted(input_path.glob("*.npy"))

    if not ecg_files:
        print(f"No .npy ECG files found in {input_path}")
        return

    all_features = []

    for ecg_file in ecg_files:
        print(f"Processing: {ecg_file.name}")
        voltage = np.load(ecg_file)

        quality = assess_signal_quality(voltage)
        if not quality["usable"]:
            print(f"  Skipped (poor quality): {quality['issues']}")
            continue

        features = extract_ecg_features(voltage)
        if "error" in features:
            print(f"  Skipped: {features['error']}")
            continue

        features["source_file"] = ecg_file.name
        features["quality"] = quality
        all_features.append(features)
        print(f"  HR={features['mean_hr_bpm']} bpm, SDNN={features['sdnn_ms']}ms, "
              f"R-amp={features['mean_r_amplitude_mv']}mV")

    # Save all features
    output_file = output_dir / "ecg_features.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_features, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted features from {len(all_features)}/{len(ecg_files)} ECG sessions")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
