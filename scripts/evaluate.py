import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

def time_to_sample_index(t: float, sample_rate: int, n_samples: int) -> int:
    if t < 0:
        raise ValueError("time values must be nonnegative")
    idx = int(round(t * sample_rate))
    return max(0, min(idx, n_samples))


def extract_time_region(signal: np.ndarray, sr: int, t1: float, t2: float):
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")

    n = len(signal)
    duration = n / sr

    if t1 >= duration:
        raise ValueError(f"t1={t1} exceeds or equals signal duration of {duration:.6f} s")
    if t2 > duration:
        raise ValueError(f"t2={t2} exceeds signal duration of {duration:.6f} s")

    start = time_to_sample_index(t1, sr, n)
    end = time_to_sample_index(t2, sr, n)

    if end <= start:
        raise ValueError("Selected evaluation region is empty after time-to-sample conversion")

    return signal[start:end], start, end

def save_region_waveform_plot(
    original: np.ndarray,
    modified: np.ndarray,
    sr: int,
    t1: float,
    t2: float,
    margin: float,
    output_path: Path,
):
    if margin < 0:
        raise ValueError("margin must be nonnegative")

    full_duration_orig = len(original) / sr
    full_duration_mod = len(modified) / sr

    view_start = max(0.0, t1 - margin)
    view_end_orig = min(full_duration_orig, t2 + margin)
    view_end_mod = min(full_duration_mod, t2 + margin)

    orig_region, orig_start, orig_end = extract_time_region(original, sr, view_start, view_end_orig)
    mod_region, mod_start, mod_end = extract_time_region(modified, sr, view_start, view_end_mod)

    t_orig = np.arange(orig_start, orig_end) / sr
    t_mod = np.arange(mod_start, mod_end) / sr

    plt.figure(figsize=(12, 6))
    plt.plot(t_orig, orig_region, label="Original")
    plt.plot(t_mod, mod_region, label="Modified", alpha=0.8)
    plt.axvline(t1, linestyle="--", linewidth=1)
    plt.axvline(t2, linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform Comparison (Region {t1:.3f}s to {t2:.3f}s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def save_region_spectrogram(
    signal: np.ndarray,
    sr: int,
    t1: float,
    t2: float,
    margin: float,
    title: str,
    output_path: Path,
):
    if margin < 0:
        raise ValueError("margin must be nonnegative")

    duration = len(signal) / sr
    view_start = max(0.0, t1 - margin)
    view_end = min(duration, t2 + margin)

    region, _, _ = extract_time_region(signal, sr, view_start, view_end)

    freqs, times, spec = spectrogram(
        region,
        fs=sr,
        window="hann",
        nperseg=512,
        noverlap=256,
        mode="magnitude",
    )

    spec_db = 20 * np.log10(spec + 1e-10)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(times + view_start, freqs, spec_db, shading="gouraud")
    plt.axvline(t1, linestyle="--", linewidth=1)
    plt.axvline(t2, linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def write_region_metrics(
    original: np.ndarray,
    modified: np.ndarray,
    sr: int,
    t1: float,
    t2: float,
    output_path: Path,
):
    original_region, _, _ = extract_time_region(original, sr, t1, t2)
    modified_region, _, _ = extract_time_region(modified, sr, t1, t2)

    metrics = {
        "region_t1_s": t1,
        "region_t2_s": t2,
        "region_requested_duration_s": t2 - t1,
        "original_region_num_samples": len(original_region),
        "modified_region_num_samples": len(modified_region),
        "original_region_duration_s": len(original_region) / sr,
        "modified_region_duration_s": len(modified_region) / sr,
        "original_region_rms": rms(original_region),
        "modified_region_rms": rms(modified_region),
        "original_region_peak_abs": float(np.max(np.abs(original_region))),
        "modified_region_peak_abs": float(np.max(np.abs(modified_region))),
    }

    with output_path.open("w") as f:
        f.write("Region Evaluation Metrics\n")
        f.write("=========================\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def load_wav_float(path: str):
    wav_path = Path(path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    sample_rate, data = wavfile.read(wav_path)

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    if data.ndim != 1:
        raise ValueError("Evaluation currently supports mono WAV files only")
    if data.size == 0:
        raise ValueError("Audio file is empty")

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        data = data.astype(np.float32) / float(scale)
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV dtype: {data.dtype}")

    if not np.all(np.isfinite(data)):
        raise ValueError("Audio contains non-finite values")

    return sample_rate, data


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def save_waveform_plot(original: np.ndarray, modified: np.ndarray, sr: int, output_path: Path):
    t_orig = np.arange(len(original)) / sr
    t_mod = np.arange(len(modified)) / sr

    plt.figure(figsize=(12, 6))
    plt.plot(t_orig, original, label="Original")
    plt.plot(t_mod, modified, label="Modified", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_single_spectrogram(signal: np.ndarray, sr: int, title: str, output_path: Path):
    freqs, times, spec = spectrogram(
        signal,
        fs=sr,
        window="hann",
        nperseg=512,
        noverlap=256,
        mode="magnitude",
    )

    spec_db = 20 * np.log10(spec + 1e-10)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(times, freqs, spec_db, shading="gouraud")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_metrics(
    original: np.ndarray,
    modified: np.ndarray,
    sr: int,
    output_path: Path,
):
    original_duration = len(original) / sr
    modified_duration = len(modified) / sr

    duration_ratio = modified_duration / original_duration if original_duration > 0 else float("nan")

    metrics = {
        "sample_rate_hz": sr,
        "original_num_samples": len(original),
        "modified_num_samples": len(modified),
        "original_duration_s": original_duration,
        "modified_duration_s": modified_duration,
        "duration_ratio_modified_over_original": duration_ratio,
        "original_rms": rms(original),
        "modified_rms": rms(modified),
        "original_peak_abs": float(np.max(np.abs(original))),
        "modified_peak_abs": float(np.max(np.abs(modified))),
    }

    with output_path.open("w") as f:
        f.write("Evaluation Metrics\n")
        f.write("==================\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate original vs modified speech audio")
    parser.add_argument("--original", required=True, help="Path to original WAV file")
    parser.add_argument("--modified", required=True, help="Path to modified WAV file")
    parser.add_argument("--report-dir", required=True, help="Directory to save plots and metrics")
    parser.add_argument("--t1", type=float, help="Optional edited-region start time in seconds for focused evaluation")
    parser.add_argument("--t2", type=float, help="Optional edited-region end time in seconds for focused evaluation")
    parser.add_argument("--margin", type=float, default=0.1, help="Extra time margin around the edited region for plots")

    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    sr_orig, original = load_wav_float(args.original)
    sr_mod, modified = load_wav_float(args.modified)

    if sr_orig != sr_mod:
        raise ValueError(
            f"Sample rates do not match: original={sr_orig}, modified={sr_mod}"
        )

    save_waveform_plot(original, modified, sr_orig, report_dir / "waveform_comparison.png")
    save_single_spectrogram(original, sr_orig, "Original Spectrogram", report_dir / "spectrogram_original.png")
    save_single_spectrogram(modified, sr_mod, "Modified Spectrogram", report_dir / "spectrogram_modified.png")
    write_metrics(original, modified, sr_orig, report_dir / "metrics.txt")

    if (args.t1 is None) != (args.t2 is None):
        raise ValueError("Provide both --t1 and --t2 for region-focused evaluation")

    if args.t1 is not None and args.t2 is not None:
        save_region_waveform_plot(
            original,
            modified,
            sr_orig,
            t1=args.t1,
            t2=args.t2,
            margin=args.margin,
            output_path=report_dir / "waveform_region_comparison.png",
        )

        save_region_spectrogram(
            original,
            sr_orig,
            t1=args.t1,
            t2=args.t2,
            margin=args.margin,
            title="Original Spectrogram (Region)",
            output_path=report_dir / "spectrogram_region_original.png",
        )

        save_region_spectrogram(
            modified,
            sr_mod,
            t1=args.t1,
            t2=args.t2,
            margin=args.margin,
            title="Modified Spectrogram (Region)",
            output_path=report_dir / "spectrogram_region_modified.png",
        )

        write_region_metrics(
            original,
            modified,
            sr_orig,
            t1=args.t1,
            t2=args.t2,
            output_path=report_dir / "metrics_region.txt",
        )

    print(f"Saved evaluation outputs to: {report_dir}")


if __name__ == "__main__":
    main()