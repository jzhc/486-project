"""
mix_noise.py  —  Multi-class noise mixing pipeline
----------------------------------------------------
Mixes a clean speech file with a noise file at a specified SNR (dB),
producing a labeled training mixture for extract_features.py.

This script supersedes the old merge.py and handles:
  • Automatic stereo → mono conversion
  • True dB-based SNR control (not a raw amplitude multiplier)
  • Multiple noise classes in one pass via CLASS_CONFIGS below

Usage
-----
1. Add your raw (48 kHz) noise WAV to the appropriate training/ subdirectory.
2. Update CLASS_CONFIGS with the new entry.
3. Run:

       python training/mix_noise.py

   Each entry produces one training mix WAV, e.g.
       training/emf_static/training_mix_emf.wav
"""

import soundfile as sf
import numpy as np
import os


# ─── Configure one entry per noise class ──────────────────────────────────────
# Each entry: (clean_speech_file, noise_file, output_mix_file, snr_db)
#
# SNR guidelines (from acoustic research):
#   plane ambiance  → 10 dB   (noise clearly audible, speech dominant)
#   EMF static      →  5 dB   (tonal interference; model needs hard examples)
#   keyboard clicks →  0 dB   (impulsive; must train at equal loudness)
#
CLASS_CONFIGS = [
    # Class 1 — Plane ambiance (existing)
    {
        "clean_file":   "training/audiobook_plane/clean_speech_5min.wav",
        "noise_file":   "sample data/plane_5min.wav",
        "output_file":  "training/audiobook_plane/training_mix_1.wav",
        "snr_db":       10.0,
        "label":        "plane_ambiance",
    },
    # Class 2 — EMF / electrical static  ← NEW
    # Record your own: hold a microphone 10–20 cm from your laptop PSU,
    # a fluorescent light, or an unshielded USB device.  5 minutes at 48 kHz.
    {
        "clean_file":   "training/audiobook_plane/clean_speech_5min.wav",
        "noise_file":   "training/emf_static/emf_raw_48k.wav",   # <-- add this file
        "output_file":  "training/emf_static/training_mix_emf.wav",
        "snr_db":       5.0,
        "label":        "emf_static",
    },
    # Class 3 — Keyboard clicks  ← NEW
    # Record your own keyboard for 5 minutes (continuous typing) at 48 kHz.
    # SNR raised from 0 dB → 8 dB: at 0 dB the noise was as loud as speech
    # (60/40 noise/speech perceptually), making the class too hard to suppress
    # and skewing the classifier.  8 dB keeps clicks audible but speech dominant.
    {
        "clean_file":   "training/audiobook_plane/clean_speech_5min.wav",
        "noise_file":   "training/keyboard_clicks/keyboard_raw_48k.wav",
        "output_file":  "training/keyboard_clicks/training_mix_keyboard.wav",
        "snr_db":       8.0,
        "label":        "keyboard_clicks",
    },
    # Class 4 — Fan hum  ← NEW
    # Stationary low-frequency drone; SNR 8 dB keeps it clearly audible while
    # speech remains dominant.  Source: sample data/fan_48k.wav (48 kHz mono).
    {
        "clean_file":   "training/audiobook_plane/clean_speech_5min.wav",
        "noise_file":   "sample data/fan_48k.wav",
        "output_file":  "training/fan_hum/training_mix_fan.wav",
        "snr_db":       8.0,
        "label":        "fan_hum",
    },
]


# ─── Core mixing function ──────────────────────────────────────────────────────
def mix_at_snr(clean_file: str, noise_file: str, output_file: str,
               snr_db: float, label: str) -> None:
    """
    Mix clean speech with noise at a specified SNR and save the result.

    SNR is defined as:
        SNR (dB) = 20 * log10(RMS_signal / RMS_noise)

    Rearranging for the required noise amplitude scale factor:
        noise_scale = RMS_signal / (RMS_noise * 10^(snr_db / 20))

    Args
    ----
    clean_file : Path to clean speech WAV (48 kHz, any channel count).
    noise_file : Path to noise WAV (48 kHz, any channel count).
    output_file: Destination path for the mixed WAV.
    snr_db     : Target Signal-to-Noise Ratio in dB.
    label      : Human-readable name, used only in log messages.
    """
    if not os.path.isfile(clean_file):
        raise FileNotFoundError(f"[{label}] Clean speech file not found: {clean_file}")
    if not os.path.isfile(noise_file):
        print(
            f"[{label}] SKIP — noise file not found: {noise_file}\n"
            f"  Record the noise and place it at that path, then re-run."
        )
        return

    print(f"\n[{label}] Mixing at {snr_db} dB SNR")

    clean_data, sr_clean = sf.read(clean_file, dtype='float32')
    noise_data, sr_noise = sf.read(noise_file, dtype='float32')

    if sr_clean != 48000:
        raise ValueError(f"[{label}] Clean file must be 48 kHz, got {sr_clean} Hz.")
    if sr_noise != 48000:
        raise ValueError(f"[{label}] Noise file must be 48 kHz, got {sr_noise} Hz. "
                         "Re-run convert.py on the noise file first.")

    # Collapse stereo → mono by averaging channels
    if clean_data.ndim > 1:
        clean_data = clean_data.mean(axis=1)
        print(f"  Clean: stereo → mono")
    if noise_data.ndim > 1:
        noise_data = noise_data.mean(axis=1)
        print(f"  Noise: stereo → mono")

    # Trim to the shorter file's length
    min_length = min(len(clean_data), len(noise_data))
    clean_data = clean_data[:min_length]
    noise_data = noise_data[:min_length]
    duration_s = min_length / sr_clean
    print(f"  Duration: {duration_s:.1f} s  ({min_length} samples)")

    # Compute RMS (add epsilon to guard against silence files)
    rms_clean = float(np.sqrt(np.mean(clean_data ** 2))) + 1e-9
    rms_noise = float(np.sqrt(np.mean(noise_data ** 2))) + 1e-9

    noise_scale = rms_clean / (rms_noise * (10 ** (snr_db / 20.0)))
    print(f"  RMS clean={rms_clean:.5f} | RMS noise={rms_noise:.5f} | "
          f"noise_scale={noise_scale:.5f}")

    mixed_data = clean_data + noise_data * noise_scale

    # Hard-limit to prevent clipping artifacts
    peak = float(np.max(np.abs(mixed_data)))
    if peak > 1.0:
        print(f"  Peak={peak:.3f} > 1.0 — normalising to prevent clipping.")
        mixed_data = mixed_data / peak

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sf.write(output_file, mixed_data, sr_clean)
    print(f"  Saved → {output_file}")


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-class Noise Mixing Pipeline")
    print("=" * 60)

    for cfg in CLASS_CONFIGS:
        mix_at_snr(
            clean_file=cfg["clean_file"],
            noise_file=cfg["noise_file"],
            output_file=cfg["output_file"],
            snr_db=cfg["snr_db"],
            label=cfg["label"],
        )

    print("\nAll done! Update CLASS_MAP in extract_features.py and run it next.")
