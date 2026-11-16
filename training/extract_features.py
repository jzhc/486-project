"""
extract_features.py
-------------------
Slices audio files into 1-second chunks, extracts 40-band MFCCs, and saves
(X_features.npy, y_labels.npy) ready for training_manager.py.

Usage
-----
Run from the project root or the training/ directory.  Edit CLASS_MAP below
to add or remove noise classes, then run:

    python training/extract_features.py

Output
------
training/X_features.npy  — shape (N, 40, TIME_STEPS), float32, z-score normalised
training/y_labels.npy    — shape (N,), int64 class indices
training/class_map.txt   — human-readable label→index mapping
"""

import librosa
import numpy as np
import soundfile as sf
import os

# ---------------------------------------------------------
# CONFIGURATION — edit this map to add new noise classes.
# Key  : integer class label used by the CNN
# Value: (audio_file_path, human_readable_name)
# ---------------------------------------------------------
CLASS_MAP = {
    0: ("training/audiobook_plane/clean_speech_5min.wav",    "clean_speech"),
    1: ("training/audiobook_plane/training_mix_1.wav",        "plane_ambiance"),
    2: ("training/emf_static/training_mix_emf.wav",           "emf_static"),
    3: ("training/keyboard_clicks/training_mix_keyboard.wav", "keyboard_clicks"),
    4: ("training/fan_hum/training_mix_fan.wav",              "fan_hum"),
}

# Audio / feature settings
SAMPLE_RATE   = 48000   # All files must be resampled to 48 kHz
CHUNK_SECONDS = 1.0     # Window length fed to MFCC extractor
N_MFCC        = 40      # Number of MFCC bands (matches CNN input size)
HOP_LENGTH    = 512     # ~10.7 ms at 48 kHz; keeps time-steps consistent

OUTPUT_DIR    = "training"


# ---------------------------------------------------------
def extract_mfcc_features(audio_path: str, label: int, chunk_seconds: float = CHUNK_SECONDS):
    """
    Read one audio file, slice it into fixed-length chunks, and extract MFCCs.

    Returns
    -------
    features : list of np.ndarray, each shaped (N_MFCC, TIME_STEPS)
    labels   : list of ints
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: '{audio_path}'\n"
            "Check CLASS_MAP paths or run mix_audio (merge.py) first."
        )

    print(f"  Loading '{audio_path}'...")
    audio_data, sample_rate = sf.read(audio_path, dtype='float32')

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz in '{audio_path}'. "
            "Re-run convert.py on this file."
        )

    # Collapse stereo → mono by averaging channels
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    samples_per_chunk = int(chunk_seconds * sample_rate)

    # Pre-compute the number of time steps a full chunk produces so every
    # MFCC array has the exact same shape (required by the CNN).
    dummy_mfcc = librosa.feature.mfcc(
        y=np.zeros(samples_per_chunk, dtype=np.float32),
        sr=sample_rate,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
    )
    expected_time_steps = dummy_mfcc.shape[1]

    features, labels = [], []

    # Slide through the file in non-overlapping 1-second windows.
    # We deliberately exclude the last partial chunk to keep shapes uniform.
    n_chunks = len(audio_data) // samples_per_chunk
    for i in range(n_chunks):
        start  = i * samples_per_chunk
        chunk  = audio_data[start : start + samples_per_chunk]

        mfcc = librosa.feature.mfcc(
            y=chunk,
            sr=sample_rate,
            n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH,
        )

        # Safety: pad / trim to exactly expected_time_steps in case of rounding
        if mfcc.shape[1] < expected_time_steps:
            mfcc = np.pad(mfcc, ((0, 0), (0, expected_time_steps - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :expected_time_steps]

        features.append(mfcc)
        labels.append(label)

    print(f"    → {len(features)} chunks extracted (shape per chunk: {features[0].shape})")
    return features, labels


# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("Feature Extraction Pipeline")
    print("=" * 60)

    all_features, all_labels = [], []

    for label, (path, name) in sorted(CLASS_MAP.items()):
        print(f"\n[Class {label}] {name}")
        feats, labs = extract_mfcc_features(path, label)
        all_features.extend(feats)
        all_labels.extend(labs)

    X = np.array(all_features, dtype=np.float32)  # (N, N_MFCC, TIME_STEPS)
    y = np.array(all_labels,   dtype=np.int64)     # (N,)

    print(f"\nTotal dataset shape : X={X.shape}, y={y.shape}")
    print(f"Class distribution  : { {cls: int((y == cls).sum()) for cls in np.unique(y)} }")

    # --- Z-score normalise across the entire dataset ---
    # Fit mean/std on the training set so the CNN sees unit-variance inputs.
    # We store the stats so they can be reloaded at inference time.
    X_mean = X.mean()
    X_std  = X.std() + 1e-9   # Avoid division by zero on silent files
    X_norm = (X - X_mean) / X_std

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x_path     = os.path.join(OUTPUT_DIR, "X_features.npy")
    y_path     = os.path.join(OUTPUT_DIR, "y_labels.npy")
    stats_path = os.path.join(OUTPUT_DIR, "mfcc_stats.npz")
    map_path   = os.path.join(OUTPUT_DIR, "class_map.txt")

    np.save(x_path, X_norm)
    np.save(y_path, y)
    np.savez(stats_path, mean=X_mean, std=X_std)

    with open(map_path, "w") as f:
        for label, (_, name) in sorted(CLASS_MAP.items()):
            f.write(f"{label}\t{name}\n")

    print(f"\nSaved:")
    print(f"  {x_path}  (normalised MFCCs)")
    print(f"  {y_path}  (class labels)")
    print(f"  {stats_path}  (mean={X_mean:.4f}, std={X_std:.4f})")
    print(f"  {map_path}")
    print("\nDone! Run training_manager.py next.")


if __name__ == "__main__":
    main()
