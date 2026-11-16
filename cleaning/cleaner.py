"""
cleaner.py
----------
Main audio I/O pipeline for the Mixture-of-Experts (MoE) Denoiser.

Modes:
  control : Baseline RNNoise (single generic neural denoiser).
  custom  : MoE Pipeline:
            1. Slices audio into 1-second chunks (matching Manager CNN training size).
            2. Manager CNN classifies the dominant acoustic environment (MFCC features).
            3. Dispatches each chunk to the specialized Expert denoiser.
            4. Reassembles the denoised audio with smooth transitions.

Usage:
  python cleaning/cleaner.py --input <input.wav> --output <output.wav> --mode custom
  python cleaning/cleaner.py --input <input.wav> --output <output.wav> --mode control
"""

import os
import sys
import time
import argparse

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import librosa

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cleaning.experts import build_expert_router
from pyrnnoise import RNNoise


# ─────────────────────────────────────────────────────────────────────────────
# Manager CNN Architecture (matches training/training_manager.py)
# ─────────────────────────────────────────────────────────────────────────────
class ManagerCNN(nn.Module):
    """Lightweight CNN noise classifier."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 48000
FRAME_SIZE     = 480          # 10 ms @ 48 kHz
CHUNK_SECONDS  = 1.0          # 1-second chunks matching Manager CNN
CHUNK_SAMPLES  = int(CHUNK_SECONDS * SAMPLE_RATE)  # 48,000 samples = 100 frames
N_MFCC         = 40
HOP_LENGTH     = 512

MODEL_PATH = os.path.join(_ROOT, "training", "manager_model.pth")
STATS_PATH = os.path.join(_ROOT, "training", "mfcc_stats.npz")
CMAP_PATH  = os.path.join(_ROOT, "training", "class_map.txt")


def _load_class_map(path: str) -> dict[int, str]:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {
            int(idx): name
            for line in f if line.strip()
            for idx, name in [line.strip().split("\t")]
        }


def _load_manager(model_path: str, stats_path: str, device: torch.device):
    state = torch.load(model_path, map_location=device)
    num_classes = state["classifier.3.weight"].shape[0]
    model = ManagerCNN(num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    stats = np.load(stats_path)
    mfcc_mean = float(stats["mean"])
    mfcc_std = float(stats["std"]) + 1e-9
    return model, mfcc_mean, mfcc_std, num_classes


def _classify_chunk(
    chunk: np.ndarray,
    model: ManagerCNN,
    mfcc_mean: float,
    mfcc_std: float,
    device: torch.device,
) -> tuple[int, bool]:
    """Extracts MFCCs from a 1-second chunk and detects primary and co-occurring noise."""
    # Ensure chunk is exactly CHUNK_SAMPLES
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
    else:
        chunk = chunk[:CHUNK_SAMPLES]

    mfcc = librosa.feature.mfcc(
        y=chunk,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
    )
    # Expected time steps is 94 for 1 second @ 48kHz, hop=512
    if mfcc.shape[1] < 94:
        mfcc = np.pad(mfcc, ((0, 0), (0, 94 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :94]

    mfcc_norm = (mfcc - mfcc_mean) / mfcc_std
    tensor = torch.tensor(mfcc_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)[0]
        pred = int(torch.argmax(logits).item())

        # Check for co-occurring desk EMF static
        emf_logit = float(logits[2].item())
        other_max = max(float(logits[0].item()), float(logits[1].item()), float(logits[4].item()))
        has_desk_emf = (pred == 2) or (emf_logit > -3.5 and (emf_logit - other_max) > 1.0)

    return pred, has_desk_emf


def process_audio_file(input_file: str, output_file: str, model_type: str = "control"):
    start_time = time.time()
    print(f"\nLoading '{input_file}'...")
    audio_data, sr = sf.read(input_file, dtype="float32")

    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz. Resample audio first.")

    if audio_data.ndim > 1:
        print("  Converting stereo to mono")
        audio_data = audio_data.mean(axis=1)

    orig_length = len(audio_data)
    duration_s = orig_length / SAMPLE_RATE
    print(f"Duration: {duration_s:.2f} s  |  Processing Mode: '{model_type}'")

    # ─────────────────────────────────────────────────────────────────────────
    # CONTROL: Baseline RNNoise
    # ─────────────────────────────────────────────────────────────────────────
    if model_type == "control":
        print("Processing with Baseline RNNoise (Control)...")
        # Pad to multiple of FRAME_SIZE
        rem = len(audio_data) % FRAME_SIZE
        padded_audio = np.pad(audio_data, (0, FRAME_SIZE - rem)) if rem else audio_data

        denoiser = RNNoise(SAMPLE_RATE)
        i16 = (padded_audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        audio_2d = np.expand_dims(i16, axis=0)

        out_frames = [f[0] for _, f in denoiser.denoise_chunk(audio_2d)]
        denoised_audio = np.concatenate(out_frames).astype(np.float32) / 32767.0
        denoised_audio = denoised_audio[:orig_length]

    # ─────────────────────────────────────────────────────────────────────────
    # CUSTOM: Mixture-of-Experts (MoE) Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    elif model_type == "custom":
        if not os.path.isfile(MODEL_PATH) or not os.path.isfile(STATS_PATH):
            raise FileNotFoundError(
                f"Model weights or stats not found at '{MODEL_PATH}' / '{STATS_PATH}'. "
                "Run training/training_manager.py first."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Manager CNN on {device}...")
        manager, mfcc_mean, mfcc_std, num_classes = _load_manager(MODEL_PATH, STATS_PATH, device)
        class_map = _load_class_map(CMAP_PATH)
        print(f"  Recognized classes: {class_map}")

        router = build_expert_router()

        # Split audio into 1-second chunks (48,000 samples)
        # Pad audio so the last chunk is a full multiple of FRAME_SIZE
        rem_frame = len(audio_data) % FRAME_SIZE
        padded_audio = np.pad(audio_data, (0, FRAME_SIZE - rem_frame)) if rem_frame else audio_data

        total_chunks = int(np.ceil(len(padded_audio) / CHUNK_SAMPLES))
        print(f"Routing {total_chunks} audio segments through MoE system...")

        output_chunks = []
        class_counts = {i: 0 for i in range(num_classes)}

        for c_idx in range(total_chunks):
            start = c_idx * CHUNK_SAMPLES
            end = min(start + CHUNK_SAMPLES, len(padded_audio))
            chunk = padded_audio[start:end]

            # Ensure chunk is a multiple of FRAME_SIZE for RNNoise
            rem_c = len(chunk) % FRAME_SIZE
            if rem_c:
                chunk = np.pad(chunk, (0, FRAME_SIZE - rem_c))

            # 1. Classify with Manager CNN and detect co-occurring noise
            noise_class, has_desk_emf = _classify_chunk(chunk, manager, mfcc_mean, mfcc_std, device)
            class_counts[noise_class] = class_counts.get(noise_class, 0) + 1

            # 2. Denoise with designated Expert (KeyboardClicksExpert applies desk EMF filtering)
            denoised_chunk = router.denoise_chunk(chunk, noise_class)
            output_chunks.append(denoised_chunk[: end - start])

        denoised_audio = np.concatenate(output_chunks)[:orig_length]

        # Print routing statistics
        print("\n" + "=" * 50)
        print("MoE Routing Statistics:")
        print("=" * 50)
        for c_i in range(num_classes):
            name = class_map.get(c_i, f"class_{c_i}")
            count = class_counts.get(c_i, 0)
            pct = (100.0 * count / total_chunks) if total_chunks else 0.0
            extra = " (Desk Composite with EMF)" if c_i == 3 else ""
            print(f"  [{c_i}] {name + extra:<32}: {count:>4} chunks ({pct:5.1f}%)")
        print("=" * 50)

    else:
        raise ValueError(f"Unknown mode '{model_type}'. Choose 'control' or 'custom'.")

    # Save output audio
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    sf.write(output_file, denoised_audio, SAMPLE_RATE)
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f} s ({duration_s / elapsed:.1f}x real-time)!")
    print(f"Saved: {output_file}\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Audio Denoising Pipeline (Control vs MoE Custom)")
    p.add_argument("--input", "-i", required=True, help="Input WAV file (48 kHz mono)")
    p.add_argument("--output", "-o", required=True, help="Output destination WAV path")
    p.add_argument(
        "--mode", "-m", default="control", choices=["control", "custom"],
        help="'control' = baseline RNNoise | 'custom' = MoE router (default: control)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_audio_file(
        input_file=args.input,
        output_file=args.output,
        model_type=args.mode,
    )
