import soundfile as sf
import numpy as np


def mix_audio_files(clean_file, noise_file, output_file, target_snr_db=10.0):
    """
    Mix a clean speech file with a noise file at a specified SNR (in dB).

    SNR = 20 * log10(RMS_signal / RMS_noise), so:
        noise_scale = RMS_signal / (RMS_noise * 10^(target_snr_db / 20))

    Args:
        clean_file:     Path to the clean speech WAV (must be 48 kHz, mono or stereo).
        noise_file:     Path to the noise WAV (must be 48 kHz, mono or stereo).
        output_file:    Destination path for the mixed WAV.
        target_snr_db:  Desired Signal-to-Noise Ratio in dB.
                        Higher values = quieter noise. 0 dB = equal loudness.
                        Typical training range: 0–20 dB.
    """
    print(f"Loading {clean_file} and {noise_file}...")
    clean_data, sr_clean = sf.read(clean_file, dtype='float32')
    noise_data, sr_noise = sf.read(noise_file, dtype='float32')

    # Ensure sample rates match the 48 kHz requirement
    if sr_clean != sr_noise:
        raise ValueError(
            f"Sample rates do not match! Clean: {sr_clean} Hz, Noise: {sr_noise} Hz. "
            "Please resample both files to 48000 Hz using convert.py first."
        )

    # Collapse stereo → mono by averaging channels, keeping arrays 1-D
    if clean_data.ndim > 1:
        clean_data = clean_data.mean(axis=1)
    if noise_data.ndim > 1:
        noise_data = noise_data.mean(axis=1)

    # Trim both arrays to the same length (use the shorter one)
    min_length = min(len(clean_data), len(noise_data))
    clean_data = clean_data[:min_length]
    noise_data = noise_data[:min_length]

    # Compute RMS of each signal (add tiny epsilon to avoid division-by-zero on silence)
    rms_clean = np.sqrt(np.mean(clean_data ** 2)) + 1e-9
    rms_noise = np.sqrt(np.mean(noise_data ** 2)) + 1e-9

    # Scale the noise so the mix achieves the requested SNR
    noise_scale = rms_clean / (rms_noise * (10 ** (target_snr_db / 20.0)))
    print(
        f"Mixing at {target_snr_db} dB SNR "
        f"(noise amplitude scale factor: {noise_scale:.4f})..."
    )
    mixed_data = clean_data + noise_data * noise_scale

    # Hard-limit to [-1.0, 1.0] to prevent clipping artifacts
    max_amplitude = np.max(np.abs(mixed_data))
    if max_amplitude > 1.0:
        print("Peak exceeds 1.0 — normalizing to prevent clipping...")
        mixed_data = mixed_data / max_amplitude

    print(f"Saving mixed training data to {output_file}...")
    sf.write(output_file, mixed_data, sr_clean)
    print("Done!")


if __name__ == "__main__":
    # Update these filenames to match your actual 5-minute clips
    mix_audio_files(
        clean_file="audiobook_5min.wav",    # Pure speech file
        noise_file="plane_5min.wav",        # Pure noise file
        output_file="training_mix_1.wav",   # Final mixed output
        target_snr_db=10.0,                 # 10 dB SNR — noise is clearly audible but speech dominates
    )
