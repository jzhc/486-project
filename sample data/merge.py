import soundfile as sf
import numpy as np

def mix_audio_files(clean_file, noise_file, output_file, noise_multiplier=0.5):
    print(f"Loading {clean_file} and {noise_file}...")
    
    # Read both audio files directly into NumPy arrays
    clean_data, sr_clean = sf.read(clean_file, dtype='float32')
    noise_data, sr_noise = sf.read(noise_file, dtype='float32')
    
    # Ensure sample rates match the 48 kHz requirement
    if sr_clean != sr_noise:
        raise ValueError(f"Sample rates do not match! Clean: {sr_clean}, Noise: {sr_noise}")
        
    # Trim both arrays to the exact same length (the length of the shortest file)
    min_length = min(len(clean_data), len(noise_data))
    clean_data = clean_data[:min_length]
    noise_data = noise_data[:min_length]
    
    print(f"Mixing tracks together with a noise multiplier of {noise_multiplier}...")
    
    # Mathematically add the arrays together. 
    # The noise_multiplier acts as your Signal-to-Noise Ratio (SNR) control.
    mixed_data = clean_data + (noise_data * noise_multiplier)
    
    # Prevent audio clipping (if the combined volume goes above 1.0 or below -1.0)
    max_amplitude = np.max(np.abs(mixed_data))
    if max_amplitude > 1.0:
        print("Combined volume exceeds 1.0. Normalizing audio to prevent clipping...")
        mixed_data = mixed_data / max_amplitude
        
    print(f"Saving mixed training data to {output_file}...")
    sf.write(output_file, mixed_data, sr_clean)
    print("Done!")

if __name__ == "__main__":
    # Update these filenames to match your actual 5-minute clips
    mix_audio_files(
        clean_file="audiobook_5min.wav",   # Your pure speech file
        noise_file="plane_5min.wav",    # Your pure noise file
        output_file="training_mix_1.wav",     # The final merged output
        noise_multiplier=0.5                  # 0.5 means noise is at 50% volume. 1.0 is 100%.
    )
