import soundfile as sf
import numpy as np
from pyrnnoise import RNNoise

def process_audio_file(input_file, output_file, model_type="control"):
    """
    Reads an audio file, processes it through a chosen ML model, and saves the output.
    """
    print(f"Loading {input_file}...")
    audio_data, sample_rate = sf.read(input_file, dtype='float32')
    
    if sample_rate != 48000:
        raise ValueError(f"Expected 48000 Hz, got {sample_rate} Hz. Please resample first.")

    # Convert to 16-bit PCM format scale 
    audio_data = (audio_data * 32767).astype(np.int16)

    # Note: While the RNNoise window is 20ms, it overlaps by 50%. 
    # Therefore, the algorithm actually processes audio in 10ms steps (480 samples)
    frame_size = 480 
    
    # Pad the audio so it divides perfectly into 480-sample frames
    padding = frame_size - (len(audio_data) % frame_size)
    if padding != frame_size:
        audio_data = np.pad(audio_data, (0, padding), mode='constant')

    print(f"Processing with {model_type} model...")
    
    # ---------------------------------------------------------
    # ALGORITHM 1: The Control Model (Baseline RNNoise)
    # ---------------------------------------------------------
    if model_type == "control":
        denoiser = RNNoise(48000)
        
        # pyrnnoise expects a 2D array shaped [channels, samples]
        audio_data_2d = np.expand_dims(audio_data, axis=0)
        denoised_frames = []
        
        # denoise_chunk automatically handles the channel setup and 480-sample loops!
        for vad_prob, denoised_frame in denoiser.denoise_chunk(audio_data_2d):
            # denoised_frame is returned as shape [1, 480]
            denoised_frames.append(denoised_frame[0]) 
            
        # Stitch the 480-sample chunks back into a 1D array
        denoised_audio = np.concatenate(denoised_frames)

    # ---------------------------------------------------------
    # ALGORITHM 2: Your Future Alternative Model
    # ---------------------------------------------------------
    elif model_type == "custom":
        print("Custom model not yet implemented! Drop your PyTorch/TensorFlow model here.")
        denoised_audio = np.zeros_like(audio_data, dtype=np.float32)
        
        # For your custom model, you will loop through frame_size
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i : i + frame_size]
            # 1. Extract features (e.g., Bark bands or MFCCs) 
            # 2. Feed features to your custom LSTM or CNN model
            # 3. Apply the output gain mask to the audio frame
            # denoised_audio[i : i + frame_size] = your_output_frame
            pass

    else:
        raise ValueError("Invalid model_type. Choose 'control' or 'custom'.")

    # Normalize audio back to float32 range [-1.0, 1.0] for saving
    denoised_audio = denoised_audio / 32767.0

    print(f"Saving denoised audio to {output_file}...")
    sf.write(output_file, denoised_audio, sample_rate)
    print("Done!")

if __name__ == "__main__":
    process_audio_file(
        input_file="noisy_input.wav", 
        output_file="cleaned_output.wav", 
        model_type="control" 
    )
