import librosa
import soundfile as sf

input_file = "office speech noisy.flac" # Replace with your actual file name
output_file = "noisy_input.wav"

print(f"Loading and resampling {input_file} to 48kHz...")
# librosa.load will automatically convert the 96kHz flac to a 48kHz numpy array
audio_data, sample_rate = librosa.load(input_file, sr=48000)

print(f"Saving as {output_file}...")
# Save it as a standard WAV file
sf.write(output_file, audio_data, 48000)
print("Done! You can now use this with test_io.py")
