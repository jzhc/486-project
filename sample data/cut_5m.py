import soundfile as sf

def cut_first_five_minutes(input_file, output_file):
    print(f"Loading {input_file}...")
    
    # Read the audio directly into a NumPy array
    audio_data, sample_rate = sf.read(input_file, dtype='float32')
    
    # Calculate the exact number of samples for 5 minutes (300 seconds)
    end_sample = 300 * sample_rate
    
    # Slice the array from the beginning to the 5-minute mark
    cut_data = audio_data[:end_sample]
    
    print(f"Saving the first 5 minutes to {output_file}...")
    sf.write(output_file, cut_data, sample_rate)
    print("Done!")

if __name__ == "__main__":
    cut_first_five_minutes(
        input_file="plane_ambiance_48k.wav", # Update with your 48kHz file name
        output_file="plane_5min.wav" 
    )
