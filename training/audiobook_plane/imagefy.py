import librosa
import numpy as np
import soundfile as sf
import os

def extract_mfcc_features(input_file, label, chunk_seconds=1.0):
    print(f"Extracting features from {input_file}...")
    
    # Read the audio array
    audio_data, sample_rate = sf.read(input_file, dtype='float32')
    
    # Calculate how many samples make up one chunk (e.g., 48,000 for 1 second)
    samples_per_chunk = int(chunk_seconds * sample_rate)
    
    features = []
    labels = []
    
    # Loop through the audio file in 1-second steps
    for i in range(0, len(audio_data) - samples_per_chunk, samples_per_chunk):
        chunk = audio_data[i : i + samples_per_chunk]
        
        # Use librosa to extract MFCCs from this specific 1-second chunk
        # n_mfcc=40 creates a 2D array with 40 frequency bands
        mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=40)
        
        features.append(mfcc)
        labels.append(label)
        
    return features, labels

if __name__ == "__main__":
    # 1. Extract features for Class 0 (Clean Speech)
    clean_features, clean_labels = extract_mfcc_features(
        input_file="clean_speech_5min.wav", 
        label=0  
    )
    
    # 2. Extract features for Class 1 (Plane Noise Mix)
    noise_features, noise_labels = extract_mfcc_features(
        input_file="training_mix_1.wav", 
        label=1  
    )
    
    # Combine the lists together
    X = np.array(clean_features + noise_features)
    y = np.array(clean_labels + noise_labels)
    
    print(f"Total dataset shape: {X.shape}") 
    # Example output: (600, 40, 94) -> 600 chunks, 40 MFCC bands, 94 time steps
    
    # Save the extracted features to disk
    print("Saving dataset to disk as .npy files...")
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)
    print("Done! You are ready to build the CNN manager.")
