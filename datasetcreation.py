import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Function to extract detailed acoustic features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load audio
    features = {}
    
    # Fundamental frequencies
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    features['MDVP:Fo(Hz)'] = np.mean(pitches) if len(pitches) > 0 else np.nan
    features['MDVP:Fhi(Hz)'] = np.max(pitches) if len(pitches) > 0 else np.nan
    features['MDVP:Flo(Hz)'] = np.min(pitches) if len(pitches) > 0 else np.nan

    # Jitter-related features
    zero_crossings = librosa.zero_crossings(y, pad=False)
    jitter_std = np.std(zero_crossings)
    jitter_mean = np.mean(zero_crossings)
    features['MDVP:Jitter(%)'] = jitter_std / jitter_mean if jitter_mean != 0 else np.nan
    features['MDVP:Jitter(Abs)'] = jitter_std if jitter_std > 0 else np.nan
    features['MDVP:RAP'] = jitter_std / (len(zero_crossings) + 1e-6)
    features['MDVP:PPQ'] = jitter_std / np.sqrt(len(zero_crossings) + 1e-6)
    features['Jitter:DDP'] = jitter_std * 3

    # Shimmer-related features
    amplitude = librosa.amplitude_to_db(np.abs(y), ref=np.max)
    shimmer_std = np.std(amplitude)
    shimmer_mean = np.mean(amplitude)
    features['MDVP:Shimmer'] = shimmer_std / shimmer_mean if shimmer_mean != 0 else np.nan
    features['MDVP:Shimmer(dB)'] = shimmer_std
    features['Shimmer:APQ3'] = shimmer_std / 3
    features['Shimmer:APQ5'] = shimmer_std / 5
    features['MDVP:APQ'] = shimmer_std / len(amplitude)
    features['Shimmer:DDA'] = shimmer_std * 3

    # Noise-to-Harmonic Ratio (NHR)
    harmonic, percussive = librosa.effects.hpss(y)
    features['NHR'] = np.mean(percussive) / (np.mean(harmonic) + 1e-6)
    features['HNR'] = np.mean(harmonic) / (np.mean(percussive) + 1e-6)

    # Nonlinear dynamic features
    features['RPDE'] = entropy(pitches) if len(pitches) > 0 else np.nan
    features['DFA'] = librosa.feature.rms(y=y).mean()  # Approximation for DFA

    # Spread and PPE (variations in pitch)
    features['spread1'] = np.std(pitches) if len(pitches) > 0 else np.nan
    features['spread2'] = np.var(pitches) if len(pitches) > 0 else np.nan
    features['D2'] = np.percentile(pitches, 99) if len(pitches) > 0 else np.nan
    features['PPE'] = np.mean(np.abs(pitches - np.mean(pitches))) if len(pitches) > 0 else np.nan

    return features

# Function to create dataset for healthy candidates
def create_dataset(audio_folder, output_csv, label=1):
    data = []

    for file_name in os.listdir(audio_folder):
        if file_name.endswith((".wav", ".mp3", ".ogg")):
            file_path = os.path.join(audio_folder, file_name)

            # Extract features
            features = extract_features(file_path)
            
            # Add label and file name
            features['status'] = label  # 0 for Healthy, 1 for Parkinson's
            features['File_Name'] = file_name
            
            data.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Dataset for label={label} saved to {output_csv}")

# Paths
audio_folder = os.getcwd()+"\\PD_AH"  
output_csv = "parkinsons_dataset.csv"
create_dataset(audio_folder, output_csv, label=1)

audio_folder = os.getcwd()+"\\HC_AH"  
output_csv = "healthy_dataset.csv"
# Generate dataset for healthy candidates
create_dataset(audio_folder, output_csv, label=0)

# Combine both datasets
healthy_df = pd.read_csv("healthy_dataset.csv")
parkinsons_df = pd.read_csv("parkinsons_dataset.csv")

# Concatenate the two DataFrames
combined_df = pd.concat([healthy_df, parkinsons_df], ignore_index=True)

# Save the final combined dataset
combined_df.to_csv("all_dataset.csv", index=False)
print("Combined dataset saved as combined_voice_dataset.csv")