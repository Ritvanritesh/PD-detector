import streamlit as st
import librosa
import numpy as np
import pickle
import joblib
import os
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Constants
RECORDING_PATH = "AudioFiles/recorded_audio.wav"
UPLOAD_PATH = "AudioFiles/uploaded_audio.wav"
UPLOAD_DIR = "AudioFiles"

# Ensure the directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Function to load the model and scaler
def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load ML model and scaler
model = joblib.load('models/xgb_clf_new.joblib')
scaler = joblib.load('models/scaler.joblib')

# Feature extraction function
def extract_features(file_path):
    try:
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
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

# Function to save uploaded file
def save_uploaded_file(uploadedfile):
    try:
        file_path = os.path.join(UPLOAD_DIR, uploadedfile.name)
        with open(file_path, "wb") as f:
            f.write(uploadedfile.getbuffer())
        #st.success(f"File saved successfully at: {file_path}")
        return file_path
    except Exception as e:
        #st.error(f"Error saving uploaded file: {e}")
        return None

# Voice recording function
def record_voice(duration=3, samplerate=44100):
    st.info("Recording for 3 seconds... Speak now! (Prolonged enunciation of the vowel /a/)")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        write(RECORDING_PATH, samplerate, audio_data)
        st.success(f"Recording saved successfully at: {RECORDING_PATH}")
    except Exception as e:
        st.error(f"Error during recording: {e}")

# Streamlit app layout
st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Parkinson's Disease Detection from Voice")
st.markdown("""Analyze voice recordings to detect the likelihood of Parkinson's disease using machine learning.""")

# Option for recording or uploading voice
option = st.radio("Choose Input Method", ("Upload an Audio File", "Record Voice"))

# Inputs for age and gender
# age = st.number_input("Enter your age", min_value=1, max_value=120, value=30)
# gender = st.selectbox("Select your gender", ("Male", "Female"))

if option == "Upload an Audio File":
    uploaded_file = st.file_uploader("Upload a voice recording (.wav or .mp3)", type=["wav", "mp3"])
    if uploaded_file:
        audio_path = save_uploaded_file(uploaded_file)
        st.audio(audio_path, format="audio/wav")
elif option == "Record Voice":
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🎙️ Record Voice"):
            record_voice()
    with col2:
        if st.button("🔁 Re-record Voice"):
            if os.path.exists(RECORDING_PATH):
                os.remove(RECORDING_PATH)
            record_voice()

    if os.path.exists(RECORDING_PATH):
        st.audio(RECORDING_PATH, format="audio/wav")
        audio_path = RECORDING_PATH
    else:
        st.info("No voice recorded yet.")

# Prediction Section with Feature Display
if 'audio_path' in locals() and os.path.exists(audio_path):
    st.markdown("---")
    st.subheader("📊 Prediction Results and Extracted Features")
    
    if model and scaler:
        features = extract_features(audio_path)
        if features is not None:
            try:
                # Prepare feature data for scaling
                feature_array = np.array(list(features.values())).reshape(1, -1)
                scaled_features = scaler.transform(feature_array)
                
                # Predict with model
                prediction = model.predict(scaled_features)[0]
                
                # Display prediction results
                if prediction == 1:
                    st.error("🔴 High likelihood of Parkinson's detected. Please consult a doctor.")
                else:
                    st.success("🟢 Low likelihood of Parkinson's detected. Stay healthy!")
                
                # Display feature data
                st.markdown("### Extracted Feature Values")
                feature_df = pd.DataFrame([features], index=["Values"]).T
                feature_df.columns = ["Feature Value"]
                st.dataframe(feature_df.style.format("{:.4f}"))  # Display with 4 decimal places
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            st.warning("Feature extraction failed.")
    else:
        st.warning("Model or scaler file is missing! Prediction is not available.")

# Footer
st.markdown("---")
st.write("🔍 **Note:** This app is for demonstration purposes and is not a substitute for professional medical advice.")
