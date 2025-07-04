import streamlit as st
import librosa
import numpy as np
import pandas as pd
from scipy.stats import entropy
import joblib
import warnings
from io import BytesIO
import os

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stProgress > div > div > div > div {
        background-color: #4e79a7;
    }
    .st-bb {
        background-color: #f0f2f6;
    }
    .st-at {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("models/xgb_clf_new.joblib")

def extract_features(audio_bytes):
    features = {
        "MDVP:Fo(Hz)": 0, "MDVP:Fhi(Hz)": 0, "MDVP:Flo(Hz)": 0,
        "MDVP:Jitter(%)": 0, "MDVP:Jitter(Abs)": 0, "MDVP:RAP": 0,
        "MDVP:PPQ": 0, "Jitter:DDP": 0, "MDVP:Shimmer": 0,
        "MDVP:Shimmer(dB)": 0, "Shimmer:APQ3": 0, "Shimmer:APQ5": 0,
        "MDVP:APQ": 0, "Shimmer:DDA": 0, "NHR": 0, "HNR": 0,
        "RPDE": 0, "DFA": 0, "spread1": 0, "spread2": 0,
        "D2": 0, "PPE": 0
    }

    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes.getvalue())

        y, sr = librosa.load("temp_audio.wav", sr=22050)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]

        if len(pitches) < 20:
            st.warning("Audio too short or lacks pitch content.")
            return None

        features["MDVP:Fo(Hz)"] = np.mean(pitches)
        features["MDVP:Fhi(Hz)"] = np.max(pitches)
        features["MDVP:Flo(Hz)"] = np.min(pitches)

        zero_crossings = librosa.zero_crossings(y, pad=False)
        jitter_std = np.std(zero_crossings)
        jitter_mean = np.mean(zero_crossings)
        features['MDVP:Jitter(%)'] = jitter_std / jitter_mean if jitter_mean != 0 else 0
        features['MDVP:Jitter(Abs)'] = jitter_std
        features['MDVP:RAP'] = jitter_std / (len(zero_crossings) + 1e-6)
        features['MDVP:PPQ'] = jitter_std / np.sqrt(len(zero_crossings) + 1e-6)
        features['Jitter:DDP'] = jitter_std * 3

        amplitude = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        shimmer_std = np.std(amplitude)
        shimmer_mean = np.mean(amplitude)
        features['MDVP:Shimmer'] = shimmer_std / shimmer_mean if shimmer_mean != 0 else 0
        features['MDVP:Shimmer(dB)'] = shimmer_std
        features['Shimmer:APQ3'] = shimmer_std / 3
        features['Shimmer:APQ5'] = shimmer_std / 5
        features['MDVP:APQ'] = shimmer_std / len(amplitude)
        features['Shimmer:DDA'] = shimmer_std * 3

        harmonic, percussive = librosa.effects.hpss(y)
        features['NHR'] = np.mean(percussive) / (np.mean(harmonic) + 1e-6)
        features['HNR'] = np.mean(harmonic) / (np.mean(percussive) + 1e-6)

        features['RPDE'] = entropy(pitches)
        features['DFA'] = librosa.feature.rms(y=y).mean()
        features['spread1'] = np.std(pitches)
        features['spread2'] = np.var(pitches)
        features['D2'] = np.percentile(pitches, 99)
        features['PPE'] = np.mean(np.abs(pitches - np.mean(pitches)))

        return features

    except Exception as e:
        st.error("Feature extraction failed. Please make sure the audio is valid and clear.")
        return None
    finally:
        try:
            os.remove("temp_audio.wav")
        except:
            pass

def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("""
    Upload a short recording of you saying "ahhh" for 3–5 seconds.
    We'll analyze it for vocal biomarkers of Parkinson's Disease.
    """)

    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Record "ahhh" for 3–5 seconds
        2. Save as WAV (16-bit PCM recommended)
        3. Upload using the uploader below
        """)

    uploaded_file = st.file_uploader("Upload WAV Audio File", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Analyze Voice"):
            with st.spinner("Extracting features and predicting..."):
                features = extract_features(uploaded_file)

                if features:
                    df = pd.DataFrame([features])
                    try:
                        model = load_model()
                        prediction = model.predict(df)[0]
                        proba = model.predict_proba(df)[0][1]

                        st.subheader("Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction", "🧠 Parkinson's Detected" if prediction else "✅ Healthy", f"{proba*100:.1f}% Confidence")
                        with col2:
                            risk = "High" if proba > 0.7 else "Medium" if proba > 0.5 else "Low"
                            st.metric("Risk Level", risk)

                        with st.expander("Feature Breakdown"):
                            st.dataframe(df.T.style.background_gradient(cmap="Blues"))
                    except Exception as e:
                        st.error("Prediction failed. Please check the model file or feature compatibility.")

if __name__ == "__main__":
    main()
