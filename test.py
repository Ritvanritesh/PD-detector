import streamlit as st
import librosa
import numpy as np
# import parselmouth
# import opensmile
import pandas as pd
from scipy.stats import entropy
import joblib
import warnings
from io import BytesIO
import os
os.environ["PARSE_MOUTH_NO_GOOGLE"] = "1"  # Disable problematic dependency
os.environ["NO_SOUNDFILE_WARNING"] = "1"   # Disable soundfile warnings

# Rest of your app code...

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better appearance
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

def safe_praat_call(func, *args, default=0):
    """Safe wrapper for Praat calls with error handling"""
    try:
        return func(*args)
    except Exception as e:
        warnings.warn(f"Praat call failed: {str(e)}")
        return default

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    return joblib.load("models/xgb_clf_new.joblib")

def extract_features(audio_bytes):
    """Extract features from audio bytes"""
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
        # Save to temp file for processing
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes.getvalue())
            
        # Load audio
        y, sr = librosa.load("temp_audio.wav", sr=22050)
        snd = parselmouth.Sound("temp_audio.wav")
        
        if snd.duration < 0.5:  # Require at least 0.5s audio
            st.warning("Audio too short for analysis (minimum 0.5 seconds)")
            return None

        # Pitch analysis
        with st.spinner("Analyzing pitch..."):
            pitch = snd.to_pitch()
            f0_values = pitch.selected_array['frequency']
            f0_values = f0_values[f0_values > 0]
            
            if len(f0_values) > 10:
                features.update({
                    "MDVP:Fo(Hz)": np.mean(f0_values),
                    "MDVP:Fhi(Hz)": np.max(f0_values),
                    "MDVP:Flo(Hz)": np.min(f0_values),
                    "PPE": entropy(f0_values) if len(f0_values) > 1 else 0
                })

        # Jitter/Shimmer analysis
        with st.spinner("Analyzing voice quality..."):
            point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
            jitter_local = safe_praat_call(parselmouth.praat.call, point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = safe_praat_call(parselmouth.praat.call, [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            features.update({
                "MDVP:Jitter(%)": jitter_local,
                "MDVP:Shimmer": shimmer_local,
                "MDVP:Jitter(Abs)": jitter_local,
                "MDVP:RAP": jitter_local,
                "MDVP:PPQ": jitter_local,
                "Jitter:DDP": jitter_local * 3,
                "Shimmer:APQ3": shimmer_local,
                "Shimmer:APQ5": shimmer_local,
                "MDVP:APQ": shimmer_local,
                "Shimmer:DDA": shimmer_local * 3
            })

        # Harmonicity
        with st.spinner("Analyzing harmonics..."):
            harmonicity = snd.to_harmonicity_cc()
            hnr = safe_praat_call(parselmouth.praat.call, harmonicity, "Get mean", 0, 0)
            features.update({
                "HNR": hnr,
                "NHR": 1/hnr if hnr > 0 else 0
            })

        # OpenSMILE features
        with st.spinner("Extracting advanced features..."):
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            smile_features = smile.process_file("temp_audio.wav")
            
            features.update({
                "RPDE": smile_features.get("F0final_sma_de_pctl1", [0])[0],
                "DFA": smile_features.get("ShimmerLocal_sma_de_stddev", [0])[0],
                "spread1": smile_features.get("F0final_sma_amean", [0])[0] - np.std(f0_values) if len(f0_values) > 0 else 0,
                "spread2": smile_features.get("F0final_sma_de_range", [0])[0],
                "D2": smile_features.get("pcm_loudness_sma_de_amean", [0])[0]
            })

        return features

    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None
    finally:
        # Clean up temp file
        try:
            os.remove("temp_audio.wav")
        except:
            pass

def main():
    st.title("🧠 Parkinson's Disease Voice Analysis")
    st.markdown("""
    This app analyzes voice characteristics to assess Parkinson's disease risk.
    Upload a WAV audio file of sustained vowel phonation (say 'ahhh' for 3-5 seconds).
    """)

    # Sidebar with info
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Record yourself saying "ahhh" for 3-5 seconds
        2. Save as WAV format (16-bit PCM recommended)
        3. Upload the file below
        4. View your results
        """)
        
        st.markdown("""
        ### About
        This tool analyzes:
        - Pitch variations (F0)
        - Voice tremor (jitter/shimmer)
        - Harmonic-to-noise ratio
        - Nonlinear dynamic features
        """)

    # File uploader
    uploaded_file = st.file_uploader("Upload WAV Audio File", type=["wav"])

    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')

        # Process file
        if st.button("Analyze Voice"):
            with st.spinner("Processing audio..."):
                try:
                    # Extract features
                    features = extract_features(uploaded_file)
                    
                    if features is not None:
                        # Make prediction
                        model = load_model()
                        df = pd.DataFrame([features])
                        prediction = model.predict(df)[0]
                        proba = model.predict_proba(df)[0][1]
                        
                        # Display results
                        st.subheader("Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Prediction", 
                                "🧠 Parkinson's Detected" if prediction == 1 else "✅ Healthy",
                                f"{proba*100:.1f}% confidence"
                            )
                            
                        with col2:
                            risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.5 else "Low"
                            st.metric("Risk Level", risk_level)
                        
                        # Interpretation
                        if proba > 0.7:
                            st.warning("This result suggests higher risk of Parkinson's. Please consult a neurologist for clinical evaluation.")
                        elif proba > 0.5:
                            st.info("Borderline result. Consider repeating the test or consulting a specialist.")
                        else:
                            st.success("Results suggest low risk of Parkinson's disease.")
                        
                        # Show feature details in expander
                        with st.expander("View Detailed Analysis"):
                            st.dataframe(df.T.style.background_gradient(cmap='Blues'))
                            
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import os
    os.environ["PATH"] += os.pathsep + os.path.join(os.getcwd(), 'Praat')
    main()
