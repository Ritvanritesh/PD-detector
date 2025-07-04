import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import parselmouth

# Load trained model
model_path = "parkinsons_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please upload 'parkinsons_model.pkl'.")
else:
    try:
        model = pickle.load(open(model_path, "rb"))
        
        def extract_features(file_path):
            try:
                snd = parselmouth.Sound(file_path)
                
                # Convert stereo to mono if needed
                if snd.n_channels != 1:
                    samples = np.mean(snd.values, axis=0)
                    snd = parselmouth.Sound(values=samples, sampling_frequency=snd.sampling_frequency)
                
                # Extract pitch first
                pitch = snd.to_pitch()
                
                # Create PointProcess from sound directly (not from pitch)
                point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
                
                # Fundamental frequency features
                mean_f0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
                max_f0 = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz")
                min_f0 = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz")
            
                # Jitter measures
                jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
                
                # Shimmer measures
                shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq3 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq5 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                
                # Harmonicity (HNR)
                harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                hnr_value = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
            
                # Create feature vector
                features = {
                    'MDVP:Fo(Hz)': mean_f0,
                    'MDVP:Fhi(Hz)': max_f0,
                    'MDVP:Flo(Hz)': min_f0,
                    'MDVP:Jitter(%)': jitter_local,
                    'MDVP:Jitter(Abs)': 0,  # Placeholder
                    'MDVP:RAP': jitter_rap,
                    'MDVP:PPQ': jitter_ppq5,
                    'Jitter:DDP': jitter_rap*3,
                    'MDVP:Shimmer': shimmer_local,
                    'MDVP:Shimmer(dB)': 0,  # Placeholder
                    'Shimmer:APQ3': shimmer_apq3,
                    'Shimmer:APQ5': shimmer_apq5,
                    'MDVP:APQ': 0,  # Placeholder
                    'Shimmer:DDA': shimmer_apq3*3,
                    'NHR': 0,  # Placeholder
                    'HNR': hnr_value,
                    'RPDE': 0,  # Not extractable
                    'DFA': 0,  # Not extractable
                    'spread1': 0,  # Not extractable
                    'spread2': 0,  # Not extractable
                    'D2': 0,  # Not extractable
                    'PPE': 0  # Not extractable
                }
                
                return pd.DataFrame([features])
            
            except Exception as e:
                st.error(f"⚠️ Feature extraction failed: {str(e)}")
                return None

        # Streamlit UI
        st.title("🧠 Parkinson's Detection from Voice")
        st.write("Upload a `.wav` file of a person saying 'aaaah' for 3-5 seconds.")

        uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')

            # Save temporary file
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())

            features = extract_features("temp_audio.wav")

            if features is not None:
                # Make prediction
                try:
                    prediction = model.predict(features)
                    prediction_proba = model.predict_proba(features)
                    
                    # Display results
                    st.subheader("Results")
                    if prediction[0] == 1:
                        st.error(f"🧠 Parkinson's Detected! (Probability: {prediction_proba[0][1]:.2%})")
                    else:
                        st.success(f"✅ Healthy! (Probability: {prediction_proba[0][0]:.2%})")
                    
                    st.write("### Extracted Features")
                    st.dataframe(features)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    
            # Clean up temporary file
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
                
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
