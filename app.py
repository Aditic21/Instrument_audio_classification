import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to convert audio files into spectrograms
def wav_to_spectrogram(file_path, pad_length=640):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    if S_DB.shape[1] < pad_length:
        pad_width = pad_length - S_DB.shape[1]
        S_DB = np.pad(S_DB, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        S_DB = S_DB[:, :pad_length]
    return S_DB

# Load the model
model = load_model('best_model.h5')

# Define the class names based on the model's training
class_names = ['Sound_Drum', 'Sound_Guitar', 'Sound_Piano', 'Sound_Violin']

# Streamlit UI
st.title("Musical Instrument Classifier")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')

    # Convert the uploaded audio file to a spectrogram
    spectrogram = wav_to_spectrogram(uploaded_file)

    # Make predictions
    spectrogram = np.expand_dims(np.expand_dims(spectrogram, axis=-1), axis=0)  # Reshape for the model
    prediction = model.predict(spectrogram)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_class_name = class_names[predicted_class_index[0]]

    # Display the prediction
    st.write(f"Predicted Class: {predicted_class_name}")