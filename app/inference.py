import base64
import io
from fastapi import HTTPException
import numpy as np
from app.utils import load_and_preprocess_data
import librosa
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

async def predict_genre(file, model):
    """
    Predict the genre of an audio file.
    
    Args:
        file (UploadFile): Audio file
        
    Returns:
        dict: Predicted genre and confidence
    """
    file_bytes = await file.read()
    try:
        # Preprocess the audio
        preprocessed_data, mel_spectrogram = load_and_preprocess_data(file_bytes)
        
        # Make prediction
        y_pred = model.predict(preprocessed_data)
        
        # Get mean predictions across chunks
        mean_predictions = np.mean(y_pred, axis=0)
        
        # Get the predicted genre index and confidence
        predicted_genre_index = np.argmax(mean_predictions)
        confidence = mean_predictions[predicted_genre_index]
        
        return {"genre": GENRES[predicted_genre_index], "confidence": float(confidence)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")
