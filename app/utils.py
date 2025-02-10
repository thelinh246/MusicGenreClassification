import librosa
import numpy as np
import tensorflow as tf
import io

def load_and_preprocess_data(file_bytes, target_shape=(150, 150)):
    """
    Load and preprocess audio file into mel spectrograms.
    
    Args:
        file_bytes (bytes): Audio file in bytes
        target_shape (tuple): Desired shape of the spectrogram
    
    Returns:
        numpy.ndarray: Preprocessed mel spectrograms
    """
    # Convert bytes to file-like object
    audio_bytes = io.BytesIO(file_bytes)
    
    # Load audio file
    audio_data, sample_rate = librosa.load(audio_bytes, sr=None)
    
    data = []
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    
    num_chunks = max(1, int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1)
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        
        chunk = audio_data[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        mel_spectrogram_resized = tf.image.resize(
            np.expand_dims(mel_spectrogram_db, axis=-1), target_shape
        ).numpy()
        
        data.append(mel_spectrogram_resized)

    return np.array(data), mel_spectrogram_db
