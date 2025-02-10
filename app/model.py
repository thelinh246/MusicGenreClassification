import tensorflow as tf

def load_model():
    """
    Load the pre-trained model from file.
    
    Returns:
        tensorflow.keras.Model: The loaded model
    """
    try:
        model = tf.keras.models.load_model("results/MusicGenreClassification.keras")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
