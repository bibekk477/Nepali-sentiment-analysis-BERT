import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax

# IMPORTANT: To load the model from your Hugging Face account, replace the placeholder
# below with your actual model ID (e.g., "your-username/your-repo-name").
# If your model is private, ensure your environment is authenticated (e.g., using huggingface-cli login).
MODEL_PATH = "Bibekk477/nepali-sentiment-bert"

# Define the labels used during training
LABELS = ["Negative", "Neutral", "Positive"]

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the model and tokenizer from the specified path (either local or Hugging Face Hub ID)."""
    global model, tokenizer
    try:
        print(f"Loading model from: {MODEL_PATH}")
        # The from_pretrained method automatically handles loading from Hugging Face Hub
        # when MODEL_PATH is a model ID string.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real scenario, you might want to raise the exception or exit
        # For this setup, we proceed but log the error
        model = None
        tokenizer = None


def predict_sentiment(text: str):
    """
    Predicts the sentiment label and the probabilities for all three classes.

    Args:
        text: The input text string to analyze.

    Returns:
        A tuple: (predicted_label: str, probabilities: dict)
    """
    if model is None or tokenizer is None:
        return "Model Not Loaded", {label: 0.0 for label in LABELS}

    try:
        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Get model outputs (logits)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = softmax(logits, dim=1)[0].tolist()

        # Get the predicted index
        predicted_index = torch.argmax(logits, dim=1).item()

        # Get the predicted label
        predicted_label = LABELS[predicted_index]

        # Format probabilities into a dictionary
        prob_dict = {LABELS[i]: round(probabilities[i], 4) for i in range(len(LABELS))}

        return predicted_label, prob_dict

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction Failed", {label: 0.0 for label in LABELS}


# Load the model immediately when the module is imported
load_model()
