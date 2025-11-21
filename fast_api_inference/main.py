from fastapi import FastAPI
from pydantic import BaseModel
from fast_api_inference.model_inference import predict_sentiment
from typing import Dict, Any

# Define the data models for request and response
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    probabilities: Dict[str, float]

app = FastAPI(title="Nepali Sentiment Analysis API")

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """
    Predicts the sentiment for the given text and returns
    the label and the probability distribution.
    """
    
    # Get the prediction from the model inference module
    predicted_label, probabilities = predict_sentiment(request.text)
    
    # Return the structured response
    return SentimentResponse(
        sentiment=predicted_label,
        probabilities=probabilities
    )

# You can add a startup event to check model loading status
@app.on_event("startup")
async def startup_event():
    print("FastAPI application started. Model loading status is handled by model_inference.py.")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "FastAPI Sentiment"}