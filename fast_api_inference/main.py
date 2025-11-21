from fastapi import FastAPI
from pydantic import BaseModel
from fast_api_inference.model_inference import predict_sentiment
from typing import Dict, Any
from fastapi.responses import HTMLResponse


# Define the data models for request and response
class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    probabilities: Dict[str, float]


app = FastAPI(title="Nepali Sentiment Analysis API")


# Home page
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
        <head>
            <title>Nepali Sentiment Analysis API</title>
            <style>
                body {
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    font-family: Arial, sans-serif;
                    text-align: center;
                }
                h1 {
                    color: #333;
                }
                p {
                    font-size: 18px;
                }
                a {
                    text-decoration: none;
                    color: #007BFF;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to Nepali Sentiment Analysis API</h1>
            <p>Predict via <a href='http://localhost:8001/'>gradio app</a> </p>
            <p>Check <a href='/docs'>API documentation</a> for more details.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """
    Predicts the sentiment for the given text and returns
    the label and the probability distribution.
    """

    # Get the prediction from the model inference module
    predicted_label, probabilities = predict_sentiment(request.text)

    # Return the structured response
    return SentimentResponse(sentiment=predicted_label, probabilities=probabilities)


# You can add a startup event to check model loading status
@app.on_event("startup")
async def startup_event():
    print(
        "FastAPI application started. Model loading status is handled by model_inference.py."
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "FastAPI Sentiment"}
