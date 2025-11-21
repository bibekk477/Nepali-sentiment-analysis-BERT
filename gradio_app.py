import gradio as gr
import requests
import json
import time

# FastAPI server URL and endpoint
API_URL = "http://127.0.0.1:8000/predict"

def get_sentiment(input_text):
    """
    Sends the input text to the FastAPI server and processes the response.
    Implements a simple retry mechanism in case FastAPI is not ready immediately.
    """
    if not input_text or len(input_text.strip()) == 0:
        return "Please enter some text.", {}

    # Simple retry logic for stability, as Gradio might start slightly before FastAPI
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Prepare the payload for the FastAPI endpoint
            payload = {"text": input_text}
            
            # Make the POST request
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            # Parse the JSON response
            data = response.json()
            
            # Extract results
            sentiment = data.get("sentiment", "Error")
            probabilities = data.get("probabilities", {})
            
            # Format the probabilities for display
            prob_output = json.dumps(probabilities, indent=4, ensure_ascii=False)
            
            return sentiment, prob_output
        
        except requests.exceptions.ConnectionError:
            print(f"Attempt {attempt+1}/{max_retries}: Connection to FastAPI failed. Retrying in 1 second...")
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            # Handle other request errors (e.g., HTTP errors, JSON decode errors)
            print(f"Request error: {e}")
            return f"API Error: {e}", {}
        
    return "Failed to connect to the sentiment API after multiple retries.", {}

# Define the Gradio Interface
with gr.Blocks(title="Nepali Sentiment Analyzer (BERT)") as app:
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #10B981;">Nepali Sentiment Analyzer</h1>
        <p style="text-align: center; color: #4B5563;">
        Enter a Nepali sentence below to get the predicted sentiment (Negative, Neutral, or Positive) 
        and the associated probability distribution from the underlying BERT model (via FastAPI).
        </p>
        """
    )
    
    with gr.Row():
        # Input component
        text_input = gr.Textbox(
            label="Input Nepali Text", 
            placeholder="उदाहरणका लागि: यो ठाउँ राम्रो छ र म फेरि आउनेछु। ",
            lines=4,
            scale=2
        )
        
        # Submit Button
    submit_button = gr.Button("Analyze Sentiment", variant="primary", scale=0)

    with gr.Row():
        # Output 1: Predicted Sentiment
        sentiment_output = gr.Label(
            label="Predicted Sentiment Label", 
            # color="#059669", # Green
            show_label=True,
            scale=1
        )
        
        # Output 2: Probabilities for all labels
        prob_output = gr.Textbox(
            label="Probability Distribution", 
            placeholder="Probabilities for Negative, Neutral, and Positive will appear here...",
            lines=6,
            interactive=False,
            scale=1
        )

    # Define the action when the button is clicked
    submit_button.click(
        fn=get_sentiment,
        inputs=[text_input],
        outputs=[sentiment_output, prob_output]
    )
    
    # Enable analysis on Enter key press
    text_input.submit(
        fn=get_sentiment,
        inputs=[text_input],
        outputs=[sentiment_output, prob_output]
    )

# Run the Gradio application
# The port is set to 8001 as required by the entrypoint.sh script
app.launch(server_name="0.0.0.0", server_port=8001,share=True)