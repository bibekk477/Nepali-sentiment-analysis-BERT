# Activate the virtual environment
Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

# Start FastAPI
Write-Host "Starting FastAPI..."
Start-Process uvicorn "fast_api_inference.main:app --host 0.0.0.0 --port 8000 --reload"

# Start Gradio
Write-Host "Starting Gradio..."
python gradio_app.py
