Write-Host "Starting FastAPI..."
uvicorn fast_api_inference.main:app --host 0.0.0.0 --port 8000 --reload

Write-Host "Starting Gradio..."
Start-Process python "gradio_app.py"
