from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastai.vision.all import *  # Import all necessary FastAI components
from PIL import Image
import io
from pathlib import Path
import pathlib
import sys

temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

app = FastAPI()

# Load model
try:
    learn = load_learner('model.pkl', cpu=True)  # Load on CPU to avoid CUDA issues
    print("Model loaded successfully:", learn.dls.vocab)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.get("/")
def read_root():
    return {"message": "Gesture recognition API is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))  # Match training preprocessing

    # Predict using FastAI model
    pred, pred_idx, probs = learn.predict(image)

    # Return prediction
    return JSONResponse({
        "prediction": str(pred),  # Convert to string for JSON serialization
        "probabilities": [float(p) for p in probs]
    })