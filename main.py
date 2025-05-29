from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastai.learner import load_learner
from PIL import Image
import io

app = FastAPI()

# Load model
learn = load_learner('./model.pkl')

@app.get("/")
def read_root():
    return {"message": "Gesture recognition API is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))

    # Predict using FastAI model
    pred = learn.predict(image)

    # Return prediction
    return JSONResponse({
        "prediction": pred[0],
        "probabilities": [float(p) for p in pred[2]]
    })
