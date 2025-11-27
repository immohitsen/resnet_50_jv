from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

app = FastAPI()

# Load Model
MODEL_PATH = "./resnet_50" # Ensure this folder is present
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
    # Get label and confidence
    label = model.config.id2label[predicted_class_idx]
    confidence = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
    
    return {
        "prediction": label,
        "confidence": confidence
    }
