from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load model (either .h5 or SavedModel)
model = tf.keras.models.load_model("./model/benign_malignant_model_v2.h5")  # or 'benign_malignant_model_v2.h5'
CLASS_NAMES = ["Benign", "Malignant"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ✅ Resize to 150x150 to match training
        image = image.resize((150, 150))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        confidence = float(prediction[0][0])
        label = CLASS_NAMES[1] if confidence > 0.5 else CLASS_NAMES[0]

        return {
            "label": label,
            "confidence_score": round(confidence, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ping")
def ping():
    return {"message": "pong"}