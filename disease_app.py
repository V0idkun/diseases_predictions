# fastapi_app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

app = FastAPI()

# Optional: enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources
model = tf.keras.models.load_model('disease_predictor_model.keras')
le = joblib.load('label_encoder.pkl')
metadata = joblib.load('disease_metadata.joblib')
feature_names = metadata['feature_name']

class SymptomInput(BaseModel):
    symptoms: list[int]  # 132 binary inputs

@app.post("/predict")
def predict_symptoms(data: SymptomInput):
    if len(data.symptoms) != len(feature_names):
        raise HTTPException(status_code=400, detail=f"Expected {len(feature_names)} features.")

    input_array = np.array(data.symptoms).reshape(1, -1)
    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction)
    disease = le.inverse_transform([predicted_class])[0]
    confidence = float(np.max(prediction) * 100)

    return {
        "disease": disease,
        "confidence": confidence,
        "message": interpret_confidence(confidence)
    }

def interpret_confidence(conf):
    if conf < 50:
        return "⚠ Low Confidence - Not very certain."
    elif 50 <= conf < 60:
        return "🟠 Moderate Confidence."
    elif 60 <= conf < 80:
        return "✅ Reasonably Confident."
    elif 80 <= conf < 95:
        return "✅✅ High Confidence."
    else:
        return "💯 Very High Confidence."

@app.post("/predict-csv")
def predict_from_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Add missing features
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]
        prediction = model.predict(df)
        predicted_classes = np.argmax(prediction, axis=1)
        diseases = le.inverse_transform(predicted_classes)
        df['Predicted_Diseases'] = diseases

        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")