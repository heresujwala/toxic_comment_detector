from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

class TextIn(BaseModel):
    text: str

app = FastAPI()

@app.post("/predict")
def predict(input: TextIn):
    vec = vectorizer.transform([input.text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]
    return {"prediction": int(prediction), "probability": float(proba)}
