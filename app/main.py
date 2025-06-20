from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.model import ToxicCommentModel
from app.feedback import save_feedback

app = FastAPI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load model
model = ToxicCommentModel()

@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")

@app.post("/predict")
def predict(payload: dict):
    return model.predict(payload["text"])

@app.post("/feedback")
def feedback(payload: dict):
    save_feedback(payload["comment"], payload["prediction"], payload["user_corrected"])
    return {"status": "success"}
