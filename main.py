from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import warnings
import joblib
from sklearn.exceptions import InconsistentVersionWarning
from model import build_features, CAT_MAP, DOW_MAP

# ── Suppress sklearn version mismatch warnings ──────────
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ── Load model using absolute path relative to BASE_DIR ─
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
FEATURES = joblib.load(os.path.join(BASE_DIR, 'features.pkl'))

app = FastAPI(
    title="YouTube Viral Predictor API",
    description="Predicts if a YouTube video will go viral. ROC-AUC: 0.9516",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request schema ─────────────────────────────────────
class VideoInput(BaseModel):
    title:         str   = "10 Things You Didn't Know About Space"
    category:      str   = "Education"
    views:         int   = 500000
    dislikes:      int   = 1000
    tag_count:     int   = 15
    publish_hour:  int   = 15
    publish_day:   str   = "Friday"
    publish_month: int   = 6

# ── Routes ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "YouTube Viral Predictor API is live 🎬"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest", "roc_auc": 0.9516}

@app.post("/predict")
def predict(video: VideoInput):
    X = build_features(video.model_dump(), FEATURES)
    prob  = round(float(model.predict_proba(X)[0][1]), 4)
    label = "viral" if prob >= 0.5 else "not_viral"

    return {
        "prediction":       label,
        "viral_probability": prob,
        "confidence":       "high" if prob > 0.75 or prob < 0.25 else "medium",
        "input_summary": {
            "title":    video.title,
            "category": video.category,
            "views":    video.views
        }
    }

@app.get("/categories")
def categories():
    return {"categories": list(CAT_MAP.keys())}

@app.get("/days")
def days():
    return {"days": list(DOW_MAP.keys())}
