from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import os
import warnings
import joblib
from sklearn.exceptions import InconsistentVersionWarning
from model import build_features, CAT_MAP, DOW_MAP

# ── Suppress sklearn version mismatch warnings ──────────
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ── Load model using absolute path ─────────────────────
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

# ── Request schema ──────────────────────────────────────
class VideoInput(BaseModel):
    title:         str = "Official Music Video"
    category:      str = "Music"
    views:         int = 20000000
    dislikes:      int = 23000
    tag_count:     int = 6
    publish_hour:  int = 15
    publish_day:   str = "Friday"
    publish_month: int = 11

    @validator('category')
    def category_must_be_valid(cls, v):
        valid = list(CAT_MAP.keys())
        if v not in valid:
            raise ValueError(f"category must be one of: {valid}")
        return v

    @validator('publish_day')
    def day_must_be_valid(cls, v):
        valid = list(DOW_MAP.keys())
        if v not in valid:
            raise ValueError(f"publish_day must be one of: {valid}")
        return v

# ── Routes ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "YouTube Viral Predictor API is live 🎬"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest", "roc_auc": 0.9516}

@app.post("/predict")
def predict(video: VideoInput):
    X    = build_features(video.model_dump(), FEATURES)
    prob = round(float(model.predict_proba(X)[0][1]), 4)

    # Change this line in predict()
    label = "viral" if prob >= 0.20 else "not_viral"

    return {
        "prediction":        label,
        "viral_probability": prob,
        "confidence": "high"   if prob > 0.30 else "medium" if prob > 0.20 else "low",
        "verdict":           "🔥 This video has strong viral potential!" if label == "viral"
                             else "📉 Low viral potential. Try Music category, 15M+ views, low dislikes.",
        "input_summary": {
            "title":        video.title,
            "category":     video.category,
            "views":        video.views,
            "dislike_ratio": round(video.dislikes / (video.views + 1), 6)
        }
    }

@app.get("/categories")
def categories():
    return {"categories": list(CAT_MAP.keys())}

@app.get("/days")
def days():
    return {"days": list(DOW_MAP.keys())}
