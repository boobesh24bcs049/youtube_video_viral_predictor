# YouTube Viral Post Predictor

A machine learning REST API that predicts whether a YouTube video will go viral based on metadata — built with Random Forest, FastAPI, and deployed live on Render.

**Live API:** https://youtube-video-viral-predictor.onrender.com  
**Swagger Docs:** https://youtube-video-viral-predictor.onrender.com/docs  
**Health Check:** https://youtube-video-viral-predictor.onrender.com/health  
**GitHub:** https://github.com/boobesh24bcs049/youtube_video_viral_predictor  
**Model:** Random Forest | **ROC-AUC:** 0.9516 | **F1 Score:** 0.6635

---

## Problem Statement

Predict whether a YouTube video will go viral (top 10% by engagement rate) **before it trends** — using only metadata available at upload time, with zero data leakage.

**Target variable:**
```
engagement_rate = (likes + comment_count) / views
viral = 1  if engagement_rate >= 90th percentile
viral = 0  otherwise
```

---

## Dataset

- **Source:** [YouTube Trending Video Statistics — Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new)
- **File used:** `USvideos.csv`
- **Size:** ~40,000 rows | 16 columns
- **Platform:** YouTube US Trending (2017–2018)

---

## Key EDA Findings

| Insight | Finding |
|---|---|
| Top viral category | Nonprofits (23%) and Comedy (21%) |
| Worst viral category | News (~1%) despite high view counts |
| Best posting hour | 14–20 UTC (9am–3pm EST) |
| Title length | Shorter titles (≤40 chars) correlate with virality |
| Tag sweet spot | 11–30 tags perform best |
| Strongest signal | Low dislike ratio — not raw views or category |

---

## Feature Engineering

Features engineered from raw metadata (no post-upload leakage):

| Feature | Description |
|---|---|
| `log_views` | Log-normalized view count |
| `dislike_ratio` | Dislikes / views |
| `short_title` | Binary: title ≤ 40 chars |
| `has_question` | Binary: title contains `?` |
| `has_exclamation` | Binary: title contains `!` |
| `caps_ratio` | Ratio of ALL CAPS words in title |
| `peak_hour` | Binary: published 14–20 UTC |
| `is_weekend` | Binary: published Saturday or Sunday |
| `publish_month` | Month of upload |
| `tag_count` | Number of tags used |
| `cat_*` | One-hot encoded category (16 categories) |

> `likes` and `comment_count` were intentionally excluded — they constitute **data leakage** as they are only known after a video trends.

---

## Feature Importance (Top 10)

| Feature | Importance |
|---|---|
| `title_length` | 0.1026 |
| `caps_ratio` | 0.0854 |
| `title_caps_count` | 0.0803 |
| `dislike_ratio` | 0.0793 |
| `publish_hour` | 0.0754 |
| `tag_count` | 0.0663 |
| `log_views` | 0.0649 |
| `views` | 0.0623 |
| `dislikes` | 0.0619 |
| `category_id` | 0.0528 |

> Key insight: The model is primarily title-driven. Title length, caps usage, and punctuation matter more than category or raw view count.

---

## Model Comparison

| Model | ROC-AUC | F1 (viral) |
|---|---|---|
| Logistic Regression | 0.7850 | 0.3188 |
| Decision Tree | 0.7981 | 0.3488 |
| XGBoost | 0.9497 | 0.6191 |
| **Random Forest**  | **0.9516** | **0.6635** |

All models trained with `class_weight='balanced'` to handle the 90/10 class imbalance.

---

## Model Behaviour Note

The Random Forest outputs conservative probabilities (max ~0.35) because YouTube virality is driven by factors not fully captured in metadata alone — content quality, creator fanbase, thumbnail design, and external sharing all play a role. Threshold tuned to 0.20 based on F1/Recall tradeoff analysis across multiple thresholds.

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.20 | 0.3105 | 0.1846 | 0.9756 |
| 0.27 | 0.3763 | 0.2338 | 0.9634 |
| 0.35 | 0.4738 | 0.3170 | 0.9377 |
| 0.50 | 0.6635 | 0.5432 | 0.8523 |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root message |
| GET | `/health` | Model info and status |
| POST | `/predict` | Predict virality |
| GET | `/categories` | List valid categories |
| GET | `/days` | List valid days |

### Sample Request

```bash
curl -X POST "https://youtube-video-viral-predictor.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Official Music Video",
       "category": "Music",
       "views": 20000000,
       "dislikes": 23000,
       "tag_count": 6,
       "publish_hour": 15,
       "publish_day": "Friday",
       "publish_month": 11
     }'
```

### Sample Response

```json
{
  "prediction": "viral",
  "viral_probability": 0.2773,
  "confidence": "medium",
  "verdict": " This video has strong viral potential!",
  "input_summary": {
    "title": "Official Music Video",
    "category": "Music",
    "views": 20000000,
    "dislike_ratio": 0.00115
  }
}
```

### Valid Categories

```
Film, Autos, Music, Pets, Sports, Travel, Gaming, People,
Comedy, Entertainment, News, Howto, Education, Science, Nonprofits
```

---

## Project Structure

```
viral_predictor/
├── main.py              # FastAPI app + prediction endpoint
├── model.py             # Feature engineering logic
├── requirements.txt     # Dependencies
├── rf_model.pkl         # Trained Random Forest model
├── features.pkl         # Feature list (column order)
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/boobesh24bcs049/youtube_video_viral_predictor
cd youtube_video_viral_predictor
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

API live at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

---

## Call the API from Python

```python
import requests

BASE_URL = "https://youtube-video-viral-predictor.onrender.com"

payload = {
    "title": "Official Music Video",
    "category": "Music",
    "views": 20000000,
    "dislikes": 23000,
    "tag_count": 6,
    "publish_hour": 15,
    "publish_day": "Friday",
    "publish_month": 11
}

response = requests.post(f"{BASE_URL}/predict", json=payload)
print(response.json())
```

---

## Tech Stack

- **ML:** scikit-learn, XGBoost, pandas, numpy
- **API:** FastAPI, Uvicorn, Pydantic
- **Deployment:** Render
- **Development:** Google Colab, Python 3.10+

---

## Resume Bullet

> Built an end-to-end YouTube viral post prediction REST API using Random Forest (ROC-AUC: 0.9516), with feature engineering, intentional data leakage prevention, threshold tuning analysis, and deployed live via FastAPI on Render.

---

## Author

**Boobesh**  
[LinkedIn](https://www.linkedin.com/in/boobesh-k-24bcs049)
