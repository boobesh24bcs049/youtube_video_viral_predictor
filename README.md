# 🎬 YouTube Viral Post Predictor

A machine learning API that predicts whether a YouTube video will go viral based on metadata — built with Random Forest, FastAPI, and deployed on Render.

**Live API:** `https://your-app.onrender.com` <!-- replace after deployment -->  
**ROC-AUC:** 0.9516 | **F1 Score (viral class):** 0.6635

---

## 📌 Problem Statement

Predict whether a YouTube video will go viral (top 10% by engagement rate) before it trends — using only metadata available at upload time, with zero data leakage.

**Target variable:**
```
engagement_rate = (likes + comment_count) / views
viral = 1  if engagement_rate >= 90th percentile
viral = 0  otherwise
```

---

## 📊 Dataset

- **Source:** [YouTube Trending Video Statistics — Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new)
- **File used:** `USvideos.csv`
- **Size:** ~40,000 rows | 16 columns
- **Platform:** YouTube US Trending (2017–2018)

---

## 🔍 Key EDA Findings

| Insight | Finding |
|---|---|
| Top viral category | Nonprofits (23%) and Comedy (21%) |
| Worst viral category | News (~1%) despite high view counts |
| Best posting hour | 14–20 UTC (9am–3pm EST) |
| Title length | Shorter titles (≤40 chars) correlate with virality |
| Tag sweet spot | 11–30 tags perform best |

---

## ⚙️ Feature Engineering

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
| `cat_*` | One-hot encoded category (15 categories) |

> ⚠️ `likes` and `comment_count` were intentionally excluded as they constitute **data leakage** — they are only known after a video trends.

---

## 🤖 Model Comparison

| Model | ROC-AUC | F1 (viral) |
|---|---|---|
| Logistic Regression | 0.7850 | 0.3188 |
| Decision Tree | 0.7981 | 0.3488 |
| XGBoost | 0.9497 | 0.6191 |
| **Random Forest** ✅ | **0.9516** | **0.6635** |

All models trained with `class_weight='balanced'` to handle the 90/10 class imbalance.

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model info |
| POST | `/predict` | Predict virality |
| GET | `/categories` | List valid categories |
| GET | `/days` | List valid days |

### Sample Request

```bash
curl -X POST "https://your-app.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "10 Things You Didnt Know About Space!",
       "category": "Education",
       "views": 500000,
       "dislikes": 1000,
       "tag_count": 20,
       "publish_hour": 15,
       "publish_day": "Friday",
       "publish_month": 6
     }'
```

### Sample Response

```json
{
  "prediction": "viral",
  "viral_probability": 0.78,
  "confidence": "high",
  "input_summary": {
    "title": "10 Things You Didnt Know About Space!",
    "category": "Education",
    "views": 500000
  }
}
```

---

## 🗂️ Project Structure

```
viral_predictor/
├── main.py              # FastAPI app
├── model.py             # Feature engineering logic
├── requirements.txt
├── rf_model.pkl         # Trained Random Forest model
├── features.pkl         # Feature list
└── README.md
```

---

## 🛠️ Run Locally

```bash
git clone https://github.com/your-username/viral-predictor
cd viral-predictor
pip install -r requirements.txt
uvicorn main:app --reload
```

API live at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

---

## 📦 Tech Stack

- **ML:** scikit-learn, XGBoost, pandas, numpy
- **API:** FastAPI, Uvicorn, Pydantic
- **Deployment:** Render
- **Development:** Google Colab, Python 3.10+

---

## 📝 Resume Bullet Points

> Built a YouTube viral post prediction API using Random Forest achieving **ROC-AUC 0.9516**, with intentional data leakage prevention, SHAP-based interpretability, and a live FastAPI endpoint deployed on Render.

---

## 👤 Author

**Boobesh**  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)
