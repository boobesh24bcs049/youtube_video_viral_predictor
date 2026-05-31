import numpy as np
import pandas as pd

CAT_MAP = {
    'Film':1, 'Autos':2, 'Music':10, 'Pets':15,
    'Sports':17, 'Travel':19, 'Gaming':20, 'People':22,
    'Comedy':23, 'Entertainment':24, 'News':25,
    'Howto':26, 'Education':27, 'Science':28, 'Nonprofits':29
}

DOW_MAP = {
    "Monday":0, "Tuesday":1, "Wednesday":2,
    "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6
}

def build_features(data: dict, FEATURES: list) -> pd.DataFrame:
    row = {f: 0 for f in FEATURES}

    title         = data['title']
    views         = data['views']
    dislikes      = data['dislikes']
    tag_count     = data['tag_count']
    publish_hour  = data['publish_hour']
    publish_month = data['publish_month']
    dow           = DOW_MAP[data['publish_day']]
    cat_id        = CAT_MAP[data['category']]

    # Base
    row['category_id']       = cat_id
    row['views']             = views
    row['dislikes']          = dislikes
    row['tag_count']         = tag_count
    row['title_length']      = len(title)
    row['title_caps_count']  = sum(1 for c in title if c.isupper())
    row['publish_hour']      = publish_hour
    row['publish_dayofweek'] = dow
    row['publish_month']     = publish_month

    # Engineered
    row['log_views']        = np.log1p(views)
    row['dislike_ratio']    = dislikes / (views + 1)
    row['short_title']      = 1 if len(title) <= 40 else 0
    row['has_question']     = 1 if '?' in title else 0
    row['has_exclamation']  = 1 if '!' in title else 0
    row['caps_ratio']       = sum(1 for w in title.split() if w.isupper()) / max(len(title.split()), 1)
    row['is_weekend']       = 1 if dow >= 5 else 0
    row['peak_hour']        = 1 if 14 <= publish_hour <= 20 else 0
    row['comments_disabled'] = 0
    row['ratings_disabled']  = 0

    # One-hot category
    cat_col = f"cat_{data['category']}"
    if cat_col in row:
        row[cat_col] = 1

    return pd.DataFrame([row])[FEATURES]
