import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "fake_news_model.pkl")
vectorizer = joblib.load(BASE / "tfidf_vectorizer.pkl")

CLASSES = list(model.classes_)
print("MODEL CLASSES:", CLASSES)


def find_fake_label():
    """Works with 0/1 labels or text labels like 'FAKE'/'REAL'."""
    for c in CLASSES:
        if str(c).strip().lower() in ("fake", "false", "misinformation"):
            return c
    # numeric labels: change this to 1 if your dataset uses 1 = fake
    return CLASSES[0]


FAKE_LABEL = find_fake_label()
print("USING FAKE LABEL:", FAKE_LABEL)

app = Flask(__name__)
HISTORY = []


def top_terms(X, n=5):
    try:
        names = vectorizer.get_feature_names_out()
        coef = model.coef_[0]
        sign = 1 if CLASSES[1] == FAKE_LABEL else -1
        row = X.tocoo()
        scored = [(names[j], sign * coef[j] * v) for j, v in zip(row.col, row.data)]
        scored.sort(key=lambda s: -s[1])
        return [w for w, s in scored[:n] if s > 0]
    except Exception:
        return []


def check_text(text):
    X = vectorizer.transform([text])
    idx = CLASSES.index(FAKE_LABEL)

    if hasattr(model, "predict_proba"):
        fake_prob = float(model.predict_proba(X)[0][idx])
    else:
        d = float(model.decision_function(X)[0])
        fake_side = d if idx == 1 else -d
        fake_prob = 1 / (1 + np.exp(-fake_side))

    score = round((1 - fake_prob) * 100)
    if score >= 65:
        verdict = "Likely accurate"
    elif score >= 40:
        verdict = "Needs checking"
    else:
        verdict = "Likely false"
    return {"score": score, "verdict": verdict, "terms": top_terms(X)}


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/check")
def api_check():
    try:
        text = (request.get_json(silent=True) or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "Please enter some text."}), 400
        result = check_text(text)
        HISTORY.insert(0, {
            "text": text[:140],
            "score": result["score"],
            "verdict": result["verdict"],
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.get("/api/stats")
def api_stats():
    counts = {"Likely accurate": 0, "Needs checking": 0, "Likely false": 0}
    for h in HISTORY:
        counts[h["verdict"]] += 1
    total = len(HISTORY)
    flagged = counts["Needs checking"] + counts["Likely false"]
    avg = round(sum(h["score"] for h in HISTORY) / total) if total else 0
    return jsonify({
        "total": total,
        "flagged_pct": round(flagged / total * 100) if total else 0,
        "avg_score": avg,
        "counts": counts,
        "recent": HISTORY[:8],
        "trend": [h["score"] for h in reversed(HISTORY[:20])],
    })


if __name__ == "__main__":
    app.run(port=5000)