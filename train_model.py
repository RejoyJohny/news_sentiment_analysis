# train_model.py
import sys
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

csv_path = sys.argv[1] if len(sys.argv) > 1 else "sentiment_dataset.csv"
df = pd.read_csv(csv_path)
df.columns = ["text", "sentiment"]
X = df["text"].fillna("").astype(str)
y = df["sentiment"].astype(int)

pipe = make_pipeline(
    TfidfVectorizer(max_features=10000, stop_words="english"),
    LogisticRegression(max_iter=1000)
)

pipe.fit(X, y)
joblib.dump(pipe, "model.pkl")
print("Saved model.pkl")
