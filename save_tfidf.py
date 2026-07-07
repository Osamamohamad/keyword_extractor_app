import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean(text):
    text = text.lower()
    text = re.sub(r"'s\b", '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = pd.read_csv("Shuffled_Human.csv").head(2000)
df = df.dropna(subset=["Text"])
texts = df["Text"].astype(str).apply(clean)

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), stop_words="english")
tfidf.fit(texts)

with open("tfidf_model.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Saved tfidf model!")