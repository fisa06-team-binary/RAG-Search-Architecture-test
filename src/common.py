import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_vectorizer = None
_doc_vectors = None

def load_dataset(path="data/data.csv"):
    return pd.read_csv(path)

def _build_index(df):
    global _vectorizer, _doc_vectors
    _vectorizer = TfidfVectorizer()
    _doc_vectors = _vectorizer.fit_transform(df["summary"].astype(str))

def retrieve_candidates(query, df, top_k=10):
    global _vectorizer, _doc_vectors
    if _vectorizer is None or _doc_vectors is None:
        _build_index(df)

    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _doc_vectors).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    return df.iloc[top_indices]