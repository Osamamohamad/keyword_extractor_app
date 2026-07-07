import pandas as pd
import numpy as np
import re
import nltk
import spacy
import streamlit as st

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# PAGE CONFIG
st.set_page_config(
    page_title="Hybrid Keyword Extraction",
    page_icon="🧠",
    layout="wide"
)

# SETUP
@st.cache_resource
def load_models():
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    nlp_model = spacy.load("en_core_web_sm")
    nlp_model.max_length = 2_000_000

    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5",device="cpu" )
    return stop_words, nlp_model, embedding_model

STOPWORDS, nlp, bge_model = load_models()


# CLEANING
def clean(text):
    text = text.lower()
    text = re.sub(r"'s\b", '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_with_lemma(text):
    text = clean(text)
    doc = nlp(text)
    return " ".join([
        token.lemma_
        for token in doc
        if not token.is_punct
    ])

def remove_wikipedia_noise(text):
    noise_patterns = [
        r'match details.*',
        r'external links.*',
        r'see also.*',
        r'references.*',
        r'\w+\.se\b.*',
        r'\w+\.com\b.*',
        r'\w+\.org\b.*',
        r'\[https?://\S+.*?\]',
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern,'',text,
            flags=re.IGNORECASE | re.DOTALL)

    return text.strip()
import pickle

@st.cache_resource
def load_tfidf():
    with open("tfidf_model.pkl", "rb") as f:
        return pickle.load(f)

tfidf = load_tfidf()

# VALIDATION
def is_valid_phrase(words):
    if len(words) == 0:
        return False
    if len(words) > 5:
        return False
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.7:
        return False
    generic_words = {
        "result", "results", "year", "years",
        "people", "person", "group", "system",
        "information", "data", "event", "events",
        "men", "women", "nation", "nations"
    }
    useful_words = [
        w for w in words
        if w not in generic_words
    ]
    if len(useful_words) < 1:
        return False

    return True

# CANDIDATE GENERATION
def get_candidates(original_text):
    doc = nlp(original_text)
    candidates = set()
    named_entities = set()
    # Named Entities
    for ent in doc.ents:
        ent_text = ent.text.strip().lower()
        if 1 <= len(ent_text.split()) <= 5:
            candidates.add(ent_text)
            named_entities.add(ent_text)
    # Noun Chunks
    for chunk in doc.noun_chunks:
        words = []
        for token in chunk:
            word = token.text.lower().strip()
            if (not token.is_stop
                and not token.is_punct
                and not token.like_num
                and len(word) > 2):
                words.append(word)

        if is_valid_phrase(words):
            candidates.add(" ".join(words))
    return list(candidates), named_entities

# TF-IDF KEYWORDS
def tfidf_keywords(text, top_k=10):
    vec = tfidf.transform([text])
    scores = vec.toarray()[0]
    features = tfidf.get_feature_names_out()
    results = []
    for i, word in enumerate(features):
        if (
            word not in STOPWORDS
            and len(word) > 2
            and scores[i] > 0
        ):
            results.append((word, scores[i]))
    results.sort(key=lambda x: x[1],reverse=True)

    return [w for w, s in results[:top_k]]

# BGE KEYWORDS
def bge_keywords(text, candidates, top_k=5):

    if not candidates:
        return []
    text_emb = normalize(bge_model.encode([text]))
    cand_emb = normalize(bge_model.encode(candidates,batch_size=32,show_progress_bar=False))
    scores = cosine_similarity(cand_emb,text_emb).flatten()
    ranked_idx = np.argsort(scores)[::-1]

    return [(candidates[idx], scores[idx])for idx in ranked_idx[:top_k] ]

# HYBRID RANKING
def hybrid_rank(tfidf_res, bge_res, named_entities,k_value):
    scores = {}
    # TF-IDF
    for word in tfidf_res:
        scores[word] = scores.get(word, 0) + 0.6

    # BGE
    for word, sim in bge_res:
        bonus = len(word.split()) * 0.15
        scores[word] = (
            scores.get(word, 0)+ sim+ bonus)

    # Named Entity Boost
    for word in list(scores.keys()):
        if word in named_entities:
            scores[word] += 0.5

    ranked = sorted(scores.items(),key=lambda x: x[1],reverse=True)

    return ranked[:k_value]

# UI
st.title("🧠 Hybrid Semantic Keyword Extraction")
st.markdown(""" 
 """)

# Sidebar
st.sidebar.header("⚙ Settings")

top_k = st.sidebar.slider("Top Keywords",3,10,5)

# Text Input
text_input = st.text_area("📄 Enter Text",height=250)

# Run Button
if st.button("🚀 Extract Keywords"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing..."):
            text = remove_wikipedia_noise(text_input)
            text_clean = clean_with_lemma(text)
            tfidf_res = tfidf_keywords(text_clean,top_k=10)

            candidates, named_entities = get_candidates(text)
            bge_res = bge_keywords(text,candidates,top_k=top_k)

            final = hybrid_rank(tfidf_res, bge_res, named_entities, top_k)

        st.success("Extraction Completed")

        # Results
        st.subheader("🔑 Final Keywords")
        for i, (kw, score) in enumerate(final, 1):
            st.markdown(f"### {i}. {kw}")
            st.progress(min((float(score) / 2), 1.0))

            st.write(f"Score: {score:.3f}")

        # Debug Section
        with st.expander("📊 Technical Details"):

            st.write("TF-IDF Keywords:")
            st.write(tfidf_res)

            st.write("Named Entities:")
            st.write(list(named_entities))

            st.write("Candidate Count:")
            st.write(len(candidates))