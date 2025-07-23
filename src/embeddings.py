from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

def generate_embeddings(texts, cleaned_texts):
    bert_embeddings = bert_model.encode(texts)
    tfidf_features = tfidf_vectorizer.fit_transform(cleaned_texts).toarray()
    return np.hstack((bert_embeddings, tfidf_features))
