# Disaster-Prediction

A machine learning project designed to classify disaster-related text messages using NLP and supervised learning models. This helps identify critical disaster information quickly from large-scale social or text data.

Disaster-Prediction/
│
├── src/
│   ├── evaluate.py        # Model evaluation metrics (accuracy, classification report)
│   ├── model.py           # Model building and training logic
│   ├── embeddings.py      # Word embedding generation and text feature representation
│   └── preprocess.py      # Text cleaning and preprocessing functions
│
├── README.md              # Project documentation

Features:

1. Preprocesses raw text (tokenization, stopword removal, stemming)

2. Generates embeddings using word2vec, TF-IDF features for model input and passing through BERT for semantic understanding.

3. Trains ML model (XGBoost)

4. Evaluates performance with accuracy and classification metrics

Modularized code (preprocessing, embeddings, model, evaluation separated)
