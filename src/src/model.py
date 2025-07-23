from xgboost import XGBClassifier
import joblib

def train_model(X_train, y_train):
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        learning_rate=0.07,
        max_depth=4,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def predict(text, model, label_encoder, bert_model):
    bert_embedding = bert_model.encode([text])
    return label_encoder.inverse_transform(model.predict(bert_embedding))[0]
