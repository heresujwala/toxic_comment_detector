import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

class ToxicCommentModel:
    def __init__(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        vector = self.vectorizer.transform([text])
        prediction = self.model.predict(vector)[0]
        probability = self.model.predict_proba(vector)[0][1]
        return {"prediction": int(prediction), "probability": float(probability)}
