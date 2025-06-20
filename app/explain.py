import shap
import joblib

class ExplanationEngine:
    def __init__(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.explainer = shap.Explainer(self.model)

    def explain(self, text):
        vector = self.vectorizer.transform([text])
        shap_values = self.explainer(vector)
        return shap_values
