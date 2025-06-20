import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sqlite3

# Load existing data
df = pd.read_csv('data/train.csv')

# Load feedback data
conn = sqlite3.connect('feedback.db')
feedback_df = pd.read_sql_query("SELECT comment, user_corrected FROM feedback", conn)

# Merge and clean
feedback_df.columns = ['comment_text', 'toxic']
df = pd.concat([df, feedback_df], ignore_index=True)

# Vectorization
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
X = tfidf.fit_transform(df['comment_text'].fillna(""))
y = df['toxic']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, 'app/model.pkl')
joblib.dump(tfidf, 'app/vectorizer.pkl')

print("✅ Retraining complete. Artifacts updated.")
