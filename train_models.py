import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
print("Loading dataset...")
# Try-catch to handle the file location issue automatically
try:
    df = pd.read_csv('data/fake_job_postings.csv')
except FileNotFoundError:
    df = pd.read_csv('fake_job_postings.csv')

# 2. Preprocessing
print("Preprocessing data...")
columns_to_clean = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in columns_to_clean:
    df[col] = df[col].fillna('')

df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements']

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

print("Cleaning text (this may take a while)...")
df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Define 4 Models
models = {
    "Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(class_weight='balanced'))
    ]),
    "Random Forest": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ]),
    "Naive Bayes": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', MultinomialNB())
    ]),
    "SVM (Linear)": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        # SGDClassifier with loss='modified_huber' acts like a probabilistic SVM
        ('clf', SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=42))
    ])
}

# 4. Training
print("\nTraining 4 Models...")
results = {}

for name, pipeline in models.items():
    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    
    # Save Model
    filename = f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl"
    joblib.dump(pipeline, filename)

print("\nAll 4 models trained successfully!")