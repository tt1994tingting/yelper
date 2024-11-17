import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_iris
import json
import joblib

# from transformers import BertTokenizer, BertModel
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from tqdm import tqdm

filename = 'reviews_sample_25000.json'
selected_columns = ['stars', 'text']

with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data)[selected_columns]

star_counts = df['stars'].value_counts().sort_index()
print(star_counts)

X = df['text']
y = df['stars']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 5: Initialize the CatBoost Classifier
model = CatBoostClassifier(
    iterations=100,          # Number of boosting iterations
    learning_rate=0.1,       # Learning rate
    depth=6,                 # Depth of each tree
    loss_function='MultiClass',
    verbose=10               # Print training information every 10 iterations
)

# Step 6: Train the Model
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Step 9: Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


joblib.dump(model, 'catboost_model_V1.pkl')