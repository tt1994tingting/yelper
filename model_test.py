import json
import pandas as pd
import random
from   sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.metrics import accuracy_score, classification_report
import joblib

filename = 'yelp_academic_dataset_review.json'
sample_size = 100000
selected_columns = ['stars', 'text']

# Step 1: Calculate the total number of lines in the file (optional)
with open(filename, 'r', encoding='utf-8') as file:
    total_lines = sum(1 for _ in file)

# Step 2: Randomly select line indices
random_indices = set(random.sample(range(total_lines), sample_size))

# Step 3: Read the file line by line and collect the randomly selected lines
random_rows = []
with open(filename, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i in random_indices:
            random_rows.append(line)
del i

data_list = []
for line in random_rows:
    # Step 2: Parse each line as JSON and add to the list
    json_obj = json.loads(line.strip())
    data_list.append(json_obj)
del random_rows, line

df = pd.DataFrame(data_list)[selected_columns]

# 提取特征 (text) 和目标 (stars)
X_large = df['text']
y_large = df['stars']

# 加载保存的 CatBoost 模型
model = joblib.load('catboost_model_V1.pkl')
print("Model loaded successfully.")

# 将文本数据转换为 TF-IDF 特征
vectorizer = TfidfVectorizer()
X_large_vectorized = vectorizer.fit_transform(X_large)

# 使用加载的模型进行预测
y_pred_large = model.predict(X_large_vectorized)

# 评估模型在 10K 数据集上的表现
accuracy = accuracy_score(y_large, y_pred_large)
print(f"Accuracy on 10K dataset: {accuracy:.4f}")

# 打印详细的分类报告
print("\nClassification Report:")
print(classification_report(y_large, y_pred_large, zero_division=0))