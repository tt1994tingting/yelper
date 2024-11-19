import json
import pandas as pd
import torch
from torch.cuda.amp import autocast
from transformers import DistilBertTokenizerFast, DistilBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier
from tqdm import tqdm
import numpy as np
import nltk
from nltk.corpus import stopwords
import swifter
from imblearn.over_sampling import SMOTE
import joblib


# Step 1: 加载数据
filename = 'reviews_sample_25000.json'
selected_columns = ['stars', 'text']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data)[selected_columns]

# 打印各类别的样本数量
star_counts = df['stars'].value_counts().sort_index()
print("各类别样本数量:\n", star_counts)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def remove_stopwords(text):
    # 使用简单的 split() 方法进行分词。 BERT 分词器在处理文本时通常会添加一些特殊标记，如 [CLS] 和 [SEP]。这些标记并不需要参与停用词过滤, 效率很低。
    tokens = text.lower().split()
    # 去除停用词
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# swifter.apply() 会自动选择最快的执行方式，包括多线程或向量化处理，因此使用 tqdm 的进度条功能是多余的。
df['text_clean'] = df['text'].swifter.apply(remove_stopwords)

# swifter.apply() 会加载整个 DataFrame 到内存中，同时还要为新的 text_clean 列分配内存。这会导致内存占用急剧增加。
# 所以先保存
cleaned_filename = 'cleaned_reviews.csv'
df.to_csv(cleaned_filename, index=False)

del df
# gc.collect() 是 Python 内置的 垃圾回收（Garbage Collection, GC）， 强制清理引用，和所有未被引用的对象，并释放其占用的内存。
gc.collect()
# GPU需要下面这行来释放显存
torch.cuda.empty_cache()
print("内存清理完成")

df = pd.read_csv(cleaned_filename)
X = df['text_clean'].tolist()
y = df['stars'].values

# Step 2: 加载预训练的 DistilBERT 模型和分词器
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# 使用 autocast() 主要是在训练模型的时候效果显著，而在推理（inference）任务中（例如你当前的文本嵌入生成任务），收益相对较小。
# autocast() 可能会引入额外的复杂性，特别是在处理批量推理时。为了避免潜在的不稳定性（例如不同硬件设备上的不兼容问题），删除 autocast() 可以简化代码，提高稳定性。
def encode_texts(texts, tokenizer, model):
    """使用 DistilBERT 对文本进行编码，并启用混合精度计算"""
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        # with autocast():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# 保存 BERT 编码结果到文件
output_file = 'distilbert_embeddings_25000_cleanedV3.npy'
if os.path.exists(output_file):
    os.remove(output_file)

# 使用 numpy 的内存映射文件逐步保存嵌入，减少内存占用
batch_size = 128
embeddings_shape = (len(X), 768)
embeddings_array = np.memmap(output_file, dtype='float32', mode='w+', shape=embeddings_shape)

for i in tqdm(range(0, len(X), batch_size)):
    batch_texts = X[i:i + batch_size]
    embeddings = encode_texts(batch_texts, tokenizer, model)
    embeddings = embeddings.cpu().numpy()
    
    # 将嵌入保存到内存映射文件中
    start_idx = i
    end_idx = i + embeddings.shape[0]
    embeddings_array[start_idx:end_idx, :] = embeddings

# 确保数据已写入文件
del embeddings_array
print(f"Embeddings saved to {output_file}")

final_embeddings = np.memmap(output_file, dtype='float32', mode='r', shape=(25000, 768))
np.save('final_embeddings.npy', final_embeddings)


def group_stars(stars):
    if stars in [1, 2]:
        return 'low'
    elif stars == 3:
        return 'medium'
    else:  # 4, 5
        return 'high'

df['grouped_stars'] = df['stars'].apply(group_stars)

# 检查分组后的数据分布
print("\nGrouped stars distribution:")
print(df['grouped_stars'].value_counts())

y_grouped = df['grouped_stars'].values
label_mapping = {'low': 0, 'medium': 1, 'high': 2}
y_encoded = np.array([label_mapping[label] for label in y_grouped])

X_encoded = np.load('final_embeddings.npy')


# Step 4: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Step 2: 使用 SMOTE 进行上采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 检查上采样后的数据分布
unique, counts = np.unique(y_train_resampled, return_counts=True)
print(f"\nResampled class distribution: {dict(zip(unique, counts))}")

# Step 5: 初始化 CatBoost 分类器
model = CatBoostClassifier(
    iterations=1000,          # 增加迭代次数以提高准确率
    learning_rate=0.05,      # 降低学习率以减少过拟合
    depth=6,                 # 加深树的深度
    loss_function='MultiClass',
    verbose=50,               # 每 50 次迭代打印一次信息
    task_type='GPU',
    used_ram_limit='7GB',
    gpu_ram_part=0.9,
    max_ctr_complexity=3,
)

# Step 6: 训练模型
# model.fit(X_train, y_train)
print("\nTraining CatBoost model with SMOTE...")
model.fit(X_train_resampled, y_train_resampled)

# Step 7: 进行预测
y_pred = model.predict(X_test)

# Step 8: 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Step 9: 输出分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

"""
Accuracy: 0.7112

Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.77      0.78      1990
           1       0.41      0.44      0.43       989
           2       0.79      0.79      0.79      2021

    accuracy                           0.71      5000
   macro avg       0.67      0.67      0.67      5000
weighted avg       0.72      0.71      0.71      5000
"""
joblib.dump(model, 'catboost_model_smote.pkl')