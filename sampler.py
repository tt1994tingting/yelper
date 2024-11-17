# import json
# import random
# from itertools import islice

# # Yelp data downloaded from:
# # https://www.yelp.com/dataset/documentation/main
# filename = 'yelp_academic_dataset_review.json'
# output_filename = 'reviews_sample.json'
# sample_size = 10000

# # Step 1: Calculate the total number of lines in the file (optional)
# with open(filename, 'r', encoding='utf-8') as file:
#     total_lines = sum(1 for _ in file)

# # Step 2: Randomly select line indices
# random_indices = set(random.sample(range(total_lines), sample_size))

# # Step 3: Read the file line by line and collect the randomly selected lines
# random_rows = []
# with open(filename, 'r', encoding='utf-8') as file:
#     for i, line in enumerate(file):
#         if i in random_indices:
#             random_rows.append(line)

# data_list = []
# for line in random_rows:
#     # Step 2: Parse each line as JSON and add to the list
#     json_obj = json.loads(line.strip())
#     data_list.append(json_obj)

# json_output = json.dumps(data_list, ensure_ascii=False, indent=4)

# # Step 4: Write the JSON string to an output file

# with open(output_filename, 'w', encoding='utf-8') as json_file:
#     json_file.write(json_output)


import json
import random
import pandas as pd
from collections import defaultdict

filename = 'yelp_academic_dataset_review.json'
output_filename = 'reviews_sample_25000.json'
sample_size_per_class = 5000

# Step 1: 初始化字典存储每个类别的数据
data_by_stars = defaultdict(list)

# Step 2: 从文件中读取数据并按 "stars" 分类
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line.strip())
        stars = json_obj.get('stars')
        useful = int(json_obj.get('useful'))
        if (stars is not None) and (useful > 0):
            data_by_stars[stars].append(json_obj)

# Step 3: 从每个类别中随机抽取 2000 条数据（如果不足则取全部）
balanced_data = []
for stars, reviews in data_by_stars.items():
    if len(reviews) > sample_size_per_class:
        balanced_data.extend(random.sample(reviews, sample_size_per_class))
    else:
        balanced_data.extend(reviews)

# Step 4: 保存抽取的数据为 JSON 文件
with open(output_filename, 'w', encoding='utf-8') as json_file:
    json.dump(balanced_data, json_file, ensure_ascii=False, indent=4)

print(f"数据已平衡抽取，并保存至 '{output_filename}' 文件。")


