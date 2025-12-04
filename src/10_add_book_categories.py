import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# 配置路径
os.chdir("D:\\book_recommendation_system")
input_path = "data/features/book_basic_features.csv"
output_path = "data/features/book_basic_features.csv"  # 覆盖原文件

# 1. 加载书籍特征数据
book_features_df = pd.read_csv(input_path, encoding='utf-8-sig')
print(f"原始书籍特征列：{book_features_df.columns.tolist()}")

# 2. 加载书籍基础数据（获取category信息）
books_df = pd.read_csv("data/processed/books.csv", encoding='utf-8-sig')
print(f"书籍基础数据列：{books_df.columns.tolist()}")

# 3. 合并书籍基础数据，获取category（如果没有category列，用publisher替代）
if 'category' in books_df.columns:
    book_features_df = pd.merge(book_features_df, books_df[['isbn', 'category']], on='isbn', how='left')
else:
    # 若没有category列，用publisher作为替代分类（保证能生成category_enc）
    book_features_df = pd.merge(book_features_df, books_df[['isbn', 'publisher']], on='isbn', how='left')
    book_features_df.rename(columns={'publisher': 'category'}, inplace=True)

# 4. 生成category_enc（分类编码）
le = LabelEncoder()
# 先填充空值为"unknown"，再编码
book_features_df['category'] = book_features_df['category'].fillna("unknown")
book_features_df['category_enc'] = le.fit_transform(book_features_df['category'])

# 5. 处理编码后的空值（实际这里不会有空值，保险处理）
book_features_df['category_enc'] = book_features_df['category_enc'].fillna(0).astype(int)

# 6. 保存结果（覆盖原文件）
book_features_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 已成功新增category_enc列，保存到：{output_path}")
print(f"分类编码数量：{len(le.classes_)}")