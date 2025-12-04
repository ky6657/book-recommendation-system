import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 强制设置工作目录为项目根目录（关键！）
os.chdir("D:\\book_recommendation_system")

# 加载原始数据（绝对路径+指定编码，解决Kaggle数据集编码问题）
books = pd.read_csv("D:\\book_recommendation_system\\data\\raw\\Books.csv", on_bad_lines='skip', encoding='latin-1')
users = pd.read_csv("D:\\book_recommendation_system\\data\\raw\\Users.csv", on_bad_lines='skip', encoding='latin-1')
ratings = pd.read_csv("D:\\book_recommendation_system\\data\\raw\\Ratings.csv", on_bad_lines='skip', encoding='latin-1')

# 1. 查看数据维度
print("=== 数据维度 ===")
print(f"书籍数据：{books.shape}")  # (271360, 8)
print(f"用户数据：{users.shape}")  # (278858, 3)
print(f"评分数据：{ratings.shape}")  # (1149780, 3)

# 2. 查看缺失值
print("\n=== 缺失值统计 ===")
print("书籍数据缺失值：")
print(books.isnull().sum())
print("\n用户数据缺失值：")
print(users.isnull().sum())
print("\n评分数据缺失值：")
print(ratings.isnull().sum())

# 3. 评分分布
plt.figure(figsize=(10, 5))
ratings['Book-Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('书籍评分分布')
plt.xlabel('评分')
plt.ylabel('数量')
plt.savefig("D:\\book_recommendation_system\\data\\ratings_dist.png")
plt.show()

# 4. 热门书籍（评分次数前10）
top_books = ratings['ISBN'].value_counts().head(10)
print("\n=== 热门书籍（评分次数前10）===")
print(top_books)