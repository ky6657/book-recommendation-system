import pandas as pd
import re
from datetime import datetime
import os

# 强制设置工作目录为项目根目录（关键！）
os.chdir("D:\\book_recommendation_system")

# 加载数据（绝对路径+指定编码）
books = pd.read_csv("D:\\book_recommendation_system\\data\\raw\\Books.csv", on_bad_lines='skip', encoding='latin-1')
users = pd.read_csv("D:\\book_recommendation_system\\data\\raw\\Users.csv", on_bad_lines='skip', encoding='latin-1')
ratings = pd.read_csv("D:\\book_recommendation_system\\data\\raw\\Ratings.csv", on_bad_lines='skip', encoding='latin-1')

# ===================== 1. 书籍数据清洗 =====================
def clean_books(df):
    print("正在清洗书籍数据...")
    # 重命名列（简化后续使用）
    df.rename(columns={
        'ISBN': 'isbn',
        'Book-Title': 'title',
        'Book-Author': 'author',
        'Year-Of-Publication': 'publish_year',
        'Publisher': 'publisher',
        'Image-URL-S': 'cover_url_s',
        'Image-URL-M': 'cover_url_m',
        'Image-URL-L': 'cover_url_l'
    }, inplace=True)
    
    # 去重（按ISBN）
    df = df.drop_duplicates(subset=['isbn'], keep='first')
    
    # 处理出版年份（清理非数字值，如'DK Publishing Inc'）
    df['publish_year'] = pd.to_numeric(df['publish_year'], errors='coerce')
    # 填充异常年份（<1900或>2025的填充为均值）
    valid_years = df[df['publish_year'].between(1900, 2025)]['publish_year']
    df['publish_year'] = df['publish_year'].fillna(valid_years.mean()).astype(int)
    
    # 文本清洗（书名/作者去特殊符号）
    def clean_text(text):
        if pd.isna(text):
            return text
        return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\-\']', '', text).strip()
    
    df['title'] = df['title'].apply(clean_text)
    df['author'] = df['author'].apply(clean_text)
    df['publisher'] = df['publisher'].apply(clean_text)
    
    # 过滤无效数据（书名/作者为空的）
    df = df[df['title'].notna() & df['author'].notna()]
    
    return df

# ===================== 2. 用户数据清洗 =====================
def clean_users(df):
    print("正在清洗用户数据...")
    # 重命名列
    df.rename(columns={
        'User-ID': 'user_id',
        'Location': 'location',
        'Age': 'age'
    }, inplace=True)
    
    # 处理年龄（填充缺失值为-1，后续编码）
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    # 过滤异常年龄（<5或>100的设为-1）
    df['age'] = df['age'].apply(lambda x: x if 5 <= x <= 100 else -1)
    
    # 提取城市（从location中拆分，如'usa, new york' → 'new york'）
    def extract_city(loc):
        if pd.isna(loc):
            return 'unknown'
        parts = loc.split(',')
        if len(parts) >= 2:
            return parts[-2].strip()  # 取倒数第二个部分（城市）
        return 'unknown'
    
    df['city'] = df['location'].apply(extract_city)
    
    # 去重（按user_id）
    df = df.drop_duplicates(subset=['user_id'], keep='first')
    
    return df

# ===================== 3. 评分数据清洗 =====================
def clean_ratings(df):
    print("正在清洗评分数据...")
    # 重命名列
    df.rename(columns={
        'User-ID': 'user_id',
        'ISBN': 'isbn',
        'Book-Rating': 'rating'
    }, inplace=True)
    
    # 过滤0分（未评分数据）
    df = df[df['rating'] > 0]
    
    # 过滤无效用户/书籍（不在清洗后的用户/书籍列表中的）
    clean_books_df = clean_books(books.copy())
    clean_users_df = clean_users(users.copy())
    df = df[df['user_id'].isin(clean_users_df['user_id'])]
    df = df[df['isbn'].isin(clean_books_df['isbn'])]
    
    # 去重
    df = df.drop_duplicates(subset=['user_id', 'isbn'], keep='first')
    
    return df

# ===================== 执行清洗并保存 =====================
if __name__ == "__main__":
    # 先创建processed文件夹（如果不存在）
    processed_dir = "D:\\book_recommendation_system\\data\\processed\\"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"创建文件夹：{processed_dir}")
    
    # 清洗数据
    clean_books_df = clean_books(books)
    clean_users_df = clean_users(users)
    clean_ratings_df = clean_ratings(ratings)
    
    # 保存清洗后数据（绝对路径）
    print("正在保存清洗后书籍数据...")
    clean_books_df.to_csv(f"{processed_dir}books.csv", index=False)
    print("正在保存清洗后用户数据...")
    clean_users_df.to_csv(f"{processed_dir}users.csv", index=False)
    print("正在保存清洗后评分数据...")
    clean_ratings_df.to_csv(f"{processed_dir}ratings.csv", index=False)
    
    # 合并用户-评分-书籍数据（供后续特征工程）
    print("正在合并数据...")
    merged_data = pd.merge(clean_ratings_df, clean_users_df, on='user_id')
    merged_data = pd.merge(merged_data, clean_books_df, on='isbn')
    merged_data.to_csv(f"{processed_dir}merged_data.csv", index=False)
    
    # 输出清洗后数据维度
    print("=== 清洗后数据维度 ===")
    print(f"书籍数据：{clean_books_df.shape}")
    print(f"用户数据：{clean_users_df.shape}")
    print(f"评分数据：{clean_ratings_df.shape}")
    print(f"合并后数据：{merged_data.shape}")
    print("✅ 数据清洗完成，文件已保存到data/processed/文件夹！")