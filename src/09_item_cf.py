import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# 轻量版Item-CF：仅计算热门书的相似度，避免内存爆炸
def item_cf_recall(user_id, top_k=30):
    # 配置路径
    os.chdir("D:\\book_recommendation_system")
    ratings_path = "data/processed/ratings.csv"
    sim_cache_path = "data/features/book_sim_cache.pkl"  # 缓存相似度矩阵，避免重复计算
    
    try:
        # 1. 加载评分数据，只保留热门1000本书（核心优化）
        ratings_df = pd.read_csv(ratings_path, encoding='utf-8-sig')
        hot_books = ratings_df['isbn'].value_counts().head(1000).index.tolist()
        ratings_df = ratings_df[ratings_df['isbn'].isin(hot_books)]
        
        # 2. 检查是否有缓存的相似度矩阵，有则直接加载
        if os.path.exists(sim_cache_path):
            with open(sim_cache_path, 'rb') as f:
                book_sim_matrix = pickle.load(f)
        else:
            # 构建用户-物品评分矩阵
            user_item_matrix = ratings_df.pivot_table(
                index='user_id', columns='isbn', values='rating', fill_value=0
            )
            # 计算物品间的余弦相似度（仅1000本，内存占用约800MB）
            print("计算热门书相似度矩阵（仅一次，后续缓存复用）...")
            book_sim_matrix = cosine_similarity(user_item_matrix.T)
            book_sim_matrix = pd.DataFrame(
                book_sim_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns
            )
            # 保存缓存
            with open(sim_cache_path, 'wb') as f:
                pickle.dump(book_sim_matrix, f)
        
        # 3. 获取用户评分过的书，推荐相似书
        user_rated = ratings_df[ratings_df['user_id'] == user_id]['isbn'].tolist()
        if not user_rated:
            return []
        # 取用户评分过的第一本书，推荐相似书
        target_book = user_rated[0]
        if target_book not in book_sim_matrix.index:
            return []
        similar_books = book_sim_matrix[target_book].sort_values(ascending=False).head(top_k+len(user_rated)).index.tolist()
        # 排除用户已评分的书
        similar_books = [b for b in similar_books if b not in user_rated][:top_k]
        return similar_books
    
    except Exception as e:
        print(f"Item-CF召回异常：{e}")
        return []