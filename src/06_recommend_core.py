# 核心推荐逻辑：同作者优先+双塔+Item-CF+评分过滤（兼顾稳定+精准度）
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import importlib  # 用于动态导入数字开头模块

# 配置项目根路径
os.chdir("D:\\book_recommendation_system")
sys.path.append("D:\\book_recommendation_system\\src")

# 引入Item-CF协同过滤召回模块（兼容数字开头模块）
try:
    item_cf = importlib.import_module("09_item_cf")
    item_cf_recall = item_cf.item_cf_recall
except ImportError as e:
    print(f"Item-CF模块导入失败，将跳过该召回方式：{e}")
    item_cf_recall = lambda user_id, top_k: []  # 兜底空函数

# ====================== 加载已有文件（适配分类特征） ======================
# 1. 加载用户/书籍Embedding（你的现有文件）
user_emb = pd.read_csv("D:\\book_recommendation_system\\data\\features\\user_embedding.csv", encoding='utf-8-sig')
book_emb = pd.read_csv("D:\\book_recommendation_system\\data\\features\\book_embedding.csv", encoding='utf-8-sig')

# 2. 加载用户/书籍特征（含分类特征category_enc）
user_features = pd.read_csv("D:\\book_recommendation_system\\data\\features\\user_basic_features.csv", encoding='utf-8-sig')
book_features = pd.read_csv("D:\\book_recommendation_system\\data\\features\\book_basic_features.csv", encoding='utf-8-sig')

# 3. 加载排序模型（含分类特征的训练后模型）
try:
    ranking_model = joblib.load("D:\\book_recommendation_system\\models\\ranking_model.pkl")
except Exception as e:
    print(f"排序模型加载失败（不影响基础推荐）：{e}")
    ranking_model = None

# 4. 加载基础数据
ratings_df = pd.read_csv("D:\\book_recommendation_system\\data\\processed\\ratings.csv", encoding='utf-8-sig')
books_df = pd.read_csv("D:\\book_recommendation_system\\data\\processed\\books.csv", encoding='utf-8-sig')

# ====================== 1. 召回层：同作者优先+多源融合（提升精准度） ======================
def recall_books(user_id, top_k=100):
    print(f"Recalling books for user {user_id}...")
    recall_list = []
    user_rated = ratings_df[ratings_df['user_id'] == user_id]['isbn'].tolist()

    # 1. 同作者召回（占比40%，最贴合用户兴趣，优先度最高）
    if user_rated:
        try:
            rated_authors = books_df[books_df['isbn'].isin(user_rated)]['author'].unique()[:3]  # 限制前3个作者，避免冗余
            author_books = books_df[books_df['author'].isin(rated_authors)]['isbn'].tolist()[:40]  # 取40本
            recall_list.extend(author_books)
        except Exception as e:
            print(f"同作者召回异常：{e}")

    # 2. 双塔Embedding召回（占比30%，个性化核心）
    if user_id in user_emb['user_id'].values:
        try:
            user_vec = user_emb[user_emb['user_id'] == user_id]['embedding'].iloc[0]
            if isinstance(user_vec, str):
                user_vec = np.array(eval(user_vec))
            # 仅计算1000本热门书的相似度（保证速度）
            hot_1000_isbns = ratings_df['isbn'].value_counts().head(1000).index.tolist()
            book_emb_subset = book_emb[book_emb['isbn'].isin(hot_1000_isbns)].copy()
            # 计算相似度
            book_emb_subset['similarity'] = book_emb_subset['embedding'].apply(
                lambda x: 1 - cosine_similarity([eval(x) if isinstance(x, str) else x], [user_vec])[0][0]
            )
            tower_books = book_emb_subset.sort_values('similarity', ascending=False).head(30)['isbn'].tolist()  # 取30本
            recall_list.extend(tower_books)
        except Exception as e:
            print(f"双塔召回异常：{e}")

    # 3. Item-CF协同过滤召回（占比20%，补充行为相似性）
    cf_books = item_cf_recall(user_id, top_k=20)  # 取20本
    recall_list.extend(cf_books)

    # 4. 热门兜底（占比10%，保证覆盖度）
    hot_books = ratings_df['isbn'].value_counts().head(10).index.tolist()  # 取10本
    recall_list.extend(hot_books)

    # 去重并截取Top-100（保持优先级：同作者→双塔→CF→热门）
    recall_list = list(dict.fromkeys(recall_list))[:top_k]
    return recall_list

# ====================== 2. 排序层：评分过滤+分类特征排序（提升精准度） ======================
def rank_books(user_id, recall_books):
    print(f"Ranking books for user {user_id}...")
    if ranking_model is None or user_id not in user_features['user_id'].values:
        return recall_books
    
    rank_features = []
    valid_books = []
    for isbn in recall_books:
        book_feat = book_features[book_features['isbn'] == isbn]
        if len(book_feat) == 0:
            continue
        
        # 提取特征（用户2个+书籍4个=6个特征，含category_enc）
        user_feat_vals = user_features[user_features['user_id'] == user_id][['age_bin', 'city_enc']].iloc[0].values
        book_feat_cols = ['author_enc', 'publisher_enc', 'publish_year_norm', 'category_enc']
        book_feat_vals = book_feat[book_feat_cols].iloc[0].values
        
        # 处理缺失值
        feat = np.nan_to_num(np.concatenate([user_feat_vals, book_feat_vals]))
        rank_features.append(feat)
        valid_books.append(isbn)
    
    # 预测评分并过滤（仅保留预测评分≥3分的书，提升精准度）
    if len(rank_features) == 0:
        return recall_books
    scores = ranking_model.predict(np.array(rank_features))
    # 过滤预测评分≥3分的书，再排序
    book_score = [ (book, score) for book, score in zip(valid_books, scores) if score >= 3.0 ]
    # 若过滤后不足10本，补充召回结果（避免空推荐）
    if len(book_score) < 10:
        missing_num = 10 - len(book_score)
        remaining_books = [b for b in recall_books if b not in valid_books][:missing_num]
        book_score.extend( [ (b, 2.9) for b in remaining_books ] )
    # 排序
    book_score = sorted(book_score, key=lambda x: x[1], reverse=True)
    return [book for book, score in book_score]

# ====================== 3. 重排层：极简截取（保证速度） ======================
def rerank_books(user_id, ranked_books, top_k=10):
    print(f"Reranking books for user {user_id}...")
    return ranked_books[:top_k]

# ====================== 4. 推荐主函数 ======================
def recommend_books(user_id, top_k=10):
    try:
        recall_list = recall_books(user_id, top_k=100)
        rank_list = rank_books(user_id, recall_list)
        rerank_list = rerank_books(user_id, rank_list, top_k=top_k)
        
        # 构建最终结果
        result = []
        for isbn in rerank_list:
            book_info = books_df[books_df['isbn'] == isbn]
            if len(book_info) == 0:
                continue
            # 优化推荐理由（更精准）
            if user_id in user_emb['user_id'].values and len(ratings_df[ratings_df['user_id'] == user_id]) > 0:
                reason = "基于你的阅读偏好+同作者+相似用户推荐"
            else:
                reason = "推荐全站热门书籍"
            publish_year = int(book_info['publish_year'].iloc[0]) if not pd.isna(book_info['publish_year'].iloc[0]) else 0
            result.append({
                'isbn': isbn,
                'title': book_info['title'].iloc[0],
                'author': book_info['author'].iloc[0],
                'publisher': book_info['publisher'].iloc[0],
                'publish_year': publish_year,
                'reason': reason
            })
        return result
    except Exception as e:
        print(f"推荐异常（返回热门书）：{e}")
        hot_books = ratings_df['isbn'].value_counts().head(top_k).index.tolist()
        result = []
        for isbn in hot_books:
            book_info = books_df[books_df['isbn'] == isbn]
            if len(book_info) == 0:
                continue
            result.append({
                'isbn': isbn,
                'title': book_info['title'].iloc[0],
                'author': book_info['author'].iloc[0],
                'publisher': book_info['publisher'].iloc[0],
                'publish_year': int(book_info['publish_year'].iloc[0]) if not pd.isna(book_info['publish_year'].iloc[0]) else 0,
                'reason': "推荐全站热门书籍"
            })
        return result

# 测试函数
if __name__ == "__main__":
    rec_result = recommend_books(276726, top_k=5)
    print(f"\n用户276726的推荐结果：")
    for idx, book in enumerate(rec_result):
        print(f"{idx+1}. 《{book['title']}》 - {book['reason']}")