import pandas as pd
import numpy as np
import os
import sys

# Force set working directory
os.chdir("D:\\book_recommendation_system")
# Add src directory to path
sys.path.append("D:\\book_recommendation_system\\src")

# 动态加载数字开头的模块
recommend_core = __import__("06_recommend_core")
recommend_books = recommend_core.recommend_books

# Load ratings data (absolute path)
ratings = pd.read_csv("D:\\book_recommendation_system\\data\\processed\\ratings.csv", encoding='latin-1')

# Select 100 test users (with at least 10 ratings)
test_users = ratings.groupby('user_id').filter(lambda x: len(x) >= 10)['user_id'].unique()[:100]
print(f"Evaluating on {len(test_users)} test users...")

# Evaluation metrics
hit_count = 0  # Hit rate (recommended books rated by user)
total_recall = 0
total_precision = 0

# Evaluate each user
for i, user_id in enumerate(test_users):
    if i % 10 == 0:
        print(f"Evaluating user {i+1}/{len(test_users)}...")
    
    # Get books rated by user
    user_books = ratings[ratings['user_id'] == user_id]['isbn'].tolist()
    # Generate recommendations
    rec_books = [book['isbn'] for book in recommend_books(user_id, top_k=10)]
    # Calculate hit count
    hit = len(set(rec_books) & set(user_books))
    
    # Update metrics
    hit_count += hit / len(rec_books) if len(rec_books) > 0 else 0
    # Recall (hit / total user books)
    recall = hit / len(user_books) if len(user_books) > 0 else 0
    total_recall += recall
    # Precision (hit / total recommended books)
    precision = hit / len(rec_books) if len(rec_books) > 0 else 0
    total_precision += precision

# Average metrics
avg_hit = hit_count / len(test_users)
avg_recall = total_recall / len(test_users)
avg_precision = total_precision / len(test_users)

# Print results
print("=== Recommendation Evaluation Results ===")
print(f"Average Hit Rate: {avg_hit:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average Precision: {avg_precision:.4f}")