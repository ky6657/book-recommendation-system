import pandas as pd
import numpy as np
import os
import joblib
from lightgbm import LGBMRegressor

# 配置路径
os.chdir("D:\\book_recommendation_system")
os.makedirs("models", exist_ok=True)

# 1. 加载数据（含新增的category_enc分类特征）
user_features = pd.read_csv("data/features/user_basic_features.csv", encoding='utf-8-sig')
book_features = pd.read_csv("data/features/book_basic_features.csv", encoding='utf-8-sig')
ratings_df = pd.read_csv("data/processed/ratings.csv", encoding='utf-8-sig')

# 2. 合并训练数据
train_data = pd.merge(ratings_df, user_features, on="user_id")
train_data = pd.merge(train_data, book_features, on="isbn")

# 3. 提取特征和标签（6个特征：用户2个+书籍4个，含category_enc）
feat_cols = ['age_bin', 'city_enc', 'author_enc', 'publisher_enc', 'publish_year_norm', 'category_enc']
# 处理缺失值（避免训练报错）
train_data[feat_cols] = train_data[feat_cols].fillna(0)
X = train_data[feat_cols].values
y = train_data['rating'].values  # 预测用户评分

# 4. 训练排序模型（优化参数，提升拟合能力）
print("开始训练包含分类特征的排序模型...")
model = LGBMRegressor(
    n_estimators=150,  # 增加树的数量，提升拟合能力
    learning_rate=0.08,  # 降低学习率，避免过拟合
    max_depth=6,  # 限制树深，防止过拟合
    random_state=42,
    verbose=-1
)
model.fit(X, y)

# 5. 保存模型（覆盖原有模型）
joblib.dump(model, "models/ranking_model.pkl")
print("✅ 含分类特征的排序模型已保存：models/ranking_model.pkl")
print(f"模型输入特征数：{len(feat_cols)}（{feat_cols}）")