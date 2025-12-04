import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Dot
from tensorflow.keras.optimizers import Adam
import os
import json  # Add json for standardized embedding saving

# Force set working directory
os.chdir("D:\\book_recommendation_system")

# Create features folder if not exists
features_dir = "D:\\book_recommendation_system\\data\\features\\"
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

# Load feature data (absolute path)
book_basic = pd.read_csv(f"{features_dir}book_basic_features.csv", encoding='latin-1')
book_interact = pd.read_csv(f"{features_dir}book_interact_features.csv", encoding='latin-1')
user_basic = pd.read_csv(f"{features_dir}user_basic_features.csv", encoding='latin-1')
user_interact = pd.read_csv(f"{features_dir}user_interact_features.csv", encoding='latin-1')
merged_data = pd.read_csv("D:\\book_recommendation_system\\data\\processed\\merged_data.csv", encoding='latin-1')

# Merge features
book_features = pd.merge(book_basic, book_interact, on='isbn')
user_features = pd.merge(user_basic, user_interact, on='user_id')
train_data = pd.merge(merged_data[['user_id', 'isbn', 'rating']], user_features, on='user_id')
train_data = pd.merge(train_data, book_features, on='isbn')

# ===================== 1. Build Two-Tower Model =====================
def build_tower_model(user_feature_dim, book_feature_dim, embedding_dim=64):
    print("Building Two-Tower model...")
    # User Tower
    user_input = Input(shape=(user_feature_dim,))
    user_dense1 = Dense(128, activation='relu')(user_input)
    user_dense2 = Dense(64, activation='relu')(user_dense1)
    user_embedding = Dense(embedding_dim, activation='relu', name='user_embedding')(user_dense2)
    
    # Book Tower
    book_input = Input(shape=(book_feature_dim,))
    book_dense1 = Dense(128, activation='relu')(book_input)
    book_dense2 = Dense(64, activation='relu')(book_dense1)
    book_embedding = Dense(embedding_dim, activation='relu', name='book_embedding')(book_dense2)
    
    # Calculate cosine similarity
    similarity = Dot(axes=1, normalize=True)([user_embedding, book_embedding])
    
    # Build model
    model = Model(inputs=[user_input, book_input], outputs=similarity)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

# ===================== 2. Prepare Training Data =====================
# Feature columns
user_feature_cols = ['age_bin', 'city_enc', 'user_rating_mean', 'user_rating_count']
book_feature_cols = ['author_enc', 'publisher_enc', 'publish_year_norm', 'book_rating_mean', 'book_rating_count']

# Extract feature matrices
X_user = train_data[user_feature_cols].values
X_book = train_data[book_feature_cols].values
# Label (normalize rating to 0-1)
y = train_data['rating'].values / 10.0

# Split train/test set
X_user_train, X_user_test, X_book_train, X_book_test, y_train, y_test = train_test_split(
    X_user, X_book, y, test_size=0.2, random_state=42
)

# ===================== 3. Train Model =====================
print("Training Two-Tower model... (This may take 10-15 minutes)")
model = build_tower_model(len(user_feature_cols), len(book_feature_cols))
history = model.fit(
    [X_user_train, X_book_train], y_train,
    batch_size=256,
    epochs=10,
    validation_data=([X_user_test, X_book_test], y_test)
)

# Save model
model.save(f"{features_dir}tower_model.h5")
print(f"Model saved to {features_dir}tower_model.h5")

# ===================== 4. Generate User/Book Embeddings (Fixed Format) =====================
print("Generating user/book embeddings...")
# Extract user/book towers
user_model = Model(inputs=model.input[0], outputs=model.get_layer('user_embedding').output)
book_model = Model(inputs=model.input[1], outputs=model.get_layer('book_embedding').output)

# Generate all user embeddings (use json.dumps to standardize format)
all_user_features = user_features[user_feature_cols].values
user_embeddings = user_model.predict(all_user_features)
user_emb_df = pd.DataFrame({
    'user_id': user_features['user_id'],
    'embedding': [json.dumps(emb.tolist()) for emb in user_embeddings]  # Fixed here
})
user_emb_df.to_csv(f"{features_dir}user_embedding.csv", index=False, encoding='latin-1')

# Generate all book embeddings (use json.dumps to standardize format)
all_book_features = book_features[book_feature_cols].values
book_embeddings = book_model.predict(all_book_features)
book_emb_df = pd.DataFrame({
    'isbn': book_features['isbn'],
    'embedding': [json.dumps(emb.tolist()) for emb in book_embeddings]  # Fixed here
})
book_emb_df.to_csv(f"{features_dir}book_embedding.csv", index=False, encoding='latin-1')

print("=== Recall Model Training Completed ===")
print(f"User embeddings count: {len(user_emb_df)}")
print(f"Book embeddings count: {len(book_emb_df)}")
print(f"✅ All embeddings saved to {features_dir}")