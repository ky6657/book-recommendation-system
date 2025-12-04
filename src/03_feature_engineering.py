import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import BertTokenizer, BertModel
import torch
import os

# Force set working directory (critical!)
os.chdir("D:\\book_recommendation_system")

# Create features folder if not exists
features_dir = "D:\\book_recommendation_system\\data\\features\\"
if not os.path.exists(features_dir):
    os.makedirs(features_dir)
    print(f"Created folder: {features_dir}")

# Load cleaned data (absolute path + encoding)
books = pd.read_csv("D:\\book_recommendation_system\\data\\processed\\books.csv", encoding='latin-1')
users = pd.read_csv("D:\\book_recommendation_system\\data\\processed\\users.csv", encoding='latin-1')
merged_data = pd.read_csv("D:\\book_recommendation_system\\data\\processed\\merged_data.csv", encoding='latin-1')

# ===================== 1. Basic Feature Encoding =====================
def encode_basic_features():
    print("Encoding basic features...")
    # 1. Book feature encoding
    # Author encoding
    le_author = LabelEncoder()
    books['author_enc'] = le_author.fit_transform(books['author'].astype(str))
    # Publisher encoding
    le_publisher = LabelEncoder()
    books['publisher_enc'] = le_publisher.fit_transform(books['publisher'].astype(str))
    # Publish year normalization (0-1)
    scaler_year = MinMaxScaler()
    books['publish_year_norm'] = scaler_year.fit_transform(books[['publish_year']])
    
    # 2. User feature encoding
    # Age bin encoding (-1=unknown, 0=5-18, 1=19-30, 2=31-45, 3=46-60, 4=60+)
    def age_bin(age):
        if age == -1:
            return 0
        elif age <= 18:
            return 1
        elif age <= 30:
            return 2
        elif age <= 45:
            return 3
        elif age <= 60:
            return 4
        else:
            return 5
    users['age_bin'] = users['age'].apply(age_bin)
    # City encoding
    le_city = LabelEncoder()
    users['city_enc'] = le_city.fit_transform(users['city'].astype(str))
    
    # Save encoded data
    books[['isbn', 'author_enc', 'publisher_enc', 'publish_year_norm']].to_csv(
        f"{features_dir}book_basic_features.csv", index=False, encoding='latin-1'
    )
    users[['user_id', 'age_bin', 'city_enc']].to_csv(
        f"{features_dir}user_basic_features.csv", index=False, encoding='latin-1'
    )
    
    return books, users

# ===================== 2. Book Title Embedding (BERT) =====================
def generate_title_embedding(books_df):
    print("Generating title embeddings (BERT)... (This may take 5-10 minutes)")
    # Load English BERT (for English books)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    # Generate embeddings in batches (avoid OOM)
    embeddings = []
    batch_size = 100
    total_batches = len(books_df) // batch_size + 1
    
    for i in range(0, len(books_df), batch_size):
        batch_idx = i // batch_size + 1
        print(f"Processing batch {batch_idx}/{total_batches}...")
        
        batch = books_df.iloc[i:i+batch_size]['title'].tolist()
        # Text encoding
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            max_length=64, 
            truncation=True, 
            padding="max_length"
        )
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
        # Get CLS token vector
        batch_emb = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(batch_emb.tolist())
    
    # Save embedding
    books_df['title_embedding'] = embeddings
    books_df[['isbn', 'title_embedding']].to_csv(
        f"{features_dir}book_title_embedding.csv", index=False, encoding='latin-1'
    )
    
    return books_df

# ===================== 3. User-Book Interaction Features =====================
def generate_interaction_features(merged_data):
    print("Generating interaction features...")
    # 1. User average rating (user preference)
    user_rating_mean = merged_data.groupby('user_id')['rating'].mean().reset_index()
    user_rating_mean.columns = ['user_id', 'user_rating_mean']
    
    # 2. Book average rating (book popularity)
    book_rating_mean = merged_data.groupby('isbn')['rating'].mean().reset_index()
    book_rating_mean.columns = ['isbn', 'book_rating_mean']
    
    # 3. User rating count (user activity)
    user_rating_count = merged_data.groupby('user_id')['rating'].count().reset_index()
    user_rating_count.columns = ['user_id', 'user_rating_count']
    
    # 4. Book rating count (book popularity)
    book_rating_count = merged_data.groupby('isbn')['rating'].count().reset_index()
    book_rating_count.columns = ['isbn', 'book_rating_count']
    
    # Merge interaction features
    user_interact = pd.merge(user_rating_mean, user_rating_count, on='user_id')
    book_interact = pd.merge(book_rating_mean, book_rating_count, on='isbn')
    
    # Save
    user_interact.to_csv(f"{features_dir}user_interact_features.csv", index=False, encoding='latin-1')
    book_interact.to_csv(f"{features_dir}book_interact_features.csv", index=False, encoding='latin-1')
    
    return user_interact, book_interact

# ===================== Execute Feature Engineering =====================
if __name__ == "__main__":
    # 1. Basic feature encoding
    books_encoded, users_encoded = encode_basic_features()
    # 2. Title embedding (time-consuming)
    books_with_emb = generate_title_embedding(books_encoded)
    # 3. Interaction features
    user_interact, book_interact = generate_interaction_features(merged_data)
    
    print("=== Feature Engineering Completed ===")
    print(f"Book basic features: {books_encoded.shape}")
    print(f"User basic features: {users_encoded.shape}")
    print(f"User interaction features: {user_interact.shape}")
    print(f"Book interaction features: {book_interact.shape}")
    print(f"✅ All features saved to {features_dir}")