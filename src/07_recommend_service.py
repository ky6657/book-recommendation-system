from flask import Flask, request, jsonify, render_template
import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append("D:\\book_recommendation_system\\src")

# Force set working directory
os.chdir("D:\\book_recommendation_system")

# 动态加载推荐核心模块
recommend_core = __import__("06_recommend_core")
recommend_books = recommend_core.recommend_books

app = Flask(__name__, 
            template_folder="../templates",
            static_folder="../static")

# 定义数据文件路径
USERS_FILE = "D:\\book_recommendation_system\\data\\processed\\users.csv"
RATINGS_FILE = "D:\\book_recommendation_system\\data\\processed\\ratings.csv"

# 1. 前端页面路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

# 2. 用户注册接口
@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        user_id = int(data['user_id'])
        city = data['city']
        age = int(data['age'])

        # 检查用户ID是否已存在
        users_df = pd.read_csv(USERS_FILE, encoding='latin-1')
        if user_id in users_df['user_id'].values:
            return jsonify({'code': 400, 'message': '用户ID已存在'})

        # 新增用户数据
        new_user = pd.DataFrame({
            'user_id': [user_id],
            'location': [city],
            'age': [age],
            'city': [city]  # 和清洗后的格式一致
        })
        # 追加到文件
        new_user.to_csv(USERS_FILE, mode='a', header=False, index=False, encoding='latin-1')

        return jsonify({'code': 200, 'message': '注册成功'})
    except Exception as e:
        return jsonify({'code': 500, 'message': f'注册失败：{str(e)}'})

# 3. 评分提交接口
@app.route('/api/submit_rating', methods=['POST'])
def submit_rating():
    try:
        data = request.get_json()
        user_id = int(data['user_id'])
        isbn = data['isbn']
        rating = int(data['rating'])

        # 检查用户是否存在
        users_df = pd.read_csv(USERS_FILE, encoding='latin-1')
        if user_id not in users_df['user_id'].values:
            return jsonify({'code': 400, 'message': '用户ID不存在，请先注册'})

        # 检查书籍是否存在
        books_df = pd.read_csv("D:\\book_recommendation_system\\data\\processed\\books.csv", encoding='latin-1')
        if isbn not in books_df['isbn'].values:
            return jsonify({'code': 400, 'message': 'ISBN不存在'})

        # 新增评分数据
        new_rating = pd.DataFrame({
            'user_id': [user_id],
            'isbn': [isbn],
            'rating': [rating]
        })
        # 追加到文件
        new_rating.to_csv(RATINGS_FILE, mode='a', header=False, index=False, encoding='latin-1')

        return jsonify({'code': 200, 'message': '评分提交成功'})
    except Exception as e:
        return jsonify({'code': 500, 'message': f'提交失败：{str(e)}'})

# 4. 推荐接口
@app.route('/api/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id'))
        top_k = int(request.args.get('top_k', 6))
        result = recommend_books(user_id, top_k=top_k)
        return jsonify({'code': 200, 'message': 'success', 'data': result})
    except Exception as e:
        return jsonify({'code': 500, 'message': f'error: {str(e)}', 'data': []})

if __name__ == "__main__":
    print("Starting recommendation service...")
    print("Frontend URL: http://localhost:5000")
    print("Register URL: http://localhost:5000/register")
    app.run(host='0.0.0.0', port=5000, debug=False)