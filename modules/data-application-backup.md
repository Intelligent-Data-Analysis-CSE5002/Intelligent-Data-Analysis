---
layout: page
title: 数据应用
---

# 数据应用

> 🚀 **模块目标**：将数据分析技术应用到实际业务场景，创建端到端的智能数据应用系统

## 💼 数据应用概述

数据应用是将数据科学技术转化为实际业务价值的关键环节。本模块涵盖了从数据收集到部署生产的完整流程，包括应用架构设计、模型部署、性能监控和持续优化等核心内容。

## 🏗️ 应用架构设计

### 微服务架构

{% raw %}
```python
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import redis
import mysql.connector
from celery import Celery
import logging
import json
from datetime import datetime

# 数据服务微服务
class DataService:
    def __init__(self):
        self.app = Flask(__name__)
        self.api = Api(self.app, doc='/docs/')
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.setup_routes()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_routes(self):
        """设置API路由"""
        
        # 数据模型定义
        data_query_model = self.api.model('DataQuery', {
            'query': fields.String(required=True, description='查询条件'),
            'limit': fields.Integer(default=100, description='返回数量限制'),
            'offset': fields.Integer(default=0, description='偏移量')
        })
        
        @self.api.route('/data/query')
        class DataQueryResource(Resource):
            @self.api.expect(data_query_model)
            def post(self):
                """查询数据"""
                try:
                    data = request.get_json()
                    query = data.get('query')
                    limit = data.get('limit', 100)
                    offset = data.get('offset', 0)
                    
                    # 检查缓存
                    cache_key = f"query:{hash(query)}:{limit}:{offset}"
                    cached_result = self.redis_client.get(cache_key)
                    
                    if cached_result:
                        self.logger.info(f"返回缓存结果: {cache_key}")
                        return json.loads(cached_result)
                    
                    # 执行查询
                    result = self.execute_query(query, limit, offset)
                    
                    # 缓存结果
                    self.redis_client.setex(cache_key, 3600, json.dumps(result))
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"查询执行失败: {str(e)}")
                    return {'error': str(e)}, 500
        
        @self.api.route('/data/stats')
        class DataStatsResource(Resource):
            def get(self):
                """获取数据统计信息"""
                try:
                    stats = self.get_data_statistics()
                    return stats
                except Exception as e:
                    self.logger.error(f"统计信息获取失败: {str(e)}")
                    return {'error': str(e)}, 500
    
    def execute_query(self, query, limit, offset):
        """执行数据查询"""
        try:
            connection = mysql.connector.connect(
                host='localhost',
                database='analytics_db',
                user='user',
                password='password'
            )
            
            cursor = connection.cursor(dictionary=True)
            safe_query = self.sanitize_query(query)
            full_query = f"{safe_query} LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(full_query)
            results = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            return {
                'data': results,
                'count': len(results),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"数据库查询失败: {str(e)}")
            raise
    
    def sanitize_query(self, query):
        """清理和验证SQL查询"""
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f"不允许的关键字: {keyword}")
        
        return query
    
    def get_data_statistics(self):
        """获取数据统计信息"""
        return {
            'total_records': 1000000,
            'tables': ['users', 'orders', 'products'],
            'last_updated': datetime.now().isoformat(),
            'data_sources': ['mysql', 'redis', 'elasticsearch']
        }
```
{% endraw %}

## 🚀 模型部署

### Docker容器化部署

{% raw %}
```python
class ModelDeployment:
    def __init__(self):
        self.deployment_configs = {}
        
    def create_flask_app(self, model_path, model_name):
        """创建Flask应用"""
        app_code = f'''
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('{model_path}')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        
        return jsonify({{
            'prediction': prediction.tolist(),
            'model_name': '{model_name}',
            'version': '1.0'
        }})
        
    except Exception as e:
        return jsonify({{'error': str(e)}}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'healthy', 'model': '{model_name}'}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
'''
        return app_code
```
{% endraw %}

### A/B测试框架

{% raw %}
```python
import random
import hashlib
import json
from datetime import datetime

class ABTestFramework:
    def __init__(self):
        self.experiments = {}
        self.results = {}
        
    def create_experiment(self, experiment_id, variants, traffic_split=None):
        """创建A/B测试实验"""
        if traffic_split is None:
            split_ratio = 1.0 / len(variants)
            traffic_split = {variant: split_ratio for variant in variants}
        
        experiment = {
            'id': experiment_id,
            'variants': variants,
            'traffic_split': traffic_split,
            'start_time': datetime.now(),
            'status': 'running'
        }
        
        self.experiments[experiment_id] = experiment
        return experiment
    
    def assign_variant(self, experiment_id, user_id):
        """为用户分配变体"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        # 使用一致性哈希分配变体
        hash_input = f"{experiment_id}_{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0
        
        cumulative_probability = 0
        for variant, probability in experiment['traffic_split'].items():
            cumulative_probability += probability
            if normalized_hash <= cumulative_probability:
                return variant
        
        return list(experiment['variants'])[0]
```
{% endraw %}

## 📊 业务应用场景

### 推荐系统

{% raw %}
```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

class RecommendationSystem:
    def __init__(self):
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        
    def collaborative_filtering(self, user_item_matrix, n_recommendations=10):
        """协同过滤推荐"""
        model = NMF(n_components=50, random_state=42)
        user_features = model.fit_transform(user_item_matrix)
        item_features = model.components_
        
        predicted_ratings = np.dot(user_features, item_features)
        recommendations = {}
        
        for user_idx in range(len(user_item_matrix)):
            rated_items = np.where(user_item_matrix[user_idx] > 0)[0]
            user_predictions = predicted_ratings[user_idx]
            user_predictions[rated_items] = -1
            
            top_items = np.argsort(user_predictions)[::-1][:n_recommendations]
            recommendations[user_idx] = top_items.tolist()
        
        return recommendations
    
    def content_based_filtering(self, item_features, user_profiles, n_recommendations=10):
        """基于内容的推荐"""
        item_similarity = cosine_similarity(item_features)
        recommendations = {}
        
        for user_id, user_profile in user_profiles.items():
            user_scores = np.zeros(len(item_features))
            
            for item_id, rating in user_profile.items():
                similar_items = item_similarity[item_id]
                user_scores += similar_items * rating
            
            for item_id in user_profile.keys():
                user_scores[item_id] = -1
            
            top_items = np.argsort(user_scores)[::-1][:n_recommendations]
            recommendations[user_id] = top_items.tolist()
        
        return recommendations
```
{% endraw %}

### 智能客服系统

{% raw %}
```python
import re
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class IntelligentCustomerService:
    def __init__(self):
        self.knowledge_base = {}
        self.conversation_history = {}
        
    def build_knowledge_base(self, faq_data):
        """构建知识库"""
        for item in faq_data:
            category = item.get('category', 'general')
            question = item['question']
            answer = item['answer']
            
            if category not in self.knowledge_base:
                self.knowledge_base[category] = []
            
            self.knowledge_base[category].append({
                'question': question,
                'answer': answer,
                'tfidf_vector': None
            })
        
        self.compute_tfidf_vectors()
    
    def compute_tfidf_vectors(self):
        """计算知识库的TF-IDF向量"""
        all_questions = []
        question_mapping = []
        
        for category, items in self.knowledge_base.items():
            for i, item in enumerate(items):
                all_questions.append(item['question'])
                question_mapping.append((category, i))
        
        if all_questions:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(all_questions)
            self.vectorizer = vectorizer
            
            for idx, (category, item_idx) in enumerate(question_mapping):
                self.knowledge_base[category][item_idx]['tfidf_vector'] = tfidf_matrix[idx]
    
    def find_best_answer(self, user_question):
        """找到最佳答案"""
        if not self.knowledge_base or not hasattr(self, 'vectorizer'):
            return None
        
        user_vector = self.vectorizer.transform([user_question])
        best_answer = None
        best_similarity = 0
        
        for category in self.knowledge_base:
            for item in self.knowledge_base[category]:
                if item['tfidf_vector'] is not None:
                    similarity = cosine_similarity(user_vector, item['tfidf_vector'])[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_answer = {
                            'answer': item['answer'],
                            'confidence': similarity,
                            'source_question': item['question']
                        }
        
        if best_similarity > 0.3:
            return best_answer
        
        return None
    
    def generate_response(self, user_id, user_message):
        """生成回复"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'type': 'user',
            'message': user_message,
            'timestamp': datetime.now()
        })
        
        answer_info = self.find_best_answer(user_message)
        
        if answer_info:
            response = {
                'message': answer_info['answer'],
                'confidence': answer_info['confidence']
            }
        else:
            response = {
                'message': "抱歉，我没有找到相关信息。让我为您转接人工客服。",
                'confidence': 0.1
            }
        
        self.conversation_history[user_id].append({
            'type': 'bot',
            'message': response['message'],
            'confidence': response.get('confidence', 0),
            'timestamp': datetime.now()
        })
        
        return response
```
{% endraw %}

## 🔗 导航链接

- [返回主页](../index.html)
- [上一模块：数据生成和场景编辑](data-generation.html)
- [课程总结](../README.html)
