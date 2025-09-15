---
layout: page
title: 数据场景检索
---

# 数据场景检索

> 🔍 **模块目标**：掌握多模态数据场景检索技术，实现智能场景匹配和内容发现

## 🌟 场景检索概述

数据场景检索是指根据查询条件从大规模数据集中快速准确地找到相关场景或内容的技术。它结合了计算机视觉、自然语言处理、多媒体检索等多个领域的技术，在智能搜索、内容推荐、场景理解等应用中发挥重要作用。

## 🏗️ 检索系统架构

### 系统组件
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import torch
from sentence_transformers import SentenceTransformer
import faiss
import json
from datetime import datetime

class MultiModalRetrievalSystem:
    def __init__(self):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_features = {}
        self.text_features = {}
        self.metadata = {}
        self.indexes = {}
        
    def initialize_indexes(self):
        """初始化各种检索索引"""
        # 文本索引
        self.indexes['text'] = faiss.IndexFlatIP(384)  # 384维度
        
        # 图像索引
        self.indexes['image'] = faiss.IndexFlatIP(2048)  # ResNet特征维度
        
        # 混合索引
        self.indexes['hybrid'] = faiss.IndexFlatIP(512)  # 混合特征维度
    
    def add_data(self, data_id, text_content=None, image_path=None, metadata=None):
        """添加数据到检索系统"""
        features = {}
        
        # 处理文本特征
        if text_content:
            text_embedding = self.text_encoder.encode([text_content])
            features['text'] = text_embedding[0]
            self.text_features[data_id] = text_embedding[0]
        
        # 处理图像特征
        if image_path:
            image_features = self.extract_image_features(image_path)
            features['image'] = image_features
            self.image_features[data_id] = image_features
        
        # 存储元数据
        if metadata:
            self.metadata[data_id] = metadata
        
        return features
    
    def extract_image_features(self, image_path):
        """提取图像特征"""
        # 使用预训练的ResNet模型提取特征
        import torchvision.models as models
        import torchvision.transforms as transforms
        from PIL import Image
        
        # 加载预训练模型
        model = models.resnet50(pretrained=True)
        model.eval()
        
        # 移除最后的分类层
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载和处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # 提取特征
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = features.squeeze().numpy()
        
        return features
    
    def build_indexes(self):
        """构建所有索引"""
        # 构建文本索引
        if self.text_features:
            text_matrix = np.vstack(list(self.text_features.values()))
            self.indexes['text'].add(text_matrix.astype('float32'))
        
        # 构建图像索引
        if self.image_features:
            image_matrix = np.vstack(list(self.image_features.values()))
            self.indexes['image'].add(image_matrix.astype('float32'))
    
    def search_text(self, query, k=10):
        """文本检索"""
        query_embedding = self.text_encoder.encode([query])
        query_vector = query_embedding[0].astype('float32').reshape(1, -1)
        
        scores, indices = self.indexes['text'].search(query_vector, k)
        
        results = []
        data_ids = list(self.text_features.keys())
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(data_ids):
                data_id = data_ids[idx]
                results.append({
                    'data_id': data_id,
                    'score': float(score),
                    'metadata': self.metadata.get(data_id, {})
                })
        
        return results
    
    def search_image(self, image_path, k=10):
        """图像检索"""
        query_features = self.extract_image_features(image_path)
        query_vector = query_features.astype('float32').reshape(1, -1)
        
        scores, indices = self.indexes['image'].search(query_vector, k)
        
        results = []
        data_ids = list(self.image_features.keys())
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(data_ids):
                data_id = data_ids[idx]
                results.append({
                    'data_id': data_id,
                    'score': float(score),
                    'metadata': self.metadata.get(data_id, {})
                })
        
        return results
    
    def hybrid_search(self, text_query=None, image_path=None, text_weight=0.5, k=10):
        """混合检索"""
        combined_scores = {}
        
        # 文本检索结果
        if text_query:
            text_results = self.search_text(text_query, k*2)
            for result in text_results:
                data_id = result['data_id']
                combined_scores[data_id] = result['score'] * text_weight
        
        # 图像检索结果
        if image_path:
            image_results = self.search_image(image_path, k*2)
            for result in image_results:
                data_id = result['data_id']
                if data_id in combined_scores:
                    combined_scores[data_id] += result['score'] * (1 - text_weight)
                else:
                    combined_scores[data_id] = result['score'] * (1 - text_weight)
        
        # 排序并返回结果
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for data_id, score in sorted_results[:k]:
            results.append({
                'data_id': data_id,
                'score': score,
                'metadata': self.metadata.get(data_id, {})
            })
        
        return results
```

### 场景理解与索引
```python
class SceneUnderstandingModule:
    def __init__(self):
        self.scene_classifier = None
        self.object_detector = None
        self.caption_generator = None
        
    def analyze_scene(self, image_path):
        """全面场景分析"""
        analysis = {
            'scene_category': self.classify_scene(image_path),
            'objects': self.detect_objects(image_path),
            'spatial_layout': self.analyze_spatial_layout(image_path),
            'visual_attributes': self.extract_visual_attributes(image_path),
            'scene_caption': self.generate_caption(image_path)
        }
        return analysis
    
    def classify_scene(self, image_path):
        """场景分类"""
        # 使用预训练的场景分类模型
        categories = {
            'indoor': ['bedroom', 'kitchen', 'living_room', 'office', 'bathroom'],
            'outdoor': ['street', 'park', 'beach', 'mountain', 'urban'],
            'nature': ['forest', 'lake', 'desert', 'field', 'sky']
        }
        
        # 这里应该调用实际的场景分类模型
        # 示例返回
        return {
            'main_category': 'outdoor',
            'sub_category': 'street',
            'confidence': 0.85
        }
    
    def detect_objects(self, image_path):
        """物体检测"""
        # 使用YOLO或其他物体检测模型
        detected_objects = [
            {'class': 'car', 'confidence': 0.95, 'bbox': [100, 150, 300, 400]},
            {'class': 'person', 'confidence': 0.88, 'bbox': [50, 100, 150, 350]},
            {'class': 'tree', 'confidence': 0.75, 'bbox': [400, 50, 600, 300]}
        ]
        return detected_objects
    
    def analyze_spatial_layout(self, image_path):
        """空间布局分析"""
        layout = {
            'dominant_regions': ['sky', 'ground', 'buildings'],
            'perspective': 'street_level',
            'depth_information': {
                'foreground': ['person', 'car'],
                'background': ['buildings', 'sky']
            }
        }
        return layout
    
    def extract_visual_attributes(self, image_path):
        """提取视觉属性"""
        attributes = {
            'color_palette': ['blue', 'gray', 'green'],
            'lighting': 'daylight',
            'weather': 'clear',
            'season': 'summer',
            'time_of_day': 'afternoon'
        }
        return attributes
    
    def generate_caption(self, image_path):
        """生成场景描述"""
        # 使用图像描述生成模型
        caption = "A busy street scene with cars and pedestrians during daytime"
        return caption
    
    def create_scene_index(self, scene_analysis):
        """创建场景检索索引"""
        index_data = {
            'scene_tags': [
                scene_analysis['scene_category']['main_category'],
                scene_analysis['scene_category']['sub_category']
            ],
            'object_tags': [obj['class'] for obj in scene_analysis['objects']],
            'attribute_tags': [
                scene_analysis['visual_attributes']['lighting'],
                scene_analysis['visual_attributes']['weather'],
                scene_analysis['visual_attributes']['time_of_day']
            ],
            'text_description': scene_analysis['scene_caption']
        }
        return index_data
```

## 🔍 高级检索技术

### 语义检索
```python
from transformers import CLIPModel, CLIPProcessor
import torch

class SemanticRetrievalEngine:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        
    def encode_text(self, text_list):
        """编码文本为向量"""
        inputs = self.clip_processor(text=text_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def encode_image(self, image_list):
        """编码图像为向量"""
        inputs = self.clip_processor(images=image_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def compute_similarity(self, text_features, image_features):
        """计算文本-图像相似度"""
        similarity = torch.matmul(
            torch.tensor(text_features), 
            torch.tensor(image_features).T
        )
        return similarity.numpy()
    
    def cross_modal_search(self, query_text, image_database, top_k=10):
        """跨模态检索"""
        # 编码查询文本
        text_features = self.encode_text([query_text])
        
        # 编码图像数据库
        image_features = self.encode_image(image_database)
        
        # 计算相似度
        similarities = self.compute_similarity(text_features, image_features)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_index': idx,
                'similarity_score': similarities[0][idx],
                'image_path': image_database[idx] if isinstance(image_database[idx], str) else None
            })
        
        return results
```

### 基于图结构的检索
```python
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

class GraphBasedRetrieval:
    def __init__(self):
        self.knowledge_graph = nx.Graph()
        self.entity_embeddings = {}
        self.relation_types = ['similar_to', 'contains', 'located_in', 'related_to']
        
    def build_knowledge_graph(self, entities, relations):
        """构建知识图谱"""
        # 添加实体节点
        for entity in entities:
            self.knowledge_graph.add_node(
                entity['id'], 
                type=entity['type'],
                attributes=entity.get('attributes', {})
            )
        
        # 添加关系边
        for relation in relations:
            self.knowledge_graph.add_edge(
                relation['source'],
                relation['target'],
                relation_type=relation['type'],
                weight=relation.get('weight', 1.0)
            )
    
    def entity_expansion(self, query_entities, max_hops=2):
        """实体扩展"""
        expanded_entities = set(query_entities)
        
        for hop in range(max_hops):
            current_entities = expanded_entities.copy()
            for entity in current_entities:
                if entity in self.knowledge_graph:
                    neighbors = list(self.knowledge_graph.neighbors(entity))
                    expanded_entities.update(neighbors)
        
        return list(expanded_entities)
    
    def graph_based_ranking(self, query_entities, candidate_entities):
        """基于图的排序"""
        scores = {}
        
        for candidate in candidate_entities:
            score = 0
            
            for query_entity in query_entities:
                if query_entity in self.knowledge_graph and candidate in self.knowledge_graph:
                    try:
                        # 计算最短路径长度
                        path_length = nx.shortest_path_length(
                            self.knowledge_graph, query_entity, candidate
                        )
                        # 距离越近分数越高
                        score += 1.0 / (1 + path_length)
                    except nx.NetworkXNoPath:
                        # 没有路径连接
                        continue
            
            scores[candidate] = score
        
        # 按分数排序
        ranked_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_entities
    
    def personalized_pagerank(self, query_entities, alpha=0.15):
        """个性化PageRank"""
        # 创建个性化向量
        personalization = {}
        for node in self.knowledge_graph.nodes():
            if node in query_entities:
                personalization[node] = 1.0 / len(query_entities)
            else:
                personalization[node] = 0.0
        
        # 计算个性化PageRank
        pagerank_scores = nx.pagerank(
            self.knowledge_graph,
            personalization=personalization,
            alpha=alpha
        )
        
        return pagerank_scores
```

### 深度学习检索
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepRetrievalModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=512):
        super(DeepRetrievalModel, self).__init__()
        
        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 交互层
        self.interaction_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 相似度预测层
        self.similarity_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, image_features):
        # 编码
        text_encoded = self.text_encoder(text_features)
        image_encoded = self.image_encoder(image_features)
        
        # 交互注意力
        text_attended, _ = self.interaction_layer(
            text_encoded.unsqueeze(1),
            image_encoded.unsqueeze(1),
            image_encoded.unsqueeze(1)
        )
        text_attended = text_attended.squeeze(1)
        
        image_attended, _ = self.interaction_layer(
            image_encoded.unsqueeze(1),
            text_encoded.unsqueeze(1),
            text_encoded.unsqueeze(1)
        )
        image_attended = image_attended.squeeze(1)
        
        # 融合特征
        fused_features = torch.cat([text_attended, image_attended], dim=-1)
        
        # 预测相似度
        similarity = self.similarity_predictor(fused_features)
        
        return similarity

class DeepRetrievalTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_step(self, text_features, image_features, labels):
        self.optimizer.zero_grad()
        
        predictions = self.model(text_features, image_features)
        loss = self.criterion(predictions.squeeze(), labels.float())
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text_features, image_features, labels in test_loader:
                predictions = self.model(text_features, image_features)
                loss = self.criterion(predictions.squeeze(), labels.float())
                
                total_loss += loss.item()
                predicted = (predictions.squeeze() > 0.5).long()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return avg_loss, accuracy
```

## 🎯 特定场景应用

### 智能监控检索
```python
class SecurityRetrievalSystem:
    def __init__(self):
        self.alert_patterns = {
            'suspicious_behavior': ['loitering', 'running', 'fighting'],
            'object_detection': ['weapon', 'bag', 'vehicle'],
            'crowd_analysis': ['gathering', 'dispersal', 'density']
        }
        
    def analyze_security_footage(self, video_path, time_range=None):
        """分析安防视频"""
        analysis_results = []
        
        # 视频帧提取和分析
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.read()[0]:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 每隔30帧分析一次（约1秒）
            if frame_count % 30 == 0:
                timestamp = frame_count / 30
                
                # 行为分析
                behaviors = self.detect_behaviors(frame)
                
                # 物体检测
                objects = self.detect_suspicious_objects(frame)
                
                # 人群分析
                crowd_info = self.analyze_crowd(frame)
                
                analysis_results.append({
                    'timestamp': timestamp,
                    'behaviors': behaviors,
                    'objects': objects,
                    'crowd_info': crowd_info,
                    'alert_level': self.calculate_alert_level(behaviors, objects, crowd_info)
                })
        
        cap.release()
        return analysis_results
    
    def detect_behaviors(self, frame):
        """检测可疑行为"""
        # 这里应该调用实际的行为识别模型
        behaviors = [
            {'type': 'walking', 'confidence': 0.9, 'person_id': 1},
            {'type': 'loitering', 'confidence': 0.7, 'person_id': 2}
        ]
        return behaviors
    
    def detect_suspicious_objects(self, frame):
        """检测可疑物体"""
        objects = [
            {'type': 'bag', 'confidence': 0.8, 'bbox': [100, 100, 200, 200]},
            {'type': 'person', 'confidence': 0.95, 'bbox': [50, 50, 150, 300]}
        ]
        return objects
    
    def analyze_crowd(self, frame):
        """人群分析"""
        crowd_info = {
            'person_count': 15,
            'density': 'medium',
            'movement_direction': 'north',
            'activity_level': 'normal'
        }
        return crowd_info
    
    def calculate_alert_level(self, behaviors, objects, crowd_info):
        """计算警报级别"""
        alert_score = 0
        
        # 行为评分
        for behavior in behaviors:
            if behavior['type'] in self.alert_patterns['suspicious_behavior']:
                alert_score += behavior['confidence'] * 2
        
        # 物体评分
        for obj in objects:
            if obj['type'] in self.alert_patterns['object_detection']:
                alert_score += obj['confidence'] * 3
        
        # 人群评分
        if crowd_info['density'] == 'high':
            alert_score += 1
        
        # 确定警报级别
        if alert_score > 3:
            return 'high'
        elif alert_score > 1.5:
            return 'medium'
        else:
            return 'low'
    
    def query_incidents(self, query_type, time_range, location=None):
        """查询历史事件"""
        # 这里应该连接到事件数据库
        incidents = [
            {
                'timestamp': '2024-01-15 14:30:00',
                'type': 'suspicious_behavior',
                'location': 'entrance_gate',
                'alert_level': 'medium',
                'description': 'Person loitering near entrance for extended period'
            },
            {
                'timestamp': '2024-01-15 16:45:00',
                'type': 'object_detection',
                'location': 'parking_lot',
                'alert_level': 'high',
                'description': 'Unattended bag detected in parking area'
            }
        ]
        
        return incidents
```

### 电商场景检索
```python
class EcommerceSceneRetrieval:
    def __init__(self):
        self.product_attributes = ['color', 'style', 'brand', 'category', 'price_range']
        self.scene_contexts = ['indoor', 'outdoor', 'casual', 'formal', 'seasonal']
        
    def visual_search(self, query_image_path, filters=None):
        """视觉搜索商品"""
        # 提取查询图像特征
        query_features = self.extract_product_features(query_image_path)
        
        # 场景理解
        scene_context = self.understand_scene_context(query_image_path)
        
        # 商品检索
        similar_products = self.find_similar_products(query_features, scene_context, filters)
        
        return similar_products
    
    def extract_product_features(self, image_path):
        """提取商品特征"""
        features = {
            'color_palette': ['red', 'black', 'white'],
            'style_tags': ['casual', 'modern'],
            'category': 'clothing',
            'visual_features': np.random.rand(512)  # 实际应该是深度学习特征
        }
        return features
    
    def understand_scene_context(self, image_path):
        """理解场景上下文"""
        context = {
            'setting': 'outdoor',
            'occasion': 'casual',
            'season': 'summer',
            'weather': 'sunny'
        }
        return context
    
    def find_similar_products(self, query_features, scene_context, filters=None):
        """查找相似商品"""
        # 模拟商品数据库查询
        products = [
            {
                'product_id': 'P001',
                'name': 'Summer Casual T-Shirt',
                'category': 'clothing',
                'price': 29.99,
                'colors': ['red', 'blue', 'white'],
                'style_tags': ['casual', 'summer'],
                'similarity_score': 0.95
            },
            {
                'product_id': 'P002',
                'name': 'Outdoor Adventure Jacket',
                'category': 'clothing',
                'price': 89.99,
                'colors': ['black', 'green'],
                'style_tags': ['outdoor', 'adventure'],
                'similarity_score': 0.82
            }
        ]
        
        # 应用过滤条件
        if filters:
            filtered_products = []
            for product in products:
                if self.apply_filters(product, filters):
                    filtered_products.append(product)
            products = filtered_products
        
        # 按相似度排序
        products.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return products
    
    def apply_filters(self, product, filters):
        """应用搜索过滤条件"""
        for filter_type, filter_value in filters.items():
            if filter_type == 'price_range':
                min_price, max_price = filter_value
                if not (min_price <= product['price'] <= max_price):
                    return False
            elif filter_type == 'category':
                if product['category'] != filter_value:
                    return False
            elif filter_type == 'color':
                if filter_value not in product['colors']:
                    return False
        
        return True
    
    def recommendation_by_scene(self, user_profile, current_scene):
        """基于场景的商品推荐"""
        recommendations = []
        
        # 根据场景推荐
        if current_scene['setting'] == 'outdoor':
            if current_scene['season'] == 'summer':
                recommendations.extend(['sunglasses', 'hat', 'sunscreen'])
            elif current_scene['season'] == 'winter':
                recommendations.extend(['jacket', 'gloves', 'boots'])
        
        # 根据用户历史偏好调整
        if user_profile.get('preferred_brands'):
            # 过滤推荐品牌
            pass
        
        return recommendations
```

## 📊 性能优化

### 索引优化
```python
class IndexOptimizer:
    def __init__(self):
        self.index_types = ['flat', 'ivf', 'hnsw', 'pq']
        
    def benchmark_indexes(self, data, queries, index_configs):
        """基准测试不同索引"""
        results = {}
        
        for config in index_configs:
            index_type = config['type']
            params = config.get('params', {})
            
            # 构建索引
            start_time = time.time()
            index = self.build_index(data, index_type, params)
            build_time = time.time() - start_time
            
            # 测试查询性能
            start_time = time.time()
            search_results = []
            for query in queries:
                results_per_query = index.search(query, k=10)
                search_results.append(results_per_query)
            search_time = time.time() - start_time
            
            # 计算准确率
            accuracy = self.calculate_accuracy(search_results, ground_truth=None)
            
            results[index_type] = {
                'build_time': build_time,
                'search_time': search_time,
                'accuracy': accuracy,
                'memory_usage': self.get_memory_usage(index)
            }
        
        return results
    
    def build_index(self, data, index_type, params):
        """构建指定类型的索引"""
        dim = data.shape[1]
        
        if index_type == 'flat':
            index = faiss.IndexFlatIP(dim)
        elif index_type == 'ivf':
            nlist = params.get('nlist', 100)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        elif index_type == 'hnsw':
            m = params.get('m', 16)
            index = faiss.IndexHNSWFlat(dim, m)
        elif index_type == 'pq':
            m = params.get('m', 8)
            nbits = params.get('nbits', 8)
            index = faiss.IndexPQ(dim, m, nbits)
        
        index.add(data.astype('float32'))
        return index
    
    def optimize_query_processing(self, queries):
        """优化查询处理"""
        # 查询缓存
        query_cache = {}
        
        # 批量处理
        batch_size = 32
        batched_queries = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        
        # 查询重写和扩展
        optimized_queries = []
        for query in queries:
            if query in query_cache:
                optimized_queries.append(query_cache[query])
            else:
                expanded_query = self.expand_query(query)
                query_cache[query] = expanded_query
                optimized_queries.append(expanded_query)
        
        return optimized_queries
    
    def expand_query(self, query):
        """查询扩展"""
        # 同义词扩展
        synonyms = self.get_synonyms(query)
        
        # 相关词扩展
        related_terms = self.get_related_terms(query)
        
        expanded_query = {
            'original': query,
            'synonyms': synonyms,
            'related_terms': related_terms
        }
        
        return expanded_query
```

### 缓存策略
```python
import redis
import pickle
from functools import wraps

class RetrievalCache:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.default_ttl = 3600  # 1小时
        
    def cache_key(self, query, params=None):
        """生成缓存键"""
        import hashlib
        key_data = f"{query}_{params}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, query, params=None):
        """获取缓存结果"""
        key = self.cache_key(query, params)
        cached_data = self.redis_client.get(key)
        
        if cached_data:
            return pickle.loads(cached_data)
        return None
    
    def cache_result(self, query, result, params=None, ttl=None):
        """缓存结果"""
        key = self.cache_key(query, params)
        ttl = ttl or self.default_ttl
        
        self.redis_client.setex(key, ttl, pickle.dumps(result))
    
    def cache_decorator(self, ttl=None):
        """缓存装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
                
                # 尝试获取缓存
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return pickle.loads(cached_result)
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                self.redis_client.setex(
                    cache_key, 
                    ttl or self.default_ttl, 
                    pickle.dumps(result)
                )
                
                return result
            return wrapper
        return decorator

# 使用示例
cache = RetrievalCache()

@cache.cache_decorator(ttl=7200)  # 缓存2小时
def expensive_search_operation(query, filters):
    # 执行耗时的搜索操作
    time.sleep(2)  # 模拟耗时操作
    return f"Result for {query} with filters {filters}"
```

## 📈 评估指标

### 检索性能评估
```python
class RetrievalEvaluator:
    def __init__(self):
        pass
    
    def precision_at_k(self, retrieved_items, relevant_items, k):
        """计算P@K"""
        retrieved_k = retrieved_items[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_items))
        return relevant_retrieved / k if k > 0 else 0
    
    def recall_at_k(self, retrieved_items, relevant_items, k):
        """计算R@K"""
        retrieved_k = retrieved_items[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_items))
        return relevant_retrieved / len(relevant_items) if len(relevant_items) > 0 else 0
    
    def average_precision(self, retrieved_items, relevant_items):
        """计算平均精度"""
        if not relevant_items:
            return 0
        
        precision_sum = 0
        relevant_count = 0
        
        for i, item in enumerate(retrieved_items):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items)
    
    def mean_average_precision(self, all_retrieved, all_relevant):
        """计算MAP"""
        ap_scores = []
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            ap = self.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    def normalized_dcg(self, retrieved_items, relevance_scores, k=None):
        """计算NDCG"""
        if k:
            retrieved_items = retrieved_items[:k]
            relevance_scores = relevance_scores[:k]
        
        # 计算DCG
        dcg = 0
        for i, item in enumerate(retrieved_items):
            if item in relevance_scores:
                relevance = relevance_scores[item]
                dcg += (2**relevance - 1) / np.log2(i + 2)
        
        # 计算IDCG
        ideal_relevance = sorted(relevance_scores.values(), reverse=True)
        idcg = 0
        for i, relevance in enumerate(ideal_relevance):
            idcg += (2**relevance - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate_retrieval_system(self, test_queries, ground_truth, system_results):
        """综合评估检索系统"""
        metrics = {
            'precision_at_1': [],
            'precision_at_5': [],
            'precision_at_10': [],
            'recall_at_10': [],
            'map': [],
            'ndcg_at_10': []
        }
        
        for query_id, retrieved in system_results.items():
            if query_id in ground_truth:
                relevant = ground_truth[query_id]
                
                # 计算各种指标
                metrics['precision_at_1'].append(
                    self.precision_at_k(retrieved, relevant, 1)
                )
                metrics['precision_at_5'].append(
                    self.precision_at_k(retrieved, relevant, 5)
                )
                metrics['precision_at_10'].append(
                    self.precision_at_k(retrieved, relevant, 10)
                )
                metrics['recall_at_10'].append(
                    self.recall_at_k(retrieved, relevant, 10)
                )
                metrics['map'].append(
                    self.average_precision(retrieved, relevant)
                )
        
        # 计算平均值
        final_metrics = {}
        for metric_name, values in metrics.items():
            final_metrics[metric_name] = np.mean(values)
        
        return final_metrics
```

## 🔗 导航链接

- [返回主页](../index.html)
- [上一模块：数据标注](data-annotation.html)
- [下一模块：数据与场景可视化](data-visualization.html)
