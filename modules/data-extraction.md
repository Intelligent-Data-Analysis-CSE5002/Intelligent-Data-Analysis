---
layout: page
title: 数据提取与分析
---

# 数据提取与分析

> 🎯 **模块目标**：学习数据提取技术和高级分析方法，从多源数据中获取价值

## 🔍 数据提取概述

数据提取是数据分析流程的关键第一步，涉及从各种数据源中获取、转换和加载数据。本模块将介绍现代数据提取技术和分析策略。

## 📊 数据源类型

### 结构化数据源
- **关系型数据库**：MySQL, PostgreSQL, Oracle
- **NoSQL数据库**：MongoDB, Cassandra, Redis
- **数据仓库**：Snowflake, BigQuery, Redshift
- **文件系统**：CSV, Excel, Parquet, JSON

### 非结构化数据源
- **文本数据**：日志文件、文档、社交媒体
- **图像数据**：照片、医学影像、卫星图像
- **音频数据**：语音记录、音乐、环境声音
- **视频数据**：监控录像、直播流、教学视频

### 实时数据流
- **消息队列**：Kafka, RabbitMQ, Apache Pulsar
- **API接口**：REST API, GraphQL, WebSocket
- **IoT设备**：传感器数据、设备状态
- **社交媒体**：Twitter API, Facebook Graph API

## 🛠️ 数据提取技术

### SQL数据提取
```sql
-- 基础数据查询
SELECT 
    customer_id,
    order_date,
    total_amount,
    product_category
FROM orders 
WHERE order_date >= '2024-01-01'
AND total_amount > 100;

-- 复杂分析查询
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(total_amount) as total_sales,
        COUNT(*) as order_count
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
),
growth_analysis AS (
    SELECT 
        month,
        total_sales,
        LAG(total_sales) OVER (ORDER BY month) as prev_month_sales,
        (total_sales - LAG(total_sales) OVER (ORDER BY month)) / 
        LAG(total_sales) OVER (ORDER BY month) * 100 as growth_rate
    FROM monthly_sales
)
SELECT * FROM growth_analysis
WHERE growth_rate IS NOT NULL
ORDER BY month;
```

### Python数据提取
```python
import pandas as pd
import requests
import sqlalchemy
from sqlalchemy import create_engine
import pymongo
from bs4 import BeautifulSoup

# 数据库连接
engine = create_engine('postgresql://user:password@localhost/database')

# SQL查询
def extract_from_sql(query):
    """从SQL数据库提取数据"""
    df = pd.read_sql(query, engine)
    return df

# API数据提取
def extract_from_api(url, headers=None):
    """从REST API提取数据"""
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API请求失败: {response.status_code}")

# MongoDB数据提取
def extract_from_mongodb(collection_name, query=None):
    """从MongoDB提取数据"""
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['database_name']
    collection = db[collection_name]
    
    cursor = collection.find(query or {})
    data = list(cursor)
    return pd.DataFrame(data)

# 网页爬取
def extract_from_web(url):
    """从网页提取数据"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 提取表格数据
    tables = soup.find_all('table')
    dfs = []
    for table in tables:
        df = pd.read_html(str(table))[0]
        dfs.append(df)
    
    return dfs
```

### 文件数据提取
```python
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# CSV文件处理
def process_csv_files(directory):
    """批量处理CSV文件"""
    csv_files = Path(directory).glob('*.csv')
    dfs = []
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = file_path.name
            dfs.append(df)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return pd.concat(dfs, ignore_index=True)

# JSON文件处理
def extract_from_json(file_path):
    """提取JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 展平嵌套JSON
    def flatten_json(nested_json, separator='_'):
        out = {}
        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    flatten(x[a], name + a + separator)
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + separator)
                    i += 1
            else:
                out[name[:-1]] = x
        flatten(nested_json)
        return out
    
    flattened = flatten_json(data)
    return pd.DataFrame([flattened])

# XML文件处理
def extract_from_xml(file_path):
    """提取XML数据"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    data = []
    for child in root:
        record = {}
        for subchild in child:
            record[subchild.tag] = subchild.text
        data.append(record)
    
    return pd.DataFrame(data)
```

## 🔄 数据转换与清洗

### ETL流水线
```python
import pandas as pd
from datetime import datetime
import re

class DataETLPipeline:
    def __init__(self):
        self.transformations = []
    
    def add_transformation(self, func):
        """添加转换函数"""
        self.transformations.append(func)
        return self
    
    def extract(self, source_config):
        """数据提取"""
        if source_config['type'] == 'sql':
            return extract_from_sql(source_config['query'])
        elif source_config['type'] == 'api':
            return pd.DataFrame(extract_from_api(source_config['url']))
        elif source_config['type'] == 'file':
            return pd.read_csv(source_config['path'])
        else:
            raise ValueError(f"不支持的数据源类型: {source_config['type']}")
    
    def transform(self, df):
        """数据转换"""
        for transformation in self.transformations:
            df = transformation(df)
        return df
    
    def load(self, df, target_config):
        """数据加载"""
        if target_config['type'] == 'sql':
            df.to_sql(target_config['table'], engine, if_exists='replace')
        elif target_config['type'] == 'file':
            df.to_csv(target_config['path'], index=False)
        else:
            raise ValueError(f"不支持的目标类型: {target_config['type']}")

# 常用转换函数
def clean_text_columns(df):
    """清洗文本列"""
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        # 去除多余空格
        df[col] = df[col].astype(str).str.strip()
        # 统一大小写
        df[col] = df[col].str.lower()
        # 去除特殊字符
        df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
    
    return df

def standardize_dates(df, date_columns):
    """标准化日期格式"""
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def handle_missing_values(df, strategy='drop'):
    """处理缺失值"""
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_mean':
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        return df
    elif strategy == 'fill_mode':
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df
    else:
        return df

# 使用示例
pipeline = DataETLPipeline()
pipeline.add_transformation(clean_text_columns)\
        .add_transformation(lambda df: standardize_dates(df, ['order_date', 'delivery_date']))\
        .add_transformation(lambda df: handle_missing_values(df, 'fill_mean'))

# 执行ETL流程
source_config = {'type': 'sql', 'query': 'SELECT * FROM orders'}
target_config = {'type': 'file', 'path': 'cleaned_orders.csv'}

raw_data = pipeline.extract(source_config)
cleaned_data = pipeline.transform(raw_data)
pipeline.load(cleaned_data, target_config)
```

## 📊 高级分析技术

### 特征工程
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def advanced_feature_engineering(df):
    """高级特征工程"""
    
    # 时间特征提取
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # 聚合特征
    if 'customer_id' in df.columns:
        customer_features = df.groupby('customer_id').agg({
            'order_amount': ['mean', 'std', 'count'],
            'product_category': lambda x: x.nunique()
        }).round(2)
        
        customer_features.columns = ['avg_order_amount', 'std_order_amount', 
                                   'total_orders', 'unique_categories']
        df = df.merge(customer_features, on='customer_id', how='left')
    
    # 交互特征
    if 'price' in df.columns and 'quantity' in df.columns:
        df['total_value'] = df['price'] * df['quantity']
        df['price_per_unit'] = df['price'] / df['quantity']
    
    # 文本特征
    if 'description' in df.columns:
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_features = tfidf.fit_transform(df['description'])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(100)])
        df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    
    return df
```

### 异常检测
```python
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def detect_anomalies(df, method='isolation_forest'):
    """异常检测"""
    
    # 选择数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_columns].fillna(0)
    
    if method == 'isolation_forest':
        # 孤立森林
        detector = IsolationForest(contamination=0.1, random_state=42)
        anomalies = detector.fit_predict(X)
        
    elif method == 'dbscan':
        # DBSCAN聚类
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        detector = DBSCAN(eps=0.5, min_samples=5)
        clusters = detector.fit_predict(X_scaled)
        anomalies = np.where(clusters == -1, -1, 1)
    
    # 添加异常标记
    df['is_anomaly'] = anomalies == -1
    
    # 可视化异常点
    if X.shape[1] >= 2:
        plt.figure(figsize=(10, 6))
        normal_points = X[anomalies == 1]
        anomaly_points = X[anomalies == -1]
        
        plt.scatter(normal_points.iloc[:, 0], normal_points.iloc[:, 1], 
                   alpha=0.6, label='正常点')
        plt.scatter(anomaly_points.iloc[:, 0], anomaly_points.iloc[:, 1], 
                   alpha=0.8, color='red', label='异常点')
        plt.xlabel(numeric_columns[0])
        plt.ylabel(numeric_columns[1])
        plt.legend()
        plt.title('异常检测结果')
        plt.show()
    
    return df
```

### 关联规则挖掘
```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def market_basket_analysis(df, transaction_col, item_col, min_support=0.01):
    """购物篮分析"""
    
    # 创建事务-商品矩阵
    basket = df.groupby([transaction_col, item_col])['quantity'].sum().unstack().fillna(0)
    
    # 转换为布尔值
    basket_bool = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # 挖掘频繁项集
    frequent_itemsets = apriori(basket_bool, min_support=min_support, use_colnames=True)
    
    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    
    # 排序并展示
    rules_sorted = rules.sort_values('lift', ascending=False)
    
    print("Top 10 关联规则:")
    print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
    
    return rules_sorted

# 使用示例
# transaction_data = pd.read_csv('transactions.csv')
# rules = market_basket_analysis(transaction_data, 'transaction_id', 'product_name')
```

## 📈 实时数据分析

### 流式数据处理
```python
import kafka
from kafka import KafkaConsumer
import json
import pandas as pd
from collections import deque
import threading
import time

class RealTimeAnalyzer:
    def __init__(self, kafka_topic, bootstrap_servers=['localhost:9092']):
        self.topic = kafka_topic
        self.servers = bootstrap_servers
        self.consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.data_buffer = deque(maxlen=1000)  # 保持最近1000条记录
        self.running = False
        
    def start_analysis(self):
        """开始实时分析"""
        self.running = True
        
        # 启动数据消费线程
        consumer_thread = threading.Thread(target=self._consume_data)
        consumer_thread.start()
        
        # 启动分析线程
        analysis_thread = threading.Thread(target=self._analyze_data)
        analysis_thread.start()
        
    def _consume_data(self):
        """消费Kafka数据"""
        for message in self.consumer:
            if not self.running:
                break
            self.data_buffer.append(message.value)
    
    def _analyze_data(self):
        """分析数据"""
        while self.running:
            if len(self.data_buffer) > 10:
                # 转换为DataFrame
                df = pd.DataFrame(list(self.data_buffer))
                
                # 实时统计
                stats = {
                    'count': len(df),
                    'avg_value': df['value'].mean() if 'value' in df.columns else 0,
                    'max_value': df['value'].max() if 'value' in df.columns else 0,
                    'timestamp': time.time()
                }
                
                # 异常检测
                if 'value' in df.columns:
                    recent_values = df['value'].tail(50)
                    threshold = recent_values.mean() + 2 * recent_values.std()
                    current_value = df['value'].iloc[-1]
                    
                    if current_value > threshold:
                        print(f"异常值检测: {current_value} > {threshold}")
                
                print(f"实时统计: {stats}")
            
            time.sleep(5)  # 每5秒分析一次
    
    def stop_analysis(self):
        """停止分析"""
        self.running = False
        self.consumer.close()

# 使用示例
# analyzer = RealTimeAnalyzer('user_events')
# analyzer.start_analysis()
```

## 🔍 数据质量评估

### 自动化质量检查
```python
def assess_data_quality(df):
    """评估数据质量"""
    
    quality_report = {}
    
    # 完整性检查
    missing_data = df.isnull().sum()
    quality_report['completeness'] = {
        'missing_values': missing_data.to_dict(),
        'missing_percentage': (missing_data / len(df) * 100).to_dict()
    }
    
    # 唯一性检查
    duplicate_count = df.duplicated().sum()
    quality_report['uniqueness'] = {
        'duplicate_rows': duplicate_count,
        'duplicate_percentage': duplicate_count / len(df) * 100
    }
    
    # 一致性检查
    consistency_issues = []
    
    # 检查数值列的异常值
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            consistency_issues.append({
                'column': col,
                'issue': 'outliers',
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100
            })
    
    quality_report['consistency'] = consistency_issues
    
    # 有效性检查
    validity_issues = []
    
    # 检查日期列
    date_columns = df.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        future_dates = df[df[col] > pd.Timestamp.now()][col]
        if len(future_dates) > 0:
            validity_issues.append({
                'column': col,
                'issue': 'future_dates',
                'count': len(future_dates)
            })
    
    quality_report['validity'] = validity_issues
    
    return quality_report

# 生成质量报告
quality_report = assess_data_quality(df)
print(json.dumps(quality_report, indent=2, default=str))
```

## 🔗 相关资源

### 提取工具
- **Apache Airflow**：工作流调度和监控
- **Talend**：数据集成平台
- **Apache NiFi**：数据流处理
- **Pentaho**：商业智能套件

### 分析框架
- **Apache Spark**：大数据处理引擎
- **Dask**：并行计算库
- **Ray**：分布式计算框架
- **Apache Kafka**：流式数据平台

## 🔗 导航链接

- [返回主页](../index.html)
- [上一模块：数据分析](data-analysis.html)
- [下一模块：数据标注](data-annotation.html)
