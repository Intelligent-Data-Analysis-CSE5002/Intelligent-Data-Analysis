---
layout: page
title: 第三章：机器学习简介
---

# 第三章：机器学习简介

> 🎯 **本章目标**：理解机器学习基本原理和常用算法

## 📚 学习内容

### 3.1 机器学习概述
- 机器学习定义与分类
- 监督学习 vs 无监督学习
- 强化学习简介
- 机器学习工作流程

### 3.2 监督学习算法
- 线性回归
- 逻辑回归
- 决策树
- 随机森林
- 支持向量机

### 3.3 无监督学习算法
- K-means 聚类
- 层次聚类
- 主成分分析 (PCA)
- DBSCAN

### 3.4 模型评估与选择
- 训练集、验证集、测试集
- 交叉验证
- 评估指标
- 过拟合与欠拟合

## 💻 实践练习

### 练习1：线性回归实现
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差: {mse}')
```

### 练习2：分类算法比较
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 多个模型比较
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} 准确率: {accuracy:.3f}')
```

## 📝 作业

1. 完成房价预测项目
2. 比较不同算法性能
3. 撰写模型评估报告
4. 预习数据可视化内容

## 🔗 相关资源

- [返回课程主页](../index.html)
- [上一章：数据预处理](chapter2.html)
- [下一章：数据可视化](chapter4.html)
- [Scikit-learn 官方文档](https://scikit-learn.org/)
