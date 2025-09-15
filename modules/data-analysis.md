---
layout: page
title: 数据分析
---

# 数据分析

> 🎯 **模块目标**：深入掌握数据分析方法，挖掘数据价值和洞察

## 📊 分析概述

数据分析是从原始数据中提取有价值信息和洞察的系统性过程。本模块将介绍现代数据分析的核心方法、工具和最佳实践。

## 🔍 分析方法论

### 数据分析流程
1. **问题定义**：明确分析目标和业务需求
2. **数据收集**：获取相关数据源
3. **数据探索**：初步了解数据特征
4. **数据清洗**：处理缺失值和异常值
5. **特征工程**：创建和选择关键特征
6. **模型构建**：选择合适的分析方法
7. **结果验证**：评估分析结果的可靠性
8. **洞察提取**：总结关键发现和建议

### 分析类型
- **描述性分析**：了解"发生了什么"
- **诊断性分析**：解释"为什么发生"
- **预测性分析**：预测"将要发生什么"
- **指导性分析**：建议"应该做什么"

## 📈 统计分析基础

### 描述性统计
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv('dataset.csv')

# 基本统计信息
print(data.describe())

# 数据分布
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(data['column1'], bins=30)
plt.title('数据分布')

plt.subplot(2, 2, 2)
plt.boxplot(data['column2'])
plt.title('箱线图')

plt.subplot(2, 2, 3)
sns.scatterplot(x='x', y='y', data=data)
plt.title('散点图')

plt.subplot(2, 2, 4)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('相关性矩阵')

plt.tight_layout()
plt.show()
```

### 假设检验
```python
from scipy import stats

# t检验
group1 = data[data['group'] == 'A']['value']
group2 = data[data['group'] == 'B']['value']

t_stat, p_value = stats.ttest_ind(group1, group2)
print(f'T统计量: {t_stat:.4f}')
print(f'P值: {p_value:.4f}')

# 卡方检验
contingency_table = pd.crosstab(data['category1'], data['category2'])
chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
print(f'卡方统计量: {chi2:.4f}')
print(f'P值: {p_val:.4f}')
```

## 🤖 机器学习分析

### 监督学习
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 数据准备
X = data.drop(['target'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('特征重要性排名')
plt.show()
```

### 无监督学习
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PCA降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title('聚类结果可视化')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.show()
```

## 📊 时间序列分析

### 趋势分析
```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# 时间序列数据准备
ts_data = pd.read_csv('timeseries.csv', parse_dates=['date'], index_col='date')

# 季节性分解
decomposition = seasonal_decompose(ts_data['value'], model='additive')

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='原始数据')
decomposition.trend.plot(ax=axes[1], title='趋势')
decomposition.seasonal.plot(ax=axes[2], title='季节性')
decomposition.resid.plot(ax=axes[3], title='残差')
plt.tight_layout()
plt.show()
```

### 预测建模
```python
# ARIMA模型
model = ARIMA(ts_data['value'], order=(1, 1, 1))
fitted_model = model.fit()

# 预测
forecast = fitted_model.forecast(steps=30)
confidence_intervals = fitted_model.get_forecast(steps=30).conf_int()

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data['value'], label='历史数据')
plt.plot(forecast.index, forecast, label='预测', color='red')
plt.fill_between(confidence_intervals.index, 
                confidence_intervals.iloc[:, 0], 
                confidence_intervals.iloc[:, 1], 
                alpha=0.3, color='red', label='置信区间')
plt.legend()
plt.title('时间序列预测')
plt.show()
```

## 🔍 高级分析技术

### A/B测试分析
```python
import scipy.stats as stats

# A/B测试数据
group_a = data[data['variant'] == 'A']['conversion_rate']
group_b = data[data['variant'] == 'B']['conversion_rate']

# 统计检验
t_stat, p_value = stats.ttest_ind(group_a, group_b)
effect_size = (group_b.mean() - group_a.mean()) / np.sqrt(
    ((group_a.var() + group_b.var()) / 2)
)

print(f'A组转化率: {group_a.mean():.4f}')
print(f'B组转化率: {group_b.mean():.4f}')
print(f'效应大小: {effect_size:.4f}')
print(f'P值: {p_value:.4f}')

# 结果可视化
plt.figure(figsize=(10, 6))
plt.hist(group_a, alpha=0.7, label='A组', bins=30)
plt.hist(group_b, alpha=0.7, label='B组', bins=30)
plt.legend()
plt.title('A/B测试结果分布')
plt.xlabel('转化率')
plt.ylabel('频次')
plt.show()
```

### 生存分析
```python
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# 生存分析数据
kmf = KaplanMeierFitter()

# 按组分析
groups = data['group'].unique()
plt.figure(figsize=(10, 6))

for group in groups:
    group_data = data[data['group'] == group]
    kmf.fit(group_data['duration'], group_data['event'])
    plt.plot(kmf.timeline, kmf.survival_function_, label=f'组 {group}')

plt.xlabel('时间')
plt.ylabel('生存概率')
plt.title('生存曲线')
plt.legend()
plt.show()

# 对数秩检验
group1_data = data[data['group'] == 'A']
group2_data = data[data['group'] == 'B']

results = logrank_test(
    group1_data['duration'], group2_data['duration'],
    group1_data['event'], group2_data['event']
)
print(f'对数秩检验 P值: {results.p_value:.4f}')
```

## 📋 分析报告

### 自动化报告生成
```python
import matplotlib.pyplot as plt
from fpdf import FPDF
import seaborn as sns

class AnalysisReport:
    def __init__(self, data):
        self.data = data
        self.pdf = FPDF()
    
    def generate_summary(self):
        """生成数据摘要"""
        summary = {
            'total_records': len(self.data),
            'features': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'basic_stats': self.data.describe().to_dict()
        }
        return summary
    
    def create_visualizations(self):
        """创建可视化图表"""
        # 相关性热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title('特征相关性矩阵')
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分布图
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, col in enumerate(numeric_columns[:4]):
            row, col_idx = i // 2, i % 2
            axes[row, col_idx].hist(self.data[col], bins=30)
            axes[row, col_idx].set_title(f'{col} 分布')
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_insights(self):
        """生成洞察和建议"""
        insights = []
        
        # 数据质量洞察
        missing_ratio = self.data.isnull().sum() / len(self.data)
        high_missing = missing_ratio[missing_ratio > 0.1]
        if not high_missing.empty:
            insights.append(f"发现 {len(high_missing)} 个特征缺失值超过10%")
        
        # 相关性洞察
        corr_matrix = self.data.corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append((corr_matrix.columns[i], 
                                   corr_matrix.columns[j], 
                                   corr_matrix.iloc[i, j]))
        
        if high_corr:
            insights.append(f"发现 {len(high_corr)} 对高相关性特征")
        
        return insights

# 使用示例
report = AnalysisReport(data)
summary = report.generate_summary()
report.create_visualizations()
insights = report.generate_insights()

print("分析摘要:")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\n关键洞察:")
for insight in insights:
    print(f"- {insight}")
```

## 📊 交互式分析

### Jupyter Dashboard
```python
import ipywidgets as widgets
from IPython.display import display
import plotly.express as px

# 交互式数据探索
def interactive_analysis():
    # 创建控件
    column_selector = widgets.Dropdown(
        options=data.columns.tolist(),
        description='选择列:'
    )
    
    chart_type = widgets.RadioButtons(
        options=['直方图', '散点图', '箱线图'],
        description='图表类型:'
    )
    
    # 输出区域
    output = widgets.Output()
    
    def update_chart(change):
        with output:
            output.clear_output()
            
            selected_column = column_selector.value
            chart_style = chart_type.value
            
            if chart_style == '直方图':
                fig = px.histogram(data, x=selected_column)
            elif chart_style == '散点图':
                if len(data.columns) > 1:
                    other_col = [col for col in data.columns if col != selected_column][0]
                    fig = px.scatter(data, x=selected_column, y=other_col)
                else:
                    fig = px.histogram(data, x=selected_column)
            else:  # 箱线图
                fig = px.box(data, y=selected_column)
            
            fig.show()
    
    # 绑定事件
    column_selector.observe(update_chart, names='value')
    chart_type.observe(update_chart, names='value')
    
    # 显示界面
    display(widgets.VBox([column_selector, chart_type, output]))
    
    # 初始显示
    update_chart(None)

# 调用交互式分析
interactive_analysis()
```

## 🔗 相关资源

### 分析工具
- **Python生态系统**：Pandas, NumPy, Scikit-learn
- **R语言**：统计分析专用语言
- **SQL**：数据库查询和分析
- **Tableau/Power BI**：商业智能工具

### 在线资源
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Coursera数据科学课程](https://www.coursera.org/browse/data-science)
- [edX统计学习](https://www.edx.org/course/statistical-learning)

## 🔗 导航链接

- [返回主页](../index.html)
- [上一模块：POINTS工具介绍](points-tool.html)
- [下一模块：数据提取与分析](data-extraction.html)
