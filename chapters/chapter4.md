---
layout: page
title: 第四章：数据可视化
---

# 第四章：数据可视化

> 🎯 **本章目标**：掌握Python数据可视化技术，创建专业图表

## 📚 学习内容

### 4.1 可视化基础
- 数据可视化原则
- 图表类型选择
- 颜色理论应用
- 可视化最佳实践

### 4.2 Matplotlib 基础
- 基本图形绘制
- 图表美化技巧
- 子图布局
- 样式定制

### 4.3 Seaborn 高级可视化
- 统计图表
- 多变量可视化
- 分类数据可视化
- 时间序列可视化

### 4.4 交互式可视化
- Plotly 交互图表
- Bokeh 仪表板
- Streamlit 应用开发

## 💻 实践练习

### 练习1：基础图表绘制
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 散点图
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'], alpha=0.6)
plt.title('散点图示例')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.show()
```

### 练习2：多子图布局
```python
# 创建多子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 子图1：直方图
axes[0,0].hist(data['column1'], bins=20)
axes[0,0].set_title('数据分布')

# 子图2：箱线图
axes[0,1].boxplot(data['column2'])
axes[0,1].set_title('异常值检测')

# 子图3：时间序列
axes[1,0].plot(data['date'], data['value'])
axes[1,0].set_title('时间趋势')

# 子图4：相关性热图
sns.heatmap(data.corr(), ax=axes[1,1], annot=True)
axes[1,1].set_title('相关性分析')

plt.tight_layout()
plt.show()
```

### 练习3：交互式图表
```python
import plotly.express as px

# 交互式散点图
fig = px.scatter(data, x='x', y='y', color='category',
                 size='size', hover_data=['info'],
                 title='交互式散点图')
fig.show()

# 交互式时间序列
fig = px.line(data, x='date', y='value',
              title='交互式时间序列图')
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
```

## 📊 可视化项目

### 项目：数据故事讲述
创建一个完整的数据可视化报告，包括：

1. **数据探索阶段**
   - 数据概览图表
   - 缺失值可视化
   - 分布特征展示

2. **关键发现展示**
   - 趋势分析图
   - 对比分析图
   - 相关性分析

3. **结论与建议**
   - 核心指标仪表板
   - 预测结果可视化
   - 行动建议图表

## 📝 作业

1. 完成可视化项目作品集
2. 创建交互式仪表板
3. 制作数据故事演示
4. 准备期末项目展示

## 🎨 可视化资源

### 颜色搭配工具
- [ColorBrewer](https://colorbrewer2.org/) - 专业配色方案
- [Coolors](https://coolors.co/) - 在线配色生成器

### 图表灵感
- [Observable](https://observablehq.com/) - D3.js 图表库
- [Kaggle Notebooks](https://www.kaggle.com/code) - 优秀可视化案例

### 图标资源
- [Font Awesome](https://fontawesome.com/) - 矢量图标
- [Feather Icons](https://feathericons.com/) - 简洁线性图标

## 🔗 相关资源

- [返回课程主页](../index.html)
- [上一章：机器学习简介](chapter3.html)
- [Matplotlib 官方文档](https://matplotlib.org/)
- [Seaborn 教程](https://seaborn.pydata.org/tutorial.html)
- [Plotly 文档](https://plotly.com/python/)
