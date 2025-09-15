---
layout: page
title: 数据与场景可视化
---

# 数据与场景可视化

> 📊 **模块目标**：掌握多维数据可视化技术，创建交互式场景展示和分析界面

## 🎨 可视化概述

数据与场景可视化是将抽象的数据转换为直观图形表示的过程，帮助用户理解复杂的数据模式、趋势和关系。在智能数据分析中，有效的可视化不仅能揭示数据中的洞察，还能促进决策制定和知识发现。

## 📈 基础可视化技术

### 统计图表可视化
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class BasicVisualization:
    def __init__(self, style='seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
        
    def create_comprehensive_dashboard(self, data):
        """创建综合数据仪表板"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('时间序列', '分布直方图', '相关性热图',
                          '散点图', '箱线图', '饼图',
                          '小提琴图', '条形图', '等高线图'),
            specs=[[{"secondary_y": True}, {}, {}],
                   [{}, {}, {"type": "pie"}],
                   [{}, {}, {}]]
        )
        
        # 时间序列图
        if 'timestamp' in data.columns and 'value' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['value'], 
                          name='时间序列', line=dict(color='blue')),
                row=1, col=1
            )
        
        # 分布直方图
        if 'value' in data.columns:
            fig.add_trace(
                go.Histogram(x=data['value'], name='分布', 
                           marker_color='lightblue'),
                row=1, col=2
            )
        
        # 相关性热图
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, 
                          x=corr_matrix.columns, 
                          y=corr_matrix.columns,
                          colorscale='RdBu', name='相关性'),
                row=1, col=3
            )
        
        # 散点图
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            fig.add_trace(
                go.Scatter(x=data[col1], y=data[col2], 
                          mode='markers', name='散点图',
                          marker=dict(color='red', size=8)),
                row=2, col=1
            )
        
        # 箱线图
        if 'category' in data.columns and 'value' in data.columns:
            for category in data['category'].unique():
                subset = data[data['category'] == category]
                fig.add_trace(
                    go.Box(y=subset['value'], name=f'箱线图_{category}'),
                    row=2, col=2
                )
        
        # 饼图
        if 'category' in data.columns:
            category_counts = data['category'].value_counts()
            fig.add_trace(
                go.Pie(labels=category_counts.index, 
                      values=category_counts.values, name='分类分布'),
                row=2, col=3
            )
        
        # 更新布局
        fig.update_layout(
            height=900,
            title_text="数据分析仪表板",
            showlegend=False
        )
        
        return fig
    
    def create_animated_visualization(self, data, time_col, value_col, category_col=None):
        """创建动画可视化"""
        if category_col:
            fig = px.scatter(data, x=time_col, y=value_col, 
                           color=category_col, size='size' if 'size' in data.columns else None,
                           animation_frame=time_col,
                           title="动态散点图")
        else:
            fig = px.line(data, x=time_col, y=value_col,
                         animation_frame=time_col,
                         title="动态折线图")
        
        fig.update_layout(
            xaxis_title=time_col,
            yaxis_title=value_col,
            hovermode='closest'
        )
        
        return fig
    
    def create_3d_visualization(self, data, x_col, y_col, z_col, color_col=None):
        """创建3D可视化"""
        if color_col and color_col in data.columns:
            fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, 
                              color=color_col, size='size' if 'size' in data.columns else None)
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=data[x_col],
                y=data[y_col],
                z=data[z_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data[z_col] if z_col in data.columns else 'blue',
                    colorscale='Viridis',
                    opacity=0.8
                )
            )])
        
        fig.update_layout(
            title='3D数据可视化',
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
        return fig

# 使用示例
# 生成示例数据
np.random.seed(42)
sample_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
    'value': np.random.randn(100).cumsum(),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.randint(10, 50, 100),
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'z': np.random.randn(100)
})

visualizer = BasicVisualization()
dashboard = visualizer.create_comprehensive_dashboard(sample_data)
# dashboard.show()
```

### 地理空间可视化
```python
import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd

class GeospatialVisualization:
    def __init__(self):
        self.default_location = [39.9042, 116.4074]  # 北京
        
    def create_interactive_map(self, data, lat_col='latitude', lon_col='longitude'):
        """创建交互式地图"""
        # 创建基础地图
        center_lat = data[lat_col].mean()
        center_lon = data[lon_col].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # 添加聚类标记
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in data.iterrows():
            popup_text = f"位置: ({row[lat_col]:.4f}, {row[lon_col]:.4f})"
            if 'name' in row:
                popup_text = f"名称: {row['name']}<br>" + popup_text
            if 'value' in row:
                popup_text += f"<br>数值: {row['value']}"
            
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=popup_text,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)
        
        return m
    
    def create_heatmap(self, data, lat_col='latitude', lon_col='longitude', 
                      weight_col=None):
        """创建热力图"""
        center_lat = data[lat_col].mean()
        center_lon = data[lon_col].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10
        )
        
        # 准备热力图数据
        if weight_col and weight_col in data.columns:
            heat_data = [[row[lat_col], row[lon_col], row[weight_col]] 
                        for idx, row in data.iterrows()]
        else:
            heat_data = [[row[lat_col], row[lon_col], 1] 
                        for idx, row in data.iterrows()]
        
        # 添加热力图层
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        return m
    
    def create_choropleth_map(self, geo_data, data, key_col, value_col):
        """创建分区统计图"""
        m = folium.Map(location=self.default_location, zoom_start=6)
        
        folium.Choropleth(
            geo_data=geo_data,
            name='choropleth',
            data=data,
            columns=[key_col, value_col],
            key_on='feature.properties.name',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=value_col
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_trajectory_map(self, trajectory_data, lat_col='latitude', 
                            lon_col='longitude', time_col='timestamp'):
        """创建轨迹地图"""
        # 按时间排序
        trajectory_data = trajectory_data.sort_values(time_col)
        
        center_lat = trajectory_data[lat_col].mean()
        center_lon = trajectory_data[lon_col].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # 创建轨迹线
        trajectory_points = trajectory_data[[lat_col, lon_col]].values.tolist()
        
        folium.PolyLine(
            trajectory_points,
            color='red',
            weight=3,
            opacity=0.8
        ).add_to(m)
        
        # 添加起点和终点标记
        start_point = trajectory_points[0]
        end_point = trajectory_points[-1]
        
        folium.Marker(
            start_point,
            popup='起点',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        folium.Marker(
            end_point,
            popup='终点',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
        
        # 添加中间时间点
        for idx, row in trajectory_data.iterrows():
            if idx % 10 == 0:  # 每10个点添加一个时间标记
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=f"时间: {row[time_col]}",
                    color='blue',
                    fill=True
                ).add_to(m)
        
        return m
```

### 实时数据可视化
```python
import streamlit as st
import time
import threading
from queue import Queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RealTimeVisualization:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.data_queue = Queue()
        self.is_running = False
        
    def start_data_simulation(self):
        """启动数据模拟"""
        def generate_data():
            import random
            timestamp = time.time()
            while self.is_running:
                # 模拟多个数据流
                data_point = {
                    'timestamp': timestamp,
                    'cpu_usage': random.uniform(20, 80),
                    'memory_usage': random.uniform(30, 90),
                    'network_io': random.uniform(0, 100),
                    'disk_io': random.uniform(0, 50)
                }
                self.data_queue.put(data_point)
                time.sleep(1)
                timestamp += 1
        
        self.is_running = True
        data_thread = threading.Thread(target=generate_data)
        data_thread.daemon = True
        data_thread.start()
    
    def create_realtime_dashboard(self):
        """创建实时仪表板"""
        st.title("实时数据监控仪表板")
        
        # 控制按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("开始监控"):
                self.start_data_simulation()
        
        with col2:
            if st.button("停止监控"):
                self.is_running = False
        
        # 创建图表容器
        chart_containers = {
            'system_metrics': st.empty(),
            'performance_gauge': st.empty(),
            'trend_analysis': st.empty()
        }
        
        # 数据存储
        data_history = {
            'timestamp': [],
            'cpu_usage': [],
            'memory_usage': [],
            'network_io': [],
            'disk_io': []
        }
        
        # 实时更新循环
        while self.is_running:
            # 获取新数据
            while not self.data_queue.empty():
                new_data = self.data_queue.get()
                
                # 更新历史数据
                for key, value in new_data.items():
                    if key in data_history:
                        data_history[key].append(value)
                        
                        # 保持数据量限制
                        if len(data_history[key]) > self.max_points:
                            data_history[key] = data_history[key][-self.max_points:]
            
            if data_history['timestamp']:
                # 更新系统指标图表
                self.update_system_metrics(chart_containers['system_metrics'], data_history)
                
                # 更新性能仪表
                self.update_performance_gauges(chart_containers['performance_gauge'], data_history)
                
                # 更新趋势分析
                self.update_trend_analysis(chart_containers['trend_analysis'], data_history)
            
            time.sleep(1)
    
    def update_system_metrics(self, container, data):
        """更新系统指标图表"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU使用率', '内存使用率', '网络I/O', '磁盘I/O'),
            vertical_spacing=0.1
        )
        
        # CPU使用率
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['cpu_usage'], 
                      name='CPU', line=dict(color='red')),
            row=1, col=1
        )
        
        # 内存使用率
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['memory_usage'], 
                      name='内存', line=dict(color='blue')),
            row=1, col=2
        )
        
        # 网络I/O
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['network_io'], 
                      name='网络', line=dict(color='green')),
            row=2, col=1
        )
        
        # 磁盘I/O
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['disk_io'], 
                      name='磁盘', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        container.plotly_chart(fig, use_container_width=True)
    
    def update_performance_gauges(self, container, data):
        """更新性能仪表"""
        if data['cpu_usage']:
            current_cpu = data['cpu_usage'][-1]
            current_memory = data['memory_usage'][-1]
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=('CPU使用率', '内存使用率')
            )
            
            # CPU仪表
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current_cpu,
                    domain={'x': [0, 0.5], 'y': [0, 1]},
                    title={'text': "CPU %"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                )
            )
            
            # 内存仪表
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current_memory,
                    domain={'x': [0.5, 1], 'y': [0, 1]},
                    title={'text': "内存 %"},
                    delta={'reference': 60},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ]
                    }
                )
            )
            
            fig.update_layout(height=300)
            container.plotly_chart(fig, use_container_width=True)
```

## 🎭 场景可视化

### 3D场景重建
```python
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Scene3DVisualization:
    def __init__(self):
        self.point_clouds = []
        self.meshes = []
        
    def load_point_cloud(self, file_path):
        """加载点云数据"""
        pcd = o3d.io.read_point_cloud(file_path)
        self.point_clouds.append(pcd)
        return pcd
    
    def create_synthetic_scene(self):
        """创建合成3D场景"""
        # 创建地面平面
        ground = o3d.geometry.TriangleMesh.create_box(
            width=10, height=0.1, depth=10
        )
        ground.translate([-5, -0.05, -5])
        ground.paint_uniform_color([0.5, 0.5, 0.5])
        
        # 创建建筑物
        building1 = o3d.geometry.TriangleMesh.create_box(
            width=2, height=3, depth=2
        )
        building1.translate([1, 0, 1])
        building1.paint_uniform_color([0.8, 0.2, 0.2])
        
        building2 = o3d.geometry.TriangleMesh.create_box(
            width=1.5, height=4, depth=1.5
        )
        building2.translate([-2, 0, 2])
        building2.paint_uniform_color([0.2, 0.8, 0.2])
        
        # 创建树木（圆柱体）
        tree_trunk = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.2, height=2
        )
        tree_trunk.translate([3, 0, -2])
        tree_trunk.paint_uniform_color([0.4, 0.2, 0.1])
        
        tree_crown = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        tree_crown.translate([3, 2, -2])
        tree_crown.paint_uniform_color([0.1, 0.6, 0.1])
        
        # 组合场景
        scene_objects = [ground, building1, building2, tree_trunk, tree_crown]
        return scene_objects
    
    def visualize_scene_with_annotations(self, scene_objects, annotations=None):
        """可视化带注释的3D场景"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D场景可视化", width=1200, height=800)
        
        # 添加场景对象
        for obj in scene_objects:
            vis.add_geometry(obj)
        
        # 添加注释
        if annotations:
            for annotation in annotations:
                # 创建文本标注（这里使用球体代替文本）
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                marker.translate(annotation['position'])
                marker.paint_uniform_color([1, 1, 0])  # 黄色标记
                vis.add_geometry(marker)
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_front([0.4, -0.2, -0.9])
        ctr.set_lookat([0, 2, 0])
        ctr.set_up([0, 1, 0])
        
        vis.run()
        vis.destroy_window()
    
    def create_point_cloud_from_depth(self, depth_image, intrinsic_matrix):
        """从深度图创建点云"""
        height, width = depth_image.shape
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # 获取内参
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        
        # 计算3D坐标
        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # 组合点云
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # 过滤无效点
        valid_mask = points[:, 2] > 0
        points = points[valid_mask]
        
        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    
    def scene_semantic_segmentation(self, point_cloud, labels):
        """场景语义分割可视化"""
        # 定义颜色映射
        color_map = {
            0: [0.5, 0.5, 0.5],  # 地面 - 灰色
            1: [0.8, 0.2, 0.2],  # 建筑 - 红色
            2: [0.1, 0.6, 0.1],  # 植被 - 绿色
            3: [0.2, 0.2, 0.8],  # 天空 - 蓝色
            4: [0.8, 0.8, 0.2],  # 车辆 - 黄色
            5: [0.8, 0.4, 0.8]   # 人物 - 紫色
        }
        
        # 应用颜色
        colors = np.array([color_map.get(label, [1, 1, 1]) for label in labels])
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        return point_cloud
```

### 虚拟现实场景
```python
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import math

class VRSceneVisualization:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.camera_pos = [0, 2, 5]
        self.camera_rotation = [0, 0]
        self.objects = []
        
    def initialize_vr_environment(self):
        """初始化VR环境"""
        pygame.init()
        display = (self.width, self.height)
        pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
        
        # 设置透视投影
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)
        
        # 设置光照
        self.setup_lighting()
    
    def setup_lighting(self):
        """设置光照"""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # 环境光
        ambient_light = [0.2, 0.2, 0.2, 1.0]
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
        
        # 漫反射光
        diffuse_light = [0.8, 0.8, 0.8, 1.0]
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
        
        # 光源位置
        light_position = [2.0, 4.0, 2.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    
    def add_scene_object(self, obj_type, position, rotation=None, scale=None, color=None):
        """添加场景对象"""
        scene_object = {
            'type': obj_type,
            'position': position,
            'rotation': rotation or [0, 0, 0],
            'scale': scale or [1, 1, 1],
            'color': color or [1, 1, 1]
        }
        self.objects.append(scene_object)
    
    def render_cube(self, size=1.0):
        """渲染立方体"""
        vertices = [
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],  # 后面
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]        # 前面
        ]
        
        faces = [
            [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
            [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]
        ]
        
        glBegin(GL_QUADS)
        for face in faces:
            for vertex in face:
                glVertex3fv([v * size for v in vertices[vertex]])
        glEnd()
    
    def render_sphere(self, radius=1.0, slices=20, stacks=20):
        """渲染球体"""
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = radius * math.sin(lat0)
            zr0 = radius * math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = radius * math.sin(lat1)
            zr1 = radius * math.cos(lat1)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                
                glNormal3f(x * zr0, y * zr0, z0)
                glVertex3f(x * zr0, y * zr0, z0)
                glNormal3f(x * zr1, y * zr1, z1)
                glVertex3f(x * zr1, y * zr1, z1)
            glEnd()
    
    def render_scene(self):
        """渲染整个场景"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置相机
        glLoadIdentity()
        glRotatef(self.camera_rotation[0], 1, 0, 0)
        glRotatef(self.camera_rotation[1], 0, 1, 0)
        glTranslatef(-self.camera_pos[0], -self.camera_pos[1], -self.camera_pos[2])
        
        # 渲染所有对象
        for obj in self.objects:
            glPushMatrix()
            
            # 应用变换
            glTranslatef(*obj['position'])
            glRotatef(obj['rotation'][0], 1, 0, 0)
            glRotatef(obj['rotation'][1], 0, 1, 0)
            glRotatef(obj['rotation'][2], 0, 0, 1)
            glScalef(*obj['scale'])
            
            # 设置颜色
            glColor3f(*obj['color'])
            
            # 渲染对象
            if obj['type'] == 'cube':
                self.render_cube()
            elif obj['type'] == 'sphere':
                self.render_sphere()
            
            glPopMatrix()
        
        pygame.display.flip()
    
    def run_vr_scene(self):
        """运行VR场景"""
        self.initialize_vr_environment()
        
        # 添加一些示例对象
        self.add_scene_object('cube', [0, 0, 0], color=[1, 0, 0])
        self.add_scene_object('sphere', [3, 1, 0], color=[0, 1, 0])
        self.add_scene_object('cube', [-3, 0, 0], color=[0, 0, 1])
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_keyboard_input(event.key)
                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:
                        self.handle_mouse_movement(event.rel)
            
            self.render_scene()
            clock.tick(60)
        
        pygame.quit()
    
    def handle_keyboard_input(self, key):
        """处理键盘输入"""
        move_speed = 0.5
        
        if key == pygame.K_w:
            self.camera_pos[2] -= move_speed
        elif key == pygame.K_s:
            self.camera_pos[2] += move_speed
        elif key == pygame.K_a:
            self.camera_pos[0] -= move_speed
        elif key == pygame.K_d:
            self.camera_pos[0] += move_speed
        elif key == pygame.K_q:
            self.camera_pos[1] += move_speed
        elif key == pygame.K_e:
            self.camera_pos[1] -= move_speed
    
    def handle_mouse_movement(self, rel):
        """处理鼠标移动"""
        sensitivity = 0.5
        self.camera_rotation[1] += rel[0] * sensitivity
        self.camera_rotation[0] += rel[1] * sensitivity
        
        # 限制上下旋转角度
        self.camera_rotation[0] = max(-90, min(90, self.camera_rotation[0]))
```

## 📊 交互式数据探索

### Web交互界面
```python
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

class InteractiveDataExplorer:
    def __init__(self, data):
        self.data = data
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """设置布局"""
        self.app.layout = html.Div([
            # 标题
            html.H1("交互式数据探索器", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # 控制面板
            html.Div([
                html.Div([
                    html.Label("选择X轴变量:"),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': col, 'value': col} 
                                for col in self.data.select_dtypes(include=[np.number]).columns],
                        value=self.data.select_dtypes(include=[np.number]).columns[0]
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("选择Y轴变量:"),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=[{'label': col, 'value': col} 
                                for col in self.data.select_dtypes(include=[np.number]).columns],
                        value=self.data.select_dtypes(include=[np.number]).columns[1] 
                        if len(self.data.select_dtypes(include=[np.number]).columns) > 1 
                        else self.data.select_dtypes(include=[np.number]).columns[0]
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),
                
                html.Div([
                    html.Label("颜色分组:"),
                    dcc.Dropdown(
                        id='color-dropdown',
                        options=[{'label': '无', 'value': None}] + 
                                [{'label': col, 'value': col} 
                                 for col in self.data.select_dtypes(include=['object']).columns],
                        value=None
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
            ]),
            
            # 图表类型选择
            html.Div([
                html.Label("图表类型:", style={'marginTop': 20}),
                dcc.RadioItems(
                    id='chart-type-radio',
                    options=[
                        {'label': '散点图', 'value': 'scatter'},
                        {'label': '线图', 'value': 'line'},
                        {'label': '柱状图', 'value': 'bar'},
                        {'label': '直方图', 'value': 'histogram'},
                        {'label': '箱线图', 'value': 'box'}
                    ],
                    value='scatter',
                    inline=True
                )
            ], style={'marginTop': 20}),
            
            # 过滤器
            html.Div([
                html.Label("数据过滤:", style={'marginTop': 20}),
                html.Div(id='filter-container')
            ]),
            
            # 主图表
            dcc.Graph(id='main-chart'),
            
            # 统计信息
            html.Div([
                html.H3("统计信息"),
                html.Div(id='statistics-display')
            ], style={'marginTop': 30}),
            
            # 数据表
            html.Div([
                html.H3("数据预览"),
                html.Div(id='data-table')
            ], style={'marginTop': 30})
        ])
    
    def setup_callbacks(self):
        """设置回调函数"""
        @self.app.callback(
            Output('main-chart', 'figure'),
            [Input('x-axis-dropdown', 'value'),
             Input('y-axis-dropdown', 'value'),
             Input('color-dropdown', 'value'),
             Input('chart-type-radio', 'value')]
        )
        def update_main_chart(x_col, y_col, color_col, chart_type):
            if chart_type == 'scatter':
                fig = px.scatter(self.data, x=x_col, y=y_col, color=color_col,
                               title=f"{chart_type.title()}: {x_col} vs {y_col}")
            elif chart_type == 'line':
                fig = px.line(self.data, x=x_col, y=y_col, color=color_col,
                             title=f"{chart_type.title()}: {x_col} vs {y_col}")
            elif chart_type == 'bar':
                if color_col:
                    fig = px.bar(self.data, x=x_col, y=y_col, color=color_col,
                                title=f"{chart_type.title()}: {x_col} vs {y_col}")
                else:
                    # 对于柱状图，如果没有颜色分组，则聚合数据
                    agg_data = self.data.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(agg_data, x=x_col, y=y_col,
                                title=f"{chart_type.title()}: {x_col} vs {y_col} (平均值)")
            elif chart_type == 'histogram':
                fig = px.histogram(self.data, x=x_col, color=color_col,
                                 title=f"直方图: {x_col}")
            elif chart_type == 'box':
                if color_col:
                    fig = px.box(self.data, x=color_col, y=y_col,
                                title=f"箱线图: {y_col} by {color_col}")
                else:
                    fig = px.box(self.data, y=y_col,
                                title=f"箱线图: {y_col}")
            
            return fig
        
        @self.app.callback(
            Output('statistics-display', 'children'),
            [Input('x-axis-dropdown', 'value'),
             Input('y-axis-dropdown', 'value')]
        )
        def update_statistics(x_col, y_col):
            stats_x = self.data[x_col].describe()
            stats_y = self.data[y_col].describe()
            
            correlation = self.data[x_col].corr(self.data[y_col])
            
            return html.Div([
                html.H4(f"{x_col} 统计:"),
                html.P(f"均值: {stats_x['mean']:.2f}, 标准差: {stats_x['std']:.2f}"),
                html.H4(f"{y_col} 统计:"),
                html.P(f"均值: {stats_y['mean']:.2f}, 标准差: {stats_y['std']:.2f}"),
                html.H4("相关性:"),
                html.P(f"{x_col} 与 {y_col} 的相关系数: {correlation:.3f}")
            ])
    
    def run_server(self, debug=True, port=8050):
        """运行服务器"""
        self.app.run_server(debug=debug, port=port)
```

## 📱 移动端可视化

### 响应式图表
```python
class ResponsiveVisualization:
    def __init__(self):
        self.mobile_config = {
            'displayModeBar': False,
            'responsive': True,
            'staticPlot': False
        }
        
    def create_mobile_friendly_chart(self, data, chart_type='line'):
        """创建移动设备友好的图表"""
        if chart_type == 'line':
            fig = go.Figure()
            
            for column in data.select_dtypes(include=[np.number]).columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column,
                    line=dict(width=3)  # 加粗线条便于手机查看
                ))
        
        elif chart_type == 'bar':
            fig = px.bar(data, x=data.index, y=data.columns[0])
        
        # 移动端优化布局
        fig.update_layout(
            # 字体大小
            font=dict(size=16),
            
            # 图例位置
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            
            # 边距
            margin=dict(l=20, r=20, t=40, b=20),
            
            # 背景
            plot_bgcolor='white',
            paper_bgcolor='white',
            
            # 标题
            title=dict(
                font=dict(size=20),
                x=0.5,
                xanchor='center'
            )
        )
        
        # 坐标轴优化
        fig.update_xaxes(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        
        fig.update_yaxes(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        
        return fig
    
    def create_touch_optimized_interface(self):
        """创建触摸优化界面"""
        # 使用Streamlit创建触摸友好界面
        st.set_page_config(
            page_title="移动数据可视化",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # 大按钮样式
        st.markdown("""
        <style>
        .big-button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            width: 100%;
        }
        
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 创建大按钮
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 销售数据", key="sales", help="查看销售趋势"):
                st.session_state.current_view = "sales"
        
        with col2:
            if st.button("👥 用户分析", key="users", help="用户行为分析"):
                st.session_state.current_view = "users"
        
        with col3:
            if st.button("💰 财务报告", key="finance", help="财务数据概览"):
                st.session_state.current_view = "finance"
        
        return fig
```

## 🔗 导航链接

- [返回主页](../index.html)
- [上一模块：数据场景检索](scene-retrieval.html)
- [下一模块：数据生成和场景编辑](data-generation.html)
