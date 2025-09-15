---
layout: page
title: 数据标注
---

# 数据标注

> 🏷️ **模块目标**：掌握数据标注技术，构建高质量训练数据集

## 📖 数据标注概述

数据标注是机器学习和人工智能项目的基础工作，涉及为原始数据添加标签、注释或元数据，以便算法能够学习和识别模式。高质量的标注数据是模型性能的关键因素。

## 🎯 标注类型与应用

### 图像标注
- **分类标注**：为图像分配类别标签
- **目标检测**：标注物体边界框和类别
- **语义分割**：像素级别的类别标注
- **实例分割**：区分同类别的不同实例
- **关键点检测**：标注人体姿态、面部特征点

### 文本标注
- **情感分析**：标注文本情感倾向
- **命名实体识别**：标注人名、地名、机构名
- **关系抽取**：标注实体间的关系
- **文本分类**：为文档分配主题标签
- **机器翻译**：提供翻译对照

### 音频标注
- **语音识别**：标注语音内容
- **音乐分类**：标注音乐风格、情绪
- **环境声音**：标注声音事件和场景
- **说话人识别**：标注说话人身份

### 视频标注
- **动作识别**：标注视频中的行为动作
- **场景分割**：标注视频场景边界
- **目标跟踪**：标注目标在视频中的轨迹
- **事件检测**：标注特定事件的时间点

## 🛠️ 标注工具与平台

### 开源标注工具

#### 图像标注工具
```python
# LabelImg - 目标检测标注
# 安装：pip install labelImg
# 使用：labelImg

# 基于Python的自定义标注工具
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os

class ImageAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("图像标注工具")
        self.root.geometry("1200x800")
        
        self.image_path = ""
        self.annotations = []
        self.current_bbox = None
        self.start_x = 0
        self.start_y = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        # 菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开图像", command=self.load_image)
        file_menu.add_command(label="保存标注", command=self.save_annotations)
        file_menu.add_command(label="加载标注", command=self.load_annotations)
        
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：图像显示
        self.image_frame = tk.Frame(main_frame, bg="white")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_bbox)
        self.canvas.bind("<B1-Motion>", self.draw_bbox)
        self.canvas.bind("<ButtonRelease-1>", self.end_bbox)
        
        # 右侧：控制面板
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)
        
        # 类别选择
        tk.Label(control_frame, text="选择类别:", font=("Arial", 12)).pack(pady=5)
        self.class_var = tk.StringVar(value="person")
        self.class_entry = tk.Entry(control_frame, textvariable=self.class_var)
        self.class_entry.pack(pady=5, fill=tk.X)
        
        # 预定义类别
        classes = ["person", "car", "bike", "dog", "cat"]
        for cls in classes:
            btn = tk.Button(control_frame, text=cls, 
                          command=lambda c=cls: self.class_var.set(c))
            btn.pack(pady=2, fill=tk.X)
        
        # 标注列表
        tk.Label(control_frame, text="标注列表:", font=("Arial", 12)).pack(pady=(20, 5))
        
        self.annotation_listbox = tk.Listbox(control_frame, height=10)
        self.annotation_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 删除按钮
        tk.Button(control_frame, text="删除选中标注", 
                 command=self.delete_annotation).pack(pady=5, fill=tk.X)
        
        # 清空按钮
        tk.Button(control_frame, text="清空所有标注", 
                 command=self.clear_annotations).pack(pady=5, fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image()
    
    def display_image(self):
        if self.image_path:
            image = Image.open(self.image_path)
            # 调整图像大小以适应画布
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def start_bbox(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        if self.current_bbox:
            self.canvas.delete(self.current_bbox)
        
        self.current_bbox = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )
    
    def draw_bbox(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        
        self.canvas.coords(self.current_bbox, self.start_x, self.start_y, cur_x, cur_y)
    
    def end_bbox(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # 确保边界框有效
        if abs(end_x - self.start_x) > 5 and abs(end_y - self.start_y) > 5:
            annotation = {
                "class": self.class_var.get(),
                "bbox": [
                    min(self.start_x, end_x),
                    min(self.start_y, end_y),
                    max(self.start_x, end_x),
                    max(self.start_y, end_y)
                ]
            }
            self.annotations.append(annotation)
            self.update_annotation_list()
        else:
            self.canvas.delete(self.current_bbox)
        
        self.current_bbox = None
    
    def update_annotation_list(self):
        self.annotation_listbox.delete(0, tk.END)
        for i, ann in enumerate(self.annotations):
            self.annotation_listbox.insert(tk.END, 
                f"{i+1}. {ann['class']}: {ann['bbox']}")
    
    def delete_annotation(self):
        selection = self.annotation_listbox.curselection()
        if selection:
            index = selection[0]
            del self.annotations[index]
            self.update_annotation_list()
            self.redraw_annotations()
    
    def clear_annotations(self):
        self.annotations = []
        self.update_annotation_list()
        self.canvas.delete("bbox")
    
    def redraw_annotations(self):
        self.canvas.delete("bbox")
        for ann in self.annotations:
            bbox = ann["bbox"]
            self.canvas.create_rectangle(
                bbox[0], bbox[1], bbox[2], bbox[3],
                outline="red", width=2, tags="bbox"
            )
    
    def save_annotations(self):
        if not self.image_path:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        annotation_file = self.image_path.rsplit('.', 1)[0] + '.json'
        annotation_data = {
            "image_path": self.image_path,
            "annotations": self.annotations
        }
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        messagebox.showinfo("成功", f"标注已保存至 {annotation_file}")
    
    def load_annotations(self):
        if not self.image_path:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        annotation_file = self.image_path.rsplit('.', 1)[0] + '.json'
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.annotations = data.get("annotations", [])
                self.update_annotation_list()
                self.redraw_annotations()
        else:
            messagebox.showinfo("信息", "未找到对应的标注文件")

# 启动标注工具
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()
```

#### 文本标注工具
```python
import streamlit as st
import pandas as pd
import json
from datetime import datetime

class TextAnnotationTool:
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        if 'texts' not in st.session_state:
            st.session_state.texts = []
        if 'annotations' not in st.session_state:
            st.session_state.annotations = {}
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
    
    def load_texts(self, file):
        """加载待标注文本"""
        if file.type == "text/csv":
            df = pd.read_csv(file)
            if 'text' in df.columns:
                st.session_state.texts = df['text'].tolist()
            else:
                st.error("CSV文件必须包含'text'列")
        elif file.type == "application/json":
            data = json.load(file)
            if isinstance(data, list):
                st.session_state.texts = data
            else:
                st.error("JSON文件必须是文本列表")
        else:
            # 纯文本文件
            content = str(file.read(), "utf-8")
            st.session_state.texts = content.split('\n')
        
        st.session_state.current_index = 0
        st.success(f"成功加载 {len(st.session_state.texts)} 条文本")
    
    def annotate_sentiment(self):
        """情感分析标注"""
        if not st.session_state.texts:
            st.warning("请先上传文本文件")
            return
        
        st.subheader("情感分析标注")
        
        # 显示当前文本
        current_text = st.session_state.texts[st.session_state.current_index]
        st.text_area("当前文本", current_text, height=100, disabled=True)
        
        # 标注选项
        sentiment = st.radio(
            "选择情感标签",
            ["积极", "消极", "中性"],
            key=f"sentiment_{st.session_state.current_index}"
        )
        
        # 置信度
        confidence = st.slider("标注置信度", 0.0, 1.0, 0.8, 0.1)
        
        # 保存标注
        if st.button("保存标注"):
            text_id = st.session_state.current_index
            st.session_state.annotations[text_id] = {
                "text": current_text,
                "sentiment": sentiment,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            st.success("标注已保存")
            
            # 自动跳转到下一条
            if st.session_state.current_index < len(st.session_state.texts) - 1:
                st.session_state.current_index += 1
                st.experimental_rerun()
    
    def annotate_ner(self):
        """命名实体识别标注"""
        if not st.session_state.texts:
            st.warning("请先上传文本文件")
            return
        
        st.subheader("命名实体识别标注")
        
        current_text = st.session_state.texts[st.session_state.current_index]
        st.text_area("当前文本", current_text, height=100, disabled=True)
        
        # 实体类型
        entity_types = ["PERSON", "ORG", "GPE", "MONEY", "DATE", "TIME"]
        
        # 实体标注
        entities = []
        
        st.write("标注实体：")
        for i in range(5):  # 最多标注5个实体
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                entity_text = st.text_input(f"实体文本 {i+1}", key=f"entity_text_{i}")
            
            with col2:
                entity_type = st.selectbox(f"实体类型 {i+1}", 
                                         [""] + entity_types, key=f"entity_type_{i}")
            
            with col3:
                start_pos = st.number_input(f"开始位置 {i+1}", 
                                          min_value=0, key=f"start_{i}")
            
            with col4:
                end_pos = st.number_input(f"结束位置 {i+1}", 
                                        min_value=0, key=f"end_{i}")
            
            if entity_text and entity_type:
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "start": int(start_pos),
                    "end": int(end_pos)
                })
        
        # 保存标注
        if st.button("保存NER标注"):
            text_id = st.session_state.current_index
            st.session_state.annotations[text_id] = {
                "text": current_text,
                "entities": entities,
                "timestamp": datetime.now().isoformat()
            }
            st.success("NER标注已保存")
    
    def navigation(self):
        """导航控制"""
        if st.session_state.texts:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("上一条"):
                    if st.session_state.current_index > 0:
                        st.session_state.current_index -= 1
                        st.experimental_rerun()
            
            with col2:
                st.write(f"当前进度: {st.session_state.current_index + 1} / {len(st.session_state.texts)}")
            
            with col3:
                if st.button("下一条"):
                    if st.session_state.current_index < len(st.session_state.texts) - 1:
                        st.session_state.current_index += 1
                        st.experimental_rerun()
    
    def export_annotations(self):
        """导出标注结果"""
        if st.session_state.annotations:
            annotations_json = json.dumps(st.session_state.annotations, 
                                        indent=2, ensure_ascii=False)
            
            st.download_button(
                label="下载标注结果",
                data=annotations_json,
                file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def run(self):
        """运行主程序"""
        st.title("文本标注工具")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "上传文本文件",
            type=['txt', 'csv', 'json'],
            help="支持txt、csv、json格式"
        )
        
        if uploaded_file:
            self.load_texts(uploaded_file)
        
        # 标注类型选择
        annotation_type = st.sidebar.selectbox(
            "选择标注类型",
            ["情感分析", "命名实体识别"]
        )
        
        # 导航
        self.navigation()
        
        # 标注界面
        if annotation_type == "情感分析":
            self.annotate_sentiment()
        elif annotation_type == "命名实体识别":
            self.annotate_ner()
        
        # 导出功能
        st.sidebar.subheader("导出标注")
        self.export_annotations()
        
        # 显示标注统计
        if st.session_state.annotations:
            st.sidebar.subheader("标注统计")
            st.sidebar.write(f"已标注: {len(st.session_state.annotations)} 条")
            st.sidebar.write(f"剩余: {len(st.session_state.texts) - len(st.session_state.annotations)} 条")

# 启动应用
if __name__ == "__main__":
    app = TextAnnotationTool()
    app.run()
```

### 商业标注平台

#### 众包标注平台
- **Amazon Mechanical Turk**：亚马逊众包平台
- **Figure Eight (now Appen)**：专业数据标注服务
- **Labelbox**：端到端标注平台
- **Scale AI**：AI训练数据平台

#### 企业级标注解决方案
- **Supervisely**：计算机视觉标注平台
- **Hasty.ai**：AI辅助标注工具
- **V7 Labs**：医学图像标注专用
- **Dataloop**：数据管理和标注平台

## 📊 标注质量控制

### 质量评估指标
```python
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
import pandas as pd

class AnnotationQualityControl:
    def __init__(self):
        pass
    
    def inter_annotator_agreement(self, annotations_df):
        """计算标注者间一致性"""
        annotators = annotations_df['annotator'].unique()
        agreements = {}
        
        for i, ann1 in enumerate(annotators):
            for ann2 in annotators[i+1:]:
                # 获取两个标注者的共同标注
                ann1_data = annotations_df[annotations_df['annotator'] == ann1]
                ann2_data = annotations_df[annotations_df['annotator'] == ann2]
                
                # 合并共同项目
                common = pd.merge(ann1_data, ann2_data, on='item_id', 
                                suffixes=('_1', '_2'))
                
                if len(common) > 0:
                    # 计算Cohen's Kappa
                    kappa = cohen_kappa_score(common['label_1'], common['label_2'])
                    accuracy = accuracy_score(common['label_1'], common['label_2'])
                    
                    agreements[f"{ann1}_vs_{ann2}"] = {
                        'kappa': kappa,
                        'accuracy': accuracy,
                        'common_items': len(common)
                    }
        
        return agreements
    
    def calculate_annotation_time_stats(self, annotations_df):
        """计算标注时间统计"""
        if 'start_time' in annotations_df.columns and 'end_time' in annotations_df.columns:
            annotations_df['duration'] = (
                pd.to_datetime(annotations_df['end_time']) - 
                pd.to_datetime(annotations_df['start_time'])
            ).dt.total_seconds()
            
            stats = {
                'mean_duration': annotations_df['duration'].mean(),
                'median_duration': annotations_df['duration'].median(),
                'std_duration': annotations_df['duration'].std(),
                'min_duration': annotations_df['duration'].min(),
                'max_duration': annotations_df['duration'].max()
            }
            
            # 检测异常快速或缓慢的标注
            Q1 = annotations_df['duration'].quantile(0.25)
            Q3 = annotations_df['duration'].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = annotations_df[
                (annotations_df['duration'] < Q1 - 1.5 * IQR) |
                (annotations_df['duration'] > Q3 + 1.5 * IQR)
            ]
            
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = len(outliers) / len(annotations_df) * 100
            
            return stats
        else:
            return {"error": "缺少时间戳信息"}
    
    def detect_annotation_patterns(self, annotations_df):
        """检测标注模式异常"""
        patterns = {}
        
        # 检测标注分布
        label_distribution = annotations_df['label'].value_counts(normalize=True)
        patterns['label_distribution'] = label_distribution.to_dict()
        
        # 检测标注者偏好
        annotator_patterns = annotations_df.groupby('annotator')['label'].value_counts(normalize=True)
        patterns['annotator_preferences'] = annotator_patterns.to_dict()
        
        # 检测时间偏好
        if 'timestamp' in annotations_df.columns:
            annotations_df['hour'] = pd.to_datetime(annotations_df['timestamp']).dt.hour
            time_patterns = annotations_df.groupby('annotator')['hour'].apply(
                lambda x: x.value_counts(normalize=True).head(3)
            )
            patterns['time_preferences'] = time_patterns.to_dict()
        
        return patterns

# 使用示例
annotations_data = pd.DataFrame({
    'item_id': range(100),
    'annotator': np.random.choice(['A', 'B', 'C'], 100),
    'label': np.random.choice(['positive', 'negative', 'neutral'], 100),
    'confidence': np.random.uniform(0.5, 1.0, 100),
    'start_time': pd.date_range('2024-01-01', periods=100, freq='H'),
    'end_time': pd.date_range('2024-01-01 00:30:00', periods=100, freq='H')
})

qc = AnnotationQualityControl()
agreements = qc.inter_annotator_agreement(annotations_data)
time_stats = qc.calculate_annotation_time_stats(annotations_data)
patterns = qc.detect_annotation_patterns(annotations_data)

print("标注者间一致性:", agreements)
print("标注时间统计:", time_stats)
print("标注模式:", patterns)
```

### 主动学习与自动化标注
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np

class ActiveLearningAnnotation:
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier(n_estimators=100)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.labeled_data = []
        self.unlabeled_data = []
    
    def add_labeled_data(self, texts, labels):
        """添加已标注数据"""
        for text, label in zip(texts, labels):
            self.labeled_data.append({'text': text, 'label': label})
    
    def add_unlabeled_data(self, texts):
        """添加未标注数据"""
        for text in texts:
            self.unlabeled_data.append({'text': text})
    
    def train_model(self):
        """训练当前模型"""
        if len(self.labeled_data) < 2:
            return False
        
        texts = [item['text'] for item in self.labeled_data]
        labels = [item['label'] for item in self.labeled_data]
        
        # 特征提取
        X = self.vectorizer.fit_transform(texts)
        
        # 训练模型
        self.model.fit(X, labels)
        return True
    
    def uncertainty_sampling(self, n_samples=10):
        """不确定性采样"""
        if not self.unlabeled_data:
            return []
        
        texts = [item['text'] for item in self.unlabeled_data]
        X = self.vectorizer.transform(texts)
        
        # 获取预测概率
        probabilities = self.model.predict_proba(X)
        
        # 计算不确定性（熵）
        uncertainties = []
        for prob in probabilities:
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            uncertainties.append(entropy)
        
        # 选择最不确定的样本
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        
        selected_samples = []
        for idx in uncertain_indices:
            selected_samples.append({
                'index': idx,
                'text': self.unlabeled_data[idx]['text'],
                'uncertainty': uncertainties[idx],
                'probabilities': probabilities[idx].tolist()
            })
        
        return selected_samples
    
    def diversity_sampling(self, n_samples=10):
        """多样性采样"""
        if not self.unlabeled_data:
            return []
        
        texts = [item['text'] for item in self.unlabeled_data]
        X = self.vectorizer.transform(texts).toarray()
        
        # 选择多样性最大的样本
        selected_indices = []
        remaining_indices = list(range(len(texts)))
        
        # 随机选择第一个样本
        first_idx = np.random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # 迭代选择与已选样本差异最大的样本
        for _ in range(min(n_samples - 1, len(remaining_indices))):
            max_min_distance = -1
            best_idx = None
            
            for candidate_idx in remaining_indices:
                min_distance = float('inf')
                
                for selected_idx in selected_indices:
                    distance = np.linalg.norm(X[candidate_idx] - X[selected_idx])
                    min_distance = min(min_distance, distance)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        selected_samples = []
        for idx in selected_indices:
            selected_samples.append({
                'index': idx,
                'text': self.unlabeled_data[idx]['text']
            })
        
        return selected_samples
    
    def auto_annotate(self, confidence_threshold=0.9):
        """自动标注高置信度样本"""
        if not self.unlabeled_data:
            return []
        
        texts = [item['text'] for item in self.unlabeled_data]
        X = self.vectorizer.transform(texts)
        
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        auto_annotated = []
        indices_to_remove = []
        
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            max_confidence = np.max(prob)
            
            if max_confidence >= confidence_threshold:
                auto_annotated.append({
                    'text': texts[i],
                    'label': pred,
                    'confidence': max_confidence
                })
                indices_to_remove.append(i)
        
        # 移除自动标注的样本
        for idx in sorted(indices_to_remove, reverse=True):
            self.unlabeled_data.pop(idx)
        
        return auto_annotated

# 使用示例
# 初始化主动学习系统
al_system = ActiveLearningAnnotation()

# 添加初始标注数据
initial_texts = ["这个产品很好", "质量太差了", "还可以吧"]
initial_labels = ["positive", "negative", "neutral"]
al_system.add_labeled_data(initial_texts, initial_labels)

# 添加待标注数据
unlabeled_texts = ["非常满意", "不推荐购买", "性价比一般", "超出预期"]
al_system.add_unlabeled_data(unlabeled_texts)

# 训练初始模型
al_system.train_model()

# 获取需要人工标注的样本
uncertain_samples = al_system.uncertainty_sampling(n_samples=2)
diverse_samples = al_system.diversity_sampling(n_samples=2)

print("不确定性采样结果:", uncertain_samples)
print("多样性采样结果:", diverse_samples)

# 自动标注高置信度样本
auto_labeled = al_system.auto_annotate(confidence_threshold=0.8)
print("自动标注结果:", auto_labeled)
```

## 🎯 特定领域标注

### 医学图像标注
```python
import cv2
import numpy as np
from sklearn.cluster import KMeans

class MedicalImageAnnotation:
    def __init__(self):
        self.annotations = {}
    
    def segment_lesion(self, image_path, roi_coordinates=None):
        """病灶分割标注"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if roi_coordinates:
            # 提取感兴趣区域
            x1, y1, x2, y2 = roi_coordinates
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        # 使用K-means进行初步分割
        data = roi.reshape((-1, 1))
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(data)
        
        # 重塑为图像形状
        segmented = labels.reshape(roi.shape)
        
        return segmented, roi
    
    def annotate_anatomical_structure(self, image_path, structure_type):
        """解剖结构标注"""
        structures = {
            'heart': {'color': (255, 0, 0), 'thickness': 2},
            'lung': {'color': (0, 255, 0), 'thickness': 2},
            'liver': {'color': (0, 0, 255), 'thickness': 2},
            'kidney': {'color': (255, 255, 0), 'thickness': 2}
        }
        
        annotation = {
            'image_path': image_path,
            'structure_type': structure_type,
            'properties': structures.get(structure_type, {'color': (255, 255, 255), 'thickness': 1}),
            'timestamp': datetime.now().isoformat()
        }
        
        return annotation
    
    def measure_distance(self, point1, point2, pixel_spacing):
        """测量距离（毫米）"""
        pixel_distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        real_distance = pixel_distance * pixel_spacing
        return real_distance
    
    def calculate_area(self, contour, pixel_spacing):
        """计算面积（平方毫米）"""
        pixel_area = cv2.contourArea(contour)
        real_area = pixel_area * (pixel_spacing ** 2)
        return real_area

# 法律文档标注
class LegalDocumentAnnotation:
    def __init__(self):
        self.legal_entities = [
            'PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'MONEY',
            'LAW', 'CASE', 'COURT', 'JUDGE', 'LAWYER'
        ]
    
    def extract_legal_entities(self, text):
        """提取法律实体"""
        # 这里可以集成专门的法律NLP模型
        entities = []
        
        # 简单的规则基础实体识别示例
        import re
        
        # 日期模式
        date_pattern = r'\d{4}年\d{1,2}月\d{1,2}日'
        dates = re.finditer(date_pattern, text)
        for match in dates:
            entities.append({
                'text': match.group(),
                'type': 'DATE',
                'start': match.start(),
                'end': match.end()
            })
        
        # 金额模式
        money_pattern = r'\d+(?:,\d{3})*(?:\.\d{2})?元'
        money_matches = re.finditer(money_pattern, text)
        for match in money_matches:
            entities.append({
                'text': match.group(),
                'type': 'MONEY',
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def annotate_clause_type(self, clause_text):
        """标注条款类型"""
        clause_types = {
            '权利义务': ['权利', '义务', '责任', '权限'],
            '违约责任': ['违约', '赔偿', '损失', '责任'],
            '争议解决': ['争议', '仲裁', '诉讼', '管辖'],
            '生效条件': ['生效', '终止', '期限', '条件']
        }
        
        scores = {}
        for clause_type, keywords in clause_types.items():
            score = sum(1 for keyword in keywords if keyword in clause_text)
            scores[clause_type] = score
        
        # 返回得分最高的条款类型
        best_type = max(scores, key=scores.get)
        return best_type if scores[best_type] > 0 else '其他'
```

## 📈 标注效率优化

### 批量标注策略
```python
class BatchAnnotationStrategy:
    def __init__(self):
        self.batch_size = 50
        self.strategies = ['random', 'similar', 'diverse', 'uncertain']
    
    def create_batches(self, data, strategy='similar'):
        """创建标注批次"""
        if strategy == 'random':
            return self._random_batches(data)
        elif strategy == 'similar':
            return self._similar_batches(data)
        elif strategy == 'diverse':
            return self._diverse_batches(data)
        elif strategy == 'uncertain':
            return self._uncertain_batches(data)
    
    def _similar_batches(self, data):
        """创建相似样本批次"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        
        # 假设data是文本列表
        vectorizer = TfidfVectorizer(max_features=100)
        features = vectorizer.fit_transform(data)
        
        # 聚类
        n_clusters = len(data) // self.batch_size
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # 按聚类分组
        batches = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in batches:
                batches[cluster_id] = []
            batches[cluster_id].append(data[i])
        
        return list(batches.values())
    
    def estimate_annotation_time(self, batch, complexity_factor=1.0):
        """估计标注时间"""
        base_time_per_sample = 30  # 秒
        estimated_time = len(batch) * base_time_per_sample * complexity_factor
        return estimated_time
    
    def optimize_annotator_assignment(self, batches, annotators):
        """优化标注者分配"""
        assignments = {}
        
        for i, batch in enumerate(batches):
            # 简单的轮询分配
            annotator = annotators[i % len(annotators)]
            if annotator not in assignments:
                assignments[annotator] = []
            assignments[annotator].append({
                'batch_id': i,
                'batch': batch,
                'estimated_time': self.estimate_annotation_time(batch)
            })
        
        return assignments
```

## 📚 最佳实践

### 标注指南制定
1. **明确标注标准**：详细定义每个标签的含义和适用范围
2. **提供示例**：为每种情况提供正面和负面示例
3. **处理边界情况**：明确模糊情况的处理方式
4. **建立审核流程**：设置多级审核和质量控制机制

### 标注者培训
1. **理论培训**：介绍项目背景和标注目标
2. **实践训练**：通过示例数据进行练习
3. **一致性测试**：评估标注者间的一致性
4. **持续反馈**：定期检查和改进标注质量

### 技术工具集成
1. **版本控制**：使用Git等工具管理标注数据版本
2. **自动化检查**：开发脚本自动检测标注错误
3. **数据备份**：定期备份标注数据防止丢失
4. **进度跟踪**：实时监控标注进度和质量指标

## 🔗 导航链接

- [返回主页](../index.html)
- [上一模块：数据提取与分析](data-extraction.html)
- [下一模块：数据场景检索](scene-retrieval.html)
