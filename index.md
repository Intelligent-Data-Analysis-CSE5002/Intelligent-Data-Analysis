---
layout: default
title: 智能数据分析
description: 课程内容、课件、工具与视频一站式展示
---

<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">

<style>
.hero-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 4rem 2rem;
  text-align: center;
  color: white;
  border-radius: 0 0 50px 50px;
  box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
  margin-bottom: 3rem;
}

.hero-section img {
  width: 140px;
  height: 140px;
  border-radius: 50%;
  box-shadow: 0 8px 32px rgba(255, 255, 255, 0.3);
  margin-bottom: 2rem;
  object-fit: cover;
}

.hero-title {
  font-size: 3.5rem;
  font-weight: 800;
  margin-bottom: 1rem;
  text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}

.hero-subtitle {
  font-size: 1.4rem;
  font-weight: 300;
  opacity: 0.9;
  margin-bottom: 2rem;
}

.badge {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 25px;
  padding: 0.5rem 1.5rem;
  font-size: 1rem;
  margin: 0 0.5rem;
  display: inline-block;
  backdrop-filter: blur(10px);
}

.tabs-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 2rem;
}

.tabs {
  display: flex;
  background: #f8f9fa;
  border-radius: 15px;
  padding: 8px;
  margin-bottom: 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  overflow-x: auto;
}

.tab-button {
  flex: 1;
  padding: 1rem 2rem;
  border: none;
  background: transparent;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 600;
  font-size: 1rem;
  transition: all 0.3s ease;
  white-space: nowrap;
  min-width: 120px;
}

.tab-button.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.tab-button:hover:not(.active) {
  background: #e9ecef;
}

.tab-content {
  display: none;
  animation: fadeIn 0.5s ease-in-out;
}

.tab-content.active {
  display: block;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.content-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
  margin-top: 2rem;
}

@media (max-width: 1024px) {
  .content-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
  
  .tabs-container {
    padding: 1rem;
    margin: 0 1rem;
  }
  
  .card {
    padding: 1.5rem;
    min-height: auto;
  }
  
  .card-title {
    font-size: 1.3rem;
  }
  
  .card-icon {
    font-size: 2rem;
  }
  
  .btn-primary, .btn-secondary {
    padding: 0.7rem 1.5rem;
    font-size: 0.9rem;
    width: 100%;
    text-align: center;
    margin-bottom: 0.5rem;
  }
}

@media (max-width: 480px) {
  .tabs-container {
    margin: 0 0.5rem;
    padding: 0.5rem;
  }
  
  .tab-nav {
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  
  .tab-button {
    font-size: 0.8rem;
    padding: 0.5rem 1rem;
    min-width: auto;
    flex: 1 1 calc(50% - 0.25rem);
  }
  
  .card {
    padding: 1rem;
    margin-bottom: 1rem;
  }
  
  .card-title {
    font-size: 1.2rem;
  }
  
  .hero h1 {
    font-size: 2rem;
  }
  
  .hero p {
    font-size: 1rem;
  }
}

/* iPhone 特殊优化 */
@media only screen and (max-width: 375px) {
  .hero-section {
    padding: 2rem 1rem;
  }
  
  .tabs-container {
    margin: 0 0.25rem;
  }
  
  .tab-button {
    font-size: 0.75rem;
    padding: 0.4rem 0.8rem;
  }
  
  .content-grid {
    gap: 1rem;
  }
  
  .card {
    padding: 0.8rem;
  }
  
  .card-icon {
    font-size: 1.8rem;
  }
  
  .card-title {
    font-size: 1.1rem;
  }
  
  .card-desc {
    font-size: 0.9rem;
    line-height: 1.5;
  }
}

.card {
  background: white;
  border-radius: 20px;
  padding: 2rem;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  border: 1px solid #f0f0f0;
  min-height: 320px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
}

.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
}

.card-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  display: block;
}

.card-title {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: #333;
}

.card-desc {
  color: #666;
  line-height: 1.6;
  margin-bottom: 1.5rem;
  flex-grow: 1;
}

.card-footer {
  margin-top: auto;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 0.8rem 2rem;
  border-radius: 25px;
  text-decoration: none;
  font-weight: 600;
  display: inline-block;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
  color: white;
  text-decoration: none;
}

.btn-secondary {
  background: white;
  color: #667eea;
  padding: 0.8rem 2rem;
  border-radius: 25px;
  text-decoration: none;
  font-weight: 600;
  display: inline-block;
  transition: all 0.3s ease;
  border: 2px solid #667eea;
  margin: 0.5rem;
}

.btn-secondary:hover {
  background: #667eea;
  color: white;
  text-decoration: none;
}

.video-container {
  background: white;
  border-radius: 20px;
  padding: 1.5rem;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.video-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2rem;
}

.responsive-video {
  position: relative;
  padding-bottom: 56.25%;
  height: 0;
  overflow: hidden;
  border-radius: 15px;
}

.responsive-video iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}
</style>

<!-- Hero Section -->
<div class="hero-section">
  <img src="https://images.unsplash.com/photo-1461749280684-dccba630e2f6?auto=format&fit=crop&w=800&q=80" alt="课程封面">
  <h1 class="hero-title">智能数据分析</h1>
  <p class="hero-subtitle">探索数据的奥秘，掌握智能分析技术</p>
  <div>
    <span class="badge">📊 数据分析</span>
    <span class="badge">🤖 机器学习</span>
    <span class="badge">📈 可视化</span>
    <span class="badge">🔬 实战项目</span>
  </div>
</div>

<!-- Tabs Container -->
<div class="tabs-container">
  <div class="tabs">
    <button class="tab-button active" onclick="openTab(event, 'course-content')">📚 课程内容</button>
    <button class="tab-button" onclick="openTab(event, 'materials')">📂 课件资料</button>
    <button class="tab-button" onclick="openTab(event, 'tools')">🔧 工具资源</button>
    <button class="tab-button" onclick="openTab(event, 'videos')">🎬 视频教程</button>
    <button class="tab-button" onclick="openTab(event, 'projects')">🚀 实战项目</button>
  </div>

  <!-- Tab Content: 课程内容 -->
  <div id="course-content" class="tab-content active">
    <div class="content-grid">
      <div class="card">
        <span class="card-icon">🗃️</span>
        <h3 class="card-title">SUScape数据集介绍</h3>
        <p class="card-desc">详细介绍SUScape数据集的结构、特点和应用场景</p>
        <div class="card-footer">
          <a href="modules/suscape-dataset.html" class="btn-primary">了解详情</a>
        </div>
      </div>
      <div class="card">
        <span class="card-icon">🔧</span>
        <h3 class="card-title">标注工具介绍</h3>
        <p class="card-desc">标注工具的功能特性、使用方法和实践</p>
        <div class="card-footer">
          <a href="modules/points-tool.html" class="btn-primary">学习使用</a>
        </div>
      </div>
      <div class="card">
        <span class="card-icon">📊</span>
        <h3 class="card-title">数据分析</h3>
        <p class="card-desc">深入数据分析方法，挖掘数据价值和洞察</p>
        <div class="card-footer">
          <a href="modules/data-analysis.html" class="btn-primary">开始分析</a>
        </div>
      </div>
      <div class="card">
        <span class="card-icon">🔍</span>
        <h3 class="card-title">数据提取与分析</h3>
        <p class="card-desc">学习数据提取技术和高级分析方法</p>
        <div class="card-footer">
          <a href="modules/data-extraction.html" class="btn-primary">提取数据</a>
        </div>
      </div>
      <div class="card">
        <span class="card-icon">🏷️</span>
        <h3 class="card-title">数据标注</h3>
        <p class="card-desc">数据标注工具使用和标注质量控制方法</p>
        <div class="card-footer">
          <a href="modules/data-annotation.html" class="btn-primary">标注数据</a>
        </div>
      </div>
      <div class="card">
        <span class="card-icon">🔎</span>
        <h3 class="card-title">数据场景检索</h3>
        <p class="card-desc">高效检索和查找特定场景数据的技术方法</p>
        <div class="card-footer">
          <a href="modules/scene-retrieval.html" class="btn-primary">检索场景</a>
        </div>
      </div>
      <div class="card">
        <span class="card-icon">📈</span>
        <h3 class="card-title">数据与场景可视化</h3>
        <p class="card-desc">创建直观的数据可视化和场景展示</p>
        <div class="card-footer">
          <a href="modules/data-visualization.html" class="btn-primary">可视化</a>
        </div>
      </div>
      <div class="card">
        <span class="card-icon">⚡</span>
        <h3 class="card-title">数据生成和场景编辑</h3>
        <p class="card-desc">生成新数据和编辑现有场景的高级技术</p>
        <div class="card-footer">
          <a href="modules/data-generation.html" class="btn-primary">生成编辑</a>
        </div>
      </div>
      <div class="card">
        <span class="card-icon">🚀</span>
        <h3 class="card-title">数据应用</h3>
        <p class="card-desc">将数据分析结果应用到实际项目和业务场景</p>
        <div class="card-footer">
          <a href="modules/data-application.html" class="btn-primary">应用实践</a>
        </div>
      </div>
    </div>
  <!-- </div>
      <div class="card">
        <span class="card-icon">📊</span>
        <h3 class="card-title">第八章：数据可视化</h3>
        <p class="card-desc">使用Python和R创建精美的图表和交互式可视化</p>
        <a href="chapters/chapter4.html" class="btn-primary">开始学习</a>
      </div>
    </div>
  </div> -->

  <!-- Tab Content: 课件资料 -->
  <div id="materials" class="tab-content">
    <div class="content-grid">
      <div class="card">
        <span class="card-icon">📖</span>
        <h3 class="card-title">PPT课件</h3>
        <p class="card-desc">完整的课程幻灯片，包含理论知识和案例分析</p>
        <a href="assets/slides/chapter1.pdf" class="btn-secondary">第一章 PPT</a>
        <a href="assets/slides/chapter2.pdf" class="btn-secondary">第二章 PPT</a>
        <a href="assets/slides/chapter3.pdf" class="btn-secondary">第三章 PPT</a>
        <a href="assets/slides/chapter4.pdf" class="btn-secondary">第四章 PPT</a>
      </div>
      <div class="card">
        <span class="card-icon">📝</span>
        <h3 class="card-title">实验手册</h3>
        <p class="card-desc">详细的实验指导和代码示例</p>
        <a href="assets/lab-manual.pdf" class="btn-primary">下载手册</a>
      </div>
      <div class="card">
        <span class="card-icon">📚</span>
        <h3 class="card-title">参考资料</h3>
        <p class="card-desc">推荐教材和扩展阅读资料</p>
        <a href="resources/references.html" class="btn-primary">查看资料</a>
      </div>
    </div>
  </div>

  <!-- Tab Content: 工具资源 -->
  <div id="tools" class="tab-content">
    <div class="content-grid">
      <div class="card">
        <span class="card-icon">🐍</span>
        <h3 class="card-title">Python环境</h3>
        <p class="card-desc">完整的Python数据分析环境配置指南</p>
        <a href="https://www.anaconda.com/" target="_blank" class="btn-secondary">Anaconda</a>
        <a href="https://jupyter.org/" target="_blank" class="btn-secondary">Jupyter</a>
      </div>
      <div class="card">
        <span class="card-icon">🌐</span>
        <h3 class="card-title">在线平台</h3>
        <p class="card-desc">无需安装，直接在浏览器中编写和运行代码</p>
        <a href="https://colab.research.google.com/" target="_blank" class="btn-secondary">Google Colab</a>
        <a href="https://www.kaggle.com/" target="_blank" class="btn-secondary">Kaggle</a>
      </div>
      <div class="card">
        <span class="card-icon">📊</span>
        <h3 class="card-title">数据源</h3>
        <p class="card-desc">高质量的公开数据集</p>
        <a href="https://www.kaggle.com/datasets" target="_blank" class="btn-secondary">Kaggle Datasets</a>
        <a href="https://archive.ics.uci.edu/ml/index.php" target="_blank" class="btn-secondary">UCI ML Repository</a>
      </div>
    </div>
  </div>

  <!-- Tab Content: 视频教程 -->
  <div id="videos" class="tab-content">
    <div class="video-grid">
      <div class="video-container">
        <h3>🎯 课程介绍</h3>
        <div class="responsive-video">
          <iframe src="https://www.bilibili.com/video/BV1xxxxxx" frameborder="0" allowfullscreen></iframe>
        </div>
      </div>
      <div class="video-container">
        <h3>🔥 Python数据分析实战</h3>
        <div class="responsive-video">
          <iframe src="https://www.youtube.com/embed/xxxxxxx" frameborder="0" allowfullscreen></iframe>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab Content: 实战项目 -->
  <div id="projects" class="tab-content">
    <div class="content-grid">
      <div class="card">
        <span class="card-icon">🏠</span>
        <h3 class="card-title">房价预测项目</h3>
        <p class="card-desc">使用机器学习算法预测房价，涵盖数据清洗、特征工程、模型训练等完整流程</p>
        <a href="projects/house-price-prediction.html" class="btn-primary">查看项目</a>
      </div>
      <div class="card">
        <span class="card-icon">🛍️</span>
        <h3 class="card-title">电商用户行为分析</h3>
        <p class="card-desc">分析用户购买行为，构建用户画像和推荐系统</p>
        <a href="projects/ecommerce-analysis.html" class="btn-primary">查看项目</a>
      </div>
      <div class="card">
        <span class="card-icon">📈</span>
        <h3 class="card-title">股票市场分析</h3>
        <p class="card-desc">时间序列分析与预测，技术指标计算和可视化</p>
        <a href="projects/stock-analysis.html" class="btn-primary">查看项目</a>
      </div>
    </div>
  </div>
</div>

<script>
function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  
  // Hide all tab content
  tabcontent = document.getElementsByClassName("tab-content");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].classList.remove("active");
  }
  
  // Remove active class from all buttons
  tablinks = document.getElementsByClassName("tab-button");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].classList.remove("active");
  }
  
  // Show selected tab and mark button as active
  document.getElementById(tabName).classList.add("active");
  evt.currentTarget.classList.add("active");
}
</script>
