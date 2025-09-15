---
layout: page
title: 数据生成和场景编辑
---

# 数据生成和场景编辑

> 🎬 **模块目标**：掌握数据生成技术和场景编辑方法，创建合成数据集和虚拟场景

## 🌟 数据生成概述

数据生成是创建人工合成数据的过程，用于增强训练数据集、保护隐私、测试算法性能等目的。结合先进的生成模型和场景编辑技术，我们可以创建高质量、多样化的数据集来支持各种机器学习任务。

## 🤖 生成模型技术

### 生成对抗网络 (GAN)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=784):  # 28x28 = 784 for MNIST
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class GANTrainer:
    def __init__(self, noise_dim=100, lr=0.0002):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim
        
        # 初始化模型
        self.generator = Generator(noise_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # 优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 训练历史
        self.g_losses = []
        self.d_losses = []
    
    def train(self, dataloader, epochs=100):
        """训练GAN"""
        for epoch in range(epochs):
            for i, (real_data, _) in enumerate(dataloader):
                batch_size = real_data.size(0)
                real_data = real_data.view(batch_size, -1).to(self.device)
                
                # 训练判别器
                self.d_optimizer.zero_grad()
                
                # 真实数据
                real_labels = torch.ones(batch_size, 1).to(self.device)
                real_output = self.discriminator(real_data)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # 生成数据
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # 训练生成器
                self.g_optimizer.zero_grad()
                
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                # 记录损失
                if i % 100 == 0:
                    self.g_losses.append(g_loss.item())
                    self.d_losses.append(d_loss.item())
                    print(f'Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], '
                          f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    def generate_samples(self, num_samples=64):
        """生成样本"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim).to(self.device)
            generated_data = self.generator(noise)
            generated_data = generated_data.cpu().numpy()
        return generated_data.reshape(num_samples, 28, 28)  # 假设是28x28图像
    
    def plot_training_progress(self):
        """绘制训练进度"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        samples = self.generate_samples(16)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')
        plt.suptitle('Generated Samples')
        plt.tight_layout()
        plt.show()
```

### 变分自编码器 (VAE)
```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # 编码器
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 方差
        
        # 解码器
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAETrainer:
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE(input_dim, hidden_dim, latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE损失函数"""
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def train(self, dataloader, epochs=10):
        """训练VAE"""
        self.model.train()
        train_loss = 0
        
        for epoch in range(epochs):
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                          f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    def generate_samples(self, num_samples=64):
        """生成样本"""
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, 20).to(self.device)
            samples = self.model.decode(z)
            return samples.cpu().numpy().reshape(num_samples, 28, 28)
    
    def interpolate_latent_space(self, point1, point2, num_steps=10):
        """在潜在空间中插值"""
        self.model.eval()
        with torch.no_grad():
            # 创建插值路径
            alpha = torch.linspace(0, 1, num_steps).unsqueeze(1).to(self.device)
            interpolated = alpha * point1 + (1 - alpha) * point2
            
            # 解码插值点
            samples = self.model.decode(interpolated)
            return samples.cpu().numpy().reshape(num_steps, 28, 28)
```

### Diffusion Models
```python
import math

class DiffusionModel(nn.Module):
    def __init__(self, img_size=32, in_channels=3, model_channels=128, num_res_blocks=2):
        super(DiffusionModel, self).__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        # U-Net架构
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # 下采样
        for level in range(3):
            for _ in range(num_res_blocks):
                self.input_blocks.append(
                    ResBlock(model_channels, model_channels, model_channels * 4)
                )
            if level < 2:
                self.input_blocks.append(nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1))
        
        # 中间层
        self.middle_block = ResBlock(model_channels, model_channels, model_channels * 4)
        
        # 上采样
        self.output_blocks = nn.ModuleList()
        for level in range(3):
            for _ in range(num_res_blocks + 1):
                self.output_blocks.append(
                    ResBlock(model_channels * 2, model_channels, model_channels * 4)
                )
            if level < 2:
                self.output_blocks.append(nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1))
        
        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps):
        # 时间嵌入
        t_emb = self.time_embedding(timesteps)
        
        # U-Net前向传播
        h = x
        hs = []
        
        # 下采样
        for module in self.input_blocks:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)
        
        # 中间层
        h = self.middle_block(h, t_emb)
        
        # 上采样
        for module in self.output_blocks:
            if isinstance(module, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)
        
        return self.out(h)
    
    def time_embedding(self, timesteps):
        """时间步嵌入"""
        half_dim = 64
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).to(timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_embed(emb)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ResBlock, self).__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)

class DDPMSampler:
    def __init__(self, model, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.num_timesteps = num_timesteps
        
        # 噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, noise, timesteps):
        """向图像添加噪声"""
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])
        
        return (sqrt_alphas_cumprod[:, None, None, None] * x_0 + 
                sqrt_one_minus_alphas_cumprod[:, None, None, None] * noise)
    
    def sample(self, shape, device):
        """从噪声中采样图像"""
        x = torch.randn(shape).to(device)
        
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)
            
            # 预测噪声
            with torch.no_grad():
                pred_noise = self.model(x, t_tensor)
            
            # 去噪步骤
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
            x = x + torch.sqrt(beta) * noise
        
        return x
```

## 🏗️ 合成数据生成

### 表格数据生成
```python
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy import stats
import random

class TabularDataGenerator:
    def __init__(self):
        self.column_distributions = {}
        self.correlation_matrix = None
        
    def fit(self, data):
        """学习数据分布"""
        self.data = data
        
        # 学习每列的分布
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # 数值列：使用高斯混合模型
                gmm = GaussianMixture(n_components=3)
                gmm.fit(data[column].values.reshape(-1, 1))
                self.column_distributions[column] = {
                    'type': 'numeric',
                    'model': gmm,
                    'min': data[column].min(),
                    'max': data[column].max()
                }
            else:
                # 分类列：使用频率分布
                value_counts = data[column].value_counts(normalize=True)
                self.column_distributions[column] = {
                    'type': 'categorical',
                    'probabilities': value_counts.to_dict()
                }
        
        # 计算数值列之间的相关性
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_columns) > 1:
            self.correlation_matrix = data[numeric_columns].corr()
    
    def generate_samples(self, num_samples=1000, maintain_correlations=True):
        """生成合成样本"""
        synthetic_data = {}
        
        if maintain_correlations and self.correlation_matrix is not None:
            # 生成保持相关性的数值数据
            numeric_columns = list(self.correlation_matrix.columns)
            multivariate_normal = stats.multivariate_normal(
                mean=np.zeros(len(numeric_columns)),
                cov=self.correlation_matrix.values
            )
            
            # 生成相关的正态分布样本
            correlated_samples = multivariate_normal.rvs(num_samples)
            
            # 转换为原始分布
            for i, column in enumerate(numeric_columns):
                # 使用逆变换采样
                uniform_samples = stats.norm.cdf(correlated_samples[:, i])
                
                # 从学习的分布中采样
                if self.column_distributions[column]['type'] == 'numeric':
                    model = self.column_distributions[column]['model']
                    # 近似逆变换
                    samples = []
                    for _ in range(num_samples):
                        sample = model.sample()[0][0]
                        samples.append(sample)
                    synthetic_data[column] = samples
        else:
            # 独立生成每列
            for column, dist_info in self.column_distributions.items():
                if dist_info['type'] == 'numeric':
                    model = dist_info['model']
                    samples = model.sample(num_samples)[0].flatten()
                    # 确保在原始范围内
                    samples = np.clip(samples, dist_info['min'], dist_info['max'])
                    synthetic_data[column] = samples
                
                elif dist_info['type'] == 'categorical':
                    categories = list(dist_info['probabilities'].keys())
                    probabilities = list(dist_info['probabilities'].values())
                    samples = np.random.choice(categories, size=num_samples, p=probabilities)
                    synthetic_data[column] = samples
        
        return pd.DataFrame(synthetic_data)
    
    def generate_conditional_samples(self, conditions, num_samples=100):
        """生成条件样本"""
        synthetic_data = {}
        
        # 首先设置条件列
        for column, value in conditions.items():
            if isinstance(value, list):
                synthetic_data[column] = np.random.choice(value, num_samples)
            else:
                synthetic_data[column] = [value] * num_samples
        
        # 生成其他列
        for column, dist_info in self.column_distributions.items():
            if column not in conditions:
                if dist_info['type'] == 'numeric':
                    model = dist_info['model']
                    samples = model.sample(num_samples)[0].flatten()
                    synthetic_data[column] = samples
                elif dist_info['type'] == 'categorical':
                    categories = list(dist_info['probabilities'].keys())
                    probabilities = list(dist_info['probabilities'].values())
                    samples = np.random.choice(categories, size=num_samples, p=probabilities)
                    synthetic_data[column] = samples
        
        return pd.DataFrame(synthetic_data)
    
    def evaluate_quality(self, original_data, synthetic_data):
        """评估合成数据质量"""
        quality_metrics = {}
        
        # 统计相似性
        for column in original_data.columns:
            if column in synthetic_data.columns:
                if original_data[column].dtype in ['int64', 'float64']:
                    # KS检验
                    ks_stat, ks_p = stats.ks_2samp(
                        original_data[column].dropna(),
                        synthetic_data[column].dropna()
                    )
                    quality_metrics[f'{column}_ks_stat'] = ks_stat
                    quality_metrics[f'{column}_ks_p'] = ks_p
                    
                    # 均值和标准差比较
                    orig_mean = original_data[column].mean()
                    synth_mean = synthetic_data[column].mean()
                    quality_metrics[f'{column}_mean_diff'] = abs(orig_mean - synth_mean) / orig_mean
                    
                    orig_std = original_data[column].std()
                    synth_std = synthetic_data[column].std()
                    quality_metrics[f'{column}_std_diff'] = abs(orig_std - synth_std) / orig_std
                
                else:
                    # 分类列：分布相似性
                    orig_dist = original_data[column].value_counts(normalize=True)
                    synth_dist = synthetic_data[column].value_counts(normalize=True)
                    
                    # 计算JS散度
                    all_categories = set(orig_dist.index) | set(synth_dist.index)
                    orig_probs = [orig_dist.get(cat, 0) for cat in all_categories]
                    synth_probs = [synth_dist.get(cat, 0) for cat in all_categories]
                    
                    js_div = self.jensen_shannon_divergence(orig_probs, synth_probs)
                    quality_metrics[f'{column}_js_divergence'] = js_div
        
        # 相关性保持
        if self.correlation_matrix is not None:
            orig_corr = original_data.corr()
            synth_corr = synthetic_data.corr()
            corr_diff = np.abs(orig_corr - synth_corr).mean().mean()
            quality_metrics['correlation_preservation'] = 1 - corr_diff
        
        return quality_metrics
    
    def jensen_shannon_divergence(self, p, q):
        """计算JS散度"""
        p = np.array(p) + 1e-10  # 避免log(0)
        q = np.array(q) + 1e-10
        m = 0.5 * (p + q)
        
        return 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)

# 使用示例
# generator = TabularDataGenerator()
# generator.fit(original_data)
# synthetic_data = generator.generate_samples(1000)
# quality = generator.evaluate_quality(original_data, synthetic_data)
```

### 时间序列数据生成
```python
class TimeSeriesGenerator:
    def __init__(self):
        self.trend_model = None
        self.seasonal_model = None
        self.noise_model = None
        
    def decompose_series(self, time_series, period=12):
        """分解时间序列"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(time_series, model='additive', period=period)
        
        self.trend = decomposition.trend.dropna()
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid.dropna()
        
        return decomposition
    
    def fit_trend_model(self, trend_data):
        """拟合趋势模型"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        X = np.arange(len(trend_data)).reshape(-1, 1)
        
        # 尝试不同阶数的多项式
        best_score = -np.inf
        best_model = None
        
        for degree in [1, 2, 3]:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, trend_data)
            score = model.score(X_poly, trend_data)
            
            if score > best_score:
                best_score = score
                best_model = (poly_features, model)
        
        self.trend_model = best_model
        return best_model
    
    def generate_trend(self, length):
        """生成趋势"""
        if self.trend_model is None:
            return np.zeros(length)
        
        poly_features, model = self.trend_model
        X = np.arange(length).reshape(-1, 1)
        X_poly = poly_features.transform(X)
        
        return model.predict(X_poly)
    
    def generate_seasonal_pattern(self, length, period=12):
        """生成季节性模式"""
        if not hasattr(self, 'seasonal'):
            # 生成简单的正弦波季节性
            t = np.arange(length)
            seasonal = np.sin(2 * np.pi * t / period)
            return seasonal
        
        # 重复已学习的季节性模式
        seasonal_pattern = self.seasonal[:period].values
        full_pattern = np.tile(seasonal_pattern, length // period + 1)
        return full_pattern[:length]
    
    def generate_noise(self, length, noise_type='gaussian'):
        """生成噪声"""
        if noise_type == 'gaussian':
            if hasattr(self, 'residual'):
                noise_std = self.residual.std()
            else:
                noise_std = 0.1
            return np.random.normal(0, noise_std, length)
        
        elif noise_type == 'autoregressive':
            # AR(1) 噪声
            if hasattr(self, 'residual'):
                from statsmodels.tsa.ar_model import AutoReg
                try:
                    ar_model = AutoReg(self.residual.dropna(), lags=1).fit()
                    noise = ar_model.forecast(length)
                    return noise
                except:
                    return np.random.normal(0, 0.1, length)
            else:
                return np.random.normal(0, 0.1, length)
    
    def generate_synthetic_series(self, length, period=12, noise_type='gaussian'):
        """生成合成时间序列"""
        # 生成各个组件
        trend = self.generate_trend(length)
        seasonal = self.generate_seasonal_pattern(length, period)
        noise = self.generate_noise(length, noise_type)
        
        # 组合时间序列
        synthetic_series = trend + seasonal + noise
        
        return synthetic_series
    
    def generate_multiple_series(self, num_series, length, correlation=0.5):
        """生成多个相关的时间序列"""
        # 生成基础系列
        base_series = self.generate_synthetic_series(length)
        
        series_list = [base_series]
        
        for i in range(num_series - 1):
            # 生成相关系列
            independent_series = self.generate_synthetic_series(length)
            
            # 创建相关性
            correlated_series = (correlation * base_series + 
                               np.sqrt(1 - correlation**2) * independent_series)
            
            series_list.append(correlated_series)
        
        return np.array(series_list).T

# ARIMA时间序列生成
class ARIMAGenerator:
    def __init__(self):
        self.model = None
        
    def fit(self, time_series, order=(1, 1, 1)):
        """拟合ARIMA模型"""
        from statsmodels.tsa.arima.model import ARIMA
        
        self.model = ARIMA(time_series, order=order)
        self.fitted_model = self.model.fit()
        
        return self.fitted_model
    
    def generate_samples(self, num_samples, num_steps=100):
        """生成ARIMA样本"""
        if self.fitted_model is None:
            raise ValueError("模型未拟合")
        
        samples = []
        for _ in range(num_samples):
            # 生成预测
            forecast = self.fitted_model.forecast(steps=num_steps)
            samples.append(forecast)
        
        return np.array(samples)
    
    def simulate_scenarios(self, base_series, num_scenarios=10, horizon=24):
        """模拟未来场景"""
        scenarios = []
        
        for _ in range(num_scenarios):
            # 重新拟合模型（添加一些随机性）
            noisy_series = base_series + np.random.normal(0, 0.01, len(base_series))
            
            try:
                model = ARIMA(noisy_series, order=(1, 1, 1)).fit()
                forecast = model.forecast(steps=horizon)
                scenarios.append(forecast)
            except:
                # 如果拟合失败，使用简单的外推
                trend = np.polyfit(range(len(base_series)), base_series, 1)
                forecast = np.polyval(trend, range(len(base_series), len(base_series) + horizon))
                scenarios.append(forecast)
        
        return np.array(scenarios)
```

## 🎨 场景编辑技术

### 图像场景编辑
```python
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class ImageSceneEditor:
    def __init__(self):
        self.layers = []
        self.current_layer = 0
        
    def load_background(self, image_path):
        """加载背景图像"""
        self.background = cv2.imread(image_path)
        self.height, self.width = self.background.shape[:2]
        self.layers = [self.background.copy()]
        
    def add_object(self, object_image_path, position, scale=1.0, rotation=0):
        """添加物体到场景"""
        obj_img = cv2.imread(object_image_path, cv2.IMREAD_UNCHANGED)
        
        # 缩放
        if scale != 1.0:
            new_width = int(obj_img.shape[1] * scale)
            new_height = int(obj_img.shape[0] * scale)
            obj_img = cv2.resize(obj_img, (new_width, new_height))
        
        # 旋转
        if rotation != 0:
            center = (obj_img.shape[1] // 2, obj_img.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            obj_img = cv2.warpAffine(obj_img, rotation_matrix, 
                                   (obj_img.shape[1], obj_img.shape[0]))
        
        # 添加到场景
        self.add_layer_at_position(obj_img, position)
    
    def add_layer_at_position(self, layer_img, position):
        """在指定位置添加图层"""
        x, y = position
        h, w = layer_img.shape[:2]
        
        # 创建新图层
        new_layer = self.layers[-1].copy()
        
        # 确保位置在画布范围内
        if x + w > self.width:
            w = self.width - x
            layer_img = layer_img[:, :w]
        if y + h > self.height:
            h = self.height - y
            layer_img = layer_img[:h, :]
        if x < 0:
            layer_img = layer_img[:, -x:]
            w = layer_img.shape[1]
            x = 0
        if y < 0:
            layer_img = layer_img[-y:, :]
            h = layer_img.shape[0]
            y = 0
        
        # 处理透明度
        if layer_img.shape[2] == 4:  # RGBA
            alpha = layer_img[:, :, 3] / 255.0
            for c in range(3):
                new_layer[y:y+h, x:x+w, c] = (
                    alpha * layer_img[:, :, c] + 
                    (1 - alpha) * new_layer[y:y+h, x:x+w, c]
                )
        else:  # RGB
            new_layer[y:y+h, x:x+w] = layer_img
        
        self.layers.append(new_layer)
    
    def change_lighting(self, brightness=0, contrast=1.0):
        """改变光照"""
        current_layer = self.layers[-1].copy()
        
        # 调整亮度和对比度
        adjusted = cv2.convertScaleAbs(current_layer, alpha=contrast, beta=brightness)
        
        self.layers.append(adjusted)
    
    def add_weather_effect(self, weather_type='rain', intensity=0.5):
        """添加天气效果"""
        current_layer = self.layers[-1].copy()
        
        if weather_type == 'rain':
            # 生成雨滴
            rain_layer = np.zeros_like(current_layer)
            
            num_drops = int(1000 * intensity)
            for _ in range(num_drops):
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                length = np.random.randint(10, 30)
                
                cv2.line(rain_layer, (x, y), (x + 2, y + length), (200, 200, 200), 1)
            
            # 混合雨滴效果
            result = cv2.addWeighted(current_layer, 1 - intensity * 0.3, 
                                   rain_layer, intensity * 0.3, 0)
        
        elif weather_type == 'fog':
            # 添加雾效
            fog_layer = np.full_like(current_layer, 180)  # 灰色雾
            result = cv2.addWeighted(current_layer, 1 - intensity * 0.6, 
                                   fog_layer, intensity * 0.6, 0)
        
        elif weather_type == 'snow':
            # 生成雪花
            snow_layer = np.zeros_like(current_layer)
            
            num_flakes = int(500 * intensity)
            for _ in range(num_flakes):
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                radius = np.random.randint(1, 4)
                
                cv2.circle(snow_layer, (x, y), radius, (255, 255, 255), -1)
            
            result = cv2.addWeighted(current_layer, 1, snow_layer, intensity, 0)
        
        self.layers.append(result)
    
    def segment_and_replace_background(self, new_background_path):
        """分割并替换背景"""
        # 这里应该使用更高级的分割模型，如DeepLab或Mask R-CNN
        # 简化版本：使用GrabCut
        
        current_layer = self.layers[-1].copy()
        mask = np.zeros(current_layer.shape[:2], np.uint8)
        
        # 初始化矩形（假设主体在中央）
        rect = (self.width//4, self.height//4, self.width//2, self.height//2)
        
        # GrabCut分割
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(current_layer, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # 创建前景掩码
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 加载新背景
        new_bg = cv2.imread(new_background_path)
        new_bg = cv2.resize(new_bg, (self.width, self.height))
        
        # 合成结果
        result = new_bg.copy()
        result = result * (1 - mask2[:, :, np.newaxis]) + current_layer * mask2[:, :, np.newaxis]
        
        self.layers.append(result.astype(np.uint8))
    
    def add_text_annotation(self, text, position, font_size=30, color=(255, 255, 255)):
        """添加文本注释"""
        current_layer = self.layers[-1].copy()
        
        # 转换为PIL图像以支持更好的文本渲染
        pil_image = Image.fromarray(cv2.cvtColor(current_layer, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color)
        
        # 转换回OpenCV格式
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self.layers.append(result)
    
    def generate_variations(self, num_variations=5):
        """生成场景变化"""
        base_layer = self.layers[0].copy()
        variations = []
        
        for i in range(num_variations):
            # 随机调整
            brightness = np.random.randint(-30, 31)
            contrast = np.random.uniform(0.8, 1.2)
            
            # 应用调整
            variation = cv2.convertScaleAbs(base_layer, alpha=contrast, beta=brightness)
            
            # 随机天气效果
            weather_effects = ['rain', 'fog', 'snow', None]
            weather = np.random.choice(weather_effects)
            
            if weather:
                intensity = np.random.uniform(0.2, 0.8)
                # 应用天气效果（简化版）
                if weather == 'rain':
                    # 添加一些噪声模拟雨水
                    noise = np.random.normal(0, 10, variation.shape).astype(np.int16)
                    variation = np.clip(variation.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            variations.append(variation)
        
        return variations
    
    def export_scene(self, output_path):
        """导出当前场景"""
        current_scene = self.layers[-1]
        cv2.imwrite(output_path, current_scene)
        
    def get_current_scene(self):
        """获取当前场景"""
        return self.layers[-1]
```

### 3D场景编辑
```python
import open3d as o3d
import numpy as np

class Scene3DEditor:
    def __init__(self):
        self.scene_objects = []
        self.materials = {}
        self.lighting = []
        
    def load_3d_model(self, file_path, object_id=None):
        """加载3D模型"""
        if file_path.endswith('.ply'):
            mesh = o3d.io.read_triangle_mesh(file_path)
        elif file_path.endswith('.obj'):
            mesh = o3d.io.read_triangle_mesh(file_path)
        else:
            raise ValueError("不支持的文件格式")
        
        if object_id is None:
            object_id = f"object_{len(self.scene_objects)}"
        
        scene_object = {
            'id': object_id,
            'mesh': mesh,
            'position': np.array([0, 0, 0]),
            'rotation': np.array([0, 0, 0]),
            'scale': np.array([1, 1, 1]),
            'material': 'default'
        }
        
        self.scene_objects.append(scene_object)
        return object_id
    
    def create_primitive(self, primitive_type, size=1.0, object_id=None):
        """创建基本几何体"""
        if primitive_type == 'cube':
            mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        elif primitive_type == 'sphere':
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        elif primitive_type == 'cylinder':
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=size, height=size*2)
        elif primitive_type == 'plane':
            mesh = o3d.geometry.TriangleMesh.create_box(width=size*2, height=0.1, depth=size*2)
        else:
            raise ValueError(f"不支持的几何体类型: {primitive_type}")
        
        if object_id is None:
            object_id = f"{primitive_type}_{len(self.scene_objects)}"
        
        scene_object = {
            'id': object_id,
            'mesh': mesh,
            'position': np.array([0, 0, 0]),
            'rotation': np.array([0, 0, 0]),
            'scale': np.array([1, 1, 1]),
            'material': 'default'
        }
        
        self.scene_objects.append(scene_object)
        return object_id
    
    def transform_object(self, object_id, position=None, rotation=None, scale=None):
        """变换物体"""
        obj = self.get_object_by_id(object_id)
        if obj is None:
            return
        
        if position is not None:
            obj['position'] = np.array(position)
        if rotation is not None:
            obj['rotation'] = np.array(rotation)
        if scale is not None:
            obj['scale'] = np.array(scale)
        
        self.apply_transforms(obj)
    
    def apply_transforms(self, scene_object):
        """应用变换"""
        mesh = scene_object['mesh']
        
        # 重置变换
        mesh.translate(-mesh.get_center())
        
        # 缩放
        scale = scene_object['scale']
        mesh.scale(scale[0], center=mesh.get_center())
        
        # 旋转
        rotation = scene_object['rotation']
        if np.any(rotation != 0):
            R = mesh.get_rotation_matrix_from_xyz(rotation)
            mesh.rotate(R, center=mesh.get_center())
        
        # 平移
        position = scene_object['position']
        mesh.translate(position)
    
    def get_object_by_id(self, object_id):
        """根据ID获取物体"""
        for obj in self.scene_objects:
            if obj['id'] == object_id:
                return obj
        return None
    
    def set_material(self, object_id, color=None, texture_path=None):
        """设置材质"""
        obj = self.get_object_by_id(object_id)
        if obj is None:
            return
        
        if color is not None:
            obj['mesh'].paint_uniform_color(color)
        
        if texture_path is not None:
            # 加载纹理（简化版本）
            texture = o3d.io.read_image(texture_path)
            # Open3D的纹理支持有限，这里只是示例
            obj['texture'] = texture
    
    def add_lighting(self, light_type='point', position=[0, 5, 0], intensity=1.0, color=[1, 1, 1]):
        """添加光源"""
        light = {
            'type': light_type,
            'position': np.array(position),
            'intensity': intensity,
            'color': np.array(color)
        }
        self.lighting.append(light)
    
    def duplicate_object(self, object_id, new_id=None, offset=[1, 0, 0]):
        """复制物体"""
        original_obj = self.get_object_by_id(object_id)
        if original_obj is None:
            return None
        
        if new_id is None:
            new_id = f"{object_id}_copy_{len(self.scene_objects)}"
        
        # 深度复制网格
        new_mesh = original_obj['mesh'].copy()
        
        new_object = {
            'id': new_id,
            'mesh': new_mesh,
            'position': original_obj['position'] + np.array(offset),
            'rotation': original_obj['rotation'].copy(),
            'scale': original_obj['scale'].copy(),
            'material': original_obj['material']
        }
        
        self.scene_objects.append(new_object)
        self.apply_transforms(new_object)
        
        return new_id
    
    def create_array(self, object_id, grid_size=(3, 3), spacing=2.0):
        """创建物体阵列"""
        created_objects = []
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if i == 0 and j == 0:
                    continue  # 跳过原始物体
                
                offset = [i * spacing, 0, j * spacing]
                new_id = self.duplicate_object(object_id, 
                                             f"{object_id}_array_{i}_{j}", 
                                             offset)
                if new_id:
                    created_objects.append(new_id)
        
        return created_objects
    
    def generate_random_scene(self, num_objects=10, scene_bounds=[-10, 10]):
        """生成随机场景"""
        primitives = ['cube', 'sphere', 'cylinder']
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        
        for i in range(num_objects):
            # 随机选择几何体类型
            primitive = np.random.choice(primitives)
            size = np.random.uniform(0.5, 2.0)
            
            # 创建物体
            obj_id = self.create_primitive(primitive, size)
            
            # 随机位置
            position = np.random.uniform(scene_bounds[0], scene_bounds[1], 3)
            position[1] = max(0, position[1])  # 确保在地面以上
            
            # 随机旋转
            rotation = np.random.uniform(0, 2*np.pi, 3)
            
            # 随机缩放
            scale = np.random.uniform(0.5, 1.5, 3)
            
            # 应用变换
            self.transform_object(obj_id, position, rotation, scale)
            
            # 随机颜色
            color = np.random.choice(colors)
            self.set_material(obj_id, color)
    
    def export_scene(self, output_path):
        """导出场景"""
        combined_mesh = o3d.geometry.TriangleMesh()
        
        for obj in self.scene_objects:
            combined_mesh += obj['mesh']
        
        o3d.io.write_triangle_mesh(output_path, combined_mesh)
    
    def visualize_scene(self):
        """可视化场景"""
        geometries = []
        
        for obj in self.scene_objects:
            geometries.append(obj['mesh'])
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
        geometries.append(coordinate_frame)
        
        o3d.visualization.draw_geometries(geometries)
    
    def render_scene(self, camera_position=[0, 5, 10], resolution=(800, 600)):
        """渲染场景"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=resolution[0], height=resolution[1])
        
        for obj in self.scene_objects:
            vis.add_geometry(obj['mesh'])
        
        # 设置相机
        ctr = vis.get_view_control()
        ctr.set_front([0, -0.5, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        
        # 渲染
        vis.run()
        vis.capture_screen_image("rendered_scene.png")
        vis.destroy_window()
```

## 🔗 导航链接

- [返回主页](../index.html)
- [上一模块：数据与场景可视化](data-visualization.html)
- [下一模块：数据应用](data-application.html)
