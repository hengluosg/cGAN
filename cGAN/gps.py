from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

import warnings
warnings.filterwarnings("ignore", category=UserWarning)





# Define the one-dimensional function g(x)
def g_1d(x):
    term1 = -10 * (np.sin(0.05 * np.pi * x) ** 6) / (2 ** (((x - 90) / 50) ** 2))
    return term1

# Generate x values in the range [0, 100]
x_values = np.linspace(0, 200, 500)
y_values = g_1d(x_values)


# Create a DataFrame for seaborn
data = pd.DataFrame({
    'x': x_values,
    'g(x)': y_values
})

# # Plot using seaborn
# sns.set()
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=data, x='x', y='g(x)', color='blue', label=r"$g(x)$")
# plt.title(" g(x)", fontsize=14)
# plt.xlabel("x", fontsize=12)
# plt.ylabel("g(x)", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.show()



def objective_function(theta, x):

    return np.sum(-10 * (np.sin(0.05 * np.pi * (theta - x)) ** 6) / (2 ** ((((theta - x) - 90) / 50) ** 2)))
    

# def objective_function(theta, x):

#     return  np.sum((theta - x) ** 2)
    


# Grid sampling function
def generate_covariates(point,d):
    n = int(round(point ** (1.0 / d))) + 1
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]
    return grid_points

def first_stage(n, x_dim, K):
    X = generate_covariates(n, x_dim)  # 生成 covariates 点
    thetas = []
    objective_values = []
    
    for x in X:
        solutions = np.random.randint(1, 800, size=(K, x_dim)) / np.random.randint(1, 100, size=(K, x_dim))
        #solutions = np.random.uniform(0, 8, size=(K, x_dim))  # 随机生成 K 个解
        objectives = [objective_function(theta, x) for theta in solutions]
        print(min(objectives))
        best_theta = solutions[np.argmin(objectives)]  # 选择最优解
        thetas.append(best_theta)
        objective_values.append(objectives)

    thetas = np.array(thetas)
    X_train = np.hstack([thetas, X])  # 将 thetas 和 X 水平拼接
    y_train = np.array([objective_function(theta, x) for theta, x in zip(thetas, X)])  # 目标值 f(θ, x)

    # 使用 Kernel Ridge Regression 拟合表面
    krr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.5)
    krr.fit(X_train, y_train)  # 训练 KernelRidge 模型
    return X, thetas, krr



class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, theta_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, theta_dim)
        )
    
    def forward(self, z, x):
        return self.net(torch.cat([z, x], dim=1))

class Discriminator(nn.Module):
    def __init__(self, theta_dim, x_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim + x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, theta, x):
        return self.net(torch.cat([theta, x], dim=1))

# cGAN 训练
def train_cgan(X, thetas, z_dim, n_iterations, batch_size):
    x_dim = X.shape[1]
    theta_dim = thetas.shape[1]

    # 初始化 Generator 和 Discriminator
    G = Generator(z_dim, x_dim, theta_dim)
    D = Discriminator(theta_dim, x_dim)

    # 定义优化器
    g_optimizer = optim.Adam(G.parameters(), lr=1e-3)
    d_optimizer = optim.Adam(D.parameters(), lr=1e-3)

    # 损失函数
    bce_loss = nn.BCELoss()

    # 转换为 Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32)

    for iteration in range(n_iterations):
        for batch_idx in range(0, len(X), batch_size):
            # 获取当前批次数据
            x_batch = X_tensor[batch_idx:batch_idx + batch_size]
            theta_batch = thetas_tensor[batch_idx:batch_idx + batch_size]
            
            # 训练 Discriminator
            z = torch.randn(x_batch.size(0), z_dim)
            fake_thetas = G(z, x_batch)
            real_labels = torch.ones(x_batch.size(0), 1)
            fake_labels = torch.zeros(x_batch.size(0), 1)

            real_scores = D(theta_batch, x_batch)
            fake_scores = D(fake_thetas.detach(), x_batch)

            d_loss = bce_loss(real_scores, real_labels) + bce_loss(fake_scores, fake_labels)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 训练 Generator
            fake_scores = D(fake_thetas, x_batch)
            g_loss = bce_loss(fake_scores, real_labels)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # 打印损失
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")


    return G

def online_stage(x, G, surface_model, z_dim, M):
    """
    Online stage to find approximately optimal solution for a given covariate point x.
    
    Parameters:
    - x: ndarray, shape (x_dim,) 输入的 covariate 点
    - G: Generator
    - surface_model: callable, \hat{f}，可以是 Kernel Ridge Regression 
    - z_dim: int
    - M: int
    
    Returns:
    - theta_star: ndarray,  \theta^*(x)
    """
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # 转换为 PyTorch Tensor 并添加批次维度

    # Step 1: Generate Candidate Solutions
    candidates = []
    for _ in range(M):
        z = torch.randn(1, z_dim)  # 从标准正态分布生成噪声 z
        theta_g = G(z, x_tensor)  # 使用训练好的生成器 G 生成候选解
        candidates.append(theta_g.detach().numpy().squeeze())  # 转换为 NumPy 数组并存储

    # Step 2: Evaluate Candidate Solutions
    scores = []
    for theta in candidates:
        # 使用表面模型 \hat{f} 评估目标值
        score = surface_model.predict([np.concatenate([theta, x])])  # 将 theta 和 x 拼接作为输入
        scores.append(score)

    # Step 3: Select the Optimal Solution
    best_index = np.argmin(scores)  # 找到目标值最小的候选解索引
    theta_star = candidates[best_index]  # 对应的最优解

    return theta_star



if __name__ == "__main__":
    # 参数定义
    n = 300  # covariate 点数量
    x_dim = 5  # covariate 维度
    K = 1000  # 每个点的解的数量
    z_dim = 10  # 噪声维度
    M = 1000  # 候选解数量
    n_iterations = 1000  # cGAN 训练迭代次数
    batch_size = 16  # 批次大小
    lowbound = 0
    upbound = 8
    # First Stage: 近似最优解的选择与表面拟合
    X, thetas, krr = first_stage(n, x_dim, K)
    print(X.shape, thetas.shape, krr)
    G = train_cgan(X, thetas, z_dim, n_iterations, batch_size)

    x = np.random.uniform(0, 8, size=x_dim)

    theta_star = online_stage(x, G, krr, z_dim, M)
    print("Approximately optimal solution θ*:", theta_star)


    print("Training completed. Generator is ready to generate samples conditioned on x.")


    print(x, theta_star,objective_function(x, theta_star))