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
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# def objective_function(theta, x):

#     return np.sum(-10 * (np.sin(0.05 * np.pi * (theta - x)) ** 6) / (2 ** ((2*((theta - x) - 90) / 50) ** 2)))
    

def objective_function(theta, x):

    return  np.sum((theta - x) ** 2)
    


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
    best_theta_total = []
    for x in X:
        solutions = random_points(K, x_dim) # 随机生成 K 个解
        #solutions = np.random.uniform(lowbound, upbound, size=(K, x_dim)) 
        objectives = min([objective_function(theta, x) for theta in solutions])
        thetas.append(solutions)
        best_theta = solutions[np.argmin(objectives)]  # 选择最优解
        best_theta_total.append(best_theta)
        objective_values.append(objectives)

 
    return X, np.array(thetas), np.array(objective_values), np.array(best_theta_total)

  

class SurfaceModel(nn.Module):
    def __init__(self, input_dim):
        super(SurfaceModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个目标值
        )
    
    def forward(self, x):
        return self.net(x)

# 训练神经网络
def train_nn_surface_model(X, thetas, objective_values, x_dim, epochs=100, lr=1e-3, batch_size=32):
    # 创建训练数据
    # n, K, theta_dim = thetas.shape
    # X_repeated = np.repeat(X, K, axis=0)  # 将 X 重复 K 次
    # thetas_flat = thetas.reshape(-1, theta_dim)  # 展平 thetas
    # inputs = np.hstack([thetas_flat, X_repeated])  # 拼接 theta 和 x
    # labels = objective_values.flatten()


    n, theta_dim = thetas.shape
    inputs = np.hstack([thetas, X]) 
    labels = objective_values  # 展平目标值

    # 转换为 PyTorch 张量
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # 定义神经网络模型
    model = SurfaceModel(input_dim=theta_dim + x_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(inputs_tensor.size(0))
        for i in range(0, inputs_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_inputs = inputs_tensor[indices]
            batch_labels = labels_tensor[indices]

            # 前向传播
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model




class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, theta_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim,256),
            nn.ReLU(),
            nn.Linear(256, theta_dim)
        )
    
    def forward(self, z, x):
        return self.net(torch.cat([z, x], dim=1))

class Discriminator(nn.Module):
    def __init__(self, theta_dim, x_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim + x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, theta, x):
        return self.net(torch.cat([theta, x], dim=1))

# cWGAN 训练

def wgan_generator_loss(D_fake):
    return -torch.mean(D_fake)

def wgan_discriminator_loss(D_real, D_fake):
    return torch.mean(D_fake) - torch.mean(D_real)


def train_cgan(X, thetas, z_dim, n_iterations, batch_size):
    x_dim = X.shape[1]
    theta_dim = thetas.shape[1]

    # 初始化 Generator 和 Discriminator
    G = Generator(z_dim, x_dim, theta_dim)
    D = Discriminator(theta_dim, x_dim)

    # 定义优化器
    g_optimizer = optim.Adam(G.parameters(), lr=1e-2,  weight_decay=1e-5)
    d_optimizer = optim.Adam(D.parameters(), lr=1e-3, weight_decay=1e-5)

    # 损失函数
    #bce_loss = nn.BCELoss()

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

            # real_labels = torch.ones(x_batch.size(0), 1)
            # fake_labels = torch.zeros(x_batch.size(0), 1)

            # real_scores = D(theta_batch, x_batch)
            # fake_scores = D(fake_thetas.detach(), x_batch)

            # d_loss = bce_loss(real_scores, real_labels) + bce_loss(fake_scores, fake_labels)
            # d_optimizer.zero_grad()
            # d_loss.backward()
            # d_optimizer.step()

            d_optimizer.zero_grad()
            D_real = D(theta_batch, x_batch)
            D_fake = D(fake_thetas.detach(), x_batch)

            loss_D = wgan_discriminator_loss(D_real, D_fake)
            loss_D.backward(retain_graph=True)
            d_optimizer.step()


            # 训练 Generator
            # fake_scores = D(fake_thetas, x_batch)
            # g_loss = bce_loss(fake_scores, real_labels)
            # g_optimizer.zero_grad()
            # g_loss.backward()
            # g_optimizer.step()


            g_optimizer.zero_grad()
            D_fake = D(fake_thetas.detach(), x_batch)
            loss_G = wgan_generator_loss(D_fake)

            lambda_theta , alpha = 0.5, 0.5
            loss_theta = torch.mean((fake_thetas.detach() - theta_batch) ** 2)
            total_loss_G = lambda_theta * loss_theta + alpha * loss_G
            total_loss_G.backward()
            g_optimizer.step()

        # 打印损失
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, D Loss: {loss_D.item()}, G Loss: {total_loss_G.item()}")


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
        nn_input = torch.tensor(np.concatenate([theta, x]), dtype=torch.float32).unsqueeze(0)
        score = surface_model(nn_input).item()


        #score = surface_model.predict([np.concatenate([theta, x])])  # 将 theta 和 x 拼接作为输入
        scores.append(score)

    # Step 3: Select the Optimal Solution
    best_index = np.argmin(scores)  # 找到目标值最小的候选解索引
    theta_star = candidates[best_index]  # 对应的最优解

    return theta_star
def random_points(n, dim):
    points = []
    for _ in range(n):
        point = []
        for _ in range(dim):
            if np.random.rand() > 0.5:
                value = np.random.randint(lowbound, upbound + 1)
            else:
                value = np.random.randint(lowbound, upbound) + np.random.rand()
            point.append(value)
        points.append(point)
    return np.array(points)


if __name__ == "__main__":
    # 参数定义
    #n = 10  # covariate 点数量
    t = 10
    n_values = [2**i for i in range(6, t)]

    x_dim = 2  # covariate 维度
    K = 2000  # 每个点的解的数量
    z_dim = 10  # 噪声维度
    M = 1000  # 候选解数量
    n_iterations = 1000  # cGAN 训练迭代次数
    batch_size = 16  # 批次大小
    lowbound = 0
    upbound = 100
    # First Stage: 近似最优解的选择与表面拟合
    #X, thetas, krr = first_stage(n, x_dim, K)

    mse1 =[]
    for n in n_values:

        X, thetas, objective_values, best_theta = first_stage(n, x_dim, K)
        print("Training neural network surface model...")
        #nn_model = train_nn_surface_model(X, thetas, objective_values, x_dim)
        nn_model = train_nn_surface_model(X, best_theta, objective_values, x_dim)

        print(best_theta.shape)
        G = train_cgan(X, best_theta, z_dim, n_iterations, batch_size)


        print("Training completed. Generator is ready to generate samples conditioned on x.")

        num_test_points = 20
        test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, x_dim))
        

        theta_star = np.array([online_stage(x, G, nn_model, z_dim, M) for x in test_points])

        mse = mean_squared_error(test_points, theta_star)

        #print("Approximately optimal solution θ*:", theta_star)
        

        mse1.append(mse)
        print("MSE", mse)
    sns.set()
    sns.lineplot(x=n_values, y=mse1, marker='o')
    plt.title('MSE vs n_values', fontsize=16)
    plt.xlabel('n_values (Number of Covariate Points)', fontsize=14)
    plt.ylabel('MSE (Mean Squared Error)', fontsize=14)
    plt.xticks(ticks=n_values, labels=[r"$2^{{{}}}$".format(i) for i in range(6, t)])
    plt.grid(True)
    plt.show()



