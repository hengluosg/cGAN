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
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

# def objective_function(theta, x):
#     term1 = np.sum(np.sin(np.pi * theta) + 0.1 * (theta - x) ** 2)
#     return term1


def generate_symmetric_matrix(d, correlation_strength=0.5):
    
    A = np.random.randn(d, d)
    A = np.dot(A.T, A)  
    for i in range(d):
        A[i, i] = 1  # Set diagonal elements to 1
    for i in range(d):
        for j in range(i+1, d):
            A[i, j] = A[j, i] = correlation_strength  # Set correlation between variables
    return A

def objective_function(theta, x):
    theta1 =  A.T @ theta
    return np.sum(np.cos(2 *theta1) + x * np.sin(2 *theta1) + 0.1 * theta1**2)


# Grid sampling function
def grid_sample(d, lowbound, upbound, point):
    # 根据给定的总点数和维度，计算每个维度的点数 n
    n = int(round(point ** (1.0 / d))) +1 # 每个维度上的点数
    
    # 在每个维度生成 n 个均匀间隔的点
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    #print(len(grid_1d))
    # 在 d 维空间生成网格点
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    
    # 如果生成的点数大于目标点数，则随机选择 point 个点
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]
    #print(grid_points)
    return grid_points



def gradient(theta, x):

    theta1  = A.T @ theta
    grad_term1 = -2 * np.sin(2 * theta1)@ A.T + 2 * x * np.cos(2 * theta1)@ A.T + 0.2 * theta1@ A.T
    noise = np.random.normal(0, noise_std, covariate_dim)
    return grad_term1 + noise



# SGD optimization function
def project(theta, lowbound, upbound):
    """
    Projects theta back into the valid domain [lowbound, upbound] element-wise.
    """
    return np.clip(theta, lowbound, upbound)




def first_stage1(x, K, T, eta, d):
    """
    Optimizes using SGD for each x_i and returns K optimized theta values for each x_i.
    x: 1-dimensional array (target values)
    K: Number of different initializations for each x_i
    T: Number of SGD steps
    eta: Learning rate
    d: Dimension of theta
    """
    # Store the optimized thetas for each x_i
    all_thetas = []
    n = x.shape[0]
    
    # For each x_i (we assume x is a 2D array, each row corresponds to one x_i)
    
    x_i = x
    
    # Initialize K random starting points for theta
    thetas = [np.random.uniform(lowbound, upbound, d) for _ in range(K)]  # Random initializations
    
    # Store the optimized thetas for each x_i
    optimized_thetas = []

    # Perform SGD optimization for each initial theta
    for k in range(K):
        theta = thetas[k]  # Get initial theta
        theta_av =[]
        for t in range(T):
            grad = gradient(theta, x_i)  # Compute the gradient
            theta = theta - eta * grad  # Update theta using SGD
            theta = project(theta, lowbound=lowbound, upbound=upbound) 
            theta_av.append(theta)
        
        optimized_thetas.append(np.mean(np.array(theta_av),axis = 0))  # Store the final theta after T steps
    
    # Append the list of K optimized thetas for this x_i
    all_thetas.append(np.array(optimized_thetas))
        
    return np.array(all_thetas)  # Shape: (n, K, d)





def first_stage(x, K, T, eta, d):
   
    # Store the optimized thetas for each x_i
    all_thetas = []
    n = x.shape[0]
    print(n)
    # For each x_i (we assume x is a 2D array, each row corresponds to one x_i)
    for i in range(n):
        x_i = x[i]
        
        # Initialize K random starting points for theta
        thetas = [np.random.uniform(lowbound, upbound, d) for _ in range(K)]  # Random initializations
        
        # Store the optimized thetas for each x_i
        optimized_thetas = []

        # Perform SGD optimization for each initial theta
        for k in range(K):
            theta = thetas[k]  # Get initial theta
            theta_av =[]
            for t in range(T):
                grad = gradient(theta, x_i)  # Compute the gradient
                theta = theta - eta * grad  # Update theta using SGD
                theta = project(theta, lowbound=lowbound, upbound=upbound) 
                theta_av.append(theta)
            
            optimized_thetas.append(np.mean(np.array(theta_av),axis = 0))  # Store the final theta after T steps
        
        # Append the list of K optimized thetas for this x_i
        all_thetas.append(np.array(optimized_thetas))
        
    return np.array(all_thetas)  # Shape: (n, K, d)






def plot_theta_distribution(selected_theta):
   
    K, d = selected_theta.shape  # K是解的数量，d是theta的维度

    # 创建图形
    fig, axes = plt.subplots(1, d, figsize=(12, 4))
    if d == 1:
        axes = [axes]  # 如果只有一维，确保axes是一个列表
    
    # 绘制每个维度的分布
    for i in range(d):
        sns.histplot(selected_theta[:, i], bins=100, kde=True, ax=axes[i], color='blue', stat='density', alpha=0.7)
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.show()

    # 如果是二维数据，绘制散点图
    if d == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=selected_theta[:, 0], y=selected_theta[:, 1], color='blue', s=100, alpha=0.7)
        plt.title('Scatter plot of two dimensions')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()


def plot_theta_distribution1(x, y):
  
    K, d = y.shape  # K是解的数量，d是theta的维度

    # 创建图形
    sns.set_theme()
    fig, axes = plt.subplots(1, d, figsize=(12, 4))
    if d == 1:
        axes = [axes]  # 如果只有一维，确保axes是一个列表
    
    # 绘制每个维度的分布
    for i in range(d):
        # 绘制真实的 θ 的分布
        sns.histplot(x[:, i], bins=100, kde=True, ax=axes[i], color='blue', stat='density', alpha=0.7, label='True', fill=True)
        sns.histplot(y[:, i], bins=100, kde=True, ax=axes[i], color='red', stat='density', alpha=0.7, label='Generate-GMM', fill=True)
        #sns.histplot(z[:, i], bins=100, kde=True, ax=axes[i], color='green', stat='density',alpha=0.7, label='Generate-CWGAN', fill=True)
        
        
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_xlabel(r'$\theta{}$'.format(i+1))
        axes[i].set_ylabel('Density')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    # 如果是二维数据，绘制散点图
    if d == 2:
        plt.figure(figsize=(8, 6))
        # 绘制真实的 θ 的散点图
        sns.scatterplot(x=x[:, 0], y=x[:, 1], color='blue', s=5, alpha=0.7, label='True')
        # 绘制生成的 θ 的散点图
        sns.scatterplot(x=y[:, 0], y=y[:, 1], color='red', s=5, alpha=0.7, label='Generate-GMM')
        #sns.scatterplot(x=z[:, 0], y=z[:, 1], color='green', s=5, alpha=0.7, label='Generate-CWGAN')
        plt.title('Scatter plot of two dimensions')
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$')
        plt.legend()
        plt.show()


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

def train_nn_surface_model(X, thetas, objective_values, x_dim, epochs=100, lr=1e-2, batch_size=32):
    # 创建训练数据
    n, K, theta_dim = thetas.shape
    X_repeated = np.repeat(X, K, axis=0)  # 将 X 重复 K 次
    thetas_flat = thetas.reshape(-1, theta_dim)  # 展平 thetas
    inputs = np.hstack([thetas_flat, X_repeated])  # 拼接 theta 和 x
    labels = objective_values.flatten()


   
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


# def get_conditioned_distribution(x_new ,gmm,M,surface_model):
    
#     x_data = np.array([gmm.means_[:1] for gmm in gmm_models])  # Get the mean of each GMM as x_i
#     distances = cdist([x_new], covariates_points, 'euclidean')  # Calculate Euclidean distances
#     closest_idx = np.argmin(distances)  # Find the closest x_i
    
#     # Step 2: Use the GMM corresponding to the closest x_i to generate samples of theta
#     gmm = gmm_models[closest_idx]
#     pi = gmm.weights_  # Mixture weights
#     mu = gmm.means_  # Mean of the Gaussians
#     cov = gmm.covariances_  # Covariance matrices
    
#     # Generate M samples from the GMM (mixture of K Gaussians)
#     samples = np.zeros((M, mu.shape[1]))  # Initialize array to store samples
#     for i in range(M):
#         component = np.random.choice(len(pi), p=pi)  # Select Gaussian component based on mixture weights
#         samples[i] = np.random.multivariate_normal(mu[component], cov[component])
    
#     # Step 3: Evaluate each sample using the objective function
#     scores = []
#     for theta in samples:
#         nn_input = torch.tensor(np.concatenate([theta, x_new]), dtype=torch.float32).unsqueeze(0)
#         score = surface_model(nn_input).item()
#         scores.append(score)
   
#     # Step 4: Select the optimal solution (the one with the lowest objective function value)
#     best_idx = np.argmin(scores)
#     optimal_theta = samples[best_idx]
#     print("generate,ture",min(scores),objective_function(optimal_theta,x_new))
#     # Return the optimal solution
#     return optimal_theta,samples



def get_conditioned_distribution(x_new, gmm_models, M, surface_model):
    """
    Generates conditioned distributions using GMM and evaluates them using a surface model.
    
    Parameters:
    x_new (array): New data point.
    gmm_models (list): List of fitted GMM models.
    covariates_points (array): Points used to determine the closest GMM.
    M (int): Number of samples to generate.
    surface_model (callable): Model used to evaluate the samples.
    k (int): Number of closest covariate points to consider.
    
    Returns:
    optimal_theta (array): The optimal sample corresponding to the lowest objective function score.
    """
    k=10
    # Step 1: Calculate Euclidean distances between x_new and each of the covariates_points
    distances = cdist([x_new], covariates_points, 'euclidean')
    
    # Step 2: Find the indices of the k closest covariate points
    closest_idx = np.argsort(distances[0])[:k]  # Get indices of the k closest points
    
    # Step 3: Use the GMMs corresponding to the closest covariate points to generate samples of theta
    # Here we combine the GMMs from the k closest points (weighted averaging or other method can be used)
    samples = np.zeros((M, gmm_models[0].means_.shape[1]))  # Initialize array to store samples
    for i in range(M):
        # Randomly choose one of the k closest GMM models
        chosen_idx = np.random.choice(closest_idx)
        gmm = gmm_models[chosen_idx]
        
        pi = gmm.weights_  # Mixture weights
        mu = gmm.means_  # Mean of the Gaussians
        cov = gmm.covariances_  # Covariance matrices
        
        # Sample from the selected GMM
        component = np.random.choice(len(pi), p=pi)  # Select Gaussian component based on mixture weights
        samples[i] = np.random.multivariate_normal(mu[component], cov[component])
    
    # Step 4: Evaluate each sample using the objective function
    scores = []
    for theta in samples:
        nn_input = torch.tensor(np.concatenate([theta, x_new]), dtype=torch.float32).unsqueeze(0)
        score = surface_model(nn_input).item()
        scores.append(score)
    
    # Step 5: Select the optimal solution (the one with the lowest objective function value)
    best_idx = np.argmin(scores)
    optimal_theta = samples[best_idx]
    print("GMM:Generated optimal solution with minimum score:", min(scores))
    
    # Return the optimal sample
    return optimal_theta,samples





class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, theta_dim, k):
        super(Generator, self).__init__()
        self.k = k  # 生成 k 个 theta
        # 第一层，接受噪声 z 和条件 x
        self.fc1 = nn.Linear(z_dim + x_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        
        # 最后一层输出 k * theta_dim，即生成 k 个 theta
        self.fc5 = nn.Linear(1024, k * theta_dim)

    def forward(self, z, x):
        # 将噪声 z 和条件 x 拼接
        combined_input = torch.cat([z, x], dim=-1)
       
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        output = self.fc5(x)
       

        # 输出 k 个 theta
        #theta = self.fc5(x).view(-1, self.k, -1)  # 输出为 [batch_size, k, theta_dim]
        theta = output.view(-1, self.k, output.shape[1] // self.k) 
        return theta

class Discriminator(nn.Module):
    def __init__(self, x_dim, theta_dim, k):
        super(Discriminator, self).__init__()
        self.k = k
        self.fc1 = nn.Linear(x_dim + k * theta_dim, 512)  # 输入包含 x 和 k 个 theta
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # 输出一个标量，表示真伪

    def forward(self, theta,x):
        # 拼接样本 x 和条件 theta

        theta_flattened = theta.view(theta.size(0), -1)  # Flatten theta 维度为 [batch_size, k * theta_dim]
        
        combined_input = torch.cat([x, theta_flattened], dim=-1)  # 拼接成 [batch_size, x_dim + k * theta_dim]
        
        x = F.leaky_relu(self.fc1(combined_input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        validity = torch.sigmoid(self.fc4(x))  # 输出真假概率
        
        return validity


# cGAN 训练
# cGAN 训练 (使用 Wasserstein 损失)
def train_cgan(X, thetas, z_dim, n_iterations, batch_size):
    print(X.shape,thetas.shape)
    x_dim = X.shape[1]
    theta_dim = thetas.shape[2]
    t = thetas.shape[1]
    # 初始化 Generator 和 Discriminator
    G = Generator(z_dim, x_dim, theta_dim, t)
    D = Discriminator(theta_dim, x_dim, t)

    # 定义优化器
    g_optimizer = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))

    # 转换为 Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    thetas_tensor = torch.tensor(thetas, dtype=torch.float32)

    for iteration in range(n_iterations):
        for batch_idx in range(0, len(X), batch_size):
            # 获取当前批次数据
            x_batch = X_tensor[batch_idx:batch_idx + batch_size]
            theta_batch = thetas_tensor[batch_idx:batch_idx + batch_size]

            # 训练 Discriminator
            for _ in range(5):  # 判别器多次更新，以增强判别器的稳定性
                z = torch.randn(x_batch.size(0), z_dim)
                fake_thetas = G(z, x_batch)

                D_real = D(theta_batch, x_batch)
                D_fake = D(fake_thetas.detach(), x_batch)

                # Wasserstein 损失
                d_loss = -torch.mean(D_real) + torch.mean(D_fake)

                # 添加梯度惩罚
                gp = gradient_penalty(D, theta_batch, fake_thetas, x_batch)
                d_loss += 10 * gp  # 梯度惩罚项，权重为 10

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # 训练 Generator
            z = torch.randn(x_batch.size(0), z_dim)
            fake_thetas = G(z, x_batch)
            D_fake = D(fake_thetas, x_batch)

            # Wasserstein 生成器损失
            g_loss = -torch.mean(D_fake)

            # 生成器的总损失，包括逼近真实最优解的损失
            lambda_theta, alpha = 0.7, 0.3
            loss_theta = torch.mean((fake_thetas - theta_batch) ** 2)
            total_loss_G = lambda_theta * loss_theta + alpha * g_loss

            g_optimizer.zero_grad()
            total_loss_G.backward()
            g_optimizer.step()

        # 打印损失
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, D Loss: {d_loss.item()}, G Loss: {total_loss_G.item()}")

    return G

# 梯度惩罚函数
# Wasserstein-GP梯度惩罚
def gradient_penalty(D, real_data, fake_data, x):
    batch_size = real_data.size(0)

    # 初始化 epsilon 张量，形状为 [batch_size, 1]
    epsilon = torch.rand(batch_size, 1, 1).to(real_data.device)  # 形状为 [batch_size, 1, 1]

    # 扩展 epsilon 使其形状为 [batch_size, k, theta_dim]
    epsilon = epsilon.expand(-1, real_data.size(1), real_data.size(2))  # 展为 [batch_size, k, theta_dim]

    # 进行插值
    interpolated_data = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated_data.requires_grad_(True)

    # 判别器输出
    validity = D(interpolated_data, x)

    # 计算梯度
    gradients = torch.autograd.grad(outputs=validity, inputs=interpolated_data,
                                    grad_outputs=torch.ones_like(validity),
                                    create_graph=True, retain_graph=True)[0]

    # 计算梯度的L2范数
    gradients = gradients.view(batch_size, -1)
    grad_l2_norm = torch.norm(gradients, p=2, dim=1)
    grad_penalty = torch.mean((grad_l2_norm - 1) ** 2)
    return grad_penalty



def online_stage(x, G, surface_model, z_dim, M):
    
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # 转换为 PyTorch Tensor 并添加批次维度

    # Step 1: Generate Candidate Solutions
    candidates = []

    for _ in range(M):
        z = torch.randn(1, z_dim)  # 从标准正态分布生成噪声 z
        theta_g = G(z, x_tensor)  # 使用训练好的生成器 G 生成候选解
        theta_g = theta_g.detach().numpy().squeeze()
        theta_sample = project(theta_g, lowbound=lowbound, upbound=upbound) 
        candidates.append(theta_sample)  # 转换为 NumPy 数组并存储
   
    # Step 2: Evaluate Candidate Solutions
    scores = []
    candidates1 =[]
    for theta in candidates:
        
        # 使用表面模型 \hat{f} 评估目标值
        if covariate_dim ==1:
            theta = np.atleast_1d(theta)
        
        
        for i in range(K1):
            x = x.reshape(1, -1)  # 将 x 转换为 [1, 2]，即二维
            if K1 ==1:
                theta_i = theta.reshape(1, -1)
            else:
                theta_i = theta[i].reshape(1, -1)
            
            nn_input = torch.tensor(np.concatenate([theta_i, x], axis=1), dtype=torch.float32).unsqueeze(0)
        

            score = surface_model(nn_input).item()
            candidates1.append(theta[i])

        #score = surface_model.predict([np.concatenate([theta, x])])  # 将 theta 和 x 拼接作为输入
            scores.append(score)

    # Step 3: Select the Optimal Solution
    best_index = np.argmin(scores) 
    theta_star = candidates1[best_index]  # 对应的最优解
    #print("fake objection:{},true objection:{}".format((min(scores)),objective_function(theta_star, x)))
    true_objective_value = objective_function(theta_star, x)
    
    print(f"cGAN Fake objective: {min(scores)}")
    #print(f"Fake objective: {min(scores)}, True objective: {true_objective_value}")
    # theta_range = np.linspace(0, upbound, 200)
    # g_values = objective_function(theta,x)

    # Compute true objective function values
    # g_values = objective_function(theta_range, x)
   

    # true_objective_value = objective_function(theta_star, x)
    # print(f"Fake objective: {min(scores)}, True objective: {true_objective_value}")

    #return theta_star, candidates, scores, true_objective_value, global_min_index, local_min_indices, g_values


    return theta_star, candidates



def gaussian_kernel(X, Y, length_scale=1.0):
    X = np.array(X)
    Y = np.array(Y)
    dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
    K = np.exp(-0.5 * dists ** 2)
    return K

# KRR 模型训练函数
def train_krr(x_train, y_train, lambda_param=1e-4, length_scale=1.0):

    K = gaussian_kernel(x_train, x_train, length_scale)
    n = K.shape[0]
    alpha = np.linalg.solve(K + lambda_param * np.eye(n), y_train)
    return alpha, x_train

# KRR 预测函数
def predict_krr(x_train, alpha, x_test, length_scale=1.0):
  
    K_test = gaussian_kernel(x_test, x_train, length_scale)
    y_pred = np.dot(K_test, alpha)
    return y_pred
def krr_fit_and_predict(x_train, w_train, mu_train, sigma_train, x_test, lambda_param=1e-4, length_scale=1.0):
   
    n_samples, d_w = w_train.shape
    _, d_mu, dim_mu = mu_train.shape
    _, d_sigma, dim_sigma, _ = sigma_train.shape

    # 预测 w
    w_pred = np.zeros((x_test.shape[0], d_w))
    alpha_w = []
    for j in range(d_w):
        alpha, x_train_w = train_krr(x_train, w_train[:, j], lambda_param, length_scale)
        w_pred[:, j] = predict_krr(x_train_w, alpha, x_test, length_scale)
        alpha_w.append(alpha)

    # 预测 mu
    mu_pred = np.zeros((x_test.shape[0], d_mu, dim_mu))
    alpha_mu = []
    for i in range(d_mu):
        alpha_mu_row = []
        for j in range(dim_mu):
            alpha, x_train_mu = train_krr(x_train, mu_train[:, i, j], lambda_param, length_scale)
            mu_pred[:, i, j] = predict_krr(x_train_mu, alpha, x_test, length_scale)
            alpha_mu_row.append(alpha)
        alpha_mu.append(alpha_mu_row)

    # 预测 sigma
    sigma_pred = np.zeros((x_test.shape[0], d_sigma, dim_sigma, dim_sigma))
    alpha_sigma = []
    for i in range(d_sigma):
        alpha_sigma_row = []
        for j in range(dim_sigma):
            for k in range(dim_sigma):
                alpha, x_train_sigma = train_krr(x_train, sigma_train[:, i, j, k], lambda_param, length_scale)
                sigma_pred[:, i, j, k] = predict_krr(x_train_sigma, alpha, x_test, length_scale)
                alpha_sigma_row.append(alpha)
        alpha_sigma.append(alpha_sigma_row)

    return w_pred, mu_pred, sigma_pred






# def sample_for_single_test_point(w_pred, mu_pred, sigma_pred, num_samples=1):
#     d_w = w_pred.shape[0]
#     dim_mu = mu_pred.shape[1]
#     weights = w_pred / np.sum(w_pred)
    
#     # if weights.ndim != 1:
#     #     raise ValueError(f"Expected weights to be 1-dimensional, got {weights.ndim}-dimensional")
#     component_indices = np.random.choice(d_w, size=num_samples, p=weights)
    
#     samples = []
#     for idx in component_indices:
#         mean = mu_pred[idx]
#         cov = sigma_pred[idx]
#         sample = np.random.multivariate_normal(mean, cov)
#         samples.append(sample)
#     return np.array(samples)



def full_process(x_train, w_train, mu_train, sigma_train, x_new,surface_model):
   

    print("Online stage: Predicting GMM parameters for new input...")
    
    
    w_pred, mu_pred, sigma_pred = krr_fit_and_predict(x_train, w_train, mu_train, sigma_train, x_new)
    print("Online stage complete.")
    print(w_pred.shape, mu_pred.shape, sigma_pred.shape)
    w_pred = np.squeeze(w_pred, axis=0)  # (10,)
    mu_pred = np.squeeze(mu_pred, axis=0)  # (10, 2)
    sigma_pred = np.squeeze(sigma_pred, axis=0)  # (10, 2, 2)
    
    # samples = np.zeros((1000, mu_pred.shape[1]))

    samples = sample_for_single_test_point(w_pred, mu_pred, sigma_pred, num_samples=1000)
    
    # w_pred = w_pred / np.sum(w_pred) 
    # for i in range(1000):
    #     m = np.random.choice(mu_pred.shape[0], p=w_pred)
    #     samples[i] = np.random.multivariate_normal(mean=mu_pred[m], cov=sigma_pred[m])
        
    # Step 4: Evaluate each sample using the objective function
    scores = []
    for theta in samples:
        print(theta.shape,x_new.shape)
        
        nn_input = torch.tensor(np.concatenate([theta.reshape(1, -1), x_new], axis=1), dtype=torch.float32).unsqueeze(0)
        score = surface_model(nn_input).item()
        scores.append(score)
    
    # Step 5: Select the optimal solution (the one with the lowest objective function value)
    best_idx = np.argmin(scores)
    optimal_theta = samples[best_idx]
    print("GMM:Generated optimal solution with minimum score:", min(scores))
    
    # Return the optimal sample
    return optimal_theta,samples
    
    
    #return w_pred, mu_pred, sigma_pred


def offline_training(gmms, n_clusters):
   
    all_means = []
    all_weights = []
    all_covariances = []

    for gmm in gmms:
        all_means.append(gmm.means_)  # 提取 means_
        all_weights.append(gmm.weights_)  # 提取 weights_
        all_covariances.append(gmm.covariances_)  # 提取 covariances_

    # 将所有 GMM 的 means_ 堆叠起来
    all_means = np.vstack(all_means)  # (总分量数, 特征维度)
    all_weights = np.hstack(all_weights)  # (总分量数,)
    all_covariances = np.vstack(all_covariances)  # (总分量数, 特征维度, 特征维度)

    # 对所有分量的 means_ 进行 KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(all_means)

    # 初始化每个簇的参数
    cluster_info = {"means": [], "weights": [], "covariances": []}
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]  # 属于该簇的分量索引

        # 计算簇的均值、权重和协方差的平均值
        cluster_means = all_means[cluster_indices]
        cluster_weights = all_weights[cluster_indices]
        cluster_covariances = all_covariances[cluster_indices]

        cluster_info["means"].append(np.mean(cluster_means, axis=0))
        cluster_info["weights"].append(np.mean(cluster_weights))
        cluster_info["covariances"].append(np.mean(cluster_covariances, axis=0))

    # 转换为 numpy 数组
    cluster_info["means"] = np.array(cluster_info["means"])
    cluster_info["weights"] = np.array(cluster_info["weights"])
    cluster_info["covariances"] = np.array(cluster_info["covariances"])

    plt.figure(figsize=(8, 6))
    plt.scatter(all_means[:, 0], all_means[:, 1], c=labels, cmap='viridis', s=10, marker='x', label='GMM Means')
    for i, label in enumerate(labels):
        plt.text(all_means[i, 0] + 0.05, all_means[i, 1] + 0.05, f' {label}', color='black', fontsize=6)
    plt.scatter(cluster_info["means"][:, 0], cluster_info["means"][:, 1], c='red', s=20, marker='o', label='Cluster Centers')

    plt.legend()
    plt.grid(True)
    plt.show()

   
    return cluster_info, kmeans

def predict_and_aggregate_kmeans(kmeans_model, all_means, all_weights, all_covariances):
    
    k, n_components, dim_mu = all_means.shape
    X_means = all_means.reshape(-1, dim_mu)  # (k * n_components, dim_mu)
    predicted_labels = kmeans_model.predict(X_means)  # 形状为 (k * n_components,)
    predicted_labels = predicted_labels.reshape(k, n_components)  # (k, n_components)
    unique_labels = np.unique(predicted_labels)
    n_clusters = len(unique_labels)
    averaged_means = []
    averaged_weights = []
    averaged_covariances = []
    for label in unique_labels:
        cluster_indices = np.where(predicted_labels == label)
        cluster_means = all_means[cluster_indices]
        cluster_weights = all_weights[cluster_indices]
        cluster_covariances = all_covariances[cluster_indices]
        averaged_means.append(np.mean(cluster_means, axis=0))
        averaged_weights.append(np.mean(cluster_weights))
        averaged_covariances.append(np.mean(cluster_covariances, axis=0))
    averaged_means = np.array(averaged_means)  # (n_clusters, dim_mu)
    averaged_weights = np.array(averaged_weights)  # (n_clusters,)
    averaged_covariances = np.array(averaged_covariances)  # (n_clusters, dim_mu, dim_mu)


    print(averaged_means.shape,averaged_weights.shape,averaged_covariances.shape)
    return averaged_means, averaged_weights, averaged_covariances


def knn_predict1(kmeans_model,covariates_points,x_new, gmm_models, n_components1,surface_model,k =5):
    
    distances = pairwise_distances(covariates_points, x_new.reshape(1, -1)).flatten()
    knn_indices = np.argsort(distances)[:k]
    knn_indices = knn_indices.astype(int).flatten()
    
    gmm_models = [gmm_models[idx] for idx in knn_indices.flatten()]
    all_means = []
    all_weights = []
    all_covariances = []
    for gmm in gmm_models:
        all_means.append(gmm.means_)  
        all_weights.append(gmm.weights_)  
        all_covariances.append(gmm.covariances_)  
    all_weights = np.array(all_weights)  # (k, n_components)
    all_means = np.array(all_means)  # (k, n_components, dim_mu)
    all_covariances = np.array(all_covariances) # (k, n_components, dim_mu, dim_mu)
    
    mu_pred, w_pred,  sigma_pred = predict_and_aggregate_kmeans(kmeans_model, all_means, all_weights, all_covariances)
    averaged_gmm = GaussianMixture(n_components=n_components1, covariance_type='full')
    averaged_gmm.means_ = mu_pred
    averaged_gmm.weights_ = w_pred 
    averaged_gmm.covariances_ = sigma_pred 
    
    samples, labels = averaged_gmm.sample(n_samples=2000)
    samples = np.array([project(x, lowbound=lowbound, upbound=upbound) for x in samples])
    #samples = sample_for_single_test_point(w_pred, mu_pred, sigma_pred, num_samples=1000)
    
    #Step 4: Evaluate each sample using the objective function
    scores = []
    for theta in samples:
        #print(theta.shape,x_new.shape)
        
        nn_input = torch.tensor(np.concatenate([theta, x_new]), dtype=torch.float32).unsqueeze(0)
        score = surface_model(nn_input).item()
        scores.append(score)
    
    # Step 5: Select the optimal solution (the one with the lowest objective function value)
    best_idx = np.argmin(scores)
    optimal_theta = samples[best_idx]
    print("GMM:Generated optimal solution with minimum score:", min(scores))

    return optimal_theta, samples









if __name__ == '__main__':
    np.random.seed(42)
    lowbound = -2.5
    upbound = 2.5
    eta = 0.1  
    covariate_dim = 2
    n = 100
    K1 = 20
    T = 20
    noise_std = 1
    z_dim = 10  # 噪声维度
    M = 1000
    A = generate_symmetric_matrix(covariate_dim, correlation_strength=0.01)
    covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
    thetas = first_stage(covariates_points, K1, T, eta, covariate_dim)
    
    objectives = np.zeros((n,K1))

    for i,x in enumerate(covariates_points):
        objectives[i,:] = [objective_function(theta, x) for theta in thetas[i]]
    
    nn_model = train_nn_surface_model(covariates_points, thetas, objectives, covariate_dim)
    #plot_theta_distribution(selected_theta)
    theta1_values = np.linspace(lowbound, upbound, 100)  # 生成 theta1 的取值范围
    theta2_values = np.linspace(lowbound, upbound, 100)  # 生成 theta2 的取值范围
    theta1_grid, theta2_grid = np.meshgrid(theta1_values, theta2_values)  # 生成网格

    # 计算目标函数值
    x0 = covariates_points[0]
    objective_values = np.zeros_like(theta1_grid)
    for i in range(len(theta1_values)):
        for j in range(len(theta2_values)):
            theta = np.array([theta1_grid[i, j], theta2_grid[i, j]])
            objective_values[i, j] = objective_function(theta, x0)

    
    # plt.figure(figsize=(8, 6))
    # contour = plt.contour(theta1_grid, theta2_grid, objective_values, 20, cmap='viridis')
    # sns.scatterplot(x=selected_theta[:, 0], y=selected_theta[:, 1], color='blue', s=100, alpha=0.7)
    # plt.colorbar(contour)
    # plt.title("Objective Function Contour Plot")
    # plt.xlabel(r'$\theta_1$')
    # plt.ylabel(r'$\theta_2$')
    # plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta1_grid, theta2_grid, objective_values, cmap='viridis')

    # 设置标题和标签
    ax.set_title("Objective Function 3D Surface Plot")
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel('Objective Function Value')

    plt.show()




    w_train = []
    mu_train = []
    sigma_train = []
    # Train one GMM per x_i
    gmm_models = []
    n_components1 = 4
    for i in range(n):
        # For each x_i, the corresponding theta samples are theta_samples[i]
        X = thetas[i]  # Shape (K, 2) for each x_i
        
        # Initialize GMM model
        gmm = GaussianMixture(n_components=n_components1, covariance_type='full', random_state=42)
        # Fit the GMM on the theta samples for this x_i
        gmm.fit(X)
      
        gmm_models.append(gmm)


    

        
    # w_train = np.array(w_train)
    # mu_train = np.array(mu_train)
    # sigma_train = np.array(sigma_train)


    # n_iterations = 100
    # batch_size = 16
    # G = train_cgan(covariates_points, thetas, z_dim, n_iterations, batch_size)
    # print("Training completed. Generator is ready to generate samples conditioned on x.")
    # x_new = np.array([1.5, 2.0])
    # theta_star, candidates = online_stage(x_new, G, nn_model, z_dim, M) 
    # candidates = np.array(candidates).reshape(-1,covariate_dim)
    # print(np.array(candidates).shape)




    cluster_info, kmeans_model = offline_training(gmm_models, n_components1)

    
    print("offline over")

    #w_pred, mu_pred, sigma_pred = full_process(X, w_train, mu_train, sigma_train, x_new)
    #best_theta, generated_theta = full_process(covariates_points, w_train, mu_train, sigma_train, x_new,nn_model)
    x_new = np.array([1.5, 2.0])
    
    #generated_theta,generated_theta1,generated_theta5=  knn_predict(covariates_points, w_train, mu_train, sigma_train, x_new, 1)
   
    best_theta, generated_theta = knn_predict1(kmeans_model, covariates_points,x_new, gmm_models, n_components1,nn_model,k =5)

    

    

    #best_theta, generated_theta = get_conditioned_distribution(x_new ,gmm_models,num_samples,nn_model)
    
    #print("Generated theta:", generated_theta.shape)
    #generated_theta = generated_theta.reshape(-1,covariate_dim)
    thetas_true = first_stage1(x_new, 3000, 100, eta, covariate_dim)
    print(thetas_true.shape)
    thetas_true = thetas_true.reshape(-1,covariate_dim)

    print("true best objection",min([objective_function(theta, x_new) for theta in thetas_true]))
    print("generate best objection",objective_function(best_theta, x_new))
 
    plot_theta_distribution1(thetas_true,generated_theta)
   
    
    theta1_values = np.linspace(lowbound, upbound, 100)  # 生成 theta1 的取值范围
    theta2_values = np.linspace(lowbound, upbound, 100)  # 生成 theta2 的取值范围
    theta1_grid, theta2_grid = np.meshgrid(theta1_values, theta2_values)  # 生成网格

    # 计算目标函数值
    x0 = x_new
    objective_values = np.zeros_like(theta1_grid)
    for i in range(len(theta1_values)):
        for j in range(len(theta2_values)):
            theta = np.array([theta1_grid[i, j], theta2_grid[i, j]])
            objective_values[i, j] = objective_function(theta, x0)

    sns.set_theme()
    plt.figure(figsize=(8, 6))
   
    contour = plt.contour(theta1_grid, theta2_grid, objective_values, 20, cmap='viridis')
    sns.scatterplot(x=thetas_true[:, 0], y=thetas_true[:, 1], color='blue', s=5, alpha=0.7,label='True')
    sns.scatterplot(x=generated_theta[:, 0], y=generated_theta[:, 1], color='red', s=5, alpha=0.7,label='Generate-GMM')
    #sns.scatterplot(x=np.array(candidates)[:, 0], y=np.array(candidates)[:, 1], color='green', s=5, alpha=0.7,label='Generate-CWGAN')
    plt.colorbar(contour)
    plt.title("Objective Function Contour Plot")
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.show()



    test_points = [np.random.uniform(lowbound, upbound, covariate_dim) for _ in range(5)]

    fake =  []
    true1 = []
    for x_new in test_points:
       
        best_theta, generated_theta = knn_predict1(kmeans_model, covariates_points,x_new, gmm_models, n_components1,nn_model,k =5)

    
        
        #best_theta, generated_theta = get_conditioned_distribution(x_new ,gmm_models,num_samples,nn_model)
        
      
        thetas_true = first_stage1(x_new, 3000, 100, eta, covariate_dim)
       
        thetas_true = thetas_true.reshape(-1,covariate_dim)
        print("true best objection",min([objective_function(theta, x_new) for theta in thetas_true]))
        print("generate best objection",objective_function(best_theta, x_new))
 


        # print("true best objection",min([objective_function(theta, x_new) for theta in thetas_true]))
        # print("generate best objection",objective_function(best_theta, x_new))
        true1.append(min([objective_function(theta, x_new) for theta in thetas_true]))
        fake.append(objective_function(best_theta, x_new))

    sns.set()

    # 绘制散点图
    plt.figure(figsize=(10, 6))
   
    sns.lineplot(x=range(len(fake)), y=fake, label="generate-GMM", color='red', marker='o')
    sns.lineplot(x=range(len(true1 )), y=true1 , label="true ", color='blue', marker='s')
    # 添加标题和标签
    
    plt.xlabel("n", fontsize=12)
    plt.ylabel("value", fontsize=12)
    plt.legend()
    plt.show()