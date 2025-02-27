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



def objective_function(theta, x):
    term1 = np.sum(np.sin(np.pi * theta) + 0.2 * (theta - x) ** 2)
    return term1

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
    """
    Computes the gradient of the objective function with respect to theta.
    theta: d-dimensional vector
    x: d-dimensional vector (target values)
    """
    grad_term1 = np.pi * np.cos(np.pi * theta)  # Gradient of sin(pi*theta)
    grad_term2 = 0.4 * (theta - x)  # Gradient of 0.2 * (theta - x)^2
    noise = np.random.normal(0, noise_std, covariate_dim)
    return grad_term1 + grad_term2 + noise



# SGD optimization function
def project(theta, lowbound, upbound):
    """
    Projects theta back into the valid domain [lowbound, upbound] element-wise.
    """
    return np.clip(theta, lowbound, upbound)


def first_stage(x, K, T, eta, d):
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
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$')
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_theta_distribution1(x, y):
    """ 参数：
    - x: 真实的 θ 数据 (二维数组, shape = [K, d])
    - y: 生成的 θ 数据 (二维数组, shape = [K, d])
    """
    K, d = y.shape  # K是解的数量，d是theta的维度

    # 创建图形
    fig, axes = plt.subplots(1, d, figsize=(12, 4))
    if d == 1:
        axes = [axes]  # 如果只有一维，确保axes是一个列表
    
    # 绘制每个维度的分布
    for i in range(d):
        # 绘制真实的 θ 的分布
        sns.histplot(x[:, i], bins=100, kde=True, ax=axes[i], color='blue', stat='density', alpha=0.7, label='True')
        # 绘制生成的 θ 的分布
        sns.histplot(y[:, i], bins=100, kde=True, ax=axes[i], color='red', stat='density', alpha=0.7, label='Generated')
        
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_xlabel(r'$\theta_1$')
        axes[i].set_ylabel(r'$\theta_2$')
        
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    # 如果是二维数据，绘制散点图
    if d == 2:
        plt.figure(figsize=(8, 6))
        # 绘制真实的 θ 的散点图
        sns.scatterplot(x=x[:, 0], y=x[:, 1], color='blue', s=10, alpha=0.7, label='True')
        # 绘制生成的 θ 的散点图
        sns.scatterplot(x=y[:, 0], y=y[:, 1], color='red', s=10, alpha=0.7, label='Generated')
        
        plt.title('Scatter plot of two dimensions')
        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_1$')
        plt.legend()
        plt.show()
    

def get_conditioned_distribution(x_new ,gmm,num_samples):
    
    # 计算责任概率
    means = gmm.means_
    # print(means.shape,np.repeat(x_new, gmm.n_components, axis=0).shape)
    new_data_for_predict = np.concatenate([np.repeat(x_new, gmm.n_components, axis=0), means[:, 2:]], axis=1)

    # 使用 GMM 的 predict_proba 方法计算每个组件的责任概率
    responsibility = gmm.predict_proba(new_data_for_predict)  # (1, K)
    

    print(responsibility.shape)
    print(responsibility[0])
    # Step 3: 基于责任概率加权计算 θ 的条件分布
    # 对每个 θ 样本进行采样
    K = gmm.n_components

    theta_samples = []

    # 抽样次数
    

    

    for _ in range(num_samples):
        sampled_thetas = []
        for k in range(K):
            # 获取第 k 个高斯分布的均值和协方差
            mean = gmm.means_[k, 2:]  # 获取每个组件中对应 theta 的均值 (假设后两列为 theta)
            cov = gmm.covariances_[k, 2:, 2:]  # 获取每个组件的协方差矩阵（假设后两列为 theta 的协方差）

            # 从该高斯分布中抽样 theta
            theta_sample = np.random.multivariate_normal(mean, cov)

            sampled_thetas.append(theta_sample)
        
        # Step 3: 根据责任概率从抽样的 theta 值中加权选择
        sampled_thetas = np.array(sampled_thetas)
        weighted_theta_sample = np.dot(responsibility, sampled_thetas)  # 责任概率加权选择样本
        theta_samples.append(weighted_theta_sample)

        # Convert list to numpy array for convenience
    theta_samples = np.array(theta_samples)  
            
    
    return np.array(theta_samples)

if __name__ == '__main__':
    np.random.seed(42)
    lowbound = 0
    upbound = 4
    eta = 0.1  
    covariate_dim = 2
    n = 50
    K1 = 10
    T = 100
    noise_std = 1
    covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
    thetas = first_stage(covariates_points, K1, T, eta, covariate_dim)
    print(covariates_points.shape, thetas.shape)
    selected_theta = thetas[0]
    plot_theta_distribution(selected_theta)


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

    

    plt.figure(figsize=(15, 15))
    contour = plt.contour(theta1_grid, theta2_grid, objective_values, 20, cmap='viridis')
    sns.scatterplot(x=selected_theta[:, 0], y=selected_theta[:, 1], color='blue', s=100, alpha=0.7)
    plt.colorbar(contour)
    plt.title("Objective Function Contour Plot")
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.show()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面图
    ax.plot_surface(theta1_grid, theta2_grid, objective_values, cmap='viridis')

    # 设置标题和标签
    #ax.set_title("Objective Function 3D Surface Plot")
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel('Objective Function Value')

    plt.show()



    from sklearn.mixture import GaussianMixture
    



    # K = 9
    # # 训练 GMM：将每个 x_i 对应的 K 个解作为数据样本来训练 GMM
    # gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42)

    # # 训练时，theta_values 的形状是 (n, K, d)，需要先将其展平成一个二维数组
    # theta_flattened = thetas.reshape(-1, covariate_dim)  # (n * K, d)
    



    combined_data = []
    for i in range(n):
        for k in range(K1):
            combined_data.append(np.concatenate([covariates_points[i], thetas[i, k,:]]))  # 将每个 x 和对应的 θ 连接起来
    combined_data = np.array(combined_data)

    print(combined_data.shape)

    max_clusters = n * 9
    bic_scores = []
    data = combined_data
    for n_clusters in range(1, max_clusters+1):
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))  # 计算 BIC
    best_n_clusters = np.argmin(bic_scores) + 1
    print("最佳簇数:", best_n_clusters)

    # 可视化 BIC 曲线
    plt.plot(range(1, max_clusters+1), bic_scores)
    plt.xlabel('簇数')
    plt.ylabel('BIC')
    plt.title('BIC Score vs Number of Clusters')
    plt.show()




    # 训练 GMM
    #gmm.fit(combined_data)
    gmm = GaussianMixture(n_components=best_n_clusters, covariance_type='full', random_state=42)
    # 获取 GMM 的参数
    gmm.fit(combined_data)
    print("Means:", gmm.means_.shape)
    print("Covariances:", gmm.covariances_.shape)
    print("Weights:", gmm.weights_.shape)
    
    # labels = gmm.predict(combined_data)

    
    x_new = np.array([1.5, 2.0]) # 新的输入
    num_samples = 1000
    x_new =x_new.reshape(1,covariate_dim)
    generated_theta = get_conditioned_distribution(x_new, gmm, num_samples)

    print("Generated theta:", generated_theta.shape)
    generated_theta = generated_theta.reshape(-1,covariate_dim)
    print(x_new)
    thetas_true = first_stage(x_new, 1000, T, eta, covariate_dim)
    print(thetas_true.shape)
    thetas_true = thetas_true.reshape(-1,covariate_dim)
    plot_theta_distribution1(thetas_true,generated_theta)
    # if covariate_dim == 2:
    #     plt.figure(figsize=(8, 6))
    #     sns.scatterplot(x=generated_theta[:, 0], y=generated_theta[:, 1], color='blue', s=100, alpha=0.7)
    #     plt.title('Scatter plot of two dimensions')
    #     plt.xlabel('Dimension 1')
    #     plt.ylabel('Dimension 2')
    #     plt.show()