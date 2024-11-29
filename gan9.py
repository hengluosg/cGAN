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


# def objective_function(theta, x):

#     return np.sum(10 * (np.sin(0.05 * np.pi * (theta - x)) ** 6) / (2 ** (2*(((theta - x) - 90) / 50) ** 2))) 
    

# def objective_function(theta, x):

#     return  np.sum((theta - x) ** 2)

# def objective_function(theta, x):
#     term1 = -10 * (np.sin(0.9 * np.pi * (theta - x)) ** 6) / (2 ** (2*((theta - x) / 2) ** 2)) + 10
#     return term1

# def objective_function(theta, x):
#     term1 = (theta - x)**3 - 3*(theta - x)**2 + 2
#     return term1
def objective_function(theta, x):
    term1 = np.sin(2 * np.pi * theta) + 0.5 * (theta - x) ** 2
    return term1






# def objective_function(theta, x):
#     term1 = -25 * (np.sin(0.5 * np.pi * (theta - x)) ** 6) / (2 ** (1.5 * ((theta - x) / 10) ** 2)) + 25
#     return term1
# Grid sampling function
def generate_covariates(point,d):
    n = int(round(point ** (1.0 / d))) + 1
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]
    return grid_points

def first_stage(n,t, x_dim, K):
    X = generate_covariates(n, x_dim)  # 生成 covariates 点
   
    objective_values = []
    best_theta_total = []
    X1 =[]
    for x in X:
        for _ in range(t):
            X1.append(x)
            solutions = np.random.uniform(lowbound, upbound, size=(K, x_dim))
            objectives = [objective_function(theta, x) for theta in solutions]
            best_theta = solutions[np.argmin(objectives)]  # 选择最优解
            best_theta_total.append(best_theta)
            objective_values.append(np.min(objectives))
    #X, np.array(thetas), np.array(objective_values) fit surface
    #X, best_theta #cGAN
    return np.array(X1), np.array(objective_values), np.array(best_theta_total)


def on_stage_test(n, x_test, K):
    best_theta_total = []
    ##solutions = random_points(K, x_dim) # 随机生成 K 个解
    for i in range(n):
        solutions = np.random.uniform(lowbound, upbound, size=(K, x_dim)) 
        objectives = [objective_function(theta, x_test) for theta in solutions]
        best_theta = solutions[np.argmin(objectives)]  # 选择最优解
        best_theta_total.append(best_theta)
        
        
    return np.array(best_theta_total)





class SurfaceModel(nn.Module):
    def __init__(self, input_dim):
        super(SurfaceModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个目标值
        )
    
    def forward(self, x):
        return self.net(x)

# 训练神经网络
def train_nn_surface_model(X, thetas, objective_values, x_dim, epochs=100, lr=1e-3, batch_size=32):
    # 创建训练数据
    theta_dim = thetas.shape[1]
    # X_repeated = np.repeat(X, K, axis=0)  # 将 X 重复 K 次
    # thetas_flat = thetas.reshape(-1, theta_dim)  # 展平 thetas
    inputs = np.hstack([thetas, X])  # 拼接 theta 和 x
    labels = objective_values.flatten()  # 展平目标值

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
# cGAN 训练 (使用 Wasserstein 损失)
def train_cgan(X, thetas, z_dim, n_iterations, batch_size):
    x_dim = X.shape[1]
    theta_dim = thetas.shape[1]

    # 初始化 Generator 和 Discriminator
    G = Generator(z_dim, x_dim, theta_dim)
    D = Discriminator(theta_dim, x_dim)

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
            lambda_theta, alpha = 0.99, 0.01
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
def gradient_penalty(D, real_samples, fake_samples, x_batch):
    alpha = torch.rand(real_samples.size(0), 1)
    alpha = alpha.expand(real_samples.size())
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = D(interpolates, x_batch)
    fake = torch.ones(d_interpolates.size(), requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def plot_results_subplots(test_points, candidate_solutions, candidate_scores, true_objective_values, global_min_indices, local_min_indices, theta_range, g_values):
   
    num_test_points = len(test_points)
    cols = 2  # 每行的子图数量
    rows = (num_test_points + cols - 1) // cols  # 计算总行数
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = axes.flatten()  # 将轴展平成一维数组，便于索引

    for i, (x, candidates, scores, true_values, global_min_idx, local_min_idx, g_val) in enumerate(
        zip(test_points, candidate_solutions, candidate_scores, true_objective_values, global_min_indices, local_min_indices, g_values)):
        
        ax = axes[i]
        index = np.argmin(scores)
        # 绘制目标函数曲线
        ax.plot(theta_range, g_val, label='g(θ)', alpha=0.8)
        
        # 绘制候选解的拟合值
        ax.scatter(candidates, scores, label='Fit g(θ)', color='g', alpha=0.6)
        
        # 绘制局部极小值
        ax.scatter(theta_range[local_min_idx], g_val[local_min_idx], label='Local Minima', color='y', marker='x')
        
        # 绘制全局最小值
        ax.scatter(theta_range[global_min_idx], g_val[global_min_idx], label='Global Minimum', color='c', marker='o')
        
        # 设置标题和标签
        ax.set_title('Test Point x = {}'.format(x[0]))
        ax.set_xlabel('θ')
        ax.set_ylabel('g(θ)')
        ax.legend()
    
    # 如果子图数量不足，将多余的子图隐藏
    for j in range(len(axes)):
        if j >= num_test_points:
            axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('gan9.png')
    #plt.show()

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
        if x_dim ==1:
            theta = np.atleast_1d(theta)
        
        nn_input = torch.tensor(np.concatenate([theta, x]), dtype=torch.float32).unsqueeze(0)
        score = surface_model(nn_input).item()


        #score = surface_model.predict([np.concatenate([theta, x])])  # 将 theta 和 x 拼接作为输入
        scores.append(score)

    # Step 3: Select the Optimal Solution
    best_index = np.argmin(scores) 
    theta_star = candidates[best_index]  # 对应的最优解
    print("fake objection:{},true objection:{}".format((min(scores)),objective_function(theta_star, x)))
    
    theta_range = np.linspace(0, upbound, 200)
    g_values = objective_function(theta,x)

    # Compute true objective function values
    g_values = objective_function(theta_range, x)
    local_min_indices, _ = find_peaks(-g_values)
    global_min_index = np.argmin(g_values)

    true_objective_value = objective_function(theta_star, x)
    print(f"Fake objective: {min(scores)}, True objective: {true_objective_value}")

    return theta_star, candidates, scores, true_objective_value, global_min_index, local_min_indices, g_values

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



def plot_optimal_solution_distributions_per_dim(true_theta, generated_theta, stage):
   
    theta_dim = true_theta.shape[1]

    # 确保 generated_theta 是二维的
    if generated_theta.ndim == 1:
        generated_theta = np.expand_dims(generated_theta, axis=1)

    plt.figure(figsize=(15, theta_dim * 4))
    
    for dim in range(theta_dim):
        plt.subplot(theta_dim, 2, dim * 2 + 1)
        plt.hist(true_theta[:, dim], bins=30, alpha=0.5, label='True Optimal Solutions', color='g')
        plt.hist(generated_theta[:, dim], bins=30, alpha=0.5, label='Generated Solutions', color='b')
        plt.xlabel(f'Theta Dimension {dim + 1} Values')
        plt.ylabel('Frequency')
        plt.title(f'{stage} Stage: Distribution in Dimension {dim + 1}')
        plt.legend()


        plt.subplot(theta_dim, 2, dim * 2 + 2)
        residuals = generated_theta[:, dim] - true_theta[:, dim]
        plt.hist(residuals, bins=30, alpha=0.5, color='purple', label='Residuals (Generated - True)')
        plt.xlabel(f'Residuals (Dim {dim + 1})')
        plt.ylabel('Frequency')
        plt.title(f'{stage} Stage: Residual Distribution in Dimension {dim + 1}')
        plt.axvline(0, color='r', linestyle='--', label='Zero Residual')
        plt.legend()


        # plt.subplot(theta_dim, 2, dim * 2 + 2)
        # plt.scatter(true_theta[:, dim], generated_theta[:, dim], alpha=0.6)
        # plt.xlabel(f'True Optimal Solutions (Dim {dim + 1})')
        # plt.ylabel(f'Generated Optimal Solutions (Dim {dim + 1})')
        # plt.title(f'{stage} Stage: Scatter Plot of True vs Generated Solutions (Dim {dim + 1})')
        # plt.plot([np.min(true_theta[:, dim]), np.max(true_theta[:, dim])], 
        #          [np.min(true_theta[:, dim]), np.max(true_theta[:, dim])], 'r--')
    
    plt.tight_layout()
    plt.show()


from scipy.stats import gaussian_kde
def plot_candidates_distribution(candidates, solutions):
   
    
    candidates = np.array(candidates)
    solutions = np.array(solutions)
    if solutions.ndim == 2 and solutions.shape[1] == 1:
        solutions = solutions.flatten()  # 将二维数组展平成一维数组
    plt.figure(figsize=(8, 6))

    plt.hist(candidates, bins=100, density=True, color='lightgreen', edgecolor='black', alpha=0.6, label='cGAN Histogram')
    plt.hist(solutions, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.6, label='random sample Histogram')
    # 计算并绘制候选解的核密度估计 (KDE)，颜色设置为 darkorange
    kde_candidates = gaussian_kde(candidates)
    x_range = np.linspace(np.min(candidates), np.max(candidates), 1000)
    plt.plot(x_range, kde_candidates(x_range), color='darkorange', lw=2, label='cGAN KDE')

    # 计算并绘制解决方案的核密度估计 (KDE)，颜色设置为 blue
    kde_sol = gaussian_kde(solutions)
    x_range = np.linspace(np.min(solutions), np.max(solutions), 1000)
    plt.plot(x_range, kde_sol(x_range), color='blue', lw=2, label='random sample KDE')

    # 添加图表标题和标签
    plt.title('cGAN and random sample Solutions Distribution ', fontsize=14)
    plt.xlabel('θ', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    

    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("distribution.png")


if __name__ == "__main__":
    # 参数定义
    #n = 10  # covariate 点数量
    s,t = 5, 6
    n_values = [2**i for i in range(s, t)]

    x_dim = 1  # covariate 维度
    K = 5  # 每个点的解的数量
    z_dim = 10  # 噪声维度
    M = 10000  # 候选解数量
    n_iterations = 1000  # cGAN 训练迭代次数
    batch_size = 64  # 批次大小
    lowbound = 0
    upbound = 2
    t = 50
    # First Stage: 近似最优解的选择与表面拟合
    #X, thetas, krr = first_stage(n, x_dim, K)

    mse1 =[]
    for n in n_values:
        
        X,  objective_values, best_theta = first_stage(n,t, x_dim, K)
        print(X[:25])
        print(X.shape,objective_values.shape,best_theta.shape)
        print("Training neural network surface model...")
        nn_model = train_nn_surface_model(X, best_theta, objective_values, x_dim)
        print(best_theta.shape)
        G = train_cgan(X, best_theta, z_dim, n_iterations, batch_size)
        print("Training completed. Generator is ready to generate samples conditioned on x.")
        generated_theta_train = []
        for x in X:
            generated_theta = G(torch.randn(1, z_dim), torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            generated_theta_train.append(generated_theta.detach().numpy().squeeze())
        generated_theta_train = np.array(generated_theta_train)
        #plot_optimal_solution_distributions_per_dim(best_theta, generated_theta_train, stage="Training")



        theta_range = np.linspace(lowbound, upbound, 200) 
        num_test_points = 1
        candidate_solutions = []
        candidate_scores = []
        true_objective_values = []
        global_min_indices = []
        local_min_indices = []
        g_values = []
        test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, x_dim))
        
        for x in test_points:
            
            
            result = online_stage(x, G, nn_model, z_dim, M)
            theta_star, candidates, scores, true_value, global_min_idx, local_min_idx, g_val = result

            candidate_solutions.append(candidates)
            candidate_scores.append(scores)
            true_objective_values.append(true_value)
            global_min_indices.append(global_min_idx)
            local_min_indices.append(local_min_idx)
            g_values.append(g_val)

            print("global and local",g_val[global_min_idx],g_val[local_min_idx])
        


            solutions =  on_stage_test(10000, x, K)
            
            plot_candidates_distribution(candidates,solutions)






        plot_results_subplots(test_points, candidate_solutions, candidate_scores, true_objective_values, global_min_indices, local_min_indices, theta_range, g_values)
    # sns.set()
    # sns.lineplot(x=n_values, y=mse1, marker='o')
    # plt.title('MSE vs n_values', fontsize=16)
    # plt.xlabel('n_values (Number of Covariate Points)', fontsize=14)
    # plt.ylabel('MSE (Mean Squared Error)', fontsize=14)
    # plt.xticks(ticks=n_values, labels=[r"$2^{{{}}}$".format(i) for i in range(s, t)])
    # plt.grid(True)
    # plt.show()







