import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import kv
import seaborn as sns
import pandas as pd
from itertools import product


def F(theta, x ,xi):
    
    return np.sum((theta - x)**2) + xi

def f(theta, x, num_samples):
    np.random.seed(1)
    xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
    #xi_samples = np.random.normal(0, 0, num_samples)
    costs = [F(theta, x, xi) for xi in xi_samples]
    return np.mean(costs)


def grad_f(theta, x, num_samples):
    
    xi_samples = np.random.normal(0, 1, num_samples)  # 随机样本
    grads = [2*(theta - x) + xi for xi in xi_samples]  # F 对 theta 的导数
    return np.mean(grads, axis=0)  # 对每个样本的梯度取平均
# SGD with Learning Rate Decay

def sgd_with_lr_decay( theta_0, x, eta_0, gamma, T):
    theta = theta_0
    
    theta_values = []  # 记录每次迭代的theta值
    for t in range(1, T + 1):
        
        grad = grad_f(theta, x, num_samples)

        #grad = 2 * (theta - x)
        # 学习率衰减
        eta_t = eta_0 / (1 + gamma * t)
        
        # 更新参数
        theta = theta - eta_t * grad
        
        
        theta_values.append(theta)
        
    
    return theta_values



# 随机梯度下降算法（SGD）的示例实现
# def sgd(theta_init, x, lr, num_iters):
#     theta = theta_init
#     for _ in range(num_iters):
#         grad = 2 * (theta - x)  # 目标函数 f 的梯度
#         theta = theta - lr * grad
#     return theta




def grid_sample(d, lowbound, upbound, point):
    # 根据给定的总点数和维度，计算每个维度的点数 n
    n = int(round(point ** (1.0 / d))) +1 # 每个维度上的点数
    np.random.seed(42)
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


# 离线阶段实现
def offline_stage(n, T, eta_0, gamma, covariate_dim):
    #np.random.seed(1)
    covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
    #covariates_points = np.random.uniform(low = lowbound , high=upbound, size=(n, covariate_dim))
    #print(n, covariates_points.shape)
    #theta_estimates = []
    theta_estimates = np.ones((n,covariate_dim))
    for i in range(n):
        x_i = covariates_points[i]
        np.random.seed()
        theta_init = np.random.randn(covariate_dim)  # 初始化θ

        
        theta_values = sgd_with_lr_decay( theta_init, x_i, eta_0, gamma, T)
        
        theta_bar = np.mean(theta_values, axis=0)
        
        
        theta_estimates[i] = theta_bar
        
    return covariates_points, theta_estimates


# 在线阶段实现
def knn_online_stage(covariates_points, theta_estimates, x, k):
    # 计算新协变量点 x 与离线阶段得到的协变量点集之间的欧氏距离
    distances = euclidean_distances([x], covariates_points).flatten()
    
    # 找到 k 个最近邻的协变量点的索引
    nearest_neighbors_indices = np.argsort(distances)[:k]
    
    # 获取 k 个最近邻点的 θ 的估计值
    nearest_theta_estimates = theta_estimates[nearest_neighbors_indices]
    
    # 计算 θ 的平均值
    theta_hat = np.mean(nearest_theta_estimates, axis=0)
    
    return theta_hat


def gaussian_kernel(x, x_i, h):
    distance = np.linalg.norm(x - x_i)  
    return np.exp(-(distance ** 2) / (2 * h ** 2))

def ks_online_stage(x, x_train, theta_train, h):
    kernel_values = np.array([gaussian_kernel(x, x_i, h) for x_i in x_train])
    kernel_values = np.expand_dims(kernel_values, axis=1)  # (n, 1)
  
    numerator = np.sum(kernel_values * theta_train, axis=0)
    denominator = np.sum(kernel_values)
    if denominator == 0:
        return 0
    theta_hat = numerator / denominator
    return theta_hat



# 自定义实现的KRR算法
# def krr_online_stage(covariates_points, theta_estimates, x, lambda_param=1e-4, gamma=1.0):
#     # 计算核矩阵K_phi
#     n = covariates_points.shape[0]
#     K_phi = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             K_phi[i, j] = np.exp(-gamma * np.linalg.norm(covariates_points[i] - covariates_points[j])**2)
    
#     # 计算核向量k_phi_x
#     k_phi_x = np.array([np.exp(-gamma * np.linalg.norm(x - covariates_points[i])**2) for i in range(n)])
    
#     # 计算KRR的预测值
#     theta_x = np.dot(k_phi_x, np.linalg.inv(K_phi + lambda_param * np.eye(n)) @ theta_estimates)
    
#     return theta_x


def true_theta_function(x):
    return x


# 将数据转换为适合Seaborn使用的格式
def prepare_data_for_plot(n_values, mse_by_k, mse_krr, mse_ks, mse_lr,k_values):
    data = []
    for k in k_values:
        for i, n in enumerate(n_values):
            data.append({"n": n, "MSE": mse_by_k[k][i], "Method": f"k-NN (k={k})"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_krr[i], "Method": "KRR"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_ks[i], "Method": "KS"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_lr[i], "Method": "LR"})
    return pd.DataFrame(data)

# 准备绘图数据
# def prepare_plot_data(n_values, mse_data, variance_data, bias_data, k_values):
#     data = []
#     for k in k_values:
#         for i, n in enumerate(n_values):
#             data.append({"n": n, "MSE": mse_data[k][i], "Variance": variance_data[k][i], "Bias^2": bias_data[k][i], "Method": f"k-NN (k={k})"})
#     return pd.DataFrame(data)

def prepare_plot_data(n_values, mse_by_k, variance_by_k, bias_by_k, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr, k_values):
    data = []
    
    # 处理k-NN的MSE、Variance、Bias^2
    for k in k_values:
        for i, n in enumerate(n_values):
            data.append({"n": n, "MSE": mse_by_k[k][i], "Variance": variance_by_k[k][i], "Bias^2": bias_by_k[k][i], "Method": f"k-NN (k={k})"})
    
    # 处理KRR的MSE、Variance、Bias^2
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_krr[i], "Variance": variance_krr[i], "Bias^2": bias_krr[i], "Method": "KRR"})
    

    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_ks[i], "Variance": variance_ks[i], "Bias^2": bias_ks[i], "Method": "KS"})
    
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_lr[i], "Variance": variance_lr[i], "Bias^2": bias_lr[i], "Method": "LR"})
    
    return pd.DataFrame(data)

def compute_bias_variance_mse(true_value, predicted_values):
    predicted_values = np.array(predicted_values)
    mean_predicted = np.mean(predicted_values, axis=0)  
    bias = mean_predicted - true_value  
    print(bias)
    bias_squared = np.mean(bias ** 2)  
    variance = np.mean(np.var(predicted_values, axis=0))  
    mse = np.mean(np.mean((predicted_values - true_value) ** 2, axis=1))  
    return bias_squared, variance, mse



def plot_metrics_for_each_method(df, methods):
    sns.set_theme()
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # 创建一个2x2的图表布局
    axes = axes.flatten()  # 将2x2的数组扁平化，以便我们可以通过索引访问
    
    for i, method in enumerate(methods):
        # 筛选当前方法的数据
        
        method_df = df[df['Method'] == method].copy()
        
        # 对数据进行对数转换
        method_df['log_n'] = np.log2(method_df['n'])  # 将 n 取 log_2
        # method_df['log_MSE'] = np.log2(method_df['MSE'])  # 将 MSE 取 log_10
        # method_df['log_Bias^2'] = np.log2(method_df['Bias^2'])  # 将 Bias^2 取 log_10
        # method_df['log_Variance'] = np.log2(method_df['Variance'])  # 将 Va
        
        
        
        
        
        # 设置 x 轴为对数尺度并绘制各指标
        sns.lineplot(data=method_df, x='n', y='MSE', label='MSE', marker='o',  markersize=10,ax=axes[i], linewidth=3)
        sns.lineplot(data=method_df, x='n', y='Bias^2', label=r"Bias$^2$", marker='D', markersize=10, ax=axes[i], linewidth=3)
        sns.lineplot(data=method_df, x='n', y='Variance', label='Variance', marker='H', markersize=10, ax=axes[i], linewidth=3)
        
        # 设置标题和轴标签
        axes[i].set_title(f"{method} Performance Metrics")
        axes[i].set_xlabel('Number of covariate points (n)')
        axes[i].set_ylabel('Value')
        axes[i].set_xscale('log', base=2)  # 设置 x 轴为 log_2 规模
        #axes[i].set_yscale('log', base=2)  # 设置 y 轴为对数尺度
        axes[i].tick_params(labelsize=20)
        #axes[i].legend(title='Metric', fontsize=10)
        axes[i].grid(True)
    
    plt.tight_layout()  # 调整整体布局
    #plt.savefig('paper3.png')
    plt.show()   



def compute_inverse_kernel_matrix(covariates_points, lambda_param=1e-4, nu=1.5, length_scale=1.0):
    K_phi = matern_kernel(covariates_points, covariates_points, nu, length_scale)
    return np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))


def matern_kernel(X, Y, nu=1.5, length_scale=1.0):
    X = np.array(X)
    Y = np.array(Y)
    dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
    if nu == 0.5:
        K = np.exp(-dists)
    elif nu == 1.5:
        K = (1.0 + np.sqrt(3.0) * dists) * np.exp(-np.sqrt(3.0) * dists)
    elif nu == 2.5:
        K = (1.0 + np.sqrt(5.0) * dists + (5.0 / 3.0) * dists**2) * np.exp(-np.sqrt(5.0) * dists)
    else:
        raise ValueError("Unsupported value of nu. Use 0.5, 1.5, or 2.5.")
    return K

def krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv, nu=1.5, length_scale=1.0):
    k_phi_x = matern_kernel([x], covariates_points, nu, length_scale).flatten()
    theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
    return theta_x




"""
    Linear regression
"""

def basis_functions(x):
  
    return np.hstack([x, x ** 2, x ** 3])

def linear_regression_train(x_train, theta_train):
   
    


    Phi = np.array([basis_functions(x) for x in x_train])
    
    #  beta_hat = (Phi.T @ Phi)^(-1) @ Phi.T @ theta_train
    Phi_T_Phi_inv = np.linalg.inv(Phi.T @ Phi)  # (s, s)
    beta_hat = Phi_T_Phi_inv @ Phi.T @ theta_train  # (s, d)
    
    return beta_hat

def linear_regression_on_stage(x, beta_hat):
    
    phi_x = basis_functions(x)  # (s,)
    theta_hat = phi_x @ beta_hat  # (d,)
    
    return theta_hat



if __name__ == '__main__':
    lowbound = 0
    upbound = 8
    total_budget = 2000
    eta_0 = 0.1  # 初始学习率
    gamma = 0.01  # 学习率衰减率
    covariate_dim = 3
    num_samples = 300  # Number of samples for true theta calculation
    # 生成测试数据集 (在线阶段需要的)
    num_test_points = 1
    #np.random.seed(1)
    test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    #true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
    true_theta_values = true_theta_function(test_points)
    
    #n_values = np.arange(10,101,20)
    n_values = 2 ** np.arange(4, 9)
    # n_values = [1]
    #n_values = np.append(n_values, [150, 200,300])
 
    # 不同的 k 值
    k_values = [2]  # 可以根据需要调整

    mse_by_k = {k: [] for k in k_values}
    variance_by_k = {k: [] for k in k_values}
    bias_by_k = {k: [] for k in k_values}
    bias_krr,variance_krr, mse_krr = [],[],[]
    bias_ks,variance_ks, mse_ks = [],[],[]
    bias_lr,variance_lr, mse_lr = [],[],[]
    replication = 10  #30 50


    for n in n_values:
        covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
        
        T = total_budget // n

        K_phi_inv = compute_inverse_kernel_matrix(covariates_points, lambda_param =  1 / n, nu=1.5, length_scale=1.0)
        
        

        # 进行 replication 次的离线阶段计算
        covariates_points_list = []
        theta_estimates_list = []
        predicted_theta_values =[]
        beta_hats = []
        for i in range(replication):

            
            covariates_points, theta_estimates = offline_stage(n, T, eta_0, gamma, covariate_dim)
            
            "linear regression beta"
            beta_hat = linear_regression_train(covariates_points, theta_estimates)
            beta_hats.append(beta_hat)

            print(test_points,covariates_points[0] , theta_estimates[0],np.linalg.norm(covariates_points - theta_estimates))
            covariates_points_list.append(covariates_points)
            theta_estimates_list.append(theta_estimates)
            predicted_theta_value = np.array([krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv) for x in test_points])
            predicted_theta_values.append(predicted_theta_value)

        bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
        bias_krr.append(bias)
        variance_krr.append(variance)
        mse_krr.append(mse)
        print(f"KRR, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")


        for k in k_values:
            predicted_theta_values= []
            for i in range(replication):
                covariates_points = covariates_points_list[i]
                theta_estimates = theta_estimates_list[i]
                
                predicted_theta_value = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
                predicted_theta_values.append(predicted_theta_value)

            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
            
            print(f"k = {k}, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")
            mse_by_k[k].append(mse)
            bias_by_k[k].append(bias)
            variance_by_k[k].append(variance)



        predicted_theta_values= []
        for i in range(replication):
            h = 1.0
            covariates_points = covariates_points_list[i]
            theta_estimates = theta_estimates_list[i]
            predicted_theta_value = np.array([ks_online_stage(x, covariates_points, theta_estimates, h) for x in test_points])
            predicted_theta_values.append(predicted_theta_value)
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
        bias_ks.append(bias)
        variance_ks.append(variance)
        mse_ks.append(mse)
        print(f"KS, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")


        predicted_theta_values= []
        for i in range(replication):
            covariates_points = covariates_points_list[i]
            theta_estimates = theta_estimates_list[i] 
            predicted_theta_value = np.array([linear_regression_on_stage(x,beta_hats[i] ) for x in test_points])
            predicted_theta_values.append(predicted_theta_value)
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
        bias_lr.append(bias)
        variance_lr.append(variance)
        mse_lr.append(mse)
        print(f"LR, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")





    df = prepare_data_for_plot(n_values, mse_by_k, mse_krr, mse_ks, mse_lr, k_values)
 
    df['log2_n'] = np.log2(df['n'])

    sns.set_theme()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='MSE', hue='Method', style='Method',markers=['o', 's', 'D', '^'], dashes=False, markersize=10,linewidth=3,ci=None)
    plt.xscale('log', base=2)
    
    plt.xlabel('Number of covariate points (n)', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('MSE vs Number of covariate points (n) for different methods', fontsize=14)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(True)

    plt.text(0.95, 0.05, f'Total Budget: {total_budget}\nCovariate Dimension: {covariate_dim}', 
         transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    #plt.savefig('paper1.png')
    plt.show()


    
    df = prepare_plot_data(n_values, mse_by_k, variance_by_k, bias_by_k, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr, k_values)
    methods = ['k-NN (k=2)', 'KRR','KS','LR']
    plot_metrics_for_each_method(df, methods)

