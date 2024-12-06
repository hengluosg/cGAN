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
from sklearn.model_selection import KFold

def F(theta, x ,xi):
    return np.sum((theta - x)**2) + xi

# def grad_f(theta, x, num_samples):
    
#     xi_samples = np.random.normal(0, 4, num_samples)  # 随机样本
#     grads = [2*(theta - x) + xi for xi in xi_samples]  # F 对 theta 的导数
#     return np.mean(grads, axis=0)  # 对每个样本的梯度取平均







# """Axis parallel hyper-ellipsoid function"""

"""y = (theta - A * sqrt(x))^T * (theta - A * sqrt(x))."""
# def testfun(theta,x):
#     d = theta.shape[0]
#     sum1 = [(i+1)* (theta[i]-np.sqrt(x[i]))**2 for i in range(d)]
#     y = np.sum(sum1) 
#     return y 


# def generate_symmetric_matrix(d):
    
#     # Generate a random matrix with normal distribution
#     np.random.seed(1)
#     M = np.random.randn(d, d)
#     A = (M + M.T) / 2
#     return A


def generate_symmetric_matrix(d, correlation_strength=0.5):
    np.random.seed(1)
    A = np.random.randn(d, d)
    A = np.dot(A.T, A)  
    for i in range(d):
        A[i, i] = 1  # Set diagonal elements to 1
    for i in range(d):
        for j in range(i+1, d):
            A[i, j] = A[j, i] = correlation_strength  # Set correlation between variables
    return A


def grad_f(theta, x , num_samples):
    d = theta.shape[0]
    x_sqrt = np.sqrt(x)
    W = np.diag(np.arange(1, d+1))  # Diagonal matrix W
    # Compute the gradient: 2W (theta - A * sqrt(x))
    grad = 2 * W @ (theta - A @ x_sqrt)  # Matrix multiplication
    #grad = 2 * W @ (theta - x_sqrt)  # Matrix multiplication
    gradients = np.zeros((num_samples, d))
    for rep in range(num_samples):
        noise = np.random.normal(0, 16, d)  # Generate Gaussian noise
        gradients[rep,:] = grad  + noise
    avg_gradient = np.mean(gradients, axis=0)
    return avg_gradient


# def grad_f(theta, x , num_samples):
#     d = theta.shape[0]
    
#     gradients = np.zeros((num_samples, d))
#     for rep in range(num_samples):
#         noise = np.random.normal(0, 16, d)  # Generate Gaussian noise
#         for i in range(d):
#             gradients[rep, i] = 2 * (i + 1) * (theta[i] - (A @ np.sqrt(x))[i]) + noise[i]
#     avg_gradient = np.mean(gradients, axis=0)
#     return avg_gradient



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
        print("Warning: Denominator is zero. Returning 0 for theta_hat.")
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


# def true_theta_function(x):
#     theta = np.sqrt(x)
#     return theta

def true_theta_function(x):
    theta = A @ np.sqrt(x).T
    print(theta.shape)
    return theta.T


# 将数据转换为适合Seaborn使用的格式
def prepare_data_for_plot(n_values, mse_knn, mse_krr, mse_ks,mse_lr):
    data = []
    # for k in k_values:
    #     for i, n in enumerate(n_values):
    #         data.append({"n": n, "MSE": mse_by_k[k][i], "Method": f"k-NN (k={k})"})
    
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_knn[i]), "Method": "KNN"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_krr[i]), "Method": "KRR"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_ks[i]), "Method": "KS"})
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": np.log2(mse_lr[i]), "Method": "LR"})
    return pd.DataFrame(data)


def prepare_plot_data(n_values, mse_knn, variance_knn, bias_knn, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr):
    data = []
    
    for i, n in enumerate(n_values):
        data.append({"n": n, "MSE": mse_knn[i], "Variance": variance_knn[i], "Bias^2": bias_knn[i], "Method": f"k-NN"})


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
    
    bias_squared = np.mean(bias ** 2)  
    variance = np.mean(np.var(predicted_values, axis=0))  
    mse = np.mean(np.mean((predicted_values - true_value) ** 2, axis=1))  
    return bias_squared, variance, mse



def plot_metrics_for_each_method(df, methods ,d):
    sns.set_theme()
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # 创建一个2x2的图表布局
    axes = axes.flatten()  # 将2x2的数组扁平化，以便我们可以通过索引访问
    print(df)
    for i, method in enumerate(methods):
        # 筛选当前方法的数据
        
        method_df = df[df['Method'] == method].copy()
        

        method_df['log_n'] = np.log2(method_df['n'])  # 将 n 取 log_2
        
        sns.lineplot(data=method_df, x='n', y='MSE', label='MSE', marker='o',  markersize=10,ax=axes[i], linewidth=3)
        sns.lineplot(data=method_df, x='n', y='Bias^2', label=r"Bias$^2$", marker='D', markersize=10, ax=axes[i], linewidth=3)
        sns.lineplot(data=method_df, x='n', y='Variance', label='Variance', marker='H', markersize=10, ax=axes[i], linewidth=3)
        
        # 设置标题和轴标签
        axes[i].set_title(f"{method} Performance Metrics")
        axes[i].set_xlabel('Total budget')
        axes[i].set_ylabel('Value')
        axes[i].set_xscale('log', base=2)  # 设置 x 轴为 log_2 规模
        #axes[i].set_yscale('log', base=2)  # 设置 y 轴为对数尺度
        axes[i].tick_params(labelsize=20)
        #axes[i].legend(title='Metric', fontsize=10)
        axes[i].text(
        0.95, 0.05, 
        f'Covariate Dimension: {covariate_dim}',  # 假设 covariate_dim 是定义好的变量
        transform=axes[i].transAxes,  # 使用当前子图的坐标系
        fontsize=14, 
        verticalalignment='bottom', 
        horizontalalignment='right')
        axes[i].grid(True)
        
    plt.tight_layout()  # 调整整体布局
    plt.savefig('d{}.png'.format(d))
    #plt.show()   



# def compute_inverse_kernel_matrix(covariates_points, lambda_param=1e-4, nu=1.5, length_scale=1.0):
#     K_phi = matern_kernel(covariates_points, covariates_points, nu, length_scale)
#     return np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))


# def matern_kernel(X, Y, nu=1.5, length_scale=1.0):
#     X = np.array(X)
#     Y = np.array(Y)
#     dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
#     if nu == 0.5:
#         K = np.exp(-dists)
#     elif nu == 1.5:
#         K = (1.0 + np.sqrt(3.0) * dists) * np.exp(-np.sqrt(3.0) * dists)
#     elif nu == 2.5:
#         K = (1.0 + np.sqrt(5.0) * dists + (5.0 / 3.0) * dists**2) * np.exp(-np.sqrt(5.0) * dists)
#     else:
#         raise ValueError("Unsupported value of nu. Use 0.5, 1.5, or 2.5.")
#     return K

# def krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv, nu=1.5, length_scale=1.0):
#     k_phi_x = matern_kernel([x], covariates_points, nu, length_scale).flatten()
#     theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
#     return theta_x

def gaussian_kernel_krr(X, Y, length_scale=1.0):
    X = np.array(X)
    Y = np.array(Y)
    
    # 计算欧几里得距离
    dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
    
    # 计算高斯核
    K = np.exp(-0.5 * dists**2)
    
    return K

# def compute_inverse_kernel_matrix(covariates_points, lambda_param=1e-4, length_scale=1.0):
#     K_phi = gaussian_kernel_krr(covariates_points, covariates_points, length_scale)
#     return np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))

# def krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv, length_scale=1.0):
#     k_phi_x = gaussian_kernel_krr([x], covariates_points, length_scale).flatten()
#     theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
#     return theta_x

def compute_inverse_kernel_matrix(covariates_points, lambda_param=1e-4, length_scale=1.0):
    """Compute the inverse of the kernel matrix with regularization."""
    K_phi = gaussian_kernel_krr(covariates_points, covariates_points, length_scale)
    return np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))

def krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv, length_scale=1.0):
    """Predict theta for a new point using the trained KRR model."""
    k_phi_x = gaussian_kernel_krr([x], covariates_points, length_scale).flatten()
    theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
    return theta_x

def compute_inverse_kernel_matrix(covariates_points, lambda_param=1e-4, length_scale=1.0):
    """Compute the inverse of the kernel matrix with regularization."""
    K_phi = gaussian_kernel_krr(covariates_points, covariates_points, length_scale)
    return np.linalg.inv(K_phi + lambda_param * np.eye(K_phi.shape[0]))

def krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv, length_scale=1.0):
    """Predict theta for a new point using the trained KRR model."""
    k_phi_x = gaussian_kernel_krr([x], covariates_points, length_scale).flatten()
    theta_x = np.dot(k_phi_x, K_phi_inv) @ theta_estimates
    return theta_x

def krr_cross_validation(covariates_points, theta_estimates, lambda_vals=[0.001, 0.001, 0.01, 0.1, 1, 10], n_splits=5, length_scale=1.0):
    """Perform K-fold cross-validation for different lambda_param values."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Define KFold split
    best_lambda = None
    best_mse = float('inf')  # Initialize best MSE as infinity

   


    for lambda_param in lambda_vals:
        mse_scores = []  # List to store mean squared errors for each fold

        for train_index, val_index in kf.split(covariates_points):
            # Split data into training and validation sets
            X_train, X_val = covariates_points[train_index], covariates_points[val_index]
            y_train, y_val = theta_estimates[train_index], theta_estimates[val_index]
            
            # Compute the kernel matrix and its inverse for the training set
            K_phi_inv = compute_inverse_kernel_matrix(X_train, lambda_param=lambda_param, length_scale=length_scale)
            
            # Predict on validation set
            predicted_theta_values = np.array([krr_online_stage(X_train, y_train, x, K_phi_inv, length_scale) for x in X_val])
            
            # Compute MSE for the fold
            mse = mean_squared_error(y_val, predicted_theta_values)
            mse_scores.append(mse)

        # Calculate the average MSE across all folds for the current lambda_param
        average_mse = np.mean(mse_scores)
        #print(f"Lambda: {lambda_param}, Average MSE: {average_mse}")

        # Update the best lambda if we found a lower MSE
        if average_mse < best_mse:
            best_mse = average_mse
            best_lambda = lambda_param

    #print(f"Best Lambda: {best_lambda} with MSE: {best_mse}")
    return best_lambda










"""
    Linear regression
"""

def basis_functions(x):
  
    return np.hstack([1,x])

# def linear_regression_train(x_train, theta_train):

#     Phi = np.array([basis_functions(x) for x in x_train])
#     n = Phi.shape[1]
    
#     #  beta_hat = (Phi.T @ Phi)^(-1) @ Phi.T @ theta_train
#     Phi_T_Phi_inv = np.linalg.inv(Phi.T @ Phi+ 0.01 * np.eye(n))  # (s, s)
#     #Phi_T_Phi_inv = np.linalg.inv(Phi.T @ Phi)  # (s, s)
#     beta_hat = Phi_T_Phi_inv @ Phi.T @ theta_train  # (s, d)
#     return beta_hat


def linear_regression_train(x_train, theta_train, lambda_reg=0.01):
    # Apply basis functions to the input data (x_train)
    Phi = np.array([basis_functions(x) for x in x_train])  # Assuming basis_functions is predefined
    # Number of features (s)
    n = Phi.shape[1]
    # Compute (Phi.T @ Phi + lambda_reg * I)^(-1) @ Phi.T @ theta_train
    Phi_T_Phi_inv = np.linalg.inv(Phi.T @ Phi + lambda_reg * np.eye(n))  # Regularized inversion
    # Compute the final beta_hat using the regularized solution
    beta_hat = Phi_T_Phi_inv @ Phi.T @ theta_train
    return beta_hat

def cross_validate(x_train, theta_train, lambda_vals=[0.001,0.01, 0.1, 1, 10], k_folds=5):
    
    n_samples = len(x_train)

    k_folds = min(k_folds, n_samples)
    best_lambda = None
    best_beta_hat = None
    best_score = float('inf')  # We are minimizing MSE
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Loop over each possible value of lambda_reg
    for lambda_reg in lambda_vals:
        fold_scores = []
        
        # Perform K-fold cross-validation
        for train_idx, val_idx in kf.split(x_train):
            x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
            theta_train_fold, theta_val_fold = theta_train[train_idx], theta_train[val_idx]
            
            # Train the model on the training fold
            beta_hat = linear_regression_train(x_train_fold, theta_train_fold, lambda_reg)
            
            # Predict on the validation fold
            Phi_val = np.array([basis_functions(x) for x in x_val_fold])
            predictions = Phi_val @ beta_hat
            
            # Calculate the Mean Squared Error (MSE) for the fold
            mse = mean_squared_error(theta_val_fold, predictions)
            fold_scores.append(mse)
        
        # Calculate the average MSE for the current lambda_reg
        avg_score = np.mean(fold_scores)
        
        # Check if this lambda gives a better score
        if avg_score < best_score:
            best_score = avg_score
            best_lambda = lambda_reg
            best_beta_hat = beta_hat
    
    # Return the best regularization parameter and the corresponding beta_hat
    #print(f"Best regularization parameter: {best_lambda}")
    return best_beta_hat






def linear_regression_on_stage(x, beta_hat):
    
    phi_x = basis_functions(x)  # (s,)
    theta_hat = phi_x @ beta_hat  # (d,)
    return theta_hat


def picture_plot1(df):
    #sns.set( font_scale = 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, column in enumerate(df):
        sns.set_theme()
        sns.lineplot(data=column,ax = axes[i], x='n', y='MSE', hue='Method', style='Method',markers=[ 'o','s', 'D', '^'], dashes=False, markersize=10,linewidth=3)
        axes[i].set_xscale('log', base=2)
        axes[i].set_xlabel('Total budget', fontsize=15)
        axes[i].set_ylabel(r'$log_{2}(MSE)$', fontsize=15)
        axes[i].set_title(r'$d_x = {} $'.format(covariate_dim1[i]), fontsize=15)
        axes[i].tick_params(axis='x', labelsize=15)
        axes[i].tick_params(axis='y', labelsize=15)
        #axes[i].tick_params(labelsize=20)
        axes[i].grid(True)
        axes[i].text(
            0.95, 0.05, 
            f'Covariate Dimension: {covariate_dim1[i]}', 
            transform=axes[i].transAxes, 
            fontsize=14, 
            verticalalignment='bottom', 
            horizontalalignment='right')
        
        axes[i].tick_params(labelsize=15)
    #plt.savefig("1-4.png")
    plt.tight_layout()
    plt.savefig('total.png')
    #plt.show()



if __name__ == '__main__':
    lowbound = 0
    upbound = 2

    eta_0 = 0.1  # 初始学习率
    gamma1 = 0.01  # 学习率衰减率
    # covariate_dim = 2
    num_samples = 1  # Number of samples for true theta calculation


    num_test_points = 1
    #np.random.seed(1)
    #test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
    #true_theta_values = np.array([true_theta_function(x, num_samples) for x in test_points])
    #true_theta_values = true_theta_function(test_points)
    
    #n_values = np.arange(10,101,20)
    
    total_budget = 2 ** np.arange(4, 14)
    # n_values = [1]
    #n_values = np.append(n_values, [150, 200,300])
 
    # 不同的 k 值
      # 可以根据需要调整
    data1 = []
    covariate_dim1 = [2,5,8]
    for covariate_dim in covariate_dim1:
        if covariate_dim == covariate_dim1[0]:
            total_budget = 2 ** np.arange(4, 14)
        elif covariate_dim == covariate_dim1[1]:
            total_budget = 2 ** np.arange(5, 15)
        else:
            total_budget = 2 ** np.arange(6, 16)
        A = generate_symmetric_matrix(covariate_dim, correlation_strength=0.5)
        print(A)
        test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
        true_theta_values = true_theta_function(test_points)
        bias_knn,variance_knn, mse_knn = [],[],[]
        bias_krr,variance_krr, mse_krr = [],[],[]
        bias_ks,variance_ks, mse_ks = [],[],[]
        bias_lr,variance_lr, mse_lr = [],[],[]
        replication = 100  #30 50

        "Kernel Ridge Regression (KRR)"
        
        for i,gamma in enumerate(total_budget):
            n = int(gamma**(2/3))
            T = int(gamma / n)
            covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)
            
            # 进行 replication 次的离线阶段计算

            predicted_theta_values =[]
            beta_hats = []
            for i in range(replication):
                covariates_points, theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
                #print(test_points,covariates_points[0] , theta_estimates[0],np.linalg.norm(covariates_points - theta_estimates))
                #print(covariates_points.shape, theta_estimates.shape)
                
                
                
                best_lambda = krr_cross_validation(covariates_points, theta_estimates, lambda_vals=[0.0001, 0.001, 0.01, 0.1, 1, 10])

                # After cross-validation, use the best lambda_param and compute predictions for the test set
                K_phi_inv = compute_inverse_kernel_matrix(covariates_points, lambda_param=best_lambda, length_scale=1.0)
                predicted_theta_value = np.array([krr_online_stage(covariates_points, theta_estimates, x, K_phi_inv, length_scale=1.0) for x in test_points])
                predicted_theta_values.append(predicted_theta_value)
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
            bias_krr.append(bias)
            variance_krr.append(variance)
            mse_krr.append(mse)
            print(f"KRR, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")


        "Kernel Smoothing (KS)"
        
        for i,gamma in enumerate(total_budget):
            h = (gamma**(-1/(covariate_dim+2)))
            n = int(gamma**((covariate_dim+1)/(covariate_dim+2)))
            T = int(gamma / n)
            predicted_theta_values =[]
            for i in range(replication):
                covariates_points, theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
                predicted_theta_value = np.array([ks_online_stage(x, covariates_points, theta_estimates, h) for x in test_points])
                predicted_theta_values.append(predicted_theta_value)
                
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
            bias_ks.append(bias)
            variance_ks.append(variance)
            mse_ks.append(mse)
            print(f"KS, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")

            

        "k-Nearest Neighbors (kNN)"

        
        for i,gamma in enumerate(total_budget):
            k = int(gamma**(1/(covariate_dim+2)))
            n = int(gamma**(covariate_dim/(covariate_dim+2))) * k 
            T = int(gamma / n)
            predicted_theta_values =[]
            
            for i in range(replication):
                covariates_points, theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
                predicted_theta_value = np.array([knn_online_stage(covariates_points, theta_estimates, x, k) for x in test_points])
                predicted_theta_values.append(predicted_theta_value)

            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
            
            
            mse_knn.append(mse)
            bias_knn.append(bias)
            variance_knn.append(variance)
            print(f"KNN, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")




        "Linear Regression (LR)"
        
        for i,gamma in enumerate(total_budget):
            n = int(gamma**(1/2))
            T = int(gamma**(1/2))
            #covariates_points = grid_sample(covariate_dim, lowbound, upbound, n)

            
            predicted_theta_values =[]
            for i in range(replication):
                covariates_points, theta_estimates = offline_stage(n, T, eta_0, gamma1, covariate_dim)
                "linear regression beta"
                #beta_hat = linear_regression_train(covariates_points, theta_estimates)
                beta_hat = cross_validate(covariates_points, theta_estimates)
                beta_hats.append(beta_hat)
                predicted_theta_value = np.array([linear_regression_on_stage(x,beta_hat ) for x in test_points])
                predicted_theta_values.append(predicted_theta_value)
            bias, variance, mse = compute_bias_variance_mse(true_theta_values, predicted_theta_values)
            bias_lr.append(bias)
            variance_lr.append(variance)
            mse_lr.append(mse)
            print(f"LR, n = {n}, T = {T}, bias_squared: {bias }, variance: {variance}, mse: {mse}")


        df = prepare_data_for_plot(total_budget, mse_knn, mse_krr, mse_ks,mse_lr)
    
        #df['log2_n'] = np.log2(df['n'])

        data1.append(df)
        df = prepare_plot_data(total_budget, mse_knn, variance_knn, bias_knn, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr)
        methods = ['k-NN', 'KRR','KS','LR']
        plot_metrics_for_each_method(df, methods,covariate_dim)
    print("over")
    picture_plot1(data1)
    
    # df = prepare_plot_data(total_budget, mse_knn, variance_knn, bias_knn, mse_krr, variance_krr, bias_krr,mse_ks, variance_ks, bias_ks,mse_lr, variance_lr, bias_lr)
    # methods = ['k-NN', 'KRR','KS','LR']
    #plot_metrics_for_each_method(df, methods)

