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
    term1 = -10 * (np.sin(0.5 * np.pi * x) ** 6) / (2 ** (2*((x-20) / 20) ** 2)) + 10
    return term1

# def test_function_with_multiple_optima(theta, x):
#     return (theta - x) ** 2 + np.sin(5 * np.pi * theta)
# theta = np.linspace(0, 4, 400)
# x_fixed = 2

# # 计算函数值
# y_values = test_function_with_multiple_optima(theta, x_fixed)

# # 绘制函数图像
# plt.figure(figsize=(10, 6))
# plt.plot(theta, y_values, color='b')
# plt.title('Test Function with Multiple Local Optima')
# plt.xlabel('Theta')
# plt.ylabel('Function Value')
# plt.grid(True)

# # 标注 theta = x 的点
# plt.axvline(x=x_fixed, color='r', linestyle='--', label=f'Theta = X = {x_fixed}')
# plt.legend()

# plt.show()

# Generate x values in the range [0, 100]
# x_values = np.linspace(0, 10, 50)
# y_values = g_1d(x_values)


# # Create a DataFrame for seaborn
# data = pd.DataFrame({
#     'x': x_values,
#     'g(x)': y_values
# })

# # Plot using seaborn
# sns.set()
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=data, x='x', y='g(x)', color='blue', label=r"$g(\theta)$")
# plt.title(r"$g(\theta)$", fontsize=14)
# plt.xlabel(r"$\theta$", fontsize=12)
# plt.ylabel(r"$g(\theta)$", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks

# # 定义目标函数 g(theta)

# def g(x):
#     term1 = -10 * (np.sin(0.5 * np.pi * x) ** 6) / (2 ** (2*((x-20) / 20) ** 2)) + 10
#     return term1
# # def g(theta):
# #     return 10 * (np.sin(0.5 * np.pi * theta) ** 2) + 6

# # 创建 theta 的范围
# theta = np.linspace(0, 10, 1000)
# g_values = g(theta)

# # 使用 find_peaks 找到局部最大值 (local maxima)
# # local_max_indices, _ = find_peaks(g_values)
# local_min_indices, _ = find_peaks(-g_values)

# # 找到全局最大值和最小值
# # global_max_index = np.argmax(g_values)
# global_min_index = np.argmin(g_values)

# # 绘制原始函数和局部/全局极值点
# plt.plot(theta, g_values, label='g(θ)', color='b')

# # 绘制局部极大值点

# plt.plot(theta[local_min_indices], g_values[local_min_indices], "yx", label="Local Minima")

# # 绘制全局最大值和最小值

# plt.plot(theta[global_min_index], g_values[global_min_index], "co", label="Global Minimum")

# plt.xlabel('θ')
# plt.ylabel('g(θ)')
# plt.title('Local and Global Extrema of g(θ)')
# plt.legend()
# plt.show()


# # 定义目标函数
# def objective_function(theta, x):
#     return 10 * (np.sin(0.05 * np.pi * (theta - x)) ** 6) / (2 ** (2 * (((theta - x) - 50) / 50) ** 2))

# # 定义 theta 和 x 的范围
# theta = np.linspace(0, 100, 100)
# x = np.linspace(0, 100, 100)

# # 创建 theta 和 x 的网格
# theta_grid, x_grid = np.meshgrid(theta, x)

# # 计算函数值
# z = objective_function(theta_grid, x_grid)

# # 绘制曲面图
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(theta_grid, x_grid, z, cmap='viridis', edgecolor='none')
# ax.set_title('Objective Function Surface')
# ax.set_xlabel('Theta')
# ax.set_ylabel('X')
# ax.set_zlabel('Objective Value')

# plt.show()



# def high_dim_objective_function(theta, x):
#     # 假设 theta 和 x 都是高维向量
#     return np.sum(
#         10 * (np.sin(0.05 * np.pi * (theta - x)) ** 6) / (2 ** (2 * (((theta - x) - 50) / 50) ** 2)),
#         axis=-1
#     )

# # 为了绘制三维图，我们考虑 theta 和 x 各自有 3 个维度
# # 定义 3 维的 theta 和 x
# theta_3d = np.linspace(0, 100, 200)
# x_3d = np.linspace(0, 100, 200)

# # 创建 theta 和 x 的网格，每个维度有 3 个分量
# theta_grid_3d, x_grid_3d = np.meshgrid(theta_3d, x_3d)

# # 初始化函数值矩阵，计算高维函数值
# z_3d = np.zeros_like(theta_grid_3d)

# # 为每个点生成 3 维度的随机分量，计算函数值
# for i in range(len(theta_3d)):
#     for j in range(len(x_3d)):
#         theta_vec = np.random.uniform(0, 100, 3)
#         x_vec = np.random.uniform(0, 100, 3)
#         z_3d[i, j] = high_dim_objective_function(theta_vec, x_vec)

# # 绘制三维曲面图
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(theta_grid_3d, x_grid_3d, z_3d, cmap='viridis', edgecolor='none')
# ax.set_title('3D High-Dimensional Objective Function Surface')
# ax.set_xlabel('Theta')
# ax.set_ylabel('X')
# ax.set_zlabel('Objective Value')

# plt.show()



# from sklearn.metrics.pairwise import rbf_kernel
# import seaborn as sns
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from itertools import product
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.metrics import mean_squared_error
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import torch
# import numpy as np
# # 定义目标函数
# def objective_function(theta, x=2):
#     return (theta - x) ** 2 + torch.sin(5 * np.pi * theta)

# # 初始化变量
# theta = torch.tensor([2.0], requires_grad=True)

# # 定义优化器
# optimizer = torch.optim.Adam([theta], lr=0.01)

# # 优化循环
# for i in range(1200):
#     optimizer.zero_grad()
#     loss = objective_function(theta)
#     loss.backward()
#     optimizer.step()

# print("Optimal theta found by Adam:", theta.item())
# print("Function value at optimal theta:", objective_function(theta).item())



# import numpy as np
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt

# # Define the function
# def g(x):
#     x1, x2 = x
#     term1 = (10 * np.sin(0.05 * np.pi * x1)**6) / (2**(2 * ((x1 - 90) / 50)**2))
#     term2 = (10 * np.sin(0.05 * np.pi * x2)**6) / (2**(2 * ((x2 - 90) / 50)**2))
#     return -(term1 + term2)  # Negative for maximization

# # Define bounds for variables
# bounds = [(0, 100), (0, 100)]

# # Perform a grid search to find starting points
# x_vals = np.linspace(0, 100, 200)
# y_vals = np.linspace(0, 100, 200)
# X, Y = np.meshgrid(x_vals, y_vals)
# Z = np.array([g([x, y]) for x, y in zip(X.ravel(), Y.ravel())])
# Z = Z.reshape(X.shape)

# # Identify potential local optima
# potential_points = []
# threshold = -0.1  # Adjust to filter noise

# for i in range(1, len(x_vals) - 1):
#     for j in range(1, len(y_vals) - 1):
#         z = Z[j, i]
#         if z < threshold:
#             # Check if it's a local minimum in a 3x3 neighborhood
#             neighbors = [
#                 Z[j - 1, i], Z[j + 1, i], Z[j, i - 1], Z[j, i + 1],
#                 Z[j - 1, i - 1], Z[j - 1, i + 1], Z[j + 1, i - 1], Z[j + 1, i + 1]
#             ]
#             if z < min(neighbors):  # Local minimum condition
#                 potential_points.append((x_vals[i], y_vals[j]))

# # Refine local optima using local optimization
# local_optima = []
# for pt in potential_points:
#     res = minimize(g, pt, bounds=bounds, method="L-BFGS-B")
#     if res.success:
#         local_optima.append((tuple(res.x), -res.fun))  # Store as (point, value)

# # Filter unique optima within tolerance
# tolerance = 1e-3
# unique_optima = []
# for opt in local_optima:
#     is_unique = all(
#         np.linalg.norm(np.array(opt[0]) - np.array(u[0])) > tolerance
#         for u in unique_optima
#     )
#     if is_unique:
#         unique_optima.append(opt)

# # Print all local optima
# unique_optima.sort(key=lambda x: -x[1])  # Sort by descending objective value
# print("All local optima (sorted by value):")
# for i, opt in enumerate(unique_optima):
#     print(f"Local optimum {i + 1}: Point={opt[0]}, Value={opt[1]}")

# # Visualize the results
# plt.figure(figsize=(10, 8))
# cp = plt.contourf(X, Y, -Z, levels=50, cmap='viridis')
# plt.colorbar(cp)
# plt.title("Contour plot of g(x1, x2)")
# plt.xlabel("x1")
# plt.ylabel("x2")
# for opt in unique_optima:
#     plt.scatter(opt[0][0], opt[0][1], color='red', label=f"Optimum (Value={opt[1]:.2f})")
# plt.legend()
# plt.show()



import numpy as np

# 定义目标函数 (高维示例)
def f(theta, x):
     

    return -10 * (np.sin(0.5 * np.pi * (theta - x)) ** 6) / (2 ** (2*((theta - x) / 20) ** 2)) + 10

# 有限差分梯度计算
def finite_difference_gradient(f, theta, x, h=1e-5):
    """
    使用有限差分法计算梯度。
    f: 目标函数
    theta: 当前点 (高维变量)
    x: 样本点
    h: 有限差分的步长
    """
    grad = np.zeros_like(theta)
    for k in range(len(theta)):
        theta_plus = np.copy(theta)
        theta_plus[k] += h
        grad[k] = (f(theta +h , x) - f(theta - h , x)) / (2*h)
    return grad

# 高维优化算法
def high_dim_local_optimization(x_samples, dim, K=10, max_iter=50, h=1e-5, learning_rate=0.0001,grad_tol=1e-4):
    """
    高维空间中局部极值优化算法。
    x_samples: 样本点集
    dim: 变量维度
    K: 每个点随机生成的候选解数目
    max_iter: 梯度下降最大迭代次数
    h: 有限差分的步长
    learning_rate: 梯度下降的学习率
    """
    local_points = []
    for x in x_samples:
        # Step 1: 随机生成 K 个候选解
        solutions = [np.random.uniform(-10, 10, size=(dim,)) for _ in range(K)]
        
        print(min([f(theta, x) for theta in solutions]))
        # Step 2: 按目标函数值选择初始解
        theta_hat = min(solutions, key=lambda theta: f(theta, x))
        
        # Step 3: 基于有限差分和梯度下降优化
        for _ in range(max_iter):
            grad = finite_difference_gradient(f, theta_hat, x, h)
            grad_norm = np.linalg.norm(grad)  # 计算梯度的范数
            
            if grad_norm < grad_tol:  # 如果梯度小于设定阈值，停止迭代
                print(f"梯度收敛: {grad_norm} < {grad_tol}")
                break
            
            theta_hat = theta_hat - learning_rate * grad  # 梯度下降更新
            
        # 保存优化后的局部极值点
        local_points.append(f(theta_hat, x))
    return local_points

# 示例使用
if __name__ == "__main__":
    # 高维输入维度
    dim = 1

    # 定义样本点 (例如 5 个高维点)
    x_samples = [0]

    # 运行优化算法
    local_solutions = high_dim_local_optimization(x_samples, dim, K=100, max_iter=100, learning_rate=0.05)

    # 输出结果
    for i, solution in enumerate(local_solutions):
        print(f"样本点 {i + 1} 的局部极值点: {solution}")
