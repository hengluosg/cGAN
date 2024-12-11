# # import numpy as np

# # # 假设一些数据
# # L = 4  # 基站数
# # F = 3  # 频率数
# # U = 6  # 用户数

# # # 用户权重
# # w_u = np.ones(U)  # 假设每个用户的权重为1

# # # 基站位置 (x, y)
# # BS_positions = np.array([[0, 0], [5, 0], [10, 0], [15, 0]])  # 4个基站

# # # 用户位置 (x, y)
# # user_positions = np.array([[2, 0], [4, 0], [6, 0], [8, 0], [10, 1], [12, 0]])  # 6个用户

# # # 计算信道增益 G_ul = 1 / D_ul^2
# # def compute_channel_gain(user_pos, BS_pos):
    
    
# #     distance = np.linalg.norm(user_pos - BS_pos)

# #     return  1/(distance ** 2)


# # def compute_interference(G, p, l, u):
# #     interference = 0
# #     for l_prime in range(L):  # 遍历所有基站 l'
# #         if l_prime != l:  # 排除当前基站
# #             interference += G[l_prime, u] * np.sum(p[l_prime, :])  # 每个基站的总功率影响
# #     return interference

# # G = np.zeros((L, U))
# # for l in range(L):
# #     for u in range(U):
# #         G[l, u] = compute_channel_gain(user_positions[u], BS_positions[l])

# # eta_u = np.random.rand(U) + 0.1  # 每个用户的噪声项
# # p = np.random.rand(L, F)

# # rate_part = 0
# # energy_part = 0
# # for u in range(U):
# #     for l in range(L):
# #         interference = compute_interference(G, p, l, u)  
# #         rate_part += w_u[u] * np.log((G[l, u] * np.sum(p[l, :])) / (0.1 + interference))  # 0.1 为噪声项
# #         energy_part += np.sum(p[l, :])  


# # c_2 = 0.01
# # energy_part = c_2 * np.sum(p)

# # # 计算目标函数
# # objective = rate_part - energy_part

# # # 输出结果
# # print(f"信道增益 G: \n{G}")
# # print(f"用户速率部分 (效用函数): {rate_part}")
# # print(f"能耗部分: {energy_part}")
# # print(f"目标函数值: {objective}")



# import numpy as np

# # 假设一些数据
# L = 4  # 基站数
# F = 1  # 频率数
# U = 6  # 用户数

# # 用户权重
# w_u = np.ones(U)  # 假设每个用户的权重为1

# # 基站位置 (x, y)
# BS_positions = np.array([[0, 0], [5, 0], [10, 0], [15, 0]])  # 4个基站

# # 用户位置 (x, y)
# user_positions = np.array([[2, 0], [4, 0], [6, 0], [8, 0], [10, 1], [12, 0]])  # 6个用户

# # 计算信道增益 G_ul = 1 / D_ul^2
# def compute_channel_gain(user_pos, BS_pos):
#     distance = np.linalg.norm(user_pos - BS_pos)
#     return 1 / (distance ** 2)

# # 计算干扰项，计算基站 l 对用户 u 的干扰
# def compute_interference(G, p, l, u):
#     interference = 0
#     for l_prime in range(L):  # 遍历所有基站 l'
#         if l_prime != l:  # 排除当前基站
#             interference += G[l_prime, u] * np.sum(np.exp(p[l_prime, :]))  # 计算干扰时考虑指数化的功率
#     return interference






# G = np.zeros((L, U))
# for l in range(L):
#     for u in range(U):
#         G[l, u] = compute_channel_gain(user_positions[u], BS_positions[l])

# # 假设每个用户的噪声项
# eta_u = np.random.rand(U) + 0.1  # 每个用户的噪声项

# # 初始化功率矩阵 p (L * F)
# p = np.random.rand(L, F)

# # 将功率进行指数化
# p_exp = np.exp(p)  # 对功率进行指数化

# # 计算速率部分
# rate_part = 0

# for l in range(L):
#     for u in range(U):
#         interference = compute_interference(G, p_exp, l, u)
#         rate_part += w_u[u] * np.log((G[l, u] * p_exp[l, :].sum()) / (eta_u[u] + interference))





# c_2 = 0.1  # 假设能耗权重
# energy_part = c_2 * np.sum(np.exp(p))  # 对功率矩阵进行指数化计算能耗

# # 总目标函数
# objective_function = rate_part - energy_part





# # 输出结果
# print("Rate part:", rate_part)
# print("Energy part:", energy_part)
# print("Objective function:", objective_function)










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




# 计算信道增益 G_ul = 1 / D_ul^2
def compute_channel_gain(user_pos, BS_pos):
    distance = np.linalg.norm(user_pos - BS_pos)
    return 1 / (distance ** 2)


def compute_interference(G, p_exp, l, u, L):
    interference = 0
    for l_prime in range(L):  # 遍历所有基站
        if l_prime != l:  # 排除当前基站
            interference += G[l_prime, u] * np.sum(p_exp[l_prime, :])  # 干扰项
    return interference


def objective_function(p,user_positions):
   
    G = np.zeros((L, U))
    for l in range(L):
        for u in range(U):
            G[l, u] = compute_channel_gain(user_positions[u], BS_positions[l])
    p_exp = np.exp(p)

    rate_part = 0
    for l in range(L):
        for u in range(U):
            interference = compute_interference(G, p_exp, l, u, L)
            rate_part += w_u[u] * np.log((G[l, u] * p_exp[l, :].sum()) / (eta_u[u] + interference))

    # 计算能耗部分
    energy_part = c_2 * np.sum(p_exp)

    # 计算目标函数
    return -rate_part + energy_part



def compute_gradient(p, user_positions):
    
    G = np.zeros((L, U))
    for l in range(L):
        for u in range(U):
            G[l, u] = compute_channel_gain(user_positions[u], BS_positions[l])

    # 对功率矩阵进行指数化
    p_exp = np.exp(p)

    # 初始化梯度
    gradient = np.zeros_like(p)

    # 计算梯度
    for l in range(L):
        for f in range(F):
            grad_rate = 0
            grad_energy = c_2 * p_exp[l, f]  # 能耗部分的梯度

            # 计算速率部分的梯度
            for u in range(U):
                interference = compute_interference(G, p_exp, l, u, L)
                SINR = (G[l, u] * p_exp[l, f]) / (eta_u[u] + interference)
                d_interference = sum(
                    G[l_prime, u] * p_exp[l_prime, f] for l_prime in range(L) if l_prime != l
                )
                grad_rate += w_u[u] * (1 / SINR) * (
                    G[l, u] * p_exp[l, f] * (eta_u[u] + interference) - G[l, u] * p_exp[l, f] * d_interference
                ) / (eta_u[u] + interference) ** 2

            # 合并梯度
            gradient[l, f] = -grad_rate + grad_energy

    return gradient

def grid_sample(d, lowbound, upbound, point):
    n = int(round(point ** (1.0 / d))) +1 # 每个维度上的点数
    #np.random.seed(42)
    grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
    grid_points = np.array(list(product(grid_1d, repeat=d)))
    if len(grid_points) > point:
        indices = np.random.choice(len(grid_points), size=point, replace=False)
        grid_points = grid_points[indices]
    return grid_points

if __name__ == '__main__':
    # 全局参数设置
    np.random.seed(2)
    L = 4  # 基站数
    F = 1  # 频率数
    U = 6  # 用户数
    p_lowbound = 0.1
    p_upbound = 10
    # 初始化功率矩阵
    p = np.random.uniform(low=p_lowbound, high=p_upbound, size=(L, F))

    covariate_dim = 2
    BS_positions = np.array([[0, 0], [5, 0], [0, 5], [5, 5]])  # 基站位置
    user_positions = np.array([[2, 0], [4, 0], [6, 0], [8, 0], [10, 1], [12, 0]])  # 用户位置
    


    
    c_2 = 0.1  # 能耗权重
    w_u = np.ones(U)  # 用户权重
    eta_u = np.random.rand(U) + 0.1  # 每个用户的噪声项

    lowbound, upbound =  0 , 5

    


    # # 初始化功率矩阵
    # p = np.random.rand(L, F)
    result = objective_function(p,user_positions)
    print("Objective function:", result)
    gradient = compute_gradient(p, user_positions)

# 输出梯度
    print("Gradient of objective function:\n", gradient)

    for  _ in range(2):
        coa = grid_sample(2, lowbound, upbound, U)
        print(coa)






