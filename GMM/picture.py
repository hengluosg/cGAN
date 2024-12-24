import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义两个高斯分布的参数
mu1, sigma1 = -8, 1  # 第一个高斯分布的均值和标准差
mu2, sigma2 = 2, 2  # 第二个高斯分布的均值和标准差

# 通过平均方法得到新的高斯分布参数
mu_new = (mu1 + mu2) / 2
sigma_new = np.sqrt((sigma1**2 + sigma2**2) / 2)

# 生成x坐标
x = np.linspace(-20, 7, 1000)

# 计算三个分布的概率密度函数（PDF）
pdf1 = norm.pdf(x, mu1, sigma1)
pdf2 = norm.pdf(x, mu2, sigma2)
pdf_new = norm.pdf(x, mu_new, sigma_new)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x, pdf1, label=f'N({mu1}, {sigma1}^2)', color='blue')
plt.plot(x, pdf2, label=f'N({mu2}, {sigma2}^2)', color='red')
plt.plot(x, pdf_new, label=f'Average N({mu_new:.2f}, {sigma_new**2:.2f})', color='green', linestyle='--')
plt.title("Combining Two 1D Gaussian Distributions")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()
