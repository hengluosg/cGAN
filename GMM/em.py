import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

def initialize_gmm_kmeans(X, K):
    """
    使用 K-means 初始化 GMM 的参数。
    
    X: 输入数据，形状为 (n_samples, n_features)
    K: 高斯混合模型的组件数
    
    返回：均值 (means), 协方差矩阵 (covariances), 混合权重 (weights)
    """
    # 使用 K-means 聚类来初始化均值
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X)
    
    # 初始化均值为 K-means 聚类中心
    means = kmeans.cluster_centers_
    
    # 初始化协方差矩阵为每个簇内的协方差
    covariances = np.array([np.cov(X[kmeans.labels_ == i].T) for i in range(K)])
    
    # 初始化权重为每个簇的样本数与总样本数的比例
    weights = np.array([np.sum(kmeans.labels_ == i) for i in range(K)]) / X.shape[0]
    
    return means, covariances, weights
def fit_gmm_em(X, K, max_iters=50000, tol=1e-6):
    means, covariances, weights = initialize_gmm_kmeans(X, K)
    
    n_samples, n_features = X.shape
    log_likelihood_old = 0
    
    
    # Step 2: EM algorithm
    for iteration in range(max_iters):
        # E-step: 计算责任概率
        responsibilities = np.zeros((n_samples, K))
        
        for i in range(n_samples):
            for k in range(K):
                # 计算每个数据点在当前高斯分布下的概率
                responsibilities[i, k] = weights[k] * multivariate_normal.pdf(X[i], means[k], covariances[k])
        
        # 归一化责任概率
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        
        # M-step: 更新参数
        N_k = responsibilities.sum(axis=0)  # 每个高斯分布的总责任
        weights = N_k / n_samples
        
        for k in range(K):
            # 更新均值
            means[k] = np.dot(responsibilities[:, k], X) / N_k[k]
            
            # 更新协方差
            diff = X - means[k]
            covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k]
        
        # 计算对数似然，检查收敛
        log_likelihood_new = np.sum(np.log(responsibilities.sum(axis=1)))
        if np.abs(log_likelihood_new - log_likelihood_old) < tol:
            print(f"Converged at iteration {iteration}")
            break
        log_likelihood_old = log_likelihood_new

    return means, covariances, weights
    


# 测试 EM 算法
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成测试数据
X, _ = make_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=42)

# 使用手动实现的 EM 算法拟合 GMM
K = 3
means, covariances, weights = fit_gmm_em(X, K)

# 打印结果
print("Means:\n", means)
print("Covariances:\n", covariances)
print("Weights:\n", weights)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c='gray', s=3)
colors = ['r', 'g', 'b']
for k in range(K):
    plt.scatter(means[k][0], means[k][1], color=colors[k], label=f'Component {k+1}')
plt.legend()
plt.title('GMM components and data')
plt.show()



from sklearn.mixture import GaussianMixture
gmm_sklearn = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
gmm_sklearn.fit(X)

# 打印结果
print("Sklearn GMM Means:\n", gmm_sklearn.means_)
print("Sklearn GMM Covariances:\n", gmm_sklearn.covariances_)
print("Sklearn GMM Weights:\n", gmm_sklearn.weights_)

# 可视化 sklearn GMM 结果
plt.scatter(X[:, 0], X[:, 1], c='gray', s=3)
for k in range(K):
    plt.scatter(gmm_sklearn.means_[k][0], gmm_sklearn.means_[k][1], color=colors[k], label=f'Component {k+1}')
plt.legend()
plt.title('Sklearn GMM components and data')
plt.show()