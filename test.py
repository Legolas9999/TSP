import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 定义多个城市中心及其方差
centers = [(20, 30), (50, 50), (80, 80)]
std_devs = [5, 10, 7]
weights = [0.3, 0.5, 0.2]  # 每个中心的权重

# 生成数据点
n_samples = 1000
X = np.zeros((n_samples, 2))

for i, (center, std, weight) in enumerate(zip(centers, std_devs, weights)):
    count = int(weight * n_samples)
    X[i * count: (i + 1) * count, 0] = np.random.normal(center[0], std, count)
    X[i * count: (i + 1) * count, 1] = np.random.normal(center[1], std, count)

# 绘制城市坐标分布图
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('City Coordinates - Mixture of Gaussians')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
