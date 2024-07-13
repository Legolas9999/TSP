import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成符合正态分布的城市坐标
center_x, center_y = 100, 100  # 中心点
std_dev_x, std_dev_y = 10, 20  # 标准差

# 生成1000个城市坐标
num_points = 200
x_coords = np.random.normal(center_x, std_dev_x, num_points)
y_coords = np.random.normal(center_y, std_dev_y, num_points)

# 绘制城市坐标
plt.figure(figsize=(10, 6))
plt.scatter(x_coords, y_coords)
plt.title('City Coordinates Generated from Normal Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
#plt.grid(True)
plt.show()
