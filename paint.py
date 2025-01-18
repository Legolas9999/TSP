import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载 Excel 文件
# file_path = '统计文档/最新统计/限制的图的解是否都在对应图中.xlsx.xlsx'  # 替换为你的文件路径
# df = pd.read_excel(file_path, engine='openpyxl')



# 加载 Excel 文件，指定工作簿
file_path = 'data/1.xlsx'  # 替换为你的文件路径
sheet_name = 'LKH'  # 替换为你需要读取的工作表的名称

# 读取特定工作表
df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

# 假设工作表中的列名为 'X' 和 'Y'
x = df['index']
y = df['de-com_10']



# 假设你的 Excel 列名为 'X' 和 'Y'
# x = df['X']
# y = df['Y']

# 生成随机数据
# n = 100
# x = np.random.random(n) * 100  # 生成0到100之间的随机x坐标
# y = np.random.random(n) * 100  # 生成0到100之间的随机y坐标

# 绘制散点图
plt.scatter(x, y)

# delaunay

# 添加标题和轴标签
plt.title('Delaunay-Complete')
plt.xlabel('instance(5~200)')
plt.ylabel('energy difference')

# 显示图形
plt.show()
