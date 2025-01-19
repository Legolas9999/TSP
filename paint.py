import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载 Excel 文件
# file_path = '统计文档/最新统计/限制的图的解是否都在对应图中.xlsx.xlsx'  # 替换为你的文件路径
# df = pd.read_excel(file_path, engine='openpyxl')



# 加载 Excel 文件，指定工作簿
file_path = 'data/1.xlsx'  # 替换为你的文件路径
sheet_name = 'Concord'  # 替换为你需要读取的工作表的名称

# 读取特定工作表
df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')


#############
method = 'nei'
n = 1
##############

x = df['index']
y = df[f'{method}-com_{n}']

if method == 'de':
   
    color = 'blue'
    title = f'Delaunay-Complete({sheet_name},n={n})'

elif method == 'seg':
    color = 'green'
    title = f'Seg-Complete({sheet_name},n={n})'

elif method == 'nei':
    color = 'orange'
    title = f'Nei-Complete({sheet_name},n={n})'




##############################################



# 创建图表，设置画面大小为10x6英寸
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制散点图
ax.scatter(x, y, color=color)

# 设置图表标题和坐标轴标签
ax.set_title(title, fontsize=30)
ax.set_xlabel('instance(5~200)', fontsize=20)
ax.set_ylabel('energy difference', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)  # 设置主刻度标签的字号

# ax.set_title('Scatter Plot Example', fontsize=16, fontweight='bold', fontstyle='italic')
# ax.set_xlabel('X coordinate', fontsize=14, fontweight='bold')
# ax.set_ylabel('Y coordinate', fontsize=14, fontstyle='italic')



# 显示图形
plt.show()