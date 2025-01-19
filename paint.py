import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




# 加载 Excel 文件，指定工作簿
file_path = 'data/1.xlsx'  # 替换为你的文件路径
sheet_name = 'Concorde'  # 替换为你需要读取的工作表的名称

# 读取特定工作表
df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')


#############
n = 1
##############

x = df['index']

y1 = df[f'de-com_{n}']
y2 = df[f'seg-com_{n}']
y3 = df[f'nei-com_{n}']





fig, axes = plt.subplots(1, 3, figsize=(21, 7))  # 1 行 3 列子图

# 第 1 个子图
axes[0].scatter(x, y1, color='blue', label='Delaunay-Complete')
axes[0].set_title(f'Delaunay-Complete({sheet_name}, n={n})', fontsize=20)
axes[0].set_xlabel('instance(5~200)', fontsize=15)
axes[0].set_ylabel('energy difference', fontsize=15)
axes[0].tick_params(axis='both', which='major', labelsize=13)  # 调整刻度字体大小


# 第 2 个子图
axes[1].scatter(x, y2, color='green', label='Seg-Complete')
axes[1].set_title(f'Seg-Complete({sheet_name}, n={n})', fontsize=20)
axes[1].set_xlabel('instance(5~200)', fontsize=15)
axes[1].set_ylabel('energy difference', fontsize=15)
axes[1].tick_params(axis='both', which='major', labelsize=13)


# 第 3 个子图
axes[2].scatter(x, y3, color='orange', label='Nei-Complete')
axes[2].set_title(f'Nei-Complete({sheet_name}, n={n})', fontsize=20)
axes[2].set_xlabel('instance(5~200)', fontsize=15)
axes[2].set_ylabel('energy difference', fontsize=15)
axes[2].tick_params(axis='both', which='major', labelsize=13)

# 设置布局调整，使标题和标签不重叠
plt.tight_layout()



# 显示图形
plt.show()









#############################################################
# from matplotlib.ticker import MaxNLocator
# from matplotlib import rcParams
# rcParams['font.family'] = 'MS Gothic'
# # 加载 Excel 文件，指定工作簿
# file_path = 'data/number of edges.xlsx'  # 替换为你的文件路径
# sheet_name = 'Sheet1'  # 替换为你需要读取的工作表的名称

# # 读取特定工作表
# #df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

# # 读取特定工作表
# df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

# x = df['index']
# y1 = df[f'de']
# y2 = df[f'seg']
# y3 = df[f'nei']
# y4 = df[f'com']


# fig, ax = plt.subplots(figsize=(15, 10))



# x_ticks = range(0, 201, 25)  # 从 0 到 200，每隔 25
# y_ticks = range(0, 20001, 4000)  # 从 0 到 20，每隔 10
# ax.set_xticks(x_ticks)
# ax.set_yticks(y_ticks)



# ax.scatter(x, y1, label='Delaunay', color='blue', marker='o')
# ax.scatter(x, y2, label='Seg', color='green', marker='s')
# ax.scatter(x, y3, label='Nei', color='orange', marker='^')
# ax.scatter(x, y4, label='Complete', color='red', marker='p')

# # 设置标题和坐标轴
# ax.set_title('異なるグラフの辺の個数', fontsize=30)
# ax.set_xlabel('instance(5~200)', fontsize=20)
# ax.set_ylabel('number of edges', fontsize=20)


# # 设置坐标轴刻度标签字体大小
# ax.set_xticklabels(ax.get_xticks(), fontsize=20)
# ax.set_yticklabels(ax.get_yticks(), fontsize=20)





# # 添加图例
# ax.legend(fontsize=20)

# # 显示网格
# ax.grid(alpha=0.5)

# plt.show()

############################################



# from matplotlib.ticker import MaxNLocator
# from matplotlib import rcParams
# rcParams['font.family'] = 'MS Gothic'
# # 加载 Excel 文件，指定工作簿
# file_path = 'data/2.xlsx'  # 替换为你的文件路径
# sheet_name = '二次项'  # 替换为你需要读取的工作表的名称

# # 读取特定工作表
# #df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

# # 读取特定工作表
# df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

# x = df['index']
# y1 = df[f'comp-de_rate']
# y2 = df[f'comp-seg_rate']
# y3 = df[f'comp-nei_rate']


# fig, ax = plt.subplots(figsize=(15, 10))



# x_ticks = range(0, 201, 25)  # 从 0 到 200，每隔 25
# y_ticks = range(0, 60, 10)  # 从 0 到 20，每隔 10
# ax.set_xticks(x_ticks)
# ax.set_yticks(y_ticks)



# ax.scatter(x, y1, label='Delaunay', color='blue', marker='o')
# ax.scatter(x, y2, label='Seg', color='green', marker='s')
# ax.scatter(x, y3, label='Nei', color='orange', marker='^')

# # 设置标题和坐标轴
# ax.set_title('制限されたTSP問題の二次項数の削減率', fontsize=30)
# ax.set_xlabel('instance(5~200)', fontsize=20)
# ax.set_ylabel('Reduction Rate(%)', fontsize=20)


# # 设置坐标轴刻度标签字体大小
# ax.set_xticklabels(ax.get_xticks(), fontsize=20)
# ax.set_yticklabels(ax.get_yticks(), fontsize=20)





# # 添加图例
# ax.legend(fontsize=20)

# # 显示网格
# ax.grid(alpha=0.5)

# plt.show()
