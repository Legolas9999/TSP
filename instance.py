import pprint
import numpy as np
from itertools import combinations
import subprocess
import networkx as nx
import math
from scipy.spatial import Delaunay
import gurobipy as gp
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import json
import time
import os
import re

from pyqubo import Array

import embedding 


def gaussian_coord(n):
    #设定随机种子
    np.random.seed(n)

    # 正态分布生成2个随机浮点数,作为城市中心 100-900之间
    # 均值500， 标准差500
    while True:
        center_x, center_y = tuple(np.random.normal(500, 500, 2))
        if all(100 <= x <= 900 for x in [center_x, center_y]):
            break

    # 随机生成标准差
    # 正态分布生成2个随机浮点数,作为标准差， 100-900之间
    while True:
        scale_x, scale_y = tuple(np.random.normal(500, 1000, 2)) 
        if all(100 <= x <= 900 for x in [scale_x, scale_y]):
            break

    # 使用正态分布生成坐标 0-1000
    coordinates = np.zeros((n,2))
    for i in range(n):
        while True:
            x_coord = np.random.normal(center_x, scale_x, 1)
            y_coord = np.random.normal(center_y, scale_y, 1)
            if all(0 <= c <= 1000 for c in [x_coord, y_coord]):
                coordinates[i, 0] = x_coord[0]
                coordinates[i, 1] = y_coord[0]
                break
            
    return  coordinates

# 根据坐标创建矩阵
def dis_mat(coord):
    # 获取个数
    n = len(coord)

    # 距离矩阵
    mat = np.zeros((n, n))
    # 所有三角形组合
    combs = combinations(range(0, n), 2)

    for comb in combs:
        # 计算欧氏距离
        dij = np.sqrt(
            (coord[comb[0], 0] - coord[comb[1], 0]) ** 2
            + (coord[comb[0], 1] - coord[comb[1], 1]) ** 2
        )

        # 取整
        dij = round(dij)

        # 避免出现除对角线上有0的情况
        if dij == 0:
            dij += 1

        # 存入
        mat[comb[0], comb[1]] = dij

    # 补全矩阵
    for i in range(n):
        for j in range(n):
            if i > j:
                mat[i, j] = mat[j, i]

    return mat


# 根据完全图计算lambda
def iterate_for_complete_lamuda(n:int, dis_mat) -> int:
    # 先创建城市列表
    city_list = [i for i in range(n)]

    # 每个城市的lambda  -> list
    lambda_list_for_every_city = []
    # 每个城市最大lambda时左右城市索引列表
    max_lambda_left_right_list = []

    #以索引的方式遍历每个城市
    for node in range(n):
        #等待组合的列表(去除本身)
        combination_list = list(filter(lambda x: x != node, city_list))
       
        # 生成所有长度为2的组合
        combs = list(combinations(combination_list, 2))

        #该点的最大lambda
        max_lambda = 0

        # 保存lambda最大时左右城市
        left_city = None
        right_city = None

        # 遍历该点的所有组合
        for comb in combs:
            temp = 0.5*(dis_mat[comb[0], node] + dis_mat[node, comb[1]])
            if temp > max_lambda:
                max_lambda = temp
                left_city = comb[0]
                right_city = comb[1]

        
        # 存入当前点的最大lambda
        lambda_list_for_every_city.append(max_lambda)
        # 存入最大lambda时左右城市索引
        max_lambda_left_right_list.append((left_city, right_city))


    #向下取整处理
    lambda_list_for_every_city = list(map(lambda x : math.floor(x) + 1, lambda_list_for_every_city))

    ### 检查check
    temp_max = max(lambda_list_for_every_city)
    max_counter = lambda_list_for_every_city.count(temp_max)
    temp_index = lambda_list_for_every_city.index(temp_max)


    return max(lambda_list_for_every_city), lambda_list_for_every_city, (n*(n-1))/2, (max_lambda_left_right_list[temp_index][0], temp_index, max_lambda_left_right_list[temp_index][1])



# 通过delaunay计算lamuda，返回元组 0：最大lamuda  1：每个城市最大lamuda列表  2:lamuda最大时左右城市列表
def delaunay(
    #dis_mat, 
    cities_coord, seg1=None, seg2=None, seg3=None, nei2=None, nei3=None
) -> tuple:

    # 进行德劳内三角剖分
    tri = Delaunay(cities_coord)

    # 创建图对象
    G = nx.Graph()

    # 将剖分中的边添加到图中,每个simplex是三角形
    for simplex in tri.simplices:
        edges = [(simplex[i], simplex[j]) for i in range(3) for j in range(i + 1, 3)]
        G.add_edges_from(edges)

    # 添加seg1
    if seg1 is not None:
        G.add_edges_from(seg1)

    # 添加seg2
    if seg2 is not None:
        G.add_edges_from(seg2)

    # 添加seg3
    if seg3 is not None:
        G.add_edges_from(seg3)

    # 添加nei2
    if nei2 is not None:
        G.add_edges_from(nei2)

    # 添加nei3
    if nei3 is not None:
        G.add_edges_from(nei3)

    # 画出三角分割(如果增加了边就不是三角分割)
    # pos = {i: cities_coord[i] for i in range(cities_coord.shape[0])}
    # nx.draw(G,pos, with_labels=True, node_size=300, node_color='skyblue')
    # plt.show()

    ######################计算lamuda######################
    # # 记录每个城市的最大lamuda
    # max_lamuda = []

    # # 记录每个城市最大lamuda时左右城市索引
    # left_and_right_index_for_every_city = []

    # # 遍历每个节点城市，i为城市索引
    # for i in range(cities_coord.shape[0]):
    #     # 生成组合列表
    #     neighbors = list(combinations(list(G.neighbors(i)), 2))

    #     # 当前最大lamuda
    #     temp_max = 0

    #     # 当前最大lamuda左右组合,元组
    #     temp_left_right = ()

    #     # 遍历每个组合，计算lamuda
    #     for neighbor in neighbors:

    #         # 根据矩阵计算
    #         current_lamuda = 0.5 * (dis_mat[i, neighbor[0]] + dis_mat[i, neighbor[1]])

    #         # 向下取整!!!!
    #         current_lamuda = math.floor(current_lamuda)

    #         # 保存最大lamuda
    #         if current_lamuda > temp_max:
    #             temp_max = current_lamuda
    #             # 并且保存当前组合
    #             temp_left_right = neighbor

    #     # 保存
    #     max_lamuda.append(temp_max)
    #     # 保存左右索引
    #     left_and_right_index_for_every_city.append(temp_left_right)

    # # 不能刚好等于，要稍微大一点
    # max_lamuda = [temp + 1 for temp in max_lamuda]


    # #####检查部分
    # temp_max = max(max_lamuda)
    # temp_index = max_lamuda.index(temp_max)
    ######################计算lamuda######################

    # 之前需要计算lambda的return
    # return max(max_lamuda), max_lamuda, G, G.number_of_edges(), (left_and_right_index_for_every_city[temp_index][0], temp_index, left_and_right_index_for_every_city[temp_index][1])

    # 现在不关心lambda，只返回加边后的图
    return G




# 基于voronoi图加边，一个线段
def edges_add_seg1(cities_coord):
    # 创建 Voronoi 图
    vor = Voronoi(cities_coord)
    # voronoi_plot_2d(vor)
    # plt.show()

    # 平面分界线交点的坐标
    coord_inter = vor.vertices

    # 每条分界线两个端点索引（平面分界线交点的索引）
    ridge = vor.ridge_vertices

    # 把无穷远的分界线去除(保留有限ridge)
    ridge = [node for node in ridge if -1 not in node]

    # 计算每个有限ridge长度
    ridge_len = []
    for node in ridge:
        dis = np.sqrt(
            (coord_inter[node[0]][0] - coord_inter[node[1]][0]) ** 2
            + (coord_inter[node[0]][1] - coord_inter[node[1]][1]) ** 2
        )

        ridge_len.append(dis)

    # 设定阈值
    threshold = max(ridge_len)

    # 筛选要变成点的边
    ridge_to_node = [
        node for node in ridge if ridge_len[ridge.index(node)] <= threshold
    ]

    # 所有的面(分界线端点索引)
    regions = vor.regions

    # 等待连接的region
    regions_to_connect = []
    for node in ridge_to_node:
        # 两个端点所属的面
        node1 = []
        node2 = []
        for region in regions:
            if node[0] in region:
                node1.append(regions.index(region))
            if node[1] in region:
                node2.append(regions.index(region))
                pass

        # 转换为集合
        node1 = set(node1)
        node2 = set(node2)

        # 取互不相交的部分(避免重复)
        # temp = (node1 - node2) | (node2 - node1)
        temp = node1.symmetric_difference(node2)

        if temp not in regions_to_connect:
            regions_to_connect.append(temp)

    # 原始点所属region
    regions = vor.point_region

    # 需要连接的原始点
    nodes_to_connect = []

    for Region in regions_to_connect:
        # 存储对应的原始点索引
        temp = []
        for region in Region:
            temp.append(np.where(regions == region)[0][0])
            pass
        nodes_to_connect.append(temp)

    nodes_to_connect = list(map(lambda x: tuple(x), nodes_to_connect))
    #print(nodes_to_connect)
    return nodes_to_connect


# seg1
def edges_add_seg1_new(cities_coord):
    # 创建Voronoi 图
    vor = Voronoi(cities_coord)

    # voronoi顶点的坐标
    vor_vertices = vor.vertices

    # 每条ridge两个端点索引（voronoi顶点的索引）
    ridges = vor.ridge_vertices
    # print(ridges)

    # 带有母点索引的区域
    regions_with_mother_point = list(vor.point_region)

    # 带有voronoi顶点索引的区域
    regions_with_voronoi_vertex = vor.regions
    # print(regions_with_voronoi_vertex)

    # 需要被连接的边
    edges_to_connect = []

    # 连接关系
    connect_relation = []
    # 遍历每个voronoi顶点
    for vor_vertex in range(len(vor_vertices)):
        # 与各个voronoi顶点相邻的voronoi顶点
        temp = [
            vertex
            for ridge in ridges
            if vor_vertex in ridge
            for vertex in ridge
            if vertex != vor_vertex
        ]

        # 去除无穷远点
        temp = list(filter(lambda x: x != -1, temp))

        connect_relation.append(temp)

    #
    # print(connect_relation)

    # 存储路径
    # 遍历每个voronoi顶点
    for vor_vertex in range(len(vor_vertices)):
        routes = [
            (vor_vertex, level_1_vertex)
            for level_1_vertex in connect_relation[vor_vertex]
        ]

        # print(routes)
        # 遍历每个路径
        for route in routes:

            vertex_1_regions = [
                vertex_1_region
                for vertex_1_region in regions_with_voronoi_vertex
                if route[0] in vertex_1_region
            ]
            vertex_2_regions = [
                vertex_2_region
                for vertex_2_region in regions_with_voronoi_vertex
                if route[1] in vertex_2_region
            ]

            # 获取每个voronoi顶点相关联的region的索引
            vertex_1_region_index = []
            vertex_2_region_index = []

            for vertex_1_region, vertex_2_region in zip(
                vertex_1_regions, vertex_2_regions
            ):
                vertex_1_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_1_region)
                )
                vertex_2_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_2_region)
                )

            # 需要连接的两个区域的索引
            region_set = set(vertex_1_region_index).symmetric_difference(
                set(vertex_2_region_index)
            )

            # print(region_set)

            edges_to_connect.append(
                (
                    regions_with_mother_point.index(region_set.pop()),
                    regions_with_mother_point.index(region_set.pop()),
                )
            )
    return edges_to_connect


# 两个线段加边
def edges_add_seg2(cities_coord):
    # 创建Voronoi 图
    vor = Voronoi(cities_coord)

    # voronoi顶点的坐标
    vor_vertices = vor.vertices

    # 每条ridge两个端点索引（voronoi顶点的索引）
    ridges = vor.ridge_vertices

    # 带有母点索引的区域
    regions_with_mother_point = list(vor.point_region)

    # 带有voronoi顶点索引的区域
    regions_with_voronoi_vertex = vor.regions

    # 需要被连接的边
    edges_to_connect = []

    # 连接关系
    connect_relation = []
    # 遍历每个voronoi顶点
    for vor_vertex in range(len(vor_vertices)):
        # 与各个voronoi顶点相邻的voronoi顶点
        temp = [
            vertex
            for ridge in ridges
            if vor_vertex in ridge
            for vertex in ridge
            if vertex != vor_vertex
        ]

        connect_relation.append(temp)

    # 遍历每个voronoi顶点
    for vor_vertex in range(len(vor_vertices)):
        level_1 = [vertex for vertex in connect_relation[vor_vertex] if vertex != -1]

        level_2 = []
        for level_1_vertex in level_1:
            temp = [
                vertex
                for vertex in connect_relation[level_1_vertex]
                if vertex not in (-1, vor_vertex)
            ]

            level_2.append(temp)

        # print()
        # print(level_1)
        # print(level_2)

        # 存储路径
        routes = [
            (vor_vertex, level_1_vertex, level_2_vertex)
            for index_level_1, level_1_vertex in enumerate(level_1)
            for level_2_vertex in level_2[index_level_1]
        ]
        # print(routes)

        # 遍历每一段路径
        for route in routes:

            vertex_1_regions = [
                vertex_1_region
                for vertex_1_region in regions_with_voronoi_vertex
                if route[0] in vertex_1_region
            ]
            vertex_2_regions = [
                vertex_2_region
                for vertex_2_region in regions_with_voronoi_vertex
                if route[1] in vertex_2_region
            ]
            vertex_3_regions = [
                vertex_3_region
                for vertex_3_region in regions_with_voronoi_vertex
                if route[2] in vertex_3_region
            ]

            # 获取每个voronoi顶点相关联的region的索引
            vertex_1_region_index = []
            vertex_2_region_index = []
            vertex_3_region_index = []

            for vertex_1_region, vertex_2_region, vertex_3_region in zip(
                vertex_1_regions, vertex_2_regions, vertex_3_regions
            ):
                vertex_1_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_1_region)
                )
                vertex_2_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_2_region)
                )
                vertex_3_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_3_region)
                )

            # print(vertex_1_region_index)
            # print(vertex_2_region_index)
            # print(vertex_3_region_index)

            # front --> route[0] route[1], back --> route[1] route[2],region_list里一定会有两个元素
            front_region_list = list(
                set(vertex_1_region_index).symmetric_difference(
                    set(vertex_2_region_index)
                )
            )
            back_region_list = list(
                set(vertex_2_region_index).symmetric_difference(
                    set(vertex_3_region_index)
                )
            )

            # print(front_region_list)
            # print(back_region_list)

            # 得到需要连接的两个region的索引
            front_set = set()
            back_set = set()
            for front, back in zip(front_region_list, back_region_list):
                if route[0] in regions_with_voronoi_vertex[front]:
                    front_set.add(front)
                if route[2] in regions_with_voronoi_vertex[back]:
                    back_set.add(back)
            # print(front_set)
            # print(back_set)

            # 去重 去除相连的两个区域重合的
            set_of_this_route = front_set.union(back_set)

            # 只有长度为2的才需要连接
            if len(set_of_this_route) == 2:
                edges_to_connect.append(
                    (
                        regions_with_mother_point.index(set_of_this_route.pop()),
                        regions_with_mother_point.index(set_of_this_route.pop()),
                    )
                )

    return edges_to_connect


# 三个线段加边
def edges_add_seg3(cities_coord):
    # 创建 Voronoi 图
    vor = Voronoi(cities_coord)
    #voronoi_plot_2d(vor)
    #plt.show()

    # voronoi顶点的坐标
    vor_vertices = vor.vertices

    # 每条ridge两个端点索引（voronoi顶点的索引）
    ridges = vor.ridge_vertices

    # 把无穷远的分界线去除(保留有限ridge)
    # ridges_without_infinite = [vertex for vertex in ridges if -1 not in vertex]

    # 带有voronoi顶点索引的区域
    regions_with_voronoi_vertex = vor.regions

    # 带有母点索引的区域
    regions_with_mother_point = list(vor.point_region)

    # 需要被连接的边
    edges_to_connect = []

    # 遍历每个vor_vertex
    connect_relation = []
    for vor_vertex in range(len(vor_vertices)):

        # 与各个voronoi顶点相邻的voronoi顶点
        temp = [
            vertex
            for ridge in ridges
            if vor_vertex in ridge
            for vertex in ridge
            if vertex != vor_vertex
        ]

        connect_relation.append(temp)

    # 遍历每个voronoi顶点
    for vor_vertex in range(len(vor_vertices)):
        # 第1层（不包含-1，且不会包含自己）
        level_1 = [vertex for vertex in connect_relation[vor_vertex] if vertex != -1]

        # 第2层
        level_2 = []
        for level_1_vertex in level_1:
            temp = [
                vertex
                for vertex in connect_relation[level_1_vertex]
                if vertex not in (-1, vor_vertex)
            ]
            level_2.append(temp)

        # 第3层,列表为空代表不存在
        level_3 = []
        for index, level_2_vertices in enumerate(level_2):
            temp_2 = []
            for level_2_vertex in level_2_vertices:
                # 每个第2层的连接
                temp_1 = [
                    vertex
                    for vertex in connect_relation[level_2_vertex]
                    if vertex not in (-1, vor_vertex, level_1[index])
                ]
                temp_2.append(temp_1)
            level_3.append(temp_2)

        # 路线
        routes = [
            (vor_vertex, level_1_vertex, level_2_vertex, level_3_vertex)
            for index_level_1, level_1_vertex in enumerate(level_1)
            for index_level_2, level_2_vertex in enumerate(level_2[index_level_1])
            for level_3_vertex in level_3[index_level_1][index_level_2]
        ]
        # print()
        # print(routes)
        # 复杂写法
        # for index_level_1, level_1_vertex in enumerate(level_1):
        #     for index_level_2, level_2_vertex in enumerate(level_2[index_level_1]):
        #         for level_3_vertex in level_3[index_level_1][index_level_2]:
        #             print(vor_vertex,level_1_vertex, level_2_vertex, level_3_vertex)

        # 遍历每一段路径
        for route in routes:

            vertex_1_regions = [
                vertex_1_region
                for vertex_1_region in regions_with_voronoi_vertex
                if route[0] in vertex_1_region
            ]
            vertex_2_regions = [
                vertex_2_region
                for vertex_2_region in regions_with_voronoi_vertex
                if route[1] in vertex_2_region
            ]
            vertex_3_regions = [
                vertex_3_region
                for vertex_3_region in regions_with_voronoi_vertex
                if route[2] in vertex_3_region
            ]
            vertex_4_regions = [
                vertex_4_region
                for vertex_4_region in regions_with_voronoi_vertex
                if route[3] in vertex_4_region
            ]

            # 获取每个voronoi顶点相关联的region的索引
            vertex_1_region_index = []
            vertex_2_region_index = []
            vertex_3_region_index = []
            vertex_4_region_index = []
            for (
                vertex_1_region,
                vertex_2_region,
                vertex_3_region,
                vertex_4_region,
            ) in zip(
                vertex_1_regions, vertex_2_regions, vertex_3_regions, vertex_4_regions
            ):
                vertex_1_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_1_region)
                )
                vertex_2_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_2_region)
                )
                vertex_3_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_3_region)
                )
                vertex_4_region_index.append(
                    regions_with_voronoi_vertex.index(vertex_4_region)
                )

            # front 前面两个点 route[0] route[1], back 后面两个点 route[2] route[3],region_list里一定会有两个元素
            front_region_list = list(
                set(vertex_1_region_index).symmetric_difference(
                    set(vertex_2_region_index)
                )
            )
            back_region_list = list(
                set(vertex_3_region_index).symmetric_difference(
                    set(vertex_4_region_index)
                )
            )

            # 得到需要连接的两个region的索引
            front_set = set()
            back_set = set()
            for front, back in zip(front_region_list, back_region_list):
                if route[0] in regions_with_voronoi_vertex[front]:
                    front_set.add(front)
                if route[3] in regions_with_voronoi_vertex[back]:
                    back_set.add(back)

            # 去重
            set_of_this_route = front_set.union(back_set)

            # 只有长度为2的才需要连接
            if len(set_of_this_route) == 2:
                edges_to_connect.append(
                    (
                        regions_with_mother_point.index(set_of_this_route.pop()),
                        regions_with_mother_point.index(set_of_this_route.pop()),
                    )
                )

    return edges_to_connect


# 再次基于voronoi加边，邻居的邻居
def edges_add_nei2_error(cities_coord):
    vor = Voronoi(cities_coord)

    # voronoi边，包括射线
    ridge = vor.ridge_vertices
    
    regions = vor.regions   # [[], [1, -1, 0], [4, 2, 3], [4, 1, -1, 2], [3, 0, -1, 2], [4, 1, 0, 3]]
    point_region = vor.point_region # [2 4 5 1 3]


    # 每个节点的邻接关系，种子节点顺序
    neighbor_list = []
    for node in range(cities_coord.shape[0]): # 0-4
        # 当前种子节点对应区域索引
        current_region_index = point_region[node]  # 2

        # 当前区域所对应的voronoi顶点
        current_region_voronoi_vertex = regions[current_region_index]  # [4, 2, 3]

        # 与当前区域相邻的区域的索引
        neighbor_region = set()
        for vonoroi_vertex in current_region_voronoi_vertex: # 4,2,3
            for region in regions:   # [[], [1, -1, 0], [4, 2, 3], [4, 1, -1, 2], [3, 0, -1, 2], [4, 1, 0, 3]]
                if vonoroi_vertex in region:
                    neighbor_region.add(regions.index(region))

        # 去除自己
        neighbor_region.discard(current_region_index)
        # 加入列表
        neighbor_list.append(neighbor_region)

    print('nei2:',neighbor_list)  # 

    # 计算需要连接的边
    edge_to_connect = []
    for index in range(cities_coord.shape[0]):
        # 当前区域的邻居
        region = neighbor_list[index]

        # 当前区域的索引
        region_index = point_region[index]

        # 遍历每个邻居
        for neighbor in region:

            # 取得邻居的邻居，居然不能直接改动，得复制
            neighbors = neighbor_list[np.where(point_region == neighbor)[0][0]].copy()

            # 从中删除当前种子节点的区域
            neighbors.discard(region_index)

            # 遍历每个邻居的邻居
            for sub_neighbor in neighbors:
                temp = {index, np.where(point_region == sub_neighbor)[0][0]}
                if temp not in edge_to_connect:
                    edge_to_connect.append(temp)

    edge_to_connect = list(map(lambda x: tuple(x), edge_to_connect))
    return edge_to_connect


def edges_add_nei2(cities_coord):
    vor = Voronoi(cities_coord)

    # voronoi边，[[-1, 0], [-1, 1], [0, 1], [2, 4], [2, 3], [3, 4], [-1, 2], [1, 4], [0, 3]]
    ridges = vor.ridge_vertices

    regions = (
        vor.regions
    )  # [[], [1, -1, 0], [4, 2, 3], [4, 1, -1, 2], [3, 0, -1, 2], [4, 1, 0, 3]]

    regions_with_mother_point = list(vor.point_region)  # [2 4 5 1 3]

    # 建立邻居关系
    neighbor_relation = (
        []
    )  # [[4, 1, 2], [3, 0, 4, 2], [3, 0, 4, 1], [1, 4, 2], [3, 0, 1, 2]]
    for mother_point in range(len(cities_coord)):  # [0,1,2,3,4]
        # 当前母节点对应区域索引
        region_index_of_mother_point = regions_with_mother_point[mother_point]
        voronoi_vertice_of_region = regions[region_index_of_mother_point]  # [4, 2, 3]

        # 求当前区域的边界
        edges_of_curr_region = []  # [ [2, 4], [2, 3], [3, 4]]
        for ridge in ridges:
            if (
                ridge[0] in voronoi_vertice_of_region
                and ridge[1] in voronoi_vertice_of_region
            ):
                edges_of_curr_region.append(ridge)

        # 找哪些区域有这些边
        regions_have_edges = []  # [3,4,5]
        for ridge in edges_of_curr_region:
            for index, region in enumerate(regions):
                if (
                    ridge[0] in region
                    and ridge[1] in region
                    and index != region_index_of_mother_point
                ):  # 如果该区域有这条边
                    regions_have_edges.append(index)

        neighbor_relation.append(regions_have_edges)

    # 将邻居关系从区域索引变为母节点的索引
    for neighbor_i, neighbor in enumerate(neighbor_relation):
        for region_i, region in enumerate(neighbor):  # [3, 4, 5]
            neighbor_relation[neighbor_i][region_i] = regions_with_mother_point.index(
                region
            )
    #print(neighbor_relation)
    # for index, relation in enumerate(neighbor_relation):
    #     print(index, relation)
    # -----

    edges_to_connect = []
    for mother_point in range(len(cities_coord)):
        # 第一层
        level_1 = neighbor_relation[mother_point]

        # 第二层
        level_2 = [
            neighbor_relation[level_1_point] for level_1_point in level_1
        ]

        # print(mother_point, level_2)

        # 连接母点和第二层,自己和自己不会连接
        connect_list = [
            (mother_point, node)
            for level_2_points in level_2
            for node in level_2_points
            if node != mother_point
        ]

        # 去重
        for edge in connect_list:
            if (edge not in edges_to_connect) and (edge[1], edge[0]) not in edges_to_connect:
                edges_to_connect.append(edge)
        
    return edges_to_connect





# nei3
def edges_add_nei3(cities_coord):
    vor = Voronoi(cities_coord)

    # voronoi边，[[-1, 0], [-1, 1], [0, 1], [2, 4], [2, 3], [3, 4], [-1, 2], [1, 4], [0, 3]]
    ridges = vor.ridge_vertices

    regions = (
        vor.regions
    )  # [[], [1, -1, 0], [4, 2, 3], [4, 1, -1, 2], [3, 0, -1, 2], [4, 1, 0, 3]]

    regions_with_mother_point = list(vor.point_region)  # [2 4 5 1 3]

    # 建立邻居关系
    neighbor_relation = (
        []
    )  # [[4, 1, 2], [3, 0, 4, 2], [3, 0, 4, 1], [1, 4, 2], [3, 0, 1, 2]]
    for mother_point in range(len(cities_coord)):  # [0,1,2,3,4]
        # 当前母节点对应区域索引
        region_index_of_mother_point = regions_with_mother_point[mother_point]
        voronoi_vertice_of_region = regions[region_index_of_mother_point]  # [4, 2, 3]

        # 求当前区域的边界
        edges_of_curr_region = []  # [ [2, 4], [2, 3], [3, 4]]
        for ridge in ridges:
            if (
                ridge[0] in voronoi_vertice_of_region
                and ridge[1] in voronoi_vertice_of_region
            ):
                edges_of_curr_region.append(ridge)

        # 找哪些区域有这些边
        regions_have_edges = []  # [3,4,5]
        for ridge in edges_of_curr_region:
            for index, region in enumerate(regions):
                if (
                    ridge[0] in region
                    and ridge[1] in region
                    and index != region_index_of_mother_point
                ):  # 如果该区域有这条边
                    regions_have_edges.append(index)

        neighbor_relation.append(regions_have_edges)

    # 将邻居关系从区域索引变为母节点的索引
    for neighbor_i, neighbor in enumerate(neighbor_relation):
        for region_i, region in enumerate(neighbor):  # [3, 4, 5]
            neighbor_relation[neighbor_i][region_i] = regions_with_mother_point.index(
                region
            )
    # print('nei3:',neighbor_relation)

    # 遍历每个母节点
    # print(neighbor_relation)
    # print()
    edges_to_connect = []
    for mother_point in range(len(cities_coord)):
        # 第一层
        level_1 = neighbor_relation[mother_point]  # [4, 1, 2]

        # 第二层
        level_2 = [
            neighbor_relation[level_1_point] for level_1_point in level_1
        ]  # [[3, 0, 1, 2], [3, 0, 4, 2], [3, 0, 4, 1]]

        # 第三层
        level_3 = [
            [neighbor_relation[level_2_point] for level_2_point in level_2_points]
            for level_2_points in level_2
        ]

        # 连接母点和第三层,自己和自己不会连接
        connect_list = [
            (mother_point, node)
            for level_3_points in level_3
            for level_3_point in level_3_points
            for node in level_3_point
            if node != mother_point
        ]

        # 去重之后再添加到等待连接列表
        for edge in connect_list:
            if (edge not in edges_to_connect) and (
                (edge[1], edge[0]) not in edges_to_connect
            ):
                edges_to_connect.append(edge)

    return edges_to_connect

# 从文件读取length,针对uncomplete
def read_LKH_length_and_tour(num, method):
    with open(f"even/even_uncomplete_graph_new/{method}/tour/random{num}.txt", "r") as file:

        result = file.readlines()
        # 读取tour
        tour = result[6:-2]

        # 获取最优路径长度
        length = int(result[0].rsplit('.', 2)[1])

        #first_line = file.readline().strip()  # 读取第一行并去除两边的空白字符
        #length = int(first_line.rsplit('.', 2)[1])

    # 格式转换，去除换行符
    tour = list(map(lambda x: int(x) - 1, tour))
    return length, tour
    




# 画出最佳路径图
def read_LKH_optimal_tour_graph(num_city):
    # 从文件读取最佳路径
    with open(f"even/even_complete_graph_new/tour/random{num_city}.txt", "r") as file:
        result = file.readlines()
        # 读取tour
        tour = result[6:-2]

        # 获取最优路径长度
        length = int(result[0].rsplit('.', 2)[1])

    # 格式转换，去除换行符
    tour = list(map(lambda x: int(x) - 1, tour))

    # 创建图
    G = nx.Graph()

    # 添加边
    edges = [(tour[i], tour[(i + 1) % num_city]) for i in range(num_city)]
    G.add_edges_from(edges)

    return G, tour, length


# 判断是否是子图，只考虑边(最优路径图，添加边的图)
def is_subgraph(G_optimal, G_add_edges):

    # 边的格式转换
    G_optimal_edge_list = list(map(lambda x: set(x), list(G_optimal.edges)))
    G_add_edges_list = list(map(lambda x: set(x), list(G_add_edges.edges)))

    # 判断optimal的边是否在add_edges里，如果不在则保留
    not_in_list = list(filter(lambda x: x not in G_add_edges_list, G_optimal_edge_list))
    # print(not_in_list)
    # 列表为空时是子图
    return not not_in_list, not_in_list


# 判断是否是子图
def is_subgraph2(G_optimal, G_add_de):
    # 边的格式转换
    G_optimal_edge_set = set(map(frozenset, G_optimal.edges()))
    G_add_de_edge_set = set(map(frozenset, G_add_de.edges()))

    # 判断G_optimal的每条边是否都在G_add_de中
    not_in_set = G_optimal_edge_set - G_add_de_edge_set
    print(not_in_set)
    # 如果not_in_set为空，说明G_optimal是G_add_de的子图
    return len(not_in_set) == 0, not_in_set


# 生成包含各类数据的字典以供查阅,并且dump到json
def creat_dump_data_dic():
    dic = {}
    for i in range(5, 201):
        print(i)
        ins = instance(i)
        dic[f"{i}"] = {
            "de_edges": ins.de_edges,
            "de_seg1_seg2_seg3_edges": ins.de_seg1_seg2_seg3_edges,
            "de_nei2_nei3_edges": ins.de_nei2_nei3_edges,
            "de_lambda": ins.de_lambda,
            "de_seg1_seg2_seg3_lambda": ins.de_seg1_seg2_seg3_lambda,
            "de_nei2_nei3_lambda": ins.de_nei2_nei3_lambda,
            "de_is_subgraph": is_subgraph(ins.graph_optimal_tour, ins.graph_de)[0],
            "de_seg1_seg2_seg3_is_subgraph": is_subgraph(
                ins.graph_optimal_tour, ins.graph_de_seg1_seg2_seg3
            )[0],
            "de_nei2_nei3_is_subgraph": is_subgraph(
                ins.graph_optimal_tour, ins.graph_de_nei2_nei3
            )[0],
        }

    json.dump(dic, open("gaussain_graph_data.json", "w"), indent=4)


# 读取数据json
def read_json(n):
    with open(f"gurobi_optimal_tour_log/solution_log/random{n}.json", "r") as file:
        dic = json.load(file)
    return np.array(dic['solution_mat'])


# 基于原有距离矩阵生成missing edges的距离矩阵，用最大距离
# 返回 0：用最大值代替   1：减去最大值
def creat_dis_mat_missing_edges(n, G_add_edges, dis_mat_for_qubo, max_distance):
    # 完全图的边
    complete_edges = [{i, j} for i in range(n) for j in range(i + 1, n)]

    # 存在的边
    exist_edges = list(map(lambda x: set(x), G_add_edges.edges()))

    # 求基于完全图消失的边
    missing_edges = list(filter(lambda x: x not in exist_edges, complete_edges))

    # 创建基于消失边的距离矩阵
    for edge in missing_edges:
        temp = list(edge)
        dis_mat_for_qubo[temp[0], temp[1]] = max_distance
        dis_mat_for_qubo[temp[1], temp[0]] = max_distance

    # -----------除对角线外同时减去最大距离
    # 复制数组以保持原数组不变
    dis_mat_missing = dis_mat_for_qubo.copy()

    # 获取对角线掩码矩阵
    diag_mask = np.eye(dis_mat_for_qubo.shape[0], dtype=bool)

    # 对所有非对角线的元素减去相同的值，例如减去2
    dis_mat_for_qubo[~diag_mask] -= max_distance

    return dis_mat_missing, dis_mat_for_qubo


# 基于原有距离矩阵生成missing edges的距离矩阵，用999999
def missing_edges(n, G_add_edges):
    # 完全图的边
    complete_edges = [{i, j} for i in range(n) for j in range(i + 1, n)]

    # 存在的边
    exist_edges = list(map(lambda x: set(x), G_add_edges.edges()))

    # 求基于完全图消失的边
    missing_edges = list(filter(lambda x: x not in exist_edges, complete_edges))

    return missing_edges


# 比较最优路径是否一致，完全图的最优路径 和 非完全图的最优路径
def compare_tour(n):

    with open(f"gaussian/gaussian_complete_graph/tour/random{n}.txt", "r") as file:
        content = file.readlines()
        tour_complete = content[6:-2]
        tour_complete_energy = int(content[0].rsplit('.', 2)[1])

    with open(f"gaussian/gaussian_uncomplete_graph/de_nei2_nei3/tour/random{n}.txt", "r") as file:
        content = file.readlines()
        tour_missing = content[6:-2]
        tour_missing_energy = int(content[0].rsplit('.', 2)[1])

    tour_complete = list(map(lambda x: int(x), tour_complete))
    tour_missing = list(map(lambda x: int(x), tour_missing))

    # print(tour_complete,tour_missing)
    return tour_complete == tour_missing , tour_complete_energy, tour_missing_energy

def uniform_coord(n):
    # 随机数种子选取 确保每次生成的一致
    np.random.seed(n)
    coord = np.random.random((n, 2)) * 100

    return coord



# 输入最佳路径图
def lambda_based_on_optimal(tour, mat):
    n = len(tour)
    lambda_list = []
    for i, city in enumerate(tour):
        node_right = tour[(i + 1) % n]
        node_left = tour[i - 1]

        # 求当前节点的lambda
        current_lambda = (mat[node_left, city] + mat[city, node_right]) / 2
        current_lambda = math.floor(current_lambda)

        # 添加到list
        lambda_list.append(current_lambda)

    # 取最大值 并 +1
    max_lambda = max(lambda_list) + 1

    return max_lambda



class instance:
    def __init__(self, n):
        # 城市个数
        self.n = n
        # ---------------------------------------------
        # 城市坐标 平均分布
        self.coord = uniform_coord(self.n)
        # 距离矩阵
        self.mat = dis_mat(self.coord)
        #最大城市间距离
        self.max_distance = int(np.max(self.mat))
        # # ---------------------------------------------
        # # # 城市坐标 正态分布
        # # self.g_coord = gaussian_coord(self.n)
        # # # 距离矩阵
        # # self.g_mat = dis_mat(self.g_coord)
        # # ---------------------------------------------
        # # 为了画图的参数
        # self.graph_pos = {i: self.coord[i] for i in range(self.n)}
        # # ---------------------------------------------
        # # 基于完全图
        # result = iterate_for_complete_lamuda(self.n, self.mat)
        # self.complete_lambda = result[0]  #完全图下的lambda
        # self.complete_lambda_list = result[1]
        self.graph_complete = nx.complete_graph(self.n)  # 对应的图
        # self.complete_edges = result[2]
        # self.left_city_right_complete = result[3] # 所选取到的最大lambda时的组合

        # # ---------------------------------------------
        # 最优路径图
        result = read_LKH_optimal_tour_graph(self.n)
        self.LKH_graph_optimal = result[0]
        self.LKH_optimal_tour = result[1]
        self.LKH_optimal_length = result[2]
        # self.optimal_lambda = lambda_based_on_optimal(self.optimal_tour, self.mat)
        # # ---------------------------------------------
        # 普通的delauny
        # result = delaunay(self.mat, self.coord)
        # self.de_lambda = result[0]
        # self.de_lambda_list = result[1]
        # self.graph_de = result[2]  # 德劳内三角分割图
        # self.de_edges = result[3]
        # self.left_city_right_de = result[4]
        self.graph_de = delaunay(self.coord)
        # # ---------------------------------------------
        # 基于de + seg1 
        # result = delaunay(
        #     self.mat,
        #     self.coord,
        #     seg1=edges_add_seg1(self.coord)
        # )
        # self.de_seg1_lambda = result[0]
        # self.de_seg1_lambda_list = result[1]
        # self.graph_de_seg1 = result[2]  # 对应的图
        # self.de_seg1_edges = result[3]
        # self.graph_de_seg1 = delaunay(
        #     self.coord,
        #     seg1=edges_add_seg1(self.coord)
        # )
        # # ---------------------------------------------  
        # 基于de + seg1 + seg2
        # result = delaunay(
        #     self.mat,
        #     self.coord,
        #     seg1=edges_add_seg1(self.coord),
        #     seg2=edges_add_seg2(self.coord)
        # )
        # self.de_seg1_seg2_lambda = result[0]
        # self.de_seg1_seg2_lambda_list = result[1]
        # self.graph_de_seg1_seg2 = result[2]  # 对应的图
        # self.de_seg1_seg2_edges = result[3]

        # self.graph_de_seg1_seg2 = delaunay(
        #     self.coord,
        #     seg1=edges_add_seg1(self.coord),
        #     seg2=edges_add_seg2(self.coord)
        # )
        # # ---------------------------------------------
        # 基于de + seg1 + seg2 + seg3
        # result = delaunay(
        #     self.mat,
        #     self.coord,
        #     seg1=edges_add_seg1(self.coord),
        #     seg2=edges_add_seg2(self.coord),
        #     seg3=edges_add_seg3(self.coord)
        # )
        # self.de_seg1_seg2_seg3_lambda = result[0]
        # self.de_seg1_seg2_seg3_lambda_list = result[1]
        # self.graph_de_seg1_seg2_seg3 = result[2]  # 对应的图
        # self.de_seg1_seg2_seg3_edges = result[3]
        # self.left_city_right_seg = result[4]

        self.graph_de_seg1_seg2_seg3 = delaunay(
            self.coord,
            seg1=edges_add_seg1(self.coord),
            seg2=edges_add_seg2(self.coord),
            seg3=edges_add_seg3(self.coord)
        )
        # # ---------------------------------------------
        # 基于de + nei2 
        # result = delaunay(
        #     self.mat,
        #     self.coord,
        #     nei2=edges_add_nei2(self.coord)
        # )
        # self.de_nei2_lambda = result[0]
        # self.de_nei2_lambda_list = result[1]
        # self.graph_de_nei2 = result[2]  # 加邻居的邻居
        # self.de_nei2_edges = result[3]
        # self.left_city_right_nei2 = result[4]

        # self.graph_de_nei2 = delaunay(
        #     self.coord,
        #     nei2=edges_add_nei2(self.coord)
        # )
        # # ---------------------------------------------
        # 基于de + nei2 + nei3
        # result = delaunay(
        #     self.mat,
        #     self.coord,
        #     nei2=edges_add_nei2(self.coord),
        #     nei3=edges_add_nei3(self.coord)
        # )
        # self.de_nei2_nei3_lambda = result[0]
        # self.de_nei2_nei3_lambda_list = result[1]
        # self.graph_de_nei2_nei3 = result[2]  # 加邻居的邻居
        # self.de_nei2_nei3_edges = result[3]
        # self.left_city_right_nei = result[4]

        self.graph_de_nei2_nei3 = delaunay(
            self.coord,
            nei2=edges_add_nei2(self.coord),
            nei3=edges_add_nei3(self.coord)
        )
        # # ---------------------------------------------




        # # ---------------------------------------------
        # 基于非完全图的距离矩阵
        # Python中的可变类型在作为参数传递给函数时，因为传递的是对象的引用而不是其副本。
        # 当你在函数内部修改这些可变对象时，外部的原始对象也会被修改。

        # 返回：0:把不存在的边替换为最大距离     1:除对角线外减去最大距离

        ##########################
        # de
        # self.mat_missing_edges_de, self.mat_missing_edges_de_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de, self.mat.copy(), self.max_distance
        # )
        self.mat_missing_edges_de, self.mat_missing_edges_de_for_qubo = creat_dis_mat_missing_edges(
            self.n, self.graph_de, self.mat.copy(), self.max_distance
        )
        ##########################

        # de + seg1
        # self.mat_missing_edges_de_seg1, self.mat_missing_edges_de_seg1_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de_seg1, self.mat.copy(), self.max_distance
        # )
        # self.mat_missing_edges_de_seg1_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de_seg1, self.mat.copy(), self.max_distance
        # )

        # de + seg1 + seg2
        # self.mat_missing_edges_de_seg1_seg2, self.mat_missing_edges_de_seg1_seg2_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de_seg1_seg2, self.mat.copy(), self.max_distance
        # )
        # self.mat_missing_edges_de_seg1_seg2_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de_seg1_seg2, self.mat.copy(), self.max_distance
        # )

        # de + seg1 + seg2 + seg3
        # self.mat_missing_edges_de_seg1_seg2_seg3, self.mat_missing_edges_de_seg1_seg2_seg3_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de_seg1_seg2_seg3, self.mat.copy(), self.max_distance
        # )
        self.mat_missing_edges_de_seg1_seg2_seg3, self.mat_missing_edges_de_seg1_seg2_seg3_for_qubo = creat_dis_mat_missing_edges(
            self.n, self.graph_de_seg1_seg2_seg3, self.mat.copy(), self.max_distance
        )

        #######################
        # de + nei2
        # self.mat_missing_edges_de_nei2, self.mat_missing_edges_de_nei2_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de_nei2, self.mat.copy(), self.max_distance
        # )
        # self.mat_missing_edges_de_nei2_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de_nei2, self.mat.copy(), self.max_distance
        # )

        # de + nei2 + nei3
        # self.mat_missing_edges_de_nei2_nei3, self.mat_missing_edges_de_nei2_nei3_for_qubo = creat_dis_mat_missing_edges(
        #     self.n, self.graph_de_nei2_nei3, self.mat.copy(), self.max_distance
        # )
        self.mat_missing_edges_de_nei2_nei3, self.mat_missing_edges_de_nei2_nei3_for_qubo = creat_dis_mat_missing_edges(
            self.n, self.graph_de_nei2_nei3, self.mat.copy(), self.max_distance
        )

        #######################

        # # ---------------------------------------------
        # # 分别读取非完全图uncomplete的length和tour
        result = read_LKH_length_and_tour(self.n, 'de')
        self.LKH_de_length = result[0]
        self.LKH_de_tour = result[1]


        result = read_LKH_length_and_tour(self.n, 'seg')
        self.LKH_seg_length = result[0]
        self.LKH_seg_tour = result[1]

        result = read_LKH_length_and_tour(self.n, 'nei')
        self.LKH_nei_length = result[0]
        self.LKH_nei_tour = result[1]

        # # ---------------------------------------------
        # # 统计非完全图矩阵中最大距离出现的次数 对称矩阵除以2
        # # 并计算可以削减的二次项个数
        # self.num_quadratic_eliminated_seg = 2 * self.n * int(np.count_nonzero(self.mat_missing_edges_seg == self.max_distance) / 2)
        
        # self.num_quadratic_eliminated_nei = 2 * self.n * int(np.count_nonzero(self.mat_missing_edges_nei == self.max_distance) / 2)

        # self.num_quadratic_com = (self.n ** 2 *(self.n - 1)) + (int(np.count_nonzero(self.mat)) * self.n)
        
        # self.num_quadratic_de = (self.n ** 2 *(self.n - 1)) + (int(np.count_nonzero(self.mat_missing_edges_de_for_qubo)) * self.n)
        
        # self.num_quadratic_de_seg1_seg2_seg3 = (self.n ** 2 *(self.n - 1)) + (int(np.count_nonzero(self.mat_missing_edges_de_seg1_seg2_seg3_for_qubo)) * self.n)
        
        # self.num_quadratic_de_nei2_nei3 = (self.n ** 2 *(self.n - 1)) + (int(np.count_nonzero(self.mat_missing_edges_de_nei2_nei3_for_qubo)) * self.n)
        
        # self.num_quadratic_total = (self.n ** 2 *(self.n - 1)) * 2

    # 写入坐标
    def write_coord(self):

        with open(f"so_big_ins/random{self.n}.txt", "w") as file:
            # 遍历坐标写入文件
            for i in range(self.n):
                file.write(f"{self.coord[i,0]} {self.coord[i,1]}\r")

    # 写入矩阵
    def write_mat(self):
        # 写参数
        with open(f"even/even_uncomplete_graph_new/de/mat/random{self.n}.tsp", "w") as file:
            file.write(
                f"NAME: random{self.n}\r\
TYPE: TSP\r\
DIMENSION: {self.n}\r\
EDGE_WEIGHT_TYPE: EXPLICIT\r\
EDGE_WEIGHT_FORMAT: UPPER_DIAG_ROW\r\
EDGE_WEIGHT_SECTION\r"
            )

            # 写矩阵,只写上三角
            for i in range(self.n):
                for j in range(self.n):
                    if i <= j:
                        file.write(str(self.mat_missing_edges_de_for_qubo[i, j])[:-2] + "\r")

            file.write("EOF")

    # 写入参数文件
    def write_par(self):
        with open(f"even/even_uncomplete_graph_new/de/par/random{self.n}.par", "w") as file:
            file.write(
                f"PROBLEM_FILE = even/even_uncomplete_graph_new/de/mat/random{self.n}.tsp\r\
INITIAL_PERIOD = 1000\r\
MAX_CANDIDATES = 4\r\
MAX_TRIALS = 1000\r\
MOVE_TYPE = 6\r\
PATCHING_C = 6\r\
PATCHING_A = 5\r\
RECOMBINATION = GPX2\r\
RUNS = 10\r\
TRACE_LEVEL = 0\r\
TOUR_FILE = even/even_uncomplete_graph_new/de/tour/random{self.n}.txt"
            )

    # LKH
    def LKH(self):
        subprocess.run(["LKH-2.exe", f"even/even_complete_graph_new/par/random{self.n}.par"])
        os.system(f'echo.|LKH-2.exe even/even_complete_graph_new/par/random{self.n}.par')

    # 把控制台输出信息直接写入log文件
    def LKH_test(self):
        with open(f"even/even_uncomplete_graph_new/de/time_log/random{self.n}.txt", "w", encoding="utf-8") as log:
            process = subprocess.Popen(
            ["LKH-2.exe", f"even/even_uncomplete_graph_new/de/par/random{self.n}.par"],
            stdin=subprocess.PIPE,
            stdout=log
        )
        
            # 发送回车键以继续下一次循环
            process.communicate(input=b'\n')


    # LKH
    @staticmethod
    def static_LKH():
        subprocess.run(["LKH-2.exe", f"so_big_ins/random10000.par"])

    # 用gurobi最优化
    def gurobi(self, lambda_value, lambda_list=None):

        # 创建模型
        model = gp.Model(f"TSP_QUBO_test_{self.n}")

        # 创建二进制变量
        x = {}
        for i in range(self.n):
            for j in range(self.n):
                x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}")

        # 目标函数（使用QUBO形式）
        obj_expr = 0
        for i in range(self.n):
            for j in range(self.n):
                d_ij = self.mat[i, j]
                for t in range(self.n):
                    obj_expr += d_ij * x[i, t] * x[j, (t + 1) % self.n]

        # 城市惩罚系数为一个值
        if lambda_list == None:
            # 直接添加约束到obj，城市约束
            for i in range(self.n):
                obj_expr += self.lambda_value * (
                    (sum(x[i, t] for t in range(self.n)) - 1) ** 2
                )

        # 城市惩罚系数为列表
        else:
            # 列表形式，城市约束
            for i in range(self.n):
                obj_expr += self.de_lambda_list[i] * (
                    (sum(x[i, t] for t in range(self.n)) - 1) ** 2
                )

        # 直接添加约束到obj，时间约束
        for t in range(self.n):
            obj_expr += self.lambda_value * (
                (sum(x[i, t] for i in range(self.n)) - 1) ** 2
            )

        # 目标函数最小化
        model.setObjective(obj_expr, gp.GRB.MINIMIZE)

        # 设置求解器时间参数
        model.Params.TimeLimit = 7200

        # 设置cutoff，达到最优解就停止
        model.setParam('Cutoff', self.optimal_length)

        # 设置日志文件名
        log_file = f"gurobi_optimal_tour_log/gurobi_log/random{self.n}.log"

        # 设置求解器参数，将日志输出到文件
        model.Params.LogFile = log_file

        # 优化
        model.optimize()

        # 检查可行性
        # 到达时间限制  或者找到最优解（似乎可以不用这个条件）
        # if model.status == gp.GRB.TIME_LIMIT or model.status == gp.GRB.OPTIMAL:

        # 记录打破的制约的个数
        un_city = 0
        un_time = 0

        # 检查城市约束是否满足
        for i in range(self.n):
            if sum(x[i, t].x for t in range(self.n)) != 1:
                un_city += 1

        # 检查时间约束是否满足
        for t in range(self.n):
            if sum(x[i, t].x for i in range(self.n)) != 1:
                un_time += 1

        # 记录
        with open(f"gurobi_optimal_tour_log/gurobi_log/random{self.n}.log", "a") as file:
            file.write(f"\rcity broken:{un_city},time broken:{un_time}\r")

        # 记录当前的变量矩阵
        solution_mat = np.full((self.n, self.n), 2)
        for i in range(self.n):
            for j in range(self.n):
                solution_mat[i, j] = int(x[i, j].x)


        dic = {}
        solution_mat_list = solution_mat.tolist()
        # ndarray不能直接写入json
        dic['solution_mat'] = solution_mat_list
        dic['best_solution'] = model.ObjVal

        json.dump(dic, open(f"gurobi_optimal_tour_log/solution_log/random{self.n}.json", "w"), indent=4)




    def get_solution_mat(self):
        # 从json中读取当前解的矩阵
        with open(f"gurobi_optimal_tour_log/solution_log/random{self.n}.json", "r") as file:
            dic = json.load(file)
            array = np.array(dic['solution_mat'])
        return array



    # 用gurobi解TSP
    def gurobi_LKH(self):
        # 创建模型
        model = gp.Model(f"TSP_QUBO_test_{self.n}")

        # 创建二进制变量
        x = {}
        for i in range(self.n):
            for j in range(self.n):
                x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}")

        # 目标函数（使用QUBO形式）
        obj_expr = 0
        for i in range(self.n):
            for j in range(self.n):
                d_ij = self.mat[i, j]
                for t in range(self.n):
                    obj_expr += d_ij * x[i, t] * x[j, (t + 1) % self.n]
        pass

        # 添加约束，城市约束
        city_constraints = {}
        for i in range(self.n):
            city_constraints[i] = model.addConstr(
                gp.quicksum(x[i, t] for t in range(self.n)) == 1,
                name=f"city_constraint_{i}",
            )

        # 添加约束，时间约束
        time_constraints = {}
        for t in range(self.n):
            time_constraints[t] = model.addConstr(
                gp.quicksum(x[i, t] for i in range(self.n)) == 1,
                name=f"time_constraint_{t}",
            )

        # 目标函数最小化
        model.setObjective(obj_expr, gp.GRB.MINIMIZE)

        # 设置求解器时间参数
        model.Params.TimeLimit = 100

        # 设置日志文件名
        log_file = f"log/random{self.n}.log"

        # 设置求解器参数，将日志输出到文件
        model.Params.LogFile = log_file

        # 优化
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            print("最短路径长度:", model.objVal)
            print("最短路径:")
            tour = []

            for j in range(self.n):
                for i in range(self.n):
                    if x[i, j].x == 1:
                        tour.append(i + 1)
            print(tour)

    # 如果不是子图画出没有包含的边
    def draw_not_subgraph(self):
        result = is_subgraph(self.graph_optimal_tour, self.graph_de_seg1_seg2)
        G = self.graph_de_seg1_seg2
        # 如果不是子图
        if result[0] == False:

            # 格式化需要特别画出的边
            edge_list = list(map(lambda x: tuple(x), result[1]))

            nx.draw(
                G, self.graph_pos, with_labels=True, node_size=300, node_color="skyblue", width=0.5
            )
            # nx.draw_networkx_edges(
            #     G, self.graph_pos, edge_list, edge_color="r", width=3
            # )
            plt.show()

        else:
            return None

    # 根据nei，创建tsp的qubo模型(索引从0开始)
    def tsp_nei_qubo_model(self):

        # n
        # liner: n^2
        # quadratic（上三角）: 2n^2(n-1) 项


        # 距离矩阵（假设对称距离）
        # self.n = 4
        # self.mat = np.array([
        #     [0, 1, 2, 3],
        #     [1, 0, 4, 5],
        #     [2, 4, 0, 6],
        #     [3, 5, 6, 0]
        # ])
        # self.max_distance = 6
    


    
        # 创建二进制变量 x[i, t]，表示城市 i 是否在路径的第 t 位置上
        tsp_x = Array.create('x', shape=(self.n, self.n), vartype='BINARY')

        # 约束1：每个城市必须且只能被访问一次
        H_city = sum((sum(tsp_x[i, t] for t in range(self.n)) - 1) ** 2 for i in range(self.n))

        # 约束2：每次只能访问一个城市
        H_time = sum((sum(tsp_x[i, t] for i in range(self.n)) - 1) ** 2 for t in range(self.n))

        # 目标函数：最小化路径的总距离
        H_obj = sum(self.mat_missing_edges_nei_for_qubo[i, j] * tsp_x[i, t] * tsp_x[j, (t+1) % self.n] for i in range(self.n) for j in range(self.n) for t in range(self.n))

        # 总哈密顿量：目标函数 + 约束条件
        H = H_obj + self.max_distance * (H_city + H_time)

        # 编译模型
        model = H.compile()

        # 转换为 QUBO
        qubo, offset = model.to_qubo()

        # # 检查

        # cnt_liner = 0
        # cnt_quadratic = 0
        # for key, val in sorted(qubo.items()):
        #     # 一次项
        #     if key[0] == key[1]:
        #         cnt_liner += 1

        #     # 二次项
        #     else:
        #         cnt_quadratic += 1
        # print(cnt_quadratic)

        return qubo
    
    # 根据seg，创建tsp的qubo模型
    def tsp_seg_qubo_model(self):
    
        # 创建二进制变量 x[i, t]，表示城市 i 是否在路径的第 t 位置上
        tsp_x = Array.create('x', shape=(self.n, self.n), vartype='BINARY')

        # 约束1：每个城市必须且只能被访问一次
        H_city = sum((sum(tsp_x[i, t] for t in range(self.n)) - 1) ** 2 for i in range(self.n))

        # 约束2：每次只能访问一个城市
        H_time = sum((sum(tsp_x[i, t] for i in range(self.n)) - 1) ** 2 for t in range(self.n))

        # 目标函数：最小化路径的总距离
        H_obj = sum(self.mat_missing_edges_seg_for_qubo[i, j] * tsp_x[i, t] * tsp_x[j, (t+1) % self.n] for i in range(self.n) for j in range(self.n) for t in range(self.n))

        # 总哈密顿量：目标函数 + 约束条件
        H = H_obj + self.max_distance * (H_city + H_time)

        # 编译模型
        model = H.compile()

        # 转换为 QUBO
        qubo, offset = model.to_qubo()

        return qubo
    
    # 根据完全图，创建tsp模型
    def tsp_complete_qubo_model(self):
    
        # 创建二进制变量 x[i, t]，表示城市 i 是否在路径的第 t 位置上
        tsp_x = Array.create('x', shape=(self.n, self.n), vartype='BINARY')

        # 约束1：每个城市必须且只能被访问一次
        H_city = sum((sum(tsp_x[i, t] for t in range(self.n)) - 1) ** 2 for i in range(self.n))

        # 约束2：每次只能访问一个城市
        H_time = sum((sum(tsp_x[i, t] for i in range(self.n)) - 1) ** 2 for t in range(self.n))

        # 目标函数：最小化路径的总距离
        H_obj = sum(self.mat[i, j] * tsp_x[i, t] * tsp_x[j, (t+1) % self.n] for i in range(self.n) for j in range(self.n) for t in range(self.n))

        # 总哈密顿量：目标函数 + 约束条件
        H = H_obj + self.max_distance * (H_city + H_time)

        # 编译模型
        model = H.compile()

        # 转换为 QUBO
        qubo, offset = model.to_qubo()

        return qubo

    def tsp_qubo_model(self, mat_index):
        # 创建二进制变量 x[i, t]，表示城市 i 是否在路径的第 t 位置上
        tsp_x = Array.create('x', shape=(self.n, self.n), vartype='BINARY')

        # 约束1：每个城市必须且只能被访问一次
        H_city = sum((sum(tsp_x[i, t] for t in range(self.n)) - 1) ** 2 for i in range(self.n))

        # 约束2：每次只能访问一个城市
        H_time = sum((sum(tsp_x[i, t] for i in range(self.n)) - 1) ** 2 for t in range(self.n))

        # 目标函数：最小化路径的总距离
        # 方法  
        # 0.de    
        # 1.de+seg1  2.de+seg1+seg2   3.de+seg1+seg2+seg3   
        # 4.de+nei2  5.de+nei2+nei3   
        # 6.complete
        mat_group = [
            self.mat_missing_edges_de_for_qubo,
            self.mat_missing_edges_de_seg1_for_qubo,
            self.mat_missing_edges_de_seg1_seg2_for_qubo,
            self.mat_missing_edges_de_seg1_seg2_seg3_for_qubo,
            self.mat_missing_edges_de_nei2_for_qubo,
            self.mat_missing_edges_de_nei2_nei3_for_qubo,
            self.mat          
            ]

        # 直接取值mat_group
        H_obj = sum(mat_group[mat_index][i, j] * tsp_x[i, t] * tsp_x[j, (t+1) % self.n] for i in range(self.n) for j in range(self.n) for t in range(self.n))

        # 总哈密顿量：目标函数 + 约束条件
        H = H_obj + self.max_distance * (H_city + H_time)

        # 编译模型
        model = H.compile()

        # 转换为 QUBO
        qubo, offset = model.to_qubo()

        return qubo
        




def check():
    for i in range(5, 201):
        ins = instance(i)
        if is_subgraph(ins.graph_optimal, ins.graph_de_nei2_nei3)[0]:
            print(1)
        else:
            print(0)



# 只是为了提取time_log文件中的时间信息
def get_time_value():
    for i in range(5, 201):
        # 文件路径
        file_path = f"even/even_uncomplete_graph_new/nei/time_log/random{i}.txt"

        # 正则表达式匹配时间项
        time_pattern = r"(Time\.(min|avg|max|total) = [\d\.]+ sec\.)"

        # 提取时间信息
        time_values = {}

        # 读取文件内容并提取时间数值
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                matches = re.findall(time_pattern, line)
                for match in matches:
                    key_value = match[0].split("=")
                    key = key_value[0].strip()  # 提取 "Time.min", "Time.avg" 等
                    value = float(key_value[1].strip().split()[0])  # 提取数值部分
                    time_values[key] = value

        # 输出提取的时间信息
        # for key, value in time_values.items():
        #     print(f"{key}: {value} seconds")
        print(time_values['Time.total'])






            

def write_mat_concorde():
    for i in range(5, 201):
        ins = instance(i)
        with open(f"even/concorde_uncomplete_graph/seg/mat/random{ins.n}.tsp",'w') as file:
            file.write(f"NAME: random{ins.n}"+ "\n")
            file.write("TYPE: TSP"+ "\n")
            file.write(f"DIMENSION: {ins.n}"+ "\n")
            file.write("EDGE_WEIGHT_TYPE: EXPLICIT"+ "\n")
            file.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX"+ "\n")
            file.write("EDGE_WEIGHT_SECTION"+ "\n")


            temp = ins.mat_missing_edges_de_seg1_seg2_seg3.astype(int)

            # 写完整矩阵
            for row in temp:
                file.write(" ".join(map(str, row)) + "\n")
        
            # 写入 EOF
            file.write("EOF\n")

        

        


def embed_test():
    ############################
    # 方法  method
    # 0.de    
    # 1.de+seg1  2.de+seg1+seg2   3.de+seg1+seg2+seg3   
    # 4.de+nei2  5.de+nei2+nei3   
    # 6.complete

    method_lst = [
        'de',
        'de_seg1',
        'de_seg1_seg2',
        'de_seg1_seg2_seg3',
        'de_nei2',
        'de_nei2_nei3',
        'complete'
    ]

    # topology
    # 0.chimera  1.pegasus  2.zephyr
    topology_lst = [
        'chimera',  
        'pegasus',
        'zephyr'
    ]

    
    ############################
    # 选择方法
    method_index = 0
    method = method_lst[method_index]

    # 选择拓扑结构
    topology_index = 0
    topology = topology_lst[topology_index]
    # add的起步（最大城市大小的起步）
    max_size = 5
    ###########################


    # zephyr
    if topology == "zephyr":
        initial = 15
    
    # chimera pegasus
    else:
        initial = 16


    ################################
    # topology  size 的增加大小
    # 指定 add 0到20    21~99     100~200  201~300
    # 第二次实验 0~300
    for add in range(0, 301):
    ################################

        with open(f'embed_result3/{topology}/{topology}_{method}.txt', "a+", encoding="utf-8") as file:
            print(f"{topology} = {initial + add}", file = file)

        for i in range(max_size, 201):
            print(i)
            ins = instance(i)

            # 创建qubo模型
            qubo = ins.tsp_qubo_model(mat_index=method_index)

            # 嵌入
            result = embedding.embed_tsp(qubo, ins.n, add, topology, method)


            # 嵌入失败
            if result is False :
                max_size = i
                break




def main():
    for i in range(5, 201):
        ins = instance(i)
        print(ins.optimal_length)



if __name__ == "__main__":
    # embed_test()
    # ins = instance(5)
    # print(ins.mat)
    # print(ins.mat_missing_edges_de_for_qubo)

    main()

    
    #print(os.urandom(4))

   





    
            