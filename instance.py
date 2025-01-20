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


# 根据写入的参数添加边
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


# 两个线段加边 seg2
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


# 三个线段加边 seg3
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


# 再次基于voronoi加边，邻居的邻居  应该是逻辑有问题
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

# nei2
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

# 从文件读取LKH的结果
def read_LKH_result(num, method, multiple=None):
    '''
    函数说明:
    这里读取的是LKH的结果
    num: 城市个数
    multiple: 倍数
    method: 使用的是哪一种TSP问题(4种)  complete|de|seg|nei  其中complete代表最优解

    返回：
    G, tour, length
    '''
    if method == 'complete':
        path = f"even/LKH_complete_graph/tour/random{num}.txt"
    else:
        path = f"even/LKH_uncomplete_graph/x{multiple}/{method}/tour/random{num}.txt"



    # 从文件读取最佳路径
    with open(path, "r") as file:
        result = file.readlines()
        # 读取tour
        tour = result[6:-2]

        # 获取最优路径长度
        length = int(result[0].rsplit('.', 2)[1])

    # 格式转换，去除换行符(LKH的结果从1开始)
    tour = list(map(lambda x: int(x) - 1, tour))

    # 创建LKH得到的最优路径图
    G = nx.Graph()

    # 添加边
    edges = [(tour[i], tour[(i + 1) % num]) for i in range(num)]
    G.add_edges_from(edges)

    return G, tour, length
    

def read_concorde_result(num, method, multiple=None):
    '''
    函数说明:
    这里读取的是concorde的结果(concorde的结果从0开始)
    num: 城市个数
    method: 使用的是哪一种TSP问题(4种)  complete|de|seg|nei  其中complete代表最优解

    返回：
    G, tour, length, time
    '''
    # 是否是完全图
    if method == 'complete':
        tour_path = f"even/concorde_complete_graph/solution/random{num}.sol"
        log_path = f"even/concorde_complete_graph/time_log/random{num}_output.log"
    else:
        tour_path = f"even/concorde_uncomplete_graph/x{multiple}/{method}/solution/random{num}.sol"
        log_path = f"even/concorde_uncomplete_graph/x{multiple}/{method}/time_log/random{num}_output.log"



    # 从文件读取最佳路径
    with open(tour_path, "r") as file:

        # 打开文件并读取所有行
        lines = file.readlines()

        # 跳过第一行，因为它是城市的总数
        tour = []

        # 从第二行开始读取城市序号
        for line in lines[1:]:
            # 分割每一行，转换为整数，然后添加到列表中
            tour.extend(map(int, line.strip().split()))


    # 从log文件中只提取solution
    with open(log_path, 'r') as file:
        content = file.read()

        # 使用正则表达式查找Optimal Solution(length)
        length_match = re.search(r"Optimal Solution: (\d+\.\d+)", content)

        # 提取并打印结果
        length = length_match.group(1) if length_match else "Not found"


    # 创建LKH得到的最优路径图
    G = nx.Graph()

    # 添加边
    edges = [(tour[i], tour[(i + 1) % num]) for i in range(num)]
    G.add_edges_from(edges)

    return G, tour, int(float(length))

# 判断是否是子图，只考虑边(最优路径图，添加边的图)
def is_subgraph(G_optimal, G_add_edges):
    '''
    G_optimal: 小图
    G_add_edges: 大图
    '''

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




# 基于原有距离矩阵生成missing edges的距离矩阵，用最大距离
# 返回 0：用最大值代替   1：减去最大值
def creat_dis_mat_missing_edges(n, G_add_edges, dis_mat_for_qubo, max_distance, multiple=1):
    # 完全图的边
    complete_edges = [{i, j} for i in range(n) for j in range(i + 1, n)]

    # 存在的边
    exist_edges = list(map(lambda x: set(x), G_add_edges.edges()))

    # 求基于完全图消失的边
    missing_edges = list(filter(lambda x: x not in exist_edges, complete_edges))

    # 创建基于消失边的距离矩阵
    for edge in missing_edges:
        temp = list(edge)
        dis_mat_for_qubo[temp[0], temp[1]] = multiple * max_distance
        dis_mat_for_qubo[temp[1], temp[0]] = multiple * max_distance

    # -----------除对角线外同时减去最大距离
    # 复制数组以保持原数组不变
    dis_mat_missing = dis_mat_for_qubo.copy()

    # 获取对角线掩码矩阵
    diag_mask = np.eye(dis_mat_for_qubo.shape[0], dtype=bool)

    # 对所有非对角线的元素减去相同的值，例如减去2
    dis_mat_for_qubo[~diag_mask] -= max_distance

    return dis_mat_missing, dis_mat_for_qubo




def uniform_coord(n):
    # 随机数种子选取 确保每次生成的一致
    np.random.seed(n)
    coord = np.random.random((n, 2)) * 100

    return coord


class instance:
    def __init__(self, n):
        # 城市个数
        self.n = n
        # # ---------------------------------------------
        # 城市坐标 平均分布
        self.coord = uniform_coord(self.n)
        # # 距离矩阵
        self.mat = dis_mat(self.coord)
        # #最大城市间距离
        self.max_distance = int(np.max(self.mat))
        # # ---------------------------------------------
        # 为了画图的参数
        # self.graph_pos = {i: self.coord[i] for i in range(self.n)}
        # ---------------------------------------------
        # 对应的完全图
        self.graph_complete = nx.complete_graph(self.n)  # 对应的图
        # # ---------------------------------------------
        # 普通的delauny三角分割
        self.graph_de = delaunay(self.coord)
        # -----------------------------------------------
        # 基于de + seg1 + seg2 + seg3
        self.graph_de_seg1_seg2_seg3 = delaunay(
            self.coord,
            seg1=edges_add_seg1(self.coord),
            seg2=edges_add_seg2(self.coord),
            seg3=edges_add_seg3(self.coord)
        )        
        # # ---------------------------------------------
        # 基于de + nei2 + nei3
        self.graph_de_nei2_nei3 = delaunay(
            self.coord,
            nei2=edges_add_nei2(self.coord),
            nei3=edges_add_nei3(self.coord)
        )
        # # ---------------------------------------------
        # 基于非完全图的距离矩阵
        # Python中的可变类型在作为参数传递给函数时，因为传递的是对象的引用而不是其副本。
        # 当你在函数内部修改这些可变对象时，外部的原始对象也会被修改。
        multiple = 1
        # 返回：0:把不存在的边替换为最大距离     1:除对角线外减去最大距离
        ##########################
        # delaunay三角分割
        self.mat_missing_edges_de, self.mat_missing_edges_de_for_qubo = creat_dis_mat_missing_edges(
            self.n, self.graph_de, self.mat.copy(), self.max_distance, multiple
        )
        ##########################
        # de + seg1 + seg2 + seg3
        self.mat_missing_edges_de_seg1_seg2_seg3, self.mat_missing_edges_de_seg1_seg2_seg3_for_qubo = creat_dis_mat_missing_edges(
            self.n, self.graph_de_seg1_seg2_seg3, self.mat.copy(), self.max_distance, multiple
        )

        #######################
        # de + nei2 + nei3
        self.mat_missing_edges_de_nei2_nei3, self.mat_missing_edges_de_nei2_nei3_for_qubo = creat_dis_mat_missing_edges(
            self.n, self.graph_de_nei2_nei3, self.mat.copy(), self.max_distance, multiple
        )
        #######################
        # # 统计非完全图矩阵中最大距离出现的次数 对称矩阵除以2
        # # 并计算可以削减的二次项个数

        # 完全图的二次项个数
        # self.num_quadratic_complete = (self.n ** 2 *(self.n - 1)) * 2
  
        # de的二次项个数
        # self.num_quadratic_de = (self.n ** 2 *(self.n - 1)) + (int(np.count_nonzero(self.mat_missing_edges_de_for_qubo)) * self.n)
        
        # seg的二次项个数
        # self.num_quadratic_de_seg1_seg2_seg3 = (self.n ** 2 *(self.n - 1)) + (int(np.count_nonzero(self.mat_missing_edges_de_seg1_seg2_seg3_for_qubo)) * self.n)
        
        # nei的二次项个数
        # self.num_quadratic_de_nei2_nei3 = (self.n ** 2 *(self.n - 1)) + (int(np.count_nonzero(self.mat_missing_edges_de_nei2_nei3_for_qubo)) * self.n)
        
        

    # 写入坐标
    def write_coord(self):

        with open(f"so_big_ins/random{self.n}.txt", "w") as file:
            # 遍历坐标写入文件
            for i in range(self.n):
                file.write(f"{self.coord[i,0]} {self.coord[i,1]}\r")

    # 写入矩阵
    def write_mat(self, method ,multiple):
        # 写参数
        with open(f"even/LKH_uncomplete_graph/x{multiple}/{method}/mat/random{self.n}.tsp", "w") as file:
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
                        file.write(str(self.mat_missing_edges_de_nei2_nei3[i, j])[:-2] + "\r")

            file.write("EOF")

    # 写入参数文件
    def write_par(self, method, multiple):
        with open(f"even/LKH_uncomplete_graph/x{multiple}/{method}/par/random{self.n}.par", "w") as file:
            file.write(
                f"PROBLEM_FILE = even/LKH_uncomplete_graph/x{multiple}/{method}/mat/random{self.n}.tsp\r\
INITIAL_PERIOD = 1000\r\
MAX_CANDIDATES = 4\r\
MAX_TRIALS = 1000\r\
MOVE_TYPE = 6\r\
PATCHING_C = 6\r\
PATCHING_A = 5\r\
RECOMBINATION = GPX2\r\
RUNS = 10\r\
TRACE_LEVEL = 0\r\
TOUR_FILE = even/LKH_uncomplete_graph/x{multiple}/{method}/tour/random{self.n}.txt"
            )

    # LKH
    def LKH(self):
        subprocess.run(["LKH-2.exe", f"even/even_complete_graph_new/par/random{self.n}.par"])
        os.system(f'echo.|LKH-2.exe even/even_complete_graph_new/par/random{self.n}.par')

    # 把控制台输出信息直接写入log文件
    def LKH_test(self, mul, method):
        with open(f"even/LKH_uncomplete_graph/x{mul}/{method}/time_log/random{self.n}.txt", "w", encoding="utf-8") as log:
            process = subprocess.Popen(
            ["LKH-2.exe", f"even/LKH_uncomplete_graph/x{mul}/{method}/par/random{self.n}.par"],
            stdin=subprocess.PIPE,
            stdout=log
        )
        
            # 发送回车键以继续下一次循环
            process.communicate(input=b'\n')


    # LKH
    @staticmethod
    def static_LKH():
        subprocess.run(["LKH-2.exe", f"so_big_ins/random10000.par"])


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


    def tsp_qubo_model(self, mat_index):
        '''
        函数说明：根据不同的距离矩阵创建tsp的qubo模型

        mat_index:  
        0.de    
        1.de+seg1+seg2+seg3   
        2.de+nei2+nei3   
        3.complete
        '''
        # 创建二进制变量 x[i, t]，表示城市 i 是否在路径的第 t 位置上
        tsp_x = Array.create('x', shape=(self.n, self.n), vartype='BINARY')

        # 约束1：每个城市必须且只能被访问一次
        H_city = sum((sum(tsp_x[i, t] for t in range(self.n)) - 1) ** 2 for i in range(self.n))

        # 约束2：每次只能访问一个城市
        H_time = sum((sum(tsp_x[i, t] for i in range(self.n)) - 1) ** 2 for t in range(self.n))

        # 目标函数：最小化路径的总距离
        # 方法  
        # 0.de    
        # 1.de+seg1+seg2+seg3   
        # 2.de+nei2+nei3   
        # 3.complete
        mat_group = [
            self.mat_missing_edges_de_for_qubo,
            self.mat_missing_edges_de_seg1_seg2_seg3_for_qubo,
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
        qubo, _ = model.to_qubo()

        return qubo
        




# 只是为了提取time_log文件中的时间信息
def get_time_value():
    for i in range(5, 201):
        # 文件路径
        file_path = f"even/LKH_uncomplete_graph/nei/time_log/random{i}.txt"

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






            

def write_mat_concorde(mul, method):
    for i in range(5, 201):
        ins = instance(i)
        with open(f"even/concorde_uncomplete_graph/x{mul}/{method}/mat/random{ins.n}.tsp",'w') as file:
            file.write(f"NAME: random{ins.n}"+ "\n")
            file.write("TYPE: TSP"+ "\n")
            file.write(f"DIMENSION: {ins.n}"+ "\n")
            file.write("EDGE_WEIGHT_TYPE: EXPLICIT"+ "\n")
            file.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX"+ "\n")
            file.write("EDGE_WEIGHT_SECTION"+ "\n")


            temp = ins.mat_missing_edges_de_nei2_nei3.astype(int)

            # 写完整矩阵
            for row in temp:
                file.write(" ".join(map(str, row)) + "\n")
        
            # 写入 EOF
            file.write("EOF\n")

        

        


def embed_test():
    ############################
    # 方法  method
    # 0.de    
    # 1.de+seg1+seg2+seg3   
    # 2.de+nei2+nei3   
    # 3.complete

    method_lst = [
        'de',
        'de_seg1_seg2_seg3',
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

        with open(f'embed_result/{topology}/{topology}_{method}.txt', "a+", encoding="utf-8") as file:
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


# 使用图比较两个回路是否一致
def compare_graph(G1, G2):
    """
    return: 路径一样true， 不一样false
    """

    for node in range(len(G1.nodes)):
        if set(G1.neighbors(node)) != set(G2.neighbors(node)):
            return False
    else:
        return True



def main():
    for i in range(118, 119):
        ins = instance(i)



        #LKH_optimal_graph, LKH_optimal_tour, LKH_optimal_length = read_LKH_result(i, 'complete')
        # concorde_optimal_graph, concorde_optimal_tour, concorde_optimal_length = read_concorde_result(i, 'complete')

        # if compare_graph(LKH_optimal_graph, concorde_optimal_graph):
        #     print(1)
        # else:
        #     print(0)

        mul = 1
        method = 'nei'
        LKH_xianzhi_graph, LKH_xianzhi_tour, LKH_xianzhi_length = read_LKH_result(i, method, mul)
        concorde_xianzhi_graph, concorde_xianzhi_tour, concorde_xianzhi_length = read_concorde_result(i, method, mul)

        print(compare_graph(LKH_xianzhi_graph, concorde_xianzhi_graph))

        if is_subgraph(concorde_xianzhi_graph, ins.graph_de_nei2_nei3)[0] and is_subgraph(LKH_xianzhi_graph, ins.graph_de_nei2_nei3)[0]:

            print(1)
        else:
            print(0)

        
        print(LKH_xianzhi_length ,concorde_xianzhi_length)







def check():
    from minorminer import find_embedding
    import dwave_networkx as dnx
    ins = instance(16)

    qubo = ins.tsp_qubo_model(3)

    chimera = dnx.zephyr_graph(15)

    embedding = find_embedding(qubo, chimera) 

    plt.figure(figsize=(12, 8))
    dnx.draw_zephyr_embedding(chimera,emb = embedding, node_size = 10)

    plt.show()



if __name__ == "__main__":
    # embed_test()
    # ins = instance(5)
    # print(ins.mat)
    # print(ins.mat_missing_edges_de_for_qubo)

    check()





   





    
            