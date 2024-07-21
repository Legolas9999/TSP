from instance import instance, missing_edges
import networkx as nx
import numpy as np
from itertools import combinations

#---------------------------------------------------------------------------------
# 与自己最近的点,针对一个例子，返回一组边
def closest_point(ins):
    '''
    返回某个实例中，距离每个点最近的点
    '''
    edges = []

    for point in range(ins.n):
        row = [x for x in ins.mat[point] if x > 0]
            
        closest = min(row)
        closest_index = np.where(ins.mat[point] == closest)[0] # 可能有多个

        for node in closest_index:
            edges.append((point, node))
    
    return edges # # [(0, 7), (0, 27),...]


def check_closest():
    '''
    通过调用closest_point来确认当前加边图中有没有这条最近边
    结果确实都有
    '''
    for i in range(5, 201):
        ins = instance(i)
        result = closest_point(ins)

        result = [set(edge) for edge in result]
        edges_de = [set(edge) for edge in ins.graph_de.edges()]  # [{0, 7}, {0, 27},...]
        edges_seg = [set(edge) for edge in ins.graph_de_seg1_seg2_seg3.edges()]
        edges_nei = [set(edge) for edge in ins.graph_de_nei2_nei3.edges()]


        for edge in result:
            # if edge not in edges_de:
            #     print(edge, 'de_error')
            if edge not in edges_seg:
                print(edge, 'seg_error')
            if edge not in edges_nei:
                print(edge, 'nei_error')
        print(i)
#---------------------------------------------------------------------------------



def Gabriel_edges(ins):
    '''
    返回Gabriel图中的边
    '''
    # 总的边
    Gabriel_edges = []

    # 所有的边
    edges = list(combinations(range(ins.n), 2)) # [(0, 1), (0, 2),...]

    #检查每一个边是否在Gabriel图里
    for edge in edges:

        mid_point_x = (ins.coord[edge[0], 0] + ins.coord[edge[1], 0]) / 2
        mid_point_y = (ins.coord[edge[0], 1] + ins.coord[edge[1], 1]) / 2

        # 该边的外接圆半径
        radius = np.sqrt((mid_point_x - ins.coord[edge[0], 0])**2
                          + (mid_point_y - ins.coord[edge[0], 1])**2)
        
        # 
        flag = True

        # 检查除该边两端点外的所有点到该边中点的距离
        for point in range(ins.n):

            # 跳过该边两端点
            if point in (edge[0], edge[1]):
                continue

            # 该点到边的中点距离
            dis = np.sqrt((mid_point_x - ins.coord[point, 0])**2
                          + (mid_point_y - ins.coord[point, 1])**2)

            # 判断，一旦有其他点在该半径内，则该边的flag置为false
            if dis < radius:
                flag = False
                
        # 决定是否要加入该边
        if flag and set(edge) not in Gabriel_edges:
            Gabriel_edges.append(set(edge))

    # 格式
    Gabriel_edges = list(map(lambda x : tuple(x), Gabriel_edges))

    return Gabriel_edges # [(0, 1), (0, 2),...]
                

def check_Gabriel():
    '''
    查看Gabriel的边是否都在seg和nei里，结果ok
    '''
    for i in range(5, 201):
        ins = instance(i)
        edges = Gabriel_edges(ins) 

        edges = list(map(lambda x : set(x), edges))
        edges_de = list(map(lambda x : set(x), ins.graph_de.edges()))
        edges_seg = list(map(lambda x : set(x), ins.graph_de_seg1_seg2_seg3.edges()))
        edges_nei = list(map(lambda x : set(x), ins.graph_de_nei2_nei3.edges()))

        for edge in edges:
            # if edge not in edges_de:
            #     print(edge, 'de_error')
            if edge not in edges_seg:
                print(edge, 'seg_error')
            if edge not in edges_nei:
                print(edge, 'nei_error')

        print(i)


            
    


if __name__ == '__main__':
    check_Gabriel()