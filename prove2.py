from instance import instance, missing_edges
import networkx as nx
import numpy as np

# 与自己最近的点,针对一个例子，返回一组边
def closest_point(ins):

    edges = []

    for point in range(ins.n):
        row = [x for x in ins.mat[point] if x > 0]
            
        closest = min(row)
        closest_index = np.where(ins.mat[point] == closest)[0] # 可能有多个

        for node in closest_index:
            edges.append((point, node))
    
    return edges # # [(0, 7), (0, 27),...]

# 60
def check():
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


if __name__ == '__main__':
    check()