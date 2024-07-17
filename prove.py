from instance import instance, missing_edges
import networkx as nx
from pprint import pprint

def prove_seg(ins, total_info):
    # 不存在的边
    edges = missing_edges(ins.n, ins.graph_de_seg1_seg2_seg3)
    edges = list(map(lambda x:list(x), edges))
    
    # 该实例的信息
    dic ={}

    # 不为空时
    if edges:
        #print(edges)
        for edge in edges:
            # 每个消失的边的信息
            info = {}
            # 本身的距离
            info['self_dis'] = int(ins.mat[edge[0], edge[1]])

            # 最优路径中的实际距离（两个）
            index1 = ins.optimal_tour.index(edge[0])
            index2 = ins.optimal_tour.index(edge[1])

            # 先截取一段并计算长度
            if index1 < index2:
                sub_tour = ins.optimal_tour[index1:index2+1]
            else:
                sub_tour = ins.optimal_tour[index2:index1+1]

            part_dis = 0
            for i in range(len(sub_tour) - 1):
                part_dis += int(ins.mat[sub_tour[i], sub_tour[i+1]])

            
            # 另外一个方向的距离
            another_dis = ins.optimal_length - part_dis
            
            if part_dis < another_dis:
                info['dist1'] = part_dis
                info['dist2'] = another_dis
            else:
                info['dist1'] = another_dis
                info['dist2'] = part_dis

            # 添加每个消失边的信息(key为元组)
            dic[tuple(edge)] = info
    
    total_info[str(ins.n)] = dic


def calculate(dic):
    res = []
    for _ ,info in dic.items():
        if info:
            for _ , dis_info in info.items():
                res.append(dis_info['dist2'] / dis_info['self_dis'])
                pass

    return res

if __name__ == '__main__':
    dic = {}
    for i in range(5,201):
        print(i)
        ins = instance(i)
        prove_seg(ins, dic)
    
    
    #pprint(dic)
    res = calculate(dic)

    print(max(res),min(res))
