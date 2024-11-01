from minorminer import busclique
from minorminer import find_embedding
import dwave_networkx as dnx
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import random
import time


def embed_tsp(qubo, n, add, topology:str, method:str):

    # 先打开文件
    with open(f'embed_result3/{topology}/{topology}_{method}.txt', "a+", encoding="utf-8") as file:
        
        # 确定random_seed 1~10
        for seed in range(1,11):
            #创建topology及嵌入
            if topology == 'chimera':
                chimera = dnx.chimera_graph(16 + add)
                start = time.time() 
                embedding = find_embedding(qubo, chimera, random_seed=seed) 
                used_time = time.time() - start
            if topology == 'pegasus':
                pegasus = dnx.pegasus_graph(16 + add)
                start = time.time()
                embedding = find_embedding(qubo, pegasus, random_seed=seed)
                used_time = time.time() - start
            if topology == 'zephyr':
                zephyr = dnx.zephyr_graph(15 + add)
                start = time.time()
                embedding = find_embedding(qubo, zephyr, random_seed=seed)
                used_time = time.time() - start

            #####
            # 找到嵌入
            if embedding:
                print(f"{n},成功!  " + f'seed = {seed} '  + f'time = {round(used_time, 2)}s', file=file)
                print('#############################################',file=file)
                return True
        
            # 没找到嵌入 且 seed <= 10
            elif seed <= 10:
                print(f"{n},失败!  " + f'seed = {seed} ' + f'time = {round(used_time, 2)}s', file=file)

            # 结束十轮seed
            else:
                print(f"{n},失败!  " + f'seed = {seed} ' + f'time = {round(used_time, 2)}s', file=file)
                print('#############################################',file=file)
                return False



#########################################################################
# 嵌入tsp完全图到2000Q的chimera,n:tsp城市个数
# 默认:chimera = dnx.chimera_graph(16) 
# qubo:qubo模型 n:城市大小   add:拓扑增加   method:方法
def embed_tsp_chimera(qubo, n, add, method):

    # 选择 2000Q 量子退火机的 Chimera 拓扑
    chimera = dnx.chimera_graph(16 + add) 
    chimera = dnx.chimera_graph(n) 


    # 查找嵌入，返回嵌入映射
    embedding = find_embedding(qubo, chimera)
    

    with open(f'embed_result/chimera/chimera_{method}.txt', "a+", encoding="utf-8") as file:
        

        # 检查嵌入是否成功
        if embedding:
            print(f"{n},嵌入成功!", file=file)
            return True
            # print(embedding)
            # dnx.draw_chimera_embedding(chimera, emb = embedding,with_labels = False, node_size = 10)
            # plt.show()
        else:
            print(f"{n},嵌入失败!", file=file)
            print("", file=file)
            return False

#########################################################################


#########################################################################
#嵌入tsp完全图到2000Q的chimera,n:tsp城市个数
# 默认:pegasus = dnx.pegasus_graph(16)
def embed_tsp_pegasus(qubo, n, add, method):

    # 选择 Advantage 量子退火机的 pegasus 拓扑
    pegasus = dnx.pegasus_graph(16 + add)

    # 
    # dnx.draw_pegasus(pegasus, with_labels = False, node_size = 10)
    # plt.show()

    # 查找嵌入，返回嵌入映射
    embedding = find_embedding(qubo, pegasus)
    
    with open(f'embed_result/pegasus/pegasus_{method}.txt', "a+", encoding="utf-8") as file:
        # 检查嵌入是否成功
        if embedding:
            print(f"{n},嵌入成功!", file=file)
            return True

        else:
            print(f"{n},嵌入失败!", file=file)
            print("", file=file)
            return False

#########################################################################


#########################################################################
#嵌入tsp完全图到2000Q的chimera,n:tsp城市个数
# 默认:zephyr = dnx.zephyr_graph(15)
def embed_tsp_zephyr(qubo, n, add, method):

    # 选择 新一代 量子退火机的 zephyr 拓扑
    zephyr = dnx.zephyr_graph(15 + add)

    # dnx.draw_zephyr(zephyr, with_labels = False, node_size = 10)
    # plt.show()


    # 查找嵌入，返回嵌入映射
    embedding = find_embedding(qubo, zephyr)
    
    with open(f'embed_result/zephyr/zephyr_{method}.txt', "a+", encoding="utf-8") as file:
        # 检查嵌入是否成功
        if embedding:
            print(f"{n},嵌入成功!", file=file)
            return True

        else:
            print(f"{n},嵌入失败!", file=file)
            print("", file=file)
            return False

#########################################################################

#########################################################################
# #嵌入tsp完全图到2000Q的chimera,n:tsp城市个数
# def embed_complete_tsp_chimera(n, qubo):
#     #2000Q的chimera图
#     G = dnx.chimera_graph(16,16,4) 

#     # 
#     bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

#     #tsp的完全图
#     D = nx.complete_graph(n)

#     #寻找嵌入，若为空则嵌入失败
#     embedding = busclique.find_clique_embedding(D, G) 


#     print(embedding)
#     print(G.number_of_nodes())

#     dnx.draw_chimera(G,with_labels = True)
#     plt.show()

#     dnx.draw_chimera_embedding(G,emb = embedding,with_labels = True)
#     plt.show() 

#########################################################################


if __name__ == '__main__':
    pass