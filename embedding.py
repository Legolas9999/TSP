from turtle import circle
from minorminer import busclique
from minorminer import find_embedding
import dwave_networkx as dnx
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import random





#########################################################################
#嵌入tsp完全图到2000Q的chimera,n:tsp城市个数
def embed_complete_tsp_chimera(qubo):

    # 选择 2000Q 量子退火机的 Chimera 拓扑
    chimera = dnx.chimera_graph(16) 

    

    # 查找嵌入，返回嵌入映射
    embedding = find_embedding(qubo, chimera)
    
    # 检查嵌入是否成功
    if embedding:
        print("嵌入成功!")
        # print(embedding)
        # dnx.draw_chimera_embedding(chimera, emb = embedding,with_labels = False, node_size = 10)
        # plt.show()
    else:
        print("嵌入失败!")

#########################################################################


#########################################################################
#嵌入tsp完全图到2000Q的chimera,n:tsp城市个数
def embed_complete_tsp_pegasus(qubo):

    # 选择 Advantage 量子退火机的 pegasus 拓扑
    pegasus = dnx.pegasus_graph(16)

    # 
    # dnx.draw_pegasus(pegasus, with_labels = False, node_size = 10)
    # plt.show()

    # 查找嵌入，返回嵌入映射
    embedding = find_embedding(qubo, pegasus)
    
    # 检查嵌入是否成功
    if embedding:
        print("嵌入成功!")
        # print(embedding)
        # dnx.draw_pegasus_embedding(pegasus, emb = embedding,with_labels = False, node_size = 5)
        # plt.show()
    else:
        print("嵌入失败!")

#########################################################################


#########################################################################
#嵌入tsp完全图到2000Q的chimera,n:tsp城市个数
def embed_complete_tsp_zephyr(qubo):

    # 选择 新一代 量子退火机的 zephyr 拓扑
    zephyr = dnx.zephyr_graph(15)

    # dnx.draw_zephyr(zephyr, with_labels = False, node_size = 10)
    # plt.show()


    # 查找嵌入，返回嵌入映射
    embedding = find_embedding(qubo, zephyr)
    
    # 检查嵌入是否成功
    if embedding:
        print("嵌入成功!")
        # print(embedding)
        # dnx.draw_zephyr_embedding(zephyr, emb = embedding,with_labels = False, node_size = 5)
        # plt.show()
    else:
        print("嵌入失败!")

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
    embed_complete_tsp_chimera()