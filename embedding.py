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
def embed_complete_tsp_chimera(n):
    #2000Q的chimera图
    G = dnx.chimera_graph(16,16,4) 

    #tsp的完全图
    D = nx.complete_graph(n)

    #寻找嵌入，若为空则嵌入失败
    embedding = busclique.find_clique_embedding(D, G) 


    print(embedding)
    print(G.number_of_nodes())

    dnx.draw_chimera(G,with_labels = True)
    plt.show()

    dnx.draw_chimera_embedding(G,emb = embedding,with_labels = True)
    plt.show() 

#########################################################################


if __name__ == '__main__':
    embed_complete_tsp_chimera(4)