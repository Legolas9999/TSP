from turtle import circle
from minorminer import busclique
from minorminer import find_embedding
import dwave_networkx as dnx
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import random

#G = dnx.chimera_graph()
#dnx.draw_chimera(G,node_size = 15,node_color = 'g')
#plt.show()

#print(len(G))


#graph = nx.barabasi_albert_graph(100, 3, seed=1)  # Build a quasi-random graph
## Set node and edge values for the problem
#h = {v: 0.0 for v in graph.nodes}
#J = {edge: random.choice([-1, 1]) for edge in graph.edges}
#bqm = dimod.BQM(h, J, vartype=dimod.SPIN)

#nx.draw(graph)
#plt.show()





#grid_graph = nx.generators.lattice.grid_2d_graph(3, 3)
#C = dnx.chimera_graph(2,1)

#embedding = find_embedding(grid_graph, C)
#print(embedding)



#########################################################################
##计算图的直径
#print('the dia is :',nx.diameter(G))
#########################################################################



#########################################################################
##通用的嵌入函数find_embedding，不知道如何在图里面直观显示 
#source_graph = nx.gnp_random_graph(10, 0.3)
##source_graph = nx.random_regular_graph(d=4, n=10)
#target_graph = nx.random_regular_graph(d=3, n=30)

#embedding = find_embedding(source_graph, target_graph, random_seed=10)
#print(embedding)


#posSource = nx.circular_layout(source_graph)
#nx.draw(source_graph, posSource,with_labels=True)
#plt.show()

#posT = nx.circular_layout(target_graph)  # 使用 circular_layout 布局算法


#nx.draw(target_graph, posT, with_labels=True)
#plt.show()


#print(embedding)

#########################################################################





#########################################################################
##将一个三节点完全图嵌入到chimera
#G = dnx.chimera_graph(2,2,4)

#triangle = [(0, 1), (1, 2), (2, 0)]
#square = [(0, 1), (1, 2), (2, 3), (3, 0)]

#embedding = find_embedding(triangle, G, random_seed=10)
#print(embedding)

#dnx.draw_chimera_embedding(G,emb = embedding,with_labels = True)
#plt.show() 
#########################################################################




#########################################################################
#嵌入完全图到chimera
#G = dnx.chimera_graph(1,1,4) #chimera图
#D = nx.complete_graph(4)  #四节点完全图
#embedding = busclique.find_clique_embedding(D, G) #寻找嵌入
#print(embedding)

#dnx.draw_chimera(G,with_labels = True)
#plt.show()

#dnx.draw_chimera_embedding(G,emb = embedding,with_labels = True)
#plt.show() 

#########################################################################




#########################################################################
##draw_chimera_graph函数里的参数   embedded_graph 若设置，则有想要的结果
## 另外 interaction_edges 参数可以之画出想要的coupler

##2*2chimera图
#G = dnx.chimera_graph(2,2,4) 

##9个节点的星状图
#H_star = nx.star_graph(9) 

##寻找嵌入
#embedding_from_find_embedding = find_embedding(H_star, G) 

##打印嵌入
#print(embedding_from_find_embedding)

##打印该星状图
#nx.draw(H_star,with_labels = True)
#plt.show()

##打印chimera图
#dnx.draw_chimera(G,with_labels = True)
#plt.show()

##设置两个子图画面，分别画图
#f, axes = plt.subplots(1, 2)

##打印chimera的嵌入（没有embedded_graph参数）
#dnx.draw_chimera_embedding(G,emb = embedding_from_find_embedding , show_labels=True, ax = axes[0])

##打印chimera的嵌入（有embedded_graph参数）
#dnx.draw_chimera_embedding(G,emb = embedding_from_find_embedding , show_labels=True, embedded_graph = H_star,ax = axes[-1])
#plt.show() 


#########################################################################


#########################################################################
##不同的嵌入方法生成的嵌入不同，有一种情况是find_embedding找到了嵌入，但clique_embedding找不到嵌入

## find_embedding  和   busclique.find_clique_embedding  的比较
##  通用                       针对完全图和dwave的三种拓扑结构

#G = dnx.chimera_graph(1,1,4) #1*1chimera图
#H = nx.complete_graph(4)  #四节点完全图

##不同的方法寻找嵌入
#embedding_from_find_embedding = find_embedding(H, G) #寻找嵌入
#embedding_from_clique_embedding = busclique.find_clique_embedding(H, G) #寻找嵌入

##设置两个子图画面，分别画图
#f, axes = plt.subplots(1, 2)

##分别打印chimera的嵌入
#dnx.draw_chimera_embedding(G,emb = embedding_from_find_embedding , show_labels=True,embedded_graph=H,ax = axes[0])

#dnx.draw_chimera_embedding(G,emb = embedding_from_clique_embedding , show_labels=True,embedded_graph=H,ax = axes[-1])

#plt.show() 


#########################################################################


#########################################################################
##  busclique.find_clique_embedding  方法也可以将其他图(除了完全图之外的图)嵌入到dwave拓扑结构(三种)中

#G = dnx.chimera_graph(1,1,4) #1*1chimera图
#H_star = nx.star_graph(3)  #4节点星图

##寻找嵌入
#embedding_from_clique_embedding = busclique.find_clique_embedding(H_star, G) 

##画嵌入
#dnx.draw_chimera_embedding(G,emb = embedding_from_clique_embedding , show_labels=True,embedded_graph=H_star)
#plt.show() 

#########################################################################


#########################################################################
###查看用两种不同的方法对 星图 进行嵌入，效果依然不同


#G = dnx.chimera_graph(1,1,4) #1*1chimera图
#H_star = nx.star_graph(3)  #4节点星图  H_star = nx.star_graph(4)  #节点星图


##不同的方法寻找嵌入
#embedding_from_find_embedding = find_embedding(H_star, G) #寻找嵌入
#embedding_from_clique_embedding = busclique.find_clique_embedding(H_star, G) #寻找嵌入

##设置两个子图画面，分别画图
#f, axes = plt.subplots(1, 2)

##分别打印chimera的嵌入
#dnx.draw_chimera_embedding(G,emb = embedding_from_find_embedding , show_labels=True,embedded_graph=H_star,ax = axes[0])

#dnx.draw_chimera_embedding(G,emb = embedding_from_clique_embedding , show_labels=True,embedded_graph=H_star,ax = axes[-1])

#plt.show() 


#########################################################################
##查看用两种不同的方法对 星图 进行嵌入，效果依然不同


G = dnx.chimera_graph(2,2,4) #1*1chimera图
H = nx.complete_graph(10)  #4节点星图  H_star = nx.star_graph(4)  #节点星图

#nx.draw(H,with_labels = True)
#dnx.draw_chimera(H,with_labels = True)
#plt.show()


#print('the dia is :',nx.diameter(G))

#不同的方法寻找嵌入
embedding_from_with = find_embedding(H, G) #寻找嵌入
#embedding_from_clique_embedding = busclique.find_clique_embedding(H, G) #寻找嵌入

#embedding_from_with_test = find_embedding(H_star, G, max_no_improvement = 100 ,verbose= 2,return_overlap= True) #寻找嵌入


#设置两个子图画面，分别画图
#f, axes = plt.subplots(1, 2)
#dnx.draw_chimera(G,with_labels = True)
dnx.draw_chimera_embedding(G,emb = embedding_from_with , show_labels=True,embedded_graph=H)
#dnx.draw_chimera_embedding(G,emb = embedding_from_with , show_labels=True)

#dnx.draw_chimera_embedding(G,emb = embedding_from_clique_embedding , show_labels=True,embedded_graph=H,ax = axes[-1])

plt.show()



#分别打印chimera的嵌入
#dnx.draw_chimera_embedding(G,emb = embedding_from_find_embedding , show_labels=True,embedded_graph=H_star,ax = axes[0])

#dnx.draw_chimera_embedding(G,emb = embedding_from_clique_embedding , show_labels=True,embedded_graph=H_star,ax = axes[-1])

#plt.show() 

#########################################################################
