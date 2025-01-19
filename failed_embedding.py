"""
This file shows a visualization of the minorminer.find_embedding() algorithm.

At each step of the find_embedding() algorithm, a new chain is inserted for a node in the source graph, where chains
are allowed to overlap. After all chains have inserted, the algorithm iteratively removes and reinserts chains,
attempting to minimize the amount of overlap between them. Eventually all overlap will be removed, and the result is
a valid embedding.

In this example, a complete graph K_8 is embedded into a chimera_graph C_2.

"""

from minorminer import miner
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt


# Parameters of the demo
wait_for_input = True                   # wait for user input to advance to the next step
G = nx.complete_graph(8)                # source graph
#C = dnx.generators.chimera_graph(2)     # target graph
C = dnx.chimera_graph(2,2,4)

#可视化过程
def show_current_embedding(emb,s_graph):
    # visualize overlaps.
    #用于清除当前的图形（Figure）对象中的所有绘图内容，以便进行新的绘图。clear figure
    plt.clf()
    #根据当前emb画嵌入，overlapped_embedding允许重叠，并且我自己添加了embedded_graph参数(对完全图来说都一样吧)
    dnx.draw_chimera_embedding(C, emb=emb, overlapped_embedding=True, show_labels=True, embedded_graph = s_graph, font_size=20, node_size=500)
    plt.show()
    if wait_for_input:
        plt.pause(1)
        test = input('whatever:')
    #若没有userinput则暂停1s
    else:
        plt.pause(0.05)


def compute_bags(C, emb):
    # Given an overlapped embedding, compute the set of source nodes embedded at every target node.
    #初始化bags 是一个字典，通过遍历C中的所有节点
    bags = {v: [] for v in C.nodes()}

    #遍历emb中的键值对
    for x, chain in emb.items():
        #遍历每一个chain，并加到对应的bags中的列表中
        for v in chain:
            bags[v].append(x)
    #bags是一个字典，每一个节点的chain信息
    return bags


# Run the algorithm.开始执行算法

#用于开启交互式模式（Interactive Mode）。
#交互式模式允许在绘图之后持续显示图形，并在图形更新时进行实时更新，而不需要重新运行整个脚本或重新绘制图形。
plt.ion()

#miner???
#这一步已经找好了embedding？
#根据设定好的随机种子，只要随机种子不变，则每次嵌入的结果都是一样的
m = miner(G, C, random_seed=0)


found = False
emb = {}

print("Embedding K_8 into Chimera C(2).")

#总共三次循环,可以自行设置，循环的次数就是iteration的次数，也就是重复removing和reinsert的次数
#一般find_embedding的默认次数是 max_no_improvement (int, optional, default=10)
for iteration in range(10):
    #iteration=0  初始化阶段
    if iteration == 0:
        print("\nInitialization phase...")
    #iteration=1  已经完成初始化阶段（已经找到一个嵌入，但可能无效），开始解决overlap
    elif iteration==1:
        print("\nOverfill improvement phase...")
    

        #遍历G（要被嵌入的图）中所有顶点
    for v in G.nodes():
        #如果已经完成初始化阶段  iteration > 0
        if iteration > 0:
            # show embedding with current vertex removed.
            removal_emb = emb.copy()   #复制当前emb嵌入
            removal_emb[v] = []      #正在遍历图G的所有顶点，清零当前节点的嵌入
            show_current_embedding(removal_emb,G)

        
            
        # run one step of the algorithm.对当前节点进行嵌入
        emb = m.quickpass(varorder=[v], clear_first=False, overlap_bound=G.number_of_nodes())
        #初始阶段，如：
        #{0: [0]}   
        #{0: [0], 1: [5]}        {0: [0], 1: [5], 2: [6]}     {0: [0], 1: [5], 2: [6], 3: [7]}


        # check if we've found an embedding.
        #计算当前的bags信息，C中每个节点的嵌入信息。针对chimera进行统计，统计每个chimera节点所对应的原图的节点并以列表形式储存。比如{0: [0], 4: [], 5: [], 6: [], 7: [], 1: [], 2: [], 3: []}
        bags = compute_bags(C, emb)

        #overlap是chain的个数最大值
        overlap = max(len(bag) for bag in bags.values())

        #如果满足条件则找到embdd，进入改进环节  iteration>0（第二轮）后开始检查是否已经找到有效的嵌入，如果overlap=1就算找到了，然后进入改进环节
        if overlap==1 and iteration > 0 and not(found):
            print("\nEmbedding found. Chain length improvement phase...")
            #found 变为 True
            found = True

        show_current_embedding(emb,G)

    max_chain_length = max(len(chain) for chain in emb.values()) #一个节点模型所占用的量子比特个数的最大值
    print("\tIteration {}: max qubit fill = {}, max chain length = {}".format(iteration, overlap, max_chain_length))
