from minorminer import busclique
from minorminer import find_embedding
import dwave_networkx as dnx
import networkx as nx
import matplotlib.pyplot as plt
import dimod
import random
import time


def embed_tsp(qubo, n, add, topology:str, method:str):

        
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


         # 先打开文件，行缓冲
        with open(f'embed_result/{topology}/{topology}_{method}.txt', "a+", encoding="utf-8") as file:

            #####
            # 找到嵌入
            if embedding:
                print(f"{n},成功!  " + f'seed = {seed} '  + f'time = {round(used_time, 2)}s', file=file, flush=True)
                print('#############################################',file=file, flush=True)
                return True
        
            # 没找到嵌入 且 seed <= 10
            elif seed < 10:
                print(f"{n},失败!  " + f'seed = {seed} ' + f'time = {round(used_time, 2)}s', file=file, flush=True)
                continue
            # 结束第十轮seed
            else:
                print(f"{n},失败!  " + f'seed = {seed} ' + f'time = {round(used_time, 2)}s', file=file, flush=True)
                print('#############################################',file=file, flush=True)
                return False




if __name__ == '__main__':
    pass

    G = dnx.zephyr_graph(15)
    dnx.draw_zephyr(G,node_size = 12, node_color = 'g',with_labels=False)
    
    plt.show()