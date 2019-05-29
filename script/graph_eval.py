# Step-by-step guide for comparing the similarity of two graphs 
# 1. open a python3 conda environment
# 2. pip install numpy cython
# 3. git clone https://github.com/Jacobe2169/GMatch4py.git
# 4. cd GMatch4py
# 5. pip install .

import numpy as np
import networkx as nx
import gmatch4py as gm

def evaluate(g1, g2):
    ged = gm.GraphEditDistance(1,1,1,1) # all edit costs are equal to 1
    result=ged.compare([g1,g2], None)
    return ged.similarity(result), ged.distance(result)

def test():
    g1 = nx.complete_bipartite_graph(5,4) 
    g2 = nx.complete_bipartite_graph(6,4)
    similarity, distance = evaluate(g1, g2)
    print(similarity)
    print(distance)

if __name__ == '__main__':
    test()

    # SIMILARITY, DISTANCE = [], []
    # gt_graph_list = []
    # pd_graph_list = []
    # for i in range(len(gt_graph_list)):
    #     similarity, distance = evaluate(gt_graph_list[i], pd_graph_list[i])
    #     SIMILARITY.append(similarity)
    #     DISTANCE.append(distance)
