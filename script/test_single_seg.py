import os,sys

# add ibexHelper path
from ibexHelper.skel import CreateSkeleton,ReadSkeletons
from ibexHelper.util import GetBbox, ReadH5, WriteH5
from ibexHelper.skel2graph import GetGraphFromSkeleton
from ibexHelper.graph import ShrinkGraph_v2, GetNodeList, GetEdgeList
from ibexHelper.graph2x import Graph2H5
import h5py
import numpy as np
import networkx as nx
from scipy.ndimage.morphology import distance_transform_cdt

opt = sys.argv[1]

res = [120,128,128] # z,y,x
out_folder = '../tmp/demo/'
bfs = 'bfs'; modified_bfs=False 
edgTh = [40,1] # threshold
# 3d segment volume
seg_fn = '/mnt/coxfs01/donglai/data/JWR/snow_cell/cell128nm/neuron/cell26_d.h5'

if opt=='0': # mesh -> skeleton
    seg = ReadH5(seg_fn, 'main')
    CreateSkeleton(seg, out_folder, res, res)

elif opt=='1': # skeleton -> dense graph
    print('read skel')
    skel = ReadSkeletons(out_folder, skeleton_algorithm='thinning', downsample_resolution=res, read_edges=True)[1]

    print('save node positions')
    node_pos = np.stack(skel.get_nodes()).astype(int)
    WriteH5(out_folder+'node_pos.h5', node_pos)

    print('generate dt for edge width')
    seg = ReadH5(seg_fn, 'main')
    sz = seg.shape
    bb = GetBbox(seg>0)
    seg_b = seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]
    dt = distance_transform_cdt(seg_b, return_distances=True)

    print('generate graph')
    new_graph, wt_dict, th_dict, ph_dict = GetGraphFromSkeleton(skel, dt=dt, dt_bb=[bb[x] for x in [0,2,4]],\
                                                   modified_bfs=modified_bfs)
    
    print('save as a networkx object')
    edge_list = GetEdgeList(new_graph, wt_dict, th_dict, ph_dict)
    G = nx.Graph(shape=sz)
    # add edge attributes
    G.add_edges_from(edge_list)
    nx.write_gpickle(G, out_folder+'graph-%s.obj'%(bfs))

elif opt == '2': # reduced graph
    import networkx as nx
    G = nx.read_gpickle(out_folder+'graph-%s.obj'%(bfs))

    n0 = len(G.nodes())
    G = ShrinkGraph_v2(G, threshold=edgTh)
    n1 = len(G.nodes())
    print('#nodes: %d -> %d'%(n0,n1))
    nx.write_gpickle(G, out_folder+'graph-%s-%d-%d.obj'%(bfs,edgTh[0],10*edgTh[1]))
elif opt == '3': # generate h5 for visualization
    G = nx.read_gpickle(out_folder+'graph-%s-%d-%d.obj'%(bfs,edgTh[0],10*edgTh[1]))
    pos = ReadH5(out_folder+'node_pos.h5','main')
    vis = Graph2H5(G, pos)
    WriteH5(out_folder+'graph-%s-%d-%d.h5'%(bfs,edgTh[0],10*edgTh[1]),vis)
