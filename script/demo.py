import os,sys

# add ibexHelper path
sys.path.append('/home/donglai/lib/ibex_fork/ibexHelper')
from ibexHelper.skel import CreateSkeleton,ReadSkeletons
from ibexHelper.util import get_bb, writepkl, readpkl
from ibexHelper.skel2graph import GetGraphFromSkeleton
from ibexHelper.graph import ShrinkGraph_v2
import h5py
import numpy as np
import networkx as nx
from scipy.ndimage.morphology import distance_transform_cdt

opt = sys.argv[1]

res = [120,128,128] # z,y,x
out_folder = '../tmp/demo/'
bfs = 'bfs'; modified_bfs=False 
seg_fn = '/mnt/coxfs01/donglai/data/JWR/snow_cell/cell128nm/neuron/cell26_d.h5'

if opt=='0': # mesh -> skeleton
    seg = np.array(h5py.File(seg_fn,'r')['main'])
    CreateSkeleton(seg, out_folder, res, res)
elif opt=='1': # skeleton -> dense graph
    print('read skel')
    skel = ReadSkeletons(out_folder, skeleton_algorithm='thinning', downsample_resolution=res, read_edges=True)[1]
    print('generate dt for edge width')
    seg = np.array(h5py.File(seg_fn,'r')['main'])
    bb = get_bb(seg>0)
    seg_b = seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]
    dt = distance_transform_cdt(seg_b, return_distances=True)
    print('generate graph')
    new_graph, wt_dict, th_dict, path_dict = GetGraphFromSkeleton(skel, dt=dt, dt_bb=[bb[x] for x in [0,2,4]],\
                                                   modified_bfs=modified_bfs)
    writepkl(out_folder+'graph-%s.p'%(bfs), [new_graph, wt_dict, th_dict, path_dict])
elif opt == '2': # reduced graph
    import networkx as nx
    graph = readpkl(Ds+'graph-%s.p'%(bfs))
    edgelist = GetEdgeList(graph[0], graph[1], graph[2])
    G = nx.Graph()
    G.add_edges_from(edgelist)
    n0 = len(G.nodes())
    G, path_dict = ShrinkGraph_v2(G, threshold=edgTh, path_dict=graph[3])
    n1 = len(G.nodes())
    print('#nodes: %d -> %d'%(n0,n1))
    nx.write_gpickle(G, Ds+'graph-%s-%d-%d.obj'%(bfs,edgTh[0],10*edgTh[1]))
    writepkl(Ds+'graph-%s-%d-%d.p'%(bfs,edgTh[0],10*edgTh[1]), [path_dict])
elif opt == '0.21': # post-process graph
    import networkx as nx
    graph = readpkl(Ds+'graph-%s.p'%(bfs))
    edgelist = GetEdgeList(graph[0], graph[1], graph[2])
    G = nx.Graph()
    G.add_edges_from(edgelist)
    nx.write_gpickle(G, Ds+'graph-%s.obj'%(bfs))

