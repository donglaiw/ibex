import numpy as np
import h5py
import pickle
import networkx as nx
import sys,os
from funlib import evaluate
   
if __name__ == "__main__":
    D0 = '/n/pfister_lab2/Lab/donglai/snemi/'
    fn_pred = D0 + 'pc/test_4min_orig/zwz_init.h5'
    fn_gt = '/n/pfister_lab2/Lab/vcg_connectomics/EM/snemi/label/test-labels.h5'
    seg_pred = np.array(h5py.File(fn_pred, 'r')['main'])
    seg_gt = np.array(h5py.File(fn_gt, 'r')['main'])
   
    res = [30,6,6]
    nodes, edges = pickle.load(open(D0 + 'db/erl/skel_pts.pkl', 'rb'), encoding="latin1")
    gt_graph = nx.Graph()
    node_segment_lut = {}
    node_segment_lut_gt = {}
    cc = 0
    for k in range(len(nodes)):
        node = nodes[k]
        edge = edges[k] + cc
        for l in range(node.shape[0]):
            gt_graph.add_node(cc, skeleton_id = k, z=node[l,0]*res[0], y=node[l,1]*res[1], x=node[l,2]*res[2])
            node_segment_lut[cc] = seg_pred[node[l,0], node[l,1], node[l,2]]
            node_segment_lut_gt[cc] = seg_gt[node[l,0], node[l,1], node[l,2]]
            cc += 1
        for l in range(edge.shape[0]):
            gt_graph.add_edge(edge[l,0], edge[l,1])
    
    scores,stat = evaluate.expected_run_length(
                    skeletons=gt_graph,
                    skeleton_id_attribute='skeleton_id',
                    edge_length_attribute='length',
                    node_segment_lut=node_segment_lut,
                    skeleton_position_attributes=['z', 'y', 'x'],
                    return_merge_split_stats = True)
    scores_gt,stat_gt = evaluate.expected_run_length(
                    skeletons=gt_graph,
                    skeleton_id_attribute='skeleton_id',
                    edge_length_attribute='length',
                    node_segment_lut=node_segment_lut_gt,
                    skeleton_position_attributes=['z', 'y', 'x'],
                    return_merge_split_stats = True)

    import pdb; pdb.set_trace()
