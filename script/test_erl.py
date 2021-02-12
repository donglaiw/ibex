import numpy as np
import kimimaro
import pickle
import cv2
import networkx as nx
import sys,os
sys.path.append('../')

class SkeletonScores():

    def __init__(self):
        self.ommitted = 0
        self.split = 0
        self.merged = 0
        self.correct = 0
        self.correct_edges = {}
        
def evaluate_skeletons(skeletons, skeleton_id_attribute, node_segment_lut):
    # find all merging segments (skeleton edges on merging segments will be counted as wrong)
    # pairs of (skeleton, segment), one for each node
    skeleton_segment = np.array([
        [data[skeleton_id_attribute], node_segment_lut[n]]
        for n, data in skeletons.nodes(data=True)])
    # unique pairs of (skeleton, segment)
    skeleton_segment = np.unique(skeleton_segment, axis=0)
    # number of times that a segment was mapped to a skeleton
    segments, num_segment_skeletons = np.unique(
        skeleton_segment[:, 1],
        return_counts=True)
    # all segments that merge at least two skeletons
    merging_segments = segments[num_segment_skeletons > 1]
    merging_segments_mask = np.isin(skeleton_segment[:, 1], merging_segments)
    merged_skeletons = skeleton_segment[:, 0][merging_segments_mask]
    merging_segments = set(merging_segments)
    merges = {}
    splits = {}
    merged_segments = skeleton_segment[:, 1][merging_segments_mask]
    for segment, skeleton in zip(merged_segments, merged_skeletons):
        if segment not in merges:
            merges[segment] = []
        merges[segment].append(skeleton)
    merged_skeletons = set(np.unique(merged_skeletons))
    skeleton_scores = {}
    for u, v in skeletons.edges():
        skeleton_id = skeletons.nodes[u][skeleton_id_attribute]
        segment_u = node_segment_lut[u]
        segment_v = node_segment_lut[v]
        if skeleton_id not in skeleton_scores:
            scores = SkeletonScores()
            skeleton_scores[skeleton_id] = scores
        else:
            scores = skeleton_scores[skeleton_id]
        if segment_u == 0 or segment_v == 0:
            scores.ommitted += 1
            continue
        if segment_u != segment_v:
            scores.split += 1
            if skeleton_id not in splits:
                splits[skeleton_id] = []
            splits[skeleton_id].append((segment_u, segment_v))
        # segment_u == segment_v != 0
        segment = segment_u
        # potentially merged edge?
        if skeleton_id in merged_skeletons:
            if segment in merging_segments:
                scores.merged += 1
                continue
        scores.correct += 1
        if segment not in scores.correct_edges:
            scores.correct_edges[segment] = []
        scores.correct_edges[segment].append((u, v))
    merge_split_stats = {'merge_stats': merges,'split_stats': splits}
    return skeleton_scores, merge_split_stats
    

def expected_run_length(skeleton, seg): #takes input a skeleton and predicted segment
    
    skeleton_id_attribute = np.array(list(skeleton.keys()))
    sl = np.empty([1500,3,400])
    for i in skeleton.keys():
        np.append(sl,skeleton[i].vertices)
    #sl = np.array([skeleton[i].vertices for i in skeleton])
    G = nx.Graph()
    G.add_nodes_from(skeleton)

    sz = seg.shape 
    print(sl.shape)
    print(seg.shape)
    pos = sl[:,0]*sz[1]*sz[2] +sl[:,1]*sz[2] 
    #+ sl[:,2]
    seg_id = seg.ravel()[pos]
    node_segment_lut = {skeleton_id_attribute[i]: seg_id[i] for i in range(len(seg_id))}

    skeleton_lengths = get_skeleton_lengths(skeletons,['x', 'y', 'z'],skeleton_id_attribute,store_edge_length='length')
    total_skeletons_length = np.sum([l for _, l in skeleton_lengths.items()])
    res = evaluate_skeletons(skeletons,skeleton_id_attribute,node_segment_lut)
    skeleton_scores, merge_split_stats = res
    skeletons_erl = 0
    
    for skeleton_id, scores in skeleton_scores.items():
        skeleton_length = skeleton_lengths[skeleton_id]
        skeleton_erl = 0
        
        for segment_id, correct_edges in scores.correct_edges.items():
            correct_edges_length = np.sum([skeletons.edges[e]['length'] for e in correct_edges])
            skeleton_erl += (correct_edges_length * (correct_edges_length/skeleton_length))
            
        skeletons_erl += ((skeleton_length/total_skeletons_length) * skeleton_erl)
        
    return skeletons_erl, merge_split_stats
        
    return skeletons_erl, merge_split_stats
    
def readh5(filename, datasetname=None):
    import h5py
    fid = h5py.File(filename,'r')

    if datasetname is None:
        if sys.version[0]=='2': # py2
            datasetname = fid.keys()
        else: # py3
            datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out=[None]*len(datasetname)
        for di,d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])
    
skels = np.load("/n/home05/akemkar/ERL/kimimaro/skeleton.p", allow_pickle=True)
print(skels)
print(skels[332].vertices)

seg_pred = readh5('/n/home05/akemkar/ERL/SNEMI-train-pred.h5')
print(seg_pred.shape)
seg_pred = seg_pred[0,:, :, :]
print(seg_pred.shape)

ERL = 0
stats = dict()
ERL, stats = expected_run_length(skels,seg_pred)
