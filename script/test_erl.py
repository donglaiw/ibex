import pickle
import sys,os
from funlib import evaluate
from ibexHelper.skel2graph import GetERLDataFromSkeleton 
from ibexHelper.util import ReadH5

def test_erl(skel_pickle_path, seg_path, res= [30,6,6]):
    nodes, edges = pickle.load(open(skel_pickle_path, 'rb'), encoding="latin1")
    seg = ReadH5(seg_path)
   
    gt_graph, node_segment_lut = GetERLDataFromSkeleton(nodes, edges, [seg], res)
    
    scores = evaluate.expected_run_length(
                    skeletons=gt_graph,
                    skeleton_id_attribute='skeleton_id',
                    edge_length_attribute='length',
                    node_segment_lut=node_segment_lut[0],
                    skeleton_position_attributes=['z', 'y', 'x'])
    print('ERL:', scores)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('need an argument to select the test')
    opt = sys.argv[1]
    if opt=='0': # mesh -> skeleton
        if len(sys.argv) < 4:
            print('need two arguments: skel_pickle_path, seg_path')
        test_erl(sys.argv[2], sys.argv[3])
