import numpy as np
import pickle
# ibex imports
from ibex.transforms.seg2seg import DownsampleMapping
from ibex.skeletonization.generate_skeletons import TopologicalThinning, FindEndpointVectors, FindEdges
from ibex.utilities.dataIO import ReadH5File,ReadSkeletons

"""
    output_path (type string) 
    -> complete path to the output folder
    seg_path (type string)
    -> complete path to the segmentation file (file must be in .h5 format)
    dsmpl_res (type array)
    -> array of size 3 having resolution(in nm) of segmentation
"""

def generate_skels(output_path, seg_path, dsmpl_res=(80,80,80)):
    meta_path = output_path + "/meta.txt"
    segmentation = ReadH5File(seg_path, "main").astype(np.int64)
    grid = [len(segmentation), len(segmentation[0]), len(segmentation[0][0])]
    with open(meta_path, "w") as meta_file:
        meta_file.write("# resolution in nm\n")
        meta_file.write("%dx%dx%d\n"%(dsmpl_res[0],dsmpl_res[1],dsmpl_res[2]))
        meta_file.write("# mask filename\n")
        meta_file.write("%s main\n"%seg_path)
        meta_file.write("# grid size\n")
        meta_file.write("%dx%dx%d\n"%(grid[0],grid[1],grid[2]))

    DownsampleMapping(output_path, segmentation, output_resolution=dsmpl_res)
    TopologicalThinning(output_path, segmentation, skeleton_resolution=dsmpl_res)
    FindEndpointVectors(output_path, skeleton_resolution=dsmpl_res)
    FindEdges(output_path, skeleton_resolution=dsmpl_res)
    skeletons = ReadSkeletons(output_path)
    
    with open(output_path + 'skel.pkl', 'wb') as fp:
        pickle.dump(skeletons, fp)
