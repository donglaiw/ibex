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
    resolution (type array)
    -> array of size 3 having resolution(in nm) of segmentation
"""

def generate_skels(output_path, seg_path, resolution, ds_resolution = (80,80,80)):
    meta_path = output_path + "/meta.txt"
    segmentation = ReadH5File(seg_path, "main").astype(np.int64)
    grid = [len(segmentation), len(segmentation[0]), len(segmentation[0][0])]
    # order: xyz
    with open(meta_path, "w") as meta_file:
        meta_file.write("# resolution in nm\n")
        meta_file.write("%dx%dx%d\n"%(resolution[2],resolution[1],resolution[0]))
        meta_file.write("# mask filename\n")
        meta_file.write("%s main\n"%seg_path)
        meta_file.write("# grid size\n")
        meta_file.write("%dx%dx%d\n"%(grid[2],grid[1],grid[0]))

    DownsampleMapping(output_path, segmentation, output_resolution=ds_resolution)
    TopologicalThinning(output_path, segmentation, skeleton_resolution=ds_resolution)
    FindEndpointVectors(output_path, skeleton_resolution=ds_resolution)
    FindEdges(output_path, skeleton_resolution=ds_resolution)
    skeletons = ReadSkeletons(output_path)
    
    with open(output_path + 'skel.pkl', 'wb') as fp:
        pickle.dump(skeletons, fp)
