"""
NOTE: Credit for most of this code goes to Brian Matejek who wrote
the Ibex package, and Srujan Meesala who provided me with scripts on
how to use Ibex.

"""
import numpy as np
import ipyvolume as ipv
import time
import os
import h5py

from scipy import ndimage
from skimage.morphology import medial_axis
from scipy.ndimage.morphology import binary_fill_holes

from ibex.transforms.seg2seg import DownsampleMapping
from ibex.skeletonization.generate_skeletons import TopologicalThinning, FindEndpointVectors, FindEdges
from ibex.utilities.dataIO import ReadSkeletons

import importlib
sh = importlib.import_module('skeleton_helper')

def CreateMetaFile(seg_name, resolution, seg_shape):
    if not os.path.isdir('./meta/'):
        os.mkdir('./meta/')
    meta = open("./meta/" + seg_name +'.meta', "w")
    meta.write("# resolution in nm\n")
    meta.write("%dx%dx%d\n"%(resolution[2], resolution[1], resolution[0]))
    meta.write("# grid size\n")
    meta.write("%dx%dx%d\n"%(seg_shape[2], seg_shape[1], seg_shape[0]))
    meta.close()
    
def CheckIds(seg, verbose=False):
    """
    checks if all IDs in segmentation are consecutive
    Args:
        seg (ndarray): segmentation data
    """
    seg_ids = np.unique(seg)
    max_id = np.max(seg_ids)
    n_ids = len(seg_ids)
    try:
        assert max_id == n_ids-1
    except:
        missing_ids = np.sort(np.array(list(set(range(max_id+1)).difference(set(seg_ids)))))
        if verbose:
            print "Error! Labels in segmentation are not consecutive. %d IDs are missing"%(len(missing_ids))
            print missing_ids
            
def Relabel(seg):
    """
    Relabels a segmentation such that max ID = # objects
    Args:
        seg (ndarray): 3D segmentation
    Returns:
        seg_relabeled (ndarray)
    """
    seg_ids = np.unique(seg)
    n_ids = len(seg_ids)
    max_id = np.max(seg_ids)
    if max_id == n_ids-1:
        return seg
    missing_ids = np.sort(np.array(list(set(range(max_id+1)).difference(set(seg_ids)))))
    seg_relabel = seg
    for i in range(len(missing_ids)):
        if i==len(missing_ids)-1:
            ids_to_correct = range(missing_ids[i]+1, max_id+1)
        else:
            ids_to_correct = range(missing_ids[i]+1, missing_ids[i+1])
        for j in ids_to_correct:
            seg_relabel[seg==j] = j-(i+1)
    return seg_relabel

def CreateSkeleton(voxel_dir, seg_id, skel_dir='./', plot_type=None, return_dt=False, \
                  in_res=None, out_res=None):
    """
    This function uses Ibex to exctract the skeleton out of a voxel representation (in
    a .h5 file). It optionally stores the skeleton plot as an html file.

    ====================
    INPUTS:
    ====================
    voxel_dir: String, the directory which stores the h5 files (each h5 file represents
                one organelle). An example of the structure of this voxel_dir is:
                voxel_dir
                |
                |__1/seg.h5
                |
                |__7/seg.h5

    seg_id:    Integer, representing the ID of the organelle inside the voxel_dir. For
                example, in the directory structure shown above, 1 and 7 are seg_ids.

    skel_dir:   String, the directory where plot of the extracted skeleton is stored as
                an HTML file.

    in_res:     Tuple of three integers, representing the resolution of the input .h5.
                Default value is (30, 48, 48)

    out_res:    Tuple of three integers, representing the resolution we want to use for
                extraction of the skeleton. This is used only if you want to downsample.

    plot_type: String, 'nodes' or 'edges'. If it is 'nodes' then only nodes of the skeleton
                are plotted. If 'edges' then edges between nodes are also plotted. This can
                take a lot of time in the 'edges' mode if the skeleton is large. Default
                value is None in which case nothing is plotted.

    return_dt: Boolean, returns distance transform if True.


    ====================
    OUTPUTS:
    ====================
    skeleton:   A skeleton object.
    
    dt:         Distance transform, only returned if return_dt is True.

    """
    seg_path = os.path.join(voxel_dir, '{:.0f}/seg.h5'.format(seg_id))
    segments = np.array(h5py.File(seg_path,'r')['main']).astype(np.int64)
    seg_name = 'temp'
    in_resolution = (30, 48, 48)
    if in_res is not None:
        in_resolution = in_res
    out_resolution = in_resolution
    if out_res is not None:
        out_resolution = out_res
    
    CreateMetaFile(seg_name, in_resolution, segments.shape)
    # Following code not needed as each segment file has a single organelle
    #     CheckIds(segments)
    #     segments = Relabel(segments)
    segments[segments > 0] = 1
    segments = binary_fill_holes(segments).astype('int64')
    
    DownsampleMapping(seg_name, segments, output_resolution=out_resolution)
    TopologicalThinning(seg_name, segments, skeleton_resolution=out_resolution)
    FindEndpointVectors(seg_name, skeleton_algorithm='thinning', skeleton_resolution=out_resolution)
    FindEdges(seg_name, skeleton_algorithm='thinning', skeleton_resolution=out_resolution)
    t0_read = time.time()
    skeletons = ReadSkeletons(seg_name, skeleton_algorithm='thinning', downsample_resolution=out_resolution, read_edges=True)
    print('Read skeletons in {:.3f}s'.format(time.time() - t0_read))
    
    if plot_type is not None:
        t0_plt = time.time()
        node_list = sh.get_nodes(skeletons[1])
        nodes = np.stack(node_list).astype(float)
        junction_idx = sh.get_junctions(skeletons[1])
        junctions = nodes[junction_idx, :]
        ends = sh.get_ends(skeletons[1])
        jns_ends = np.vstack([junctions, ends])
        
        IX, IY, IZ = 2, 1, 0
        ipv.figure()
        if plot_type == 'nodes':
            nodes = ipv.scatter(nodes[:,IX], nodes[:,IY], nodes[:,IZ], \
                                size=0.5, marker='sphere', color='blue')
        elif plot_type == 'edges':
            edges = sh.get_edges(skeletons[1]).astype(float)
            for e1, e2 in edges:
                if not ((e1[IX] == e2[IX]) and (e1[IY] == e2[IY]) and (e1[IZ] == e2[IZ])):
                    ipv.plot([e1[IX], e2[IX]], [e1[IY], e2[IY]], [e1[IZ], e2[IZ]], \
                             color='blue');

        jns = ipv.scatter(jns_ends[:,IX], jns_ends[:,IY], jns_ends[:,IZ], \
                          size=0.85, marker='sphere', color='red')
        ipv.pylab.style.axes_off()
        ipv.pylab.style.box_off()

        ipv.save(os.path.join(skel_dir, 'skel' + str(seg_id) + '.html'))
        print('Plot created and saved in {:.3f}s'.format(time.time() - t0_plt))
    if return_dt:
        t0_dt = time.time()
        dt = distance_transform_edt(segments, return_distances=True)
        print('Distance Transform computed in {:.3f}s'.format(time.time() - t0_dt))
        return skeletons[1], dt
    else:
        return skeletons[1]
