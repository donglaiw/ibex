import os,sys
from ibex.transforms.seg2seg import DownsampleMapping
from ibex.skeletonization.generate_skeletons import TopologicalThinning, FindEndpointVectors, FindEdges
from ibex.utilities.dataIO import ReadSkeletons
from scipy.ndimage.morphology import binary_fill_holes

import ipyvolume as ipv
import numpy as np

# skel operation
##################
def CreateSkeleton(segment, out_folder = 'temp/', in_res=(30, 48, 48), out_res=(30, 48, 48)):
    """
    This function uses Ibex to exctract the skeleton out of a voxel representation (in
    a .h5 file). It optionally stores the skeleton plot as an html file.

    ====================
    INPUTS:
    ====================

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
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print('meta file')
    CreateMetaFile(in_res, segment.shape, out_folder)

    print('seg: pre-process')
    segment[segment > 0] = 1
    segment = binary_fill_holes(segment).astype('int64')
    
    print('seg: downsample')
    DownsampleMapping(out_folder, segment, output_resolution=out_res)
    print('skel: topological thining')
    TopologicalThinning(out_folder, segment, skeleton_resolution=out_res)
    print('graph: edge/end-pt')
    FindEndpointVectors(out_folder, skeleton_algorithm='thinning', skeleton_resolution=out_res)
    FindEdges(out_folder, skeleton_algorithm='thinning', skeleton_resolution=out_res)
 
def CreateMetaFile(resolution, seg_shape,out_folder='./'):
    meta = open(os.path.join(out_folder, 'meta.txt'), "w")
    meta.write("# resolution in nm\n")
    meta.write("%dx%dx%d\n"%(resolution[2], resolution[1], resolution[0]))
    meta.write("# grid size\n")
    meta.write("%dx%dx%d\n"%(seg_shape[2], seg_shape[1], seg_shape[0]))
    meta.close()

 
def PlotSkeleton(seg_name, plot_type='node', out_res=(30,48,48)): 
    print('Read skeletons')
    skeletons = ReadSkeletons(seg_name, skeleton_algorithm='thinning', downsample_resolution=out_res, read_edges=True)

    print('Plot skeletons')
    node_list = skeletons[1].get_nodes()
    nodes = np.stack(node_list).astype(float)
    junction_idx = skeletons[1].get_junctions()
    junctions = nodes[junction_idx, :]
    ends = skeletons[1].get_ends()
    jns_ends = np.vstack([junctions, ends])
    
    IX, IY, IZ = 2, 1, 0
    ipv.figure()
    if plot_type == 'nodes':
        nodes = ipv.scatter(nodes[:,IX], nodes[:,IY], nodes[:,IZ], \
                            size=0.5, marker='sphere', color='blue')
    elif plot_type == 'edges':
        edges = skeletons[1].get_edges().astype(float)
        for e1, e2 in edges:
            if not ((e1[IX] == e2[IX]) and (e1[IY] == e2[IY]) and (e1[IZ] == e2[IZ])):
                ipv.plot([e1[IX], e2[IX]], [e1[IY], e2[IY]], [e1[IZ], e2[IZ]], \
                         color='blue');

    jns = ipv.scatter(jns_ends[:,IX], jns_ends[:,IY], jns_ends[:,IZ], \
                      size=0.85, marker='sphere', color='red')
    ipv.pylab.style.axes_off()
    ipv.pylab.style.box_off()

    ipv.save(seg_name + '_skel.html')

def SaveMeshAsHTML(cell_dir, stl_id, mesh_dir='./'):
    in_mesh = GetMesh(cell_dir, stl_id)
    IX, IY, IZ = 2, 1, 0
    x, y, z = in_mesh.vertices[:,IX].astype(float), \
                in_mesh.vertices[:,IY].astype(float), \
                in_mesh.vertices[:,IZ].astype(float)
    triangles = in_mesh.faces
    ipv.figure()
    mesh = ipv.plot_trisurf(x, y, z, triangles=triangles, color='skyblue')
    ipv.pylab.style.axes_off()
    ipv.pylab.style.box_off()
    ipv.save(os.path.join(mesh_dir, 'mesh' + str(stl_id) + '.html'))
