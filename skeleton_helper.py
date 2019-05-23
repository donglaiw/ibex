"""

These helpers functions were written by Srujan Meesala. They all operate on
an instance of the Skeleton data structure defined in Ibex. This instance is
referred to as 'skel' in all functions below.

"""
import numpy as np


def get_nodes(skel, get_ends=False):
    """Returns N x 3 bdarray of node co-ordinates, where N is # nodes"""
    n_nodes = len(skel.joints)+len(skel.endpoints)
    nodes = np.zeros((n_nodes,3), dtype=np.int)
    for i, joint in enumerate(skel.joints):
        nodes[i,:] = [joint.iz, joint.iy, joint.ix]
    for i, endpoint in enumerate(skel.endpoints):
        nodes[-1-i,:] = [endpoint.iz, endpoint.iy, endpoint.ix]
    if not get_ends:
        return nodes
    else:
        ends = np.ones(n_nodes)
        ends[0:len(skel.joints)] = 0
        return nodes, ends

def get_edges(skel):
    """Returns E x 2 x 3 ndarray of edge co-ordinates,where E is # edges"""
    n_edges = len(skel.edges)
    edges = np.zeros((n_edges,2,3), dtype=np.int)
    # go through all edges
    for i in range(n_edges):
        edges[i,0,:] = [skel.edges[i].source.iz, skel.edges[i].source.iy, skel.edges[i].source.ix]
        edges[i,1,:] = [skel.edges[i].target.iz, skel.edges[i].target.iy, skel.edges[i].target.ix]
    return edges


def get_ends(skel):
    """Returns M x 3 bdarray of node co-ordinates, where N is # endpoints"""
    n_ends = len(skel.endpoints)
    ends = np.zeros((n_ends,3), dtype=np.int)
    for i, endpoint in enumerate(skel.endpoints):
        ends[-1-i,:] = [endpoint.iz, endpoint.iy, endpoint.ix]
    return ends

def get_adj(skel):
    """Returns non-zero elements of adjacency matrix with nodes ordered acc to the get_nodes function"""
    n_edges = len(skel.edges)
    iv_list = [joint.iv for joint in skel.joints]
    iv_list.extend([ep.iv for ep in skel.endpoints])
    adj = np.zeros((n_edges,2), dtype=np.int)
    for i in range(n_edges):
        adj[i,:] = np.array([iv_list.index(skel.edges[i].source.iv), iv_list.index(skel.edges[i].target.iv)])
    return adj

def get_junctions(skel):
    """Returns indices of junctions in node list; junctions are nodes with >2 edges"""
    n_nodes = len(skel.joints)+len(skel.endpoints)
    n_edges = len(skel.edges)
    try:
        assert n_edges > 0
    except:
        return None
    adj = skel.get_adj()
    mask = adj[:,0] != adj[:,1]
    uid, cc = np.unique(adj[mask,:], return_counts=True)
    return uid[np.where(cc>2)]

def length(skel):
    """Returns sum of all edge lengths"""
    n_edges = len(skel.edges)
    sk_length = 0
    n_edges = len(skel.edges)
    # go through all edges
    for i in range(n_edges):
        source = np.array([skel.edges[i].source.iz,
                           skel.edges[i].source.iy,
                           skel.edges[i].source.ix])
        target = np.array([skel.edges[i].target.iz,
                           skel.edges[i].target.iy,
                           skel.edges[i].target.ix])
        sk_length += np.linalg.norm(np.multiply(source-target, np.asarray(skel.resolution)))
    return sk_length
