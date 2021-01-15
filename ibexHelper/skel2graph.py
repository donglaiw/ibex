import os,sys
from ibex.transforms.seg2seg import DownsampleMapping
from ibex.skeletonization.generate_skeletons import TopologicalThinning, FindEndpointVectors, FindEdges
from ibex.utilities.dataIO import ReadSkeletons
from scipy.ndimage.morphology import binary_fill_holes

import ipyvolume as ipv
import numpy as np

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# skel -> graph
##################
def GetGraphFromSkeleton(skel, dt=None, dt_bb=[0,0,0], modified_bfs=True):
    # tree: BFS
    # not-tree: modified_bfs
    adj_mat = skel.get_adj()
    orig_graph = GetAdjDict(adj_mat)
    node_coords = skel.get_nodes()
    # dt within a bounding box
    node_coords -= dt_bb
    
    new_graph = {}
    visited = [False]*len(node_coords)
    # find a source which has at least one incident edge
    src = None
    for i in range(len(node_coords)-1, -1, -1):
        if i in orig_graph and (False or len(orig_graph[i]) > 2): 
            src = i
            break
    if src is None:
        src = next(iter(orig_graph.keys()))
    if modified_bfs:
        wt_dict, th_dict, ph_dict = ModifiedBFS([src], orig_graph, new_graph, visited, node_coords, dt=dt)
    else:
        wt_dict, th_dict, ph_dict = BFS([src], orig_graph, new_graph, visited, node_coords, dt=dt)
    return new_graph, wt_dict, th_dict, ph_dict


def AddToDict(d, p1, p2):
    if p1 not in d:
        d[p1] = []
    d[p1] += [p2]

def AddEdge(graph, p1, p2):
    AddToDict(graph, p1, p2)
    AddToDict(graph, p2, p1)

def AddEdgeAndWeight(graph, p1, p2, wt_dict, wt, th_dict, th=None):
    AddEdge(graph, p1, p2)
    wt_dict[(p1, p2)] = wt
    wt_dict[(p2, p1)] = wt
    if th is not None:
        th_dict[(p1, p2)] = th
        th_dict[(p2, p1)] = th

def GetAdjDict(adj_mat):
    """
    INPUT:  adj_mat is a compact adjacency matrix produced by Ibex,
            of dimensions |E| x 2 where each tuple is the index
            of source and target node.
    OUTPUT: Dict where keys are node indices, and for each key,
            the value is the list of adjacent nodes.
    """
    adj_dict = {}
    for p1, p2 in adj_mat:
        if p1 != p2: 
            AddToDict(adj_dict, p1, p2)
            AddToDict(adj_dict, p2, p1)
    return adj_dict


# can have cycle
def ModifiedBFS(node_list, orig_graph, new_graph, visited, node_coords, dt=None, \
                use_euclid=True, debug=False):
    def IsJunction(node, graph):
        return (len(graph[node]) == 2)
    
    def IsEdge(n1, n2, graph):
        if n1 in graph and n2 in graph[n1]:
            return True
        return False
    
    def Euclidean(n1, n2, coords):
        return np.linalg.norm(coords[n1,:] - coords[n2,:])
    
    def AvgThick(n1, n2, coords, dt):
        if dt is None: return 0.0
        x1,y1,z1 = coords[n1,:]
        x2,y2,z2 = coords[n2,:]
        return 0.5*(dt[x1,y1,z1] + dt[x2,y2,z2])
    
    def GetNext(src, adj, orig_graph, coords, use_euclid=True, dt=None):
        prev = src
        cur = adj
        path = [prev]
        weight = 1.0
        if use_euclid: 
            weight = Euclidean(prev, cur, coords)
        thickness = AvgThick(prev, cur, coords, dt)*weight
        while IsJunction(cur, orig_graph):
            nxt = orig_graph[cur][int(orig_graph[cur][0] == prev)]
            prev = cur
            cur = nxt
            cur_wt = 1.0
            if use_euclid:
                cur_wt = Euclidean(prev, cur, coords)
            weight += cur_wt
            thickness += AvgThick(prev, cur, coords, dt)*cur_wt
            path += [prev]
        path += [cur]
        try:
            thickness = thickness/weight
        except:
            print('Total edge weight should not be zero.')
        return cur, weight, thickness, path
    
    if debug:
        for key in orig_graph:
            print(key, orig_graph[key])
    new_node = max(orig_graph) 
    weight_dict = {}
    thick_dict = {}
    path_dict = {}
    while len(node_list) > 0:
        src = node_list.pop(0)
        visited[src] = True
        adj_nodes = orig_graph[src]
        if debug: print('Source {:.0f}'.format(src))
        self_loops_ = []
        for adj in adj_nodes:
            if debug: print('  Adj {:.0f}'.format(adj))
            nxt, wt, th, path = GetNext(src, adj, orig_graph, node_coords, use_euclid=use_euclid, \
                                       dt=dt)
            if debug: print('    Nxt {:.0f}'.format(nxt))
            if (not visited[nxt]) or (src == nxt):
                # if there is no edge yet between src & nxt in new_graph, add one
                if (not IsEdge(src, nxt, new_graph)) and (src != nxt):
                    AddEdge(new_graph, src, nxt)
                    if debug: print('      Add Edge {:.0f}-{:.0f}'.format(src, nxt))
                    weight_dict[(src, nxt)] = wt
                    weight_dict[(nxt, src)] = wt
                    thick_dict[(src, nxt)] = th
                    thick_dict[(nxt, src)] = th
                    path_dict[(src, nxt)] = path
                    path_dict[(nxt, src)] = path
                # if there is an edge betwen src & nxt in new_graph, add another path
                # of length two edges between src & nxt, with a new node in between. 
                # (This step allows capturing multiple paths between src and nxt.)
                # (The reason for adding a new node: Networkx won't allow multiple paths).
                elif src != nxt:
                    new_node += 1
                    AddEdgeAndWeight(new_graph, src, new_node, weight_dict, wt/2.0, \
                                    thick_dict, th)
                    AddEdgeAndWeight(new_graph, new_node, nxt, weight_dict, wt/2.0, \
                                    thick_dict, th)
                    if debug: 
                        print('      Add Edge {:.0f}-{:.0f}'.format(src, new_node))
                        print('      Add Edge {:.0f}-{:.0f}'.format(new_node, nxt))
                # If src is same as nxt then there is a self-loop in the skeleton topology.
                # This code snippet captures that loop by creating a triangular loop with
                # the src as one of the vertices and two new nodes.
                elif (src == nxt):
                    if (path not in self_loops_) and (list(reversed(path)) not in self_loops_):
                        self_loops_ += [path]
                        new_node += 1
                        n1 = new_node
                        new_node += 1
                        n2 = new_node
                        AddEdgeAndWeight(new_graph, src, n1, weight_dict, wt/3.0, \
                                        thick_dict, th)
                        AddEdgeAndWeight(new_graph, n1, n2, weight_dict, wt/3.0, \
                                        thick_dict, th)
                        AddEdgeAndWeight(new_graph, n2, nxt, weight_dict, wt/3.0, \
                                        thick_dict, th)
                        if debug: 
                            print('      Add Edge {:.0f}-{:.0f}'.format(src, n1))
                            print('      Add Edge {:.0f}-{:.0f}'.format(n1, n2))
                            print('      Add Edge {:.0f}-{:.0f}'.format(n2, nxt))
                    
                if (nxt not in node_list) and (src != nxt):
                    node_list += [nxt]
    return weight_dict, thick_dict, path_dict

# tree structure, no cycle
def BFS(node_list, orig_graph, new_graph, visited, node_coords, dt=None,\
                use_euclid=True, debug=False):
    def IsJunction(node, graph):
        return (len(graph[node]) == 2)
    
    def IsEdge(n1, n2, graph):
        if n1 in graph and n2 in graph[n1]:
            return True
        return False
    
    def Euclidean(n1, n2, coords):
        return np.linalg.norm(coords[n1,:] - coords[n2,:])
    
    def AvgThick(n1, n2, coords, dt):
        if dt is None: return 0.0
        x1,y1,z1 = coords[n1,:]
        x2,y2,z2 = coords[n2,:]
        return 0.5*(dt[x1,y1,z1] + dt[x2,y2,z2])
    
    def GetNext(src, adj, orig_graph, coords, use_euclid=True, dt=None):
        prev = src
        cur = adj
        path = [prev]
        weight = 1.0
        if use_euclid: 
            weight = Euclidean(prev, cur, coords)
        thickness = AvgThick(prev, cur, coords, dt)*weight
        while IsJunction(cur, orig_graph):
            nxt = orig_graph[cur][int(orig_graph[cur][0] == prev)]
            prev = cur
            cur = nxt
            cur_wt = 1.0
            if use_euclid:
                cur_wt = Euclidean(prev, cur, coords)
            weight += cur_wt
            thickness += AvgThick(prev, cur, coords, dt)*cur_wt
            path += [prev]
        path += [cur]
        try:
            thickness = thickness/weight
        except:
            print('Total edge weight should not be zero.')
        return cur, weight, thickness, path
    
    if debug:
        for key in orig_graph:
            print(key, orig_graph[key])
    new_node = max(orig_graph) 
    weight_dict = {}
    thick_dict = {}
    path_dict = {}
    while len(node_list) > 0:
        src = node_list.pop(0)
        visited[src] = True
        adj_nodes = orig_graph[src]
        if debug: print('Source {:.0f}'.format(src))
            
        for adj in adj_nodes:
            if debug: print('  Adj {:.0f}'.format(adj))
            nxt, wt, th, path = GetNext(src, adj, orig_graph, node_coords, use_euclid=use_euclid, \
                                       dt=dt)
            if debug: print('    Nxt {:.0f}'.format(nxt))
            if (not visited[nxt]):
                    AddEdge(new_graph, src, nxt)
                    if debug: print('      Add Edge {:.0f}-{:.0f}'.format(src, nxt))
                    weight_dict[(src, nxt)] = wt
                    weight_dict[(nxt, src)] = wt
                    thick_dict[(src, nxt)] = th
                    thick_dict[(nxt, src)] = th
                    path_dict[(src, nxt)] = path
                    path_dict[(nxt, src)] = path
                    node_list += [nxt]
                    visited[nxt] = True
    return weight_dict, thick_dict, path_dict
