import os,sys
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# post-process graph
#####################
def ShrinkGraph(G, threshold=[0,0], debug=False, prune_jns=True, percentile=None, path_dict=None):
    # abhi's version
    """ 
        Do not contract an edge if edge part of a triangle loop 
        with length shorter than the threshold.
    """
    # weight and thickness
    if percentile is not None:
        wts = np.array([d['weight'] for d in G.edges.values()])
        tks = np.array([d['thick'] for d in G.edges.values()])
        threshold[0] = GetThreshold(wts,percentile[0])
        threshold[1] = GetThreshold(tks,percentile[1])
    th_w,th_t = threshold
    def GetOrphan(a1, a2, s1, s2, G):
        orphan_, other_, orph_node_ = None, None, None
        if len(s1) == 0 and len(s2) == 2: orphan_ = a2; other_ = a1;
        if len(s2) == 0 and len(s1) == 2: orphan_ = a1; other_ = a2;
        # if no edge exists between two adjacent nodes 
        # of orphan, then orphan can be deleted.
        if orphan_ is not None:
            orph_node_ = a1
            m, n = set(G[orphan_].keys()).difference({other_})
            if n not in G[m].keys():
                return orph_node_
        return None

    delete_count = 0
    while True:
        wts = np.array([d['weight'] for d in G.edges.values()])
        tks = np.array([d['thick'] for d in G.edges.values()])
        if prune_jns:
            idx = np.hstack([np.where(wts < threshold[0])[0],\
                             np.where(tks < threshold[1])[0]])
        else:
            mask = [(len(G[e[0][0]].keys()) == 1) or (len(G[e[0][1]].keys()) == 1) \
                    for e in G.edges.items()]
            idx = np.hstack([np.where(wts < threshold[0] & mask)[0],\
                             np.where(tks < threshold[1] & mask)[0]])
        if len(idx) == 0:
            break
        u,v = None, None
        orphan_node = None
        # find a candidate edge to delete in this loop
        for i in idx:
            a1, a2 = G.edges.items()[i][0] 
            s1, s2 = set(G[a1].keys()).difference({a2}), \
                        set(G[a2].keys()).difference({a1})
            intersect_ = s1 & s2
            # If no common node in adjacency list then break
            if len(intersect_) == 0:
                u, v = a1, a2
                orphan_node = GetOrphan(a1, a2, s1, s2, G)
                break
            # Else, make sure no triangular loop of decent size breaks
            else:
                crucial_edge = False
                for n in intersect_:
                    w1 = G.get_edge_data(n, a1)['weight']
                    w2 = G.get_edge_data(n, a2)['weight']
                    # if a loop (containing this edge) with 
                    # len > 3*thresh exists, don't delete this edge
                    if w1 + w2 > 3*threshold[0] - wts[i]:
                        crucial_edge = True
                        break
                if not crucial_edge:
                    u, v = a1, a2
                    orphan_node = GetOrphan(a1, a2, s1, s2, G)
                    break
        if u is None:
            break
        # update path_dict
        if path_dict is not None:
            p0 = path_dict[(u,v)]
            path_dict.pop((u,v), None)
            path_dict.pop((v,u), None)
            # v be removed
            for vv in G.edges(v):
                v2 = vv[0] if vv[0]!=v else vv[1]
                if v2!=u:
                    p1 = path_dict[(v2,v)]
                    path_dict.pop((v,v2), None)
                    path_dict.pop((v2,v), None)
                    path_dict[(u,v2)] = p0+p1 
                    path_dict[(v2,u)] = p0+p1 

        G = nx.contracted_nodes(G, u, v, self_loops=False)
        delete_count += 1
        if orphan_node is not None:
            m, n = G[orphan_node].keys()
            wm, wn = G[orphan_node][m]['weight'], G[orphan_node][n]['weight']
            # add edge between m and n
            G.add_edge(m, n, weight=(wm+wn))
            # delete orphan node
            G.remove_node(orphan_node)
            if path_dict is not None:
                p0 = path_dict[(m,orphan_node)]
                p1 = path_dict[(n,orphan_node)]
                path_dict.pop((m,orphan_node), None)
                path_dict.pop((orphan_node,m), None)
                path_dict.pop((n,orphan_node), None)
                path_dict.pop((orphan_node,n), None)
                path_dict[(m,n)] = p0+p1 
                path_dict[(n,m)] = p0+p1 
        if debug: 
            print('Deleted edge {}-{}'.format(u, v))
            PrintSummary(G)
    # print('Total edges deleted {}'.format(delete_count))
    return G,path_dict

def ShrinkGraph_v2(G, threshold=[0,0], debug=False, percentile=None, path_dict=None):
    # donglai's improved version
    # weight and thickness
    if percentile is not None:
        wts = np.array([d['weight'] for d in G.edges.values()])
        tks = np.array([d['thick'] for d in G.edges.values()])
        threshold[0] = GetThreshold(wts, percentile[0])
        threshold[1] = GetThreshold(tks, percentile[1])
    th_w,th_t = threshold
    delete_count = 0
    iter_count = 0
    while True:
        print('Iteration %d'%(iter_count))
        wts = np.array([d['weight'] for d in G.edges.values()])
        tks = np.array([d['thick'] for d in G.edges.values()])

        # step 1: remove Junction-Endpt edge
        mask = [(len(G[e[0][0]].keys()) == 1) or (len(G[e[0][1]].keys()) == 1) \
                for e in G.edges.items()]
        idx = np.unique(np.hstack([np.asarray((wts < th_w) & mask).nonzero()[0],\
                         np.asarray((tks < th_t) & mask).nonzero()[0]]))
        if len(idx) == 0:
            print('nothing to shrink')
            break

        ees =  G.edges.items()
        kks = [ees[i][0] for i in idx]
        # TODO: based on angle, delete or contract
        for kk in kks:
            a1, a2 = kk 
            path_dict.pop((a1,a2), None)
            path_dict.pop((a2,a1), None)
            if len(G[a1].keys()) == 1:
                G.remove_node(a1)
            if len(G[a2].keys()) == 1:
                G.remove_node(a2)
            delete_count += 1

        # step 2: remove degree=2 node
        # TODO: add thick thres
        nns = G.nodes.items()
        idx = np.where(np.array([len(G.edges(i[0])) for i in nns])==2)[0]
        kks = [nns[i][0] for i in idx]
        for nn in kks:
            m, n = G[nn].keys()
            wm, wn = G[nn][m]['weight'], G[nn][n]['weight']
            tmn = (wm*G[nn][m]['thick']+ wn*G[nn][n]['thick'])/(wn+wm)
            # add edge between m and n
            G.add_edge(m, n, weight=(wm+wn), thick=tmn)
            G.remove_node(nn)
            if path_dict is not None:
                p0 = path_dict[(m,nn)]
                p1 = path_dict[(n,nn)]
                path_dict.pop((m,nn), None)
                path_dict.pop((nn,m), None)
                path_dict.pop((n,nn), None)
                path_dict.pop((nn,n), None)
                path_dict[(m,n)] = p0+p1 
                path_dict[(n,m)] = p0+p1 

            delete_count += 1
        iter_count += 1
    return G,path_dict

def PrintSummary(G):
    print('Summary')
    for e in zip(G.edges, G.edges.values()):
        edge_ = 'Edge({}, {})'.format(e[0][0], e[0][1])
        wt_ = 'Weight: {:.2f}'.format(e[1]['weight'])
        print(edge_ + ' '*(20 - len(edge_)) + wt_)

def MergeTwoEdges(G):
    while True:
        adj_, two_node_ = None, None
        for node in G:
            if len(G[node].keys()) == 2:
                adj_ = G[node].keys()
                two_node_ = node
                break
        if adj_ is None:
            break
        m, n = G[two_node_].keys()
        wm, wn = G[two_node_][m]['weight'], G[two_node_][n]['weight']
        # add edge between m and n
        G.add_edge(m, n, weight=(wm+wn))
        # delete orphan node
        G.remove_node(two_node_)
    return G

def DeLeaf(G, thresh=None, avoid_n=-1):
    leaves = []
    for n in G:
        adj_ = G[n].keys()
        if avoid_n in adj_:
            continue
        if len(adj_) == 1:
            if thresh is None:
                leaves += [n]
            elif G[n][adj_[0]]['weight'] < thresh:
                leaves += [n]
    for n in leaves:
        G.remove_node(n)
    return G

def GetWt(G):
    return np.sum([x['weight'] for x in G.edges.values()])

def GetThreshold(wts, base_percentile=50):
    threshold = np.percentile(wts, base_percentile)
    perc_wts = wts/np.sum(wts)
    base_perc_wt = np.percentile(perc_wts, base_percentile)
    pivots = [base_percentile + 0.5*(j+1) for j in range(70)]
    for p in pivots:
        if p < 100 and np.percentile(perc_wts, p) > 20*base_perc_wt:
            threshold = np.percentile(wts, p)
            break
    return threshold

def GetLeafPercentile(G, perc=50):
    leaves = []
    for n in G:
        adj_ = G[n].keys()
        if len(adj_) == 1:
            leaves += [G[n][adj_[0]]['weight']]
    t = np.percentile(leaves, perc)
    print('Threshold for leaves is {:.1f}'.format(t))
    return t

def GetNumLeaves(G, n):
    num_l = 0
    for m in G[n].keys():
        if len(G[m].keys()) == 1:
            num_l += 1
    return num_l

def GetMaxDegree(G):
    max_n, max_deg, min_leaves = None, 0, 1000
    for n in G:
        if len(G[n].keys()) > max_deg:
            max_deg = len(G[n].keys())
            max_n = n
            min_leaves = GetNumLeaves(G, n)
        elif len(G[n].keys()) == max_deg:
            num_l = GetNumLeaves(G, n)
            if num_l < min_leaves:
                max_n = n
                min_leaves = num_l
    return max_n, max_deg

def AddCenter(G, max_n):
    deg = len(G[max_n].keys())
    max_node = max(G.nodes.keys()) + 1
    G.add_nodes_from([max_node + i for i in range(deg)])
    adj_ = G[max_n].keys()
    for i, n in enumerate(adj_):
        wt = G[max_n][n]['weight']
        G.add_edge(max_node + i, n, weight=wt)
        G.add_edge(max_node + i, max_node + (i+1)%deg, weight=1.0)
    G.remove_node(max_n)
    
def PrintLeafWts(G):
    leaves = []
    for n in G:
        adj_ = G[n].keys()
        if len(adj_) == 1:
            leaves += [G[n][adj_[0]]['weight']]
    print(sorted(leaves))

def PlotGraph(G,show_wt=True,show_th=True):
    if not show_wt and not show_th:
        nx.draw_networkx(G)
    else:
        if show_th:
            edge_dict = {e[0]: '({:.1f}, {:.1f})'.format(e[1]['weight'], \
                                                         e[1]['thick']) for e in G.edges.items()}
        else:
            edge_dict = {e[0]: '{:.1f}'.format(e[1]['weight']) for e in G.edges.items()}
        pos=graphviz_layout(G)
        nx.draw_networkx(G, pos=pos, node_size=sz, ax=ax, with_labels=labels,
                        font_size=5, font_color='white', font_weight='bold') #, edge_color='red')        
def DrawGraph(ax, edgelist, sz=1, labels=False, weighted=False, threshold=1.0, show_th=False, \
             show_wts=False, save_graphx=False, gx_dir=None, seg_id=None, percentile=None, \
             percent=None, prune_jns=True, is_tree=False):
   
    def ShrinkGraph(G, threshold, debug=False, prune_jns=True):
        """ 
            Do not contract an edge if edge part of a triangle loop 
            with length shorter than the threshold.
        """
        def GetOrphan(a1, a2, s1, s2, G):
            orphan_, other_, orph_node_ = None, None, None
            if len(s1) == 0 and len(s2) == 2: orphan_ = a2; other_ = a1;
            if len(s2) == 0 and len(s1) == 2: orphan_ = a1; other_ = a2;
            # if no edge exists between two adjacent nodes 
            # of orphan, then orphan can be deleted.
            if orphan_ is not None:
                orph_node_ = a1
                m, n = set(G[orphan_].keys()).difference({other_})
                if n not in G[m].keys():
                    return orph_node_
            return None
        if threshold == 0.0: return G
        if debug: PrintSummary(G)
        
        if percentile is not None:
            wts = np.array([d['weight'] for d in G.edges.values()])
            threshold = np.percentile(wts, percentile)
            perc_wts = wts/np.sum(wts)
            base_ = np.percentile(perc_wts, percentile)
            pivots = [percentile + 0.5*(j+1) for j in range(70)]
            for p in pivots:
                if p < 100 and np.percentile(perc_wts, p) > 50*base_:
                    threshold = np.percentile(wts, p)
                    print('Threshold found at {:.2f}'.format(p))
                    break
        delete_count = 0
        while True:
            wts = np.array([d['weight'] for d in G.edges.values()])
            if prune_jns:
                idx = np.asarray(wts < threshold).nonzero()[0]
            else:
                mask = [(len(G[e[0][0]].keys()) == 1) or (len(G[e[0][1]].keys()) == 1) \
                        for e in G.edges.items()]
                idx = np.asarray((wts < threshold) & mask).nonzero()[0]
            if len(idx) == 0:
                break
            u,v = None, None
            orphan_node = None
            # find a candidate edge to delete in this loop
            for i in idx:
                a1, a2 = G.edges.items()[i][0] 
                s1, s2 = set(G[a1].keys()).difference({a2}), \
                            set(G[a2].keys()).difference({a1})
                intersect_ = s1 & s2
                # If no common node in adjacency list then break
                if len(intersect_) == 0:
                    u, v = a1, a2
                    orphan_node = GetOrphan(a1, a2, s1, s2, G)
                    break
                # Else, make sure no triangular loop of decent size breaks
                else:
                    crucial_edge = False
                    for n in intersect_:
                        w1 = G.get_edge_data(n, a1)['weight']
                        w2 = G.get_edge_data(n, a2)['weight']
                        # if a loop (containing this edge) with 
                        # len > 3*thresh exists, don't delete this edge
                        if w1 + w2 > 3*threshold - wts[i]:
                            crucial_edge = True
                            break
                    if not crucial_edge:
                        u, v = a1, a2
                        orphan_node = GetOrphan(a1, a2, s1, s2, G)
                        break
            if u is None:
                break
            G = nx.contracted_nodes(G, u, v, self_loops=False)
            delete_count += 1
            if orphan_node is not None:
                m, n = G[orphan_node].keys()
                wm, wn = G[orphan_node][m]['weight'], G[orphan_node][n]['weight']
                # add edge between m and n
                G.add_edge(m, n, weight=(wm+wn))
                # delete orphan node
                G.remove_node(orphan_node)
            if debug: 
                print('Deleted edge {}-{}'.format(u, v))
                PrintSummary(G)
        print('Total edges deleted {}'.format(delete_count))
        return G
    
    G = nx.Graph()
    if not weighted:
        G.add_edges_from(edgelist)
        pos=graphviz_layout(G)
        nx.draw_networkx(G, pos=pos, node_size=sz, ax=ax, with_labels=labels,
                        font_size=8, font_color='white', font_weight='bold', edge_color='red')    
    else:
        G.add_edges_from(edgelist)
        
        t0_shrink = time.time()

        if is_tree:
            if True:
                factor = 5.0**(-3)
                wt = GetWt(G)
                t = factor*wt
                G = MergeTwoEdges(G)

                q0 = 20
                t0 = GetLeafPercentile(G, perc=q0)
                print('Original {:.0f}%ile weight {:.2f}'.format(q0, t0))
                for p in [25, 40, 80, 40, 20]:
                    t = GetLeafPercentile(G, perc=p)
                    G = DeLeaf(G, thresh=t)
                    G = MergeTwoEdges(G)

                G = ShrinkGraph(G, threshold=t0, prune_jns=True)
                G = DeLeaf(G, thresh=t0)
                G = MergeTwoEdges(G)
                max_n, max_deg = GetMaxDegree(G)

                if max_n in G.nodes.keys():
                    AddCenter(G, max_n)
        else:
            G = ShrinkGraph(G, threshold=threshold, prune_jns=True)

        if save_graphx:
            nx.write_gpickle(G, os.path.join(gx_dir, str(seg_id) + '_networkx.obj'))
        print('Graph shrunk in {:.2f}s'.format(time.time() - t0_shrink))
        if show_th:
            edge_dict = {e[0]: '({:.1f}, {:.1f})'.format(e[1]['weight'], \
                                                         e[1]['thick']) for e in G.edges.items()}
        else:
            edge_dict = {e[0]: '{:.1f}'.format(e[1]['weight']) for e in G.edges.items()}
        pos=graphviz_layout(G)
        nx.draw_networkx(G, pos=pos, node_size=sz, ax=ax, with_labels=labels,
                        font_size=5, font_color='white', font_weight='bold') #, edge_color='red')    
        if show_wts:
            nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_dict)

def MakeAndPlotGraph(voxel_dir, seg_id, graph_dir='./', threshold=0.0, show_graph=True, \
                     save_graph=True, skel=None, dt=None, use_dt=False, show_orig=False, \
                    gx_dir=None, save_graphx=False, percentile=None, in_res=(30,30,30), \
                    out_res=(30,30,30), show_wts=False, return_graph=False, prune_jns=True, \
                    modified_bfs=True):
    """
    ==================================================
    INPUTS
    ==================================================

    voxel_dir:      String, the directory which contains voxelized representation of an organelle.
                    Example of directory structure:
                    voxel_dir
                    |
                    |__7/seg.h5
                    |
                    |__102/seg.h5
    
    seg_id:         Integer, representing ID of the organelle to be processed. In the above directory
                    structure example, 7 and 102 are the seg_ids.
    
    gx_dir:         String, directory where the networkx objects will be stored if save_graphx is True.

    save_graphx:    Boolean, if True then the networkx object will be saved in the directory
                    specified by gx_dir.

    graph_dir:      String, directory where plot of the reduced graph will be stored if save_graph is True.

    save_graph:     Boolean, if True then a plot of the reduced graph is stored inside graph_dir.

    threshold:      Float, all edges with weights below this threshold will be collapsed.

    modified_bfs:   Boolean. If True, then the ModifiedBFS algorithm is used - the use case is when the 
                    underlying organelle contains loops (e.g. mitochondrias). If False, then simple BFS
                    is used - this is for Pyramidal Cells which do not contain loops, and have a tree
                    structure.
    
    show_graph:     Boolean. If True the reduced graph is displayed as a plot. Default True.

    show_orig:      Boolean. If True the original skeleton graph (prior to running the Modified BFS algo.
                    on it) is plotted. This may take a lot of time as the skeleton graphs usually have a
                    lot of nodes. Default False.

    show_wts:       Boolean. If True, then edges in reduced graph plot are annotated with edge weights.

    
    prune_jns:      Boolean. True for Pyramidal Cells and False for Mitochondria. If True then even those
                    edges, both of whose ends are Key Nodes, will be collapsed if they are shorter than
                    threshold.
    """
    
    is_tree = False
    if not modified_bfs:
        is_tree = True
    
    # Create skeleton from voxels
    if skel is None:
        if use_dt:
            skel, dt = CreateSkeleton(voxel_dir, seg_id, return_dt=True, \
                                     in_res=in_res, out_res=out_res)
        else:
            skel = CreateSkeleton(voxel_dir, seg_id, return_dt=False, \
                                     in_res=in_res, out_res=out_res)
    # Create graph from skeleton
    t0_gr = time.time()
    new_graph, wt_dict, th_dict = GetGraphFromSkeleton(skel, dt=dt, \
                                                       modified_bfs=modified_bfs)
    print('Graph generated in {:.3f}s'.format(time.time() - t0_gr))
    
    # Create edge_lists to be used for plotting graph
    t0_ed = time.time()
    edgelist_new = GetEdgeList(new_graph, wt_dict, th_dict)
    if show_orig:
        edgelist_orig = GetEdgeList(GetAdjDict(sh.get_adj(skel)))
    print('Retrieved edgelists for plotting in {:.3f}s'.format(time.time() - t0_ed))
    
    if show_graph:
        if show_orig:
            fig, ax = plt.subplots(2,1)
            DrawGraph(ax[0], edgelist_orig, sz=10) 
            ax[0].set_title('Skeleton Graph (All Nodes)', fontsize=14, fontweight='bold');
            DrawGraph(ax[1], edgelist_new, weighted=True, threshold=threshold, gx_dir=gx_dir, \
                     seg_id=seg_id, save_graphx=save_graphx, show_wts=show_wts, percentile=percentile, \
                     prune_jns=prune_jns, is_tree=is_tree)
            ax[1].set_title('Skeleton Graph (Only Junction Nodes)', fontsize=14, fontweight='bold');
            for i in [0,1]:
                ax[i].get_xaxis().set_visible(False)
                ax[i].get_yaxis().set_visible(False)
            fig.set_size_inches(18, 14)
        else:
            fig, ax = plt.subplots()
            print('Plotting graph..', is_tree)
            DrawGraph(ax, edgelist_new, weighted=True, threshold=threshold, gx_dir=gx_dir, \
                     seg_id=seg_id, save_graphx=save_graphx, show_wts=show_wts, percentile=percentile, \
                     prune_jns=prune_jns, is_tree=is_tree)
            ax.set_title('Skeleton Graph (Only Junction Nodes)', fontsize=14, fontweight='bold');
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.set_size_inches(18, 7)
        if save_graph:
            fig.savefig(os.path.join(graph_dir, 'graph' + str(seg_id) + \
                                     '_thresh_{:.1f}.png'.format(threshold)))
    if return_graph:
        G = nx.Graph()
        G.add_edges_from(edgelist_new)
        return G
