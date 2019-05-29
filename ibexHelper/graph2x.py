import numpy as np

def Graph2H5(G, pos, ww=2):
    nn = list(G)
    def isJunc(x):
        return G.degree(x)>1

    out = np.zeros(G.graph['shape'],np.uint16)
    for k in nn:
	pt = pos[k]
        val = 1+isJunc(k)
	out[max(0,pt[0]-ww):pt[0]+ww+1,\
	    max(0,pt[1]-ww):pt[1]+ww+1,\
	    max(0,pt[2]-ww):pt[2]+ww+1] = val
    return out

def Graph2Seg(G):
    print('load node position')
    skel = ReadSkeletons(seg_name, skeleton_algorithm='thinning', downsample_resolution=res, read_edges=True)[1]
    nodes = np.stack(skel.get_nodes()).astype(int)

    print('load reduced graph')
    path_dict = readpkl(Ds+'graph-%s-%d-%d.p'%(bfs,edgTh[0],10*edgTh[1]))[0]
    G = nx.read_gpickle(Ds+'graph-%s-%d-%d.obj'%(bfs,edgTh[0],10*edgTh[1]))
    numE = len(G.edges())
    pts = [None]*numE

    rr = 1
    for ei,e in enumerate(G.edges()):
	# remove end point
	nn = list(set(path_dict[e])-set(e))[::rr] 
	pts[ei] = np.hstack([nodes[nn],ei*np.ones((len(nn),1),int)])
    pts = np.vstack(pts)

    print('load seg')
    seg = readh5('data_xt/seg_ds2_label_open1_bv.h5')
   
    print('segment id assignment')
    """
    # method 1: nearest neighbor
    ind = np.where(seg.reshape(-1)==seg_id)[0].reshape((-1,1))
    pts_seg = np.hstack(pos2zyx(ind,szx2))
    numP = len(ind)
    chunk = 10000
    numC = (numP+chunk-1) // chunk
    sid = np.zeros(numP,np.uint16)
    for cid in range(numC):
	print('%d/%d'%(cid,numC))
	sid[cid*chunk:(cid+1)*chunk] = pts[np.argmin(cdist(pts[:,:3],pts_seg[cid*chunk:(cid+1)*chunk]), axis=0),-1]

    print('output decomp')
    out = np.zeros(szx2, np.uint16).reshape(-1)
    for si in range(numE): 
	out[ind[sid==si]] = 1+si
    """

    # method 2: distance transform
    bb_r = 100
    dis = max(szx2)*np.ones(szx2, int)
    out = np.zeros(szx2, np.uint16)
    # O(E)
    for i in range(numE):
	print('%d/%d'%(i,numE))
	pt = pts[pts[:,-1]==i,:3]
	if pt.shape[0]>0:
	    bb = np.hstack([np.maximum(pt.min(axis=0)-bb_r,0),\
			    np.minimum(pt.max(axis=0)+bb_r,szx2-1)])
	    tsz = (bb[3:]-bb[:3])+1
	    # distance to background pix
	    tmp = np.ones(tsz, np.uint8).reshape(-1)
	    tind = zyx2pos(pt[:,0]-bb[0],pt[:,1]-bb[1],pt[:,2]-bb[2],tsz)
	    tmp[tind] = 0
	    # scipy: slow
	    """
	    st=time.time()
	    for j in range(10):
		cc = distance_transform_edt(tmp.reshape(tsz))
	    print(time.time()-st) # 28 sec
	    st=time.time()
	    for j in range(10):
		cc = distance_transform_cdt(tmp.reshape(tsz))
	    print(time.time()-st) # 9 sec
	    import pdb; pdb.set_trace()
	    """
	    cc = distance_transform_cdt(tmp.reshape(tsz))

	    dis_t = dis[bb[0]:bb[3]+1,bb[1]:bb[4]+1,bb[2]:bb[5]+1]
	    out_t = out[bb[0]:bb[3]+1,bb[1]:bb[4]+1,bb[2]:bb[5]+1]

	    gid = cc<dis_t
	    out_t[gid] = i+1
	    dis_t[gid] = cc[gid]

    writeh5(Ds+'decomp-%s-%d-%d_%d.h5'%(bfs,edgTh[0],10*edgTh[1],rr), relabelType(out*(seg==seg_id)).reshape(szx2))


