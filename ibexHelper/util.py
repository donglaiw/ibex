import numpy as np
import pickle

def get_bb(seg, do_count=False):
    dim = len(seg.shape)
    a=np.where(seg>0)
    if len(a)==0:
        return [-1]*dim*2
    out=[]
    for i in range(dim):
        out+=[a[i].min(), a[i].max()]
    if do_count:
        out+=[len(a[0])]
    return out

def writepkl(filename, content):
    with open(filename, "wb") as f:
        if isinstance(content, (list,)):
            for val in content:
                pickle.dump(val, f)
        else:
            pickle.dump(content, f)

def readpkl(filename):
    data = []
    with open(filename, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except:
                break
    return data


