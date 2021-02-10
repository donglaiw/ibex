import numpy as np
import pickle
import h5py

def GetBbox(seg, do_count=False):
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

def WritePkl(filename, content):
    with open(filename, "wb") as f:
        if isinstance(content, (list,)):
            for val in content:
                pickle.dump(val, f)
        else:
            pickle.dump(content, f)

def ReadPkl(filename):
    data = []
    with open(filename, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except:
                break
    return data

def WriteH5(filename, dtarray, datasetname='main'):
    fid=h5py.File(filename,'w')                                                                      
    if isinstance(datasetname, (list,)):                                                             
        for i,dd in enumerate(datasetname):                                                          
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]                                                                       
    else:                                                                                            
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype) 
        ds[:] = dtarray                                                                              
    fid.close()    

def ReadH5(filename, datasetname='main'):
    fid=h5py.File(filename,'r')
    if isinstance(datasetname, (list,)):
        out = [None] *len(datasetname)
        for i,dd in enumerate(datasetname):
            out[i] = np.array(fid[dd])
    else:                                                                                            
        sz = len(fid[datasetname].shape)
        out = np.array(fid[datasetname])
    return out

def createFolder(fpath):
    if not os.path.isdir(fpath):
            os.mkdir(fpath)
    else:
        pass

def readNpy(filename):
    if os.path.exists(filename):
        return np.load(filename)
    else:
        return None

def writeNpy(arr, filename):
    np.save(filename, arr)

def readJson(filename):
    if os.path.exists(filename):
        with open(filename) as f:
            data = json.load(f)
        return data
    else:
        return None

def writeJson(dictionary, filename):
    with open(filename, 'w') as f:
        data = json.dump(dictionary, f)
