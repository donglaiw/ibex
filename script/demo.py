import os,sys

# add ibexHelper path
sys.path.append('/home/donglai/lib/ibex_fork/ibexHelper')
from ibexHelper.skel import CreateSkeleton
import h5py
import numpy as np

opt = sys.argv[1]

res = [120,128,128] # z,y,x
if opt=='0': # mesh -> skeleton
    fn = '/mnt/coxfs01/donglai/data/JWR/snow_cell/cell128nm/neuron/cell26_d.h5'
    seg = np.array(h5py.File(fn,'r')['main'])

    CreateSkeleton(seg, '../tmp/demo', res, res)

