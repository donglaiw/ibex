import os
import numpy as np
import time
import json

from util import *

def gen_mask(seg, obj_id):
    obj_mask = np.zeros_like(seg, dtype=int)
    obj_mask[seg==obj_id] = 1
    return obj_mask

def get_bbox(seg, obj_id):
    regions = regionprops(gen_mask(seg, obj_id), cache=True)
    # check order
    import pdb; pdb.set_trace()
    return regions[0].bbox

def get_vols(label_data, ids=[]):
    """
    Returns voxel count of requested ids in segmentation
    Args:
        label_data (ndarray): input segmentation
        ids (list of ints)
    Returns:
        vols (ndarray): id #, voxel count sorted in descending order
    """
    if len(ids)==0:
        unique_ids, counts = np.unique(label_data, return_counts=True)
    else:
        unique_ids = ids
        counts = [np.count_nonzero(label_data==id) for id in ids]
    ids_sorted = np.argsort(-counts)
    unique_ids = unique_ids[ids_sorted]
    counts = counts[ids_sorted]
    return (unique_ids, counts)

class SegPrep(object):
    """
    Methods to prepare a segmentation for the error correction pipeline
    """
    def __init__(self, name, res):
        """
        Initializes seg prep task, generates .meta file for input segmentation
        Attributes:
            name (str): folder name for all outputs associated with this segmentation
            resolution (float, float, float): physical resolution in order Z, Y, X in nm
            data (3D ndarray): segmentation data
        """
        self.name = name
        self.resolution = res
        self.shape = None
        self.data = None
        self.seg_ids = None # list of segment labels
        self.n_ids = None # number of segments
        self.bbox_dict = None # bbox coordinates for each segment
        self.fiber_ids = None # list of fiber ids

        create_folder('./segs/')
        create_folder('./segs/' + self.name)
        create_folder('./meta/')
        if not os.path.exists('./meta/' + name + '.meta'):
            meta = open("./meta/" + name +'.meta', "w")
            meta.write("# resolution in nm\n")
            meta.write("%dx%dx%d\n"%(res[2], res[1], res[0]))
            meta.close()
            create_folder('./logs/')
            log = open("./logs/" + name +'.log', "w")
            log.write("created .meta file\n\n")
            log.close()

    def read(self, filename, datasetname, dsmpl=(1,1,1), block_lims=3*((None,None),)):
        """
        Args:
            filename (str)
            datasetname (str): name of group
            dsmpl (int, int, int): downsampling factor along each axis
            block_lims (tuple of 3 tuples, each with 2 ints): extents of array to read in along each axis
        """
        self.data = read3d_h5(filename, datasetname, dsmpl=dsmpl, block_lims=block_lims)
        self.shape = list(self.data.shape)
        self.seg_ids = np.unique(self.data)
        self.n_ids = len(self.seg_ids)
        # log = open("./logs/" + self.name +'.log', "a+")
        # log.write("read from parent file\n")
        # log.write(os.path.abspath(filename)+'\n')
        # log.write("downsampling factor from parent file\n")
        # log.write("%dx%dx%d\n"%(dsmpl[0], dsmpl[1], dsmpl[2]))
        # log.write("slices read from parent file\n")
        # log.write("[%d:%d, %d:%d, %d:%d]\n"%(block_lims[0][0], block_lims[0][1], block_lims[1][0], block_lims[1][1], block_lims[2][0], block_lims[2][1]))
        # log.close()

    def read_internal(self, stage=None):
        """reads previously saved segmentation"""
        if stage is None:
            self.data = read3d_h5('./segs/' + self.name + '/seg.h5', 'main')
        else:
            self.data = read3d_h5('./segs/' + self.name + "/" + stage + "-seg.h5", 'main')
        self.shape = list(self.data.shape)
        self.seg_ids = np.unique(self.data)
        self.n_ids = len(self.seg_ids)

    def set_data(self, seg):
        """assign segmentation data"""
        self.data = seg
        self.shape = list(seg.shape)
        self.seg_ids = np.unique(self.data)
        self.n_ids = len(self.seg_ids)

    def set_bbox_dict(self, bbox_dict):
        self.bbox_dict = bbox_dict

    def padzeros(self, nzeros, axis):
        """
        Pads zeros at end along chosen axis
        Args:
            nzeros (int): number of zeros
            axis (int): axis along which zeros must be appended
        """
        # TO DO : update meta file accordingly
        zero_dims = list(self.data.shape)
        zero_dims[axis] = nzeros
        padding = np.zeros(zero_dims,dtype=np.uint32)
        self.data = np.append(self.data, padding, axis=axis)
        self.shape = self.data.shape
        meta = open("./logs/" + self.name +'.log', "a+")
        log = open("./logs/" + self.name +'.log', "a+")
        log.write("zero-padded flag\n")
        log.close()

    def swapaxes(self, axis1, axis2):
        self.data = self.data.swapaxes(axis1, axis2)
        self.shape = self.data.shape
        res_swapped = [self.resolution[0], self.resolution[1], self.resolution[2]]
        res_swapped[axis1], res_swapped[axis2] = res_swapped[axis2], res_swapped[axis1]
        self.resolution = tuple(res_swapped)
        # TO DO: Change meta file as well
        log = open("./logs/" + self.name +'.log', "a+")
        log.write("swapped axes %d and %d\n"%(axis1, axis2))

    def write(self, stage=None):
        if stage is None: 
            fout = './segs/' + self.name + "/seg.h5"
        else: 
            fout = './segs/' + self.name + "/" + stage + "-seg.h5"
        writeh5(fout, 'main', self.data, compression="gzip", chunks=(1,self.data.shape[1],self.data.shape[2]))
        meta = open("./meta/" + self.name +'.meta', "a+")
        meta.write("# grid size\n")
        meta.write("%dx%dx%d\n"%(self.shape[2], self.shape[1], self.shape[0]))
        meta.close()
        log = open("./logs/" + self.name +'.log', "a+")
        log.write("saved prepared segmentation\n\n")
        log.close()

    def gen_bboxes(self):
        """
        generates bounding box for each object in segmentation
        """
        start_time = time.time()
        if self.seg_ids is None:
            self.seg_ids = np.unique(self.data)
        if self.n_ids is None:
            self.n_ids = len(self.seg_ids)
        self.bbox_dict = {}
        start_msg = "Starting bbox evaluation of %d objects\n"%(self.n_ids)
        print start_msg
        log = open("./logs/" + self.name +'.log', "a+")
        log.write(start_msg)
        bbox_list = []
        for i, seg_id in enumerate(self.seg_ids.tolist()):
            bbox_list.append(get_bbox(self.data, seg_id))
        self.bbox_dict = dict(zip(self.seg_ids, bbox_list))
        save_dict = dict(zip(["%d"%(i) for i in self.seg_ids], bbox_list))
        write_json(save_dict, './segs/'+self.name+'/bboxes.json')
        stop_msg =  "BBox evaluation time: %f\n"%(time.time()-start_time)
        print stop_msg
        log = open("./logs/" + self.name +'.log', "a+")
        log.write(stop_msg)

    def read_bboxes(self, stage=None, external_path=None):
        """
        reads in previously evaluated bounding box information 
        """
        try:
            if external_path is None and stage is None: # read from segs folder of the same segmentation
                self.bbox_dict = read_json('./segs/'+self.name+'/bboxes.json')
            elif external_path is None:
                self.bbox_dict = read_json('./segs/'+self.name+'/bboxes-'+stage+'.json')
            else: # read externally, eg. bboxes computed form lower res segmentation
                self.bbox_dict = read_json(external_path)
        except:
            print "Could not locate bounding box file"
        for i, bbox in self.bbox_dict.items(): # convert str keys to int keys
                self.bbox_dict[long(i)] = self.bbox_dict.pop(i)
        #for label in self.seg_ids.tolist(): # confirm you loaded the right bbox file for the segmentation data
        #    print label
        #    assert label in self.bbox_dict.keys()

    def find_fiber_ids(self, method="bbox-aspect-ratio", params=None):
        """
        Finds fiber IDs in segmentation by inspecting shape statistics
        Args:
            method (str): filtering statistic, currently using Haidong's heuristics
            params (dict): dict of thresholding params, will vary based on method
        TODO (Jeff): 
            add improved aspect ratio calculation from slice wise measurements
        TODO: 
            Make extrude method more efficient by looking for overlapping bboxes instead
            of doing a full volume search
        """
        self.fiber_ids = []
        start_time = time.time()
        if method=="bbox-aspect-ratio" or method=="extrude":
            # use bbox aspect ratio to select fibers
            if self.bbox_dict is None:
                self.read_bboxes()
            # extract filter params
            vol_thresh = params["vol-thresh"] # fiber volume threshold
            len_thresh = params["len-thresh"] # fiber length threshold
            if method=="extrude": 
                area_thresh = params["split-area-thresh"] # area threshold for small fiber splits 
            # begin fiber ID search
            if method=="bbox-aspect-ratio" or bbox=="extrude":
                for obj_id in self.seg_ids.tolist():
                    bbox = self.bbox_dict[obj_id]
                    bbox_vol = (bbox[5]-bbox[2])*(bbox[4]-bbox[1])*(bbox[3]-bbox[0])
                    bbox_len = (bbox[3]-bbox[0])
                    if bbox_vol < vol_thresh and bbox_len > len_thresh:
                        self.fiber_ids.append(obj_id)
                        if method=="extrude" and bbox_len < self.shape[0]:
                            # "extrude" is an optional second level of filtration after "bbox-aspect-ratio"
                            # the goal is to catch small fiber splits that were missed in the first stage
                            y_slice = slice(bbox[1], bbox[4])
                            x_slice = slice(bbox[2], bbox[5])
                            # get box extruded along z-axis to search for small splits
                            extrude_box = np.concatenate((self.data[0:bbox[0],y_slice,x_slice], 
                                                          self.data[bbox[3]:self.shape[0],y_slice,x_slice]), axis=0)
                            objs_caught = np.unique(extrude_box)
                            # keep only those objects splits an area threshold
                            for o_id in objs_caught:
                                o_bbox = self.bbox_dict[o_id]
                                o_area = (o_bbox[5]-o_bbox[2])*(o_bbox[4]-o_bbox[1])
                                if o_area < area_thresh:
                                    self.fiber_ids.append(o_id)
                self.fiber_ids = list(set(self.fiber_ids))
        if method=="dsmpl":
            # "dsmpl" catches the large cell bodies in a downsampled segmentation 
            # and zeros them out in the original segmentation
            dsmpl = params["dsmpl"]
            bvol_thresh = params["bvol-thresh"]
            seg_dsmpl = self.data[::dsmpl[0], ::dsmpl[1], ::dsmpl[2]]
            labels, vols = get_vols(seg_dsmpl)
            body_ids = [label[0] for label in labels[np.argwhere(vols>bvol_thresh)]]
            print "Found %d cell bodies"%(len(body_ids))
            self.fiber_ids = [label for label in self.seg_ids if label not in body_ids]
        else:
            print "Error: Invalid fiber extraction method"
        fout = open('./segs/'+self.name+'/fiber'+'-filt-'+method+'.ids', "w")
        for i in self.fiber_ids:
            fout.write('%d'%(i))
        if method=="bbox-aspect-ratio":
            print "Found %d fibers out of %d objects"%(len(self.fiber_ids), self.n_ids)
        elif method=="extrude":
            print "Found %d fiber components out of %d objects"%(len(self.fiber_ids), self.n_ids)
        print "Fiber extraction time: %f s"%(time.time()-start_time)          

    def filter_fibers(self):
        """
        Zeroes out non-fiber regions of segmentation
        Uses bboxes for fast data access; so assumes bboxes have already been found
        Uses fiber ids; assumes they have already been found
        """
        start_time = time.time()
        remove_ids = list(set(self.seg_ids.tolist()).difference(self.fiber_ids))
        print "Zeroing %d objects that are not fibers"%(len(remove_ids))
        for r_id in remove_ids:
            if r_id==0: 
                continue
            bbox = self.bbox_dict[r_id]
            cropped_seg = self.data[bbox[0]:bbox[3],bbox[1]:bbox[4],bbox[2]:bbox[5]]
            change_vox = list(np.nonzero(cropped_seg==r_id))
            change_vox = tuple([cv + bbox[i] for i, cv in enumerate(change_vox)])
            self.data[change_vox] = 0
            # update bbox_dict
            self.bbox_dict.pop(r_id)
        # update class ID info
        self.seg_ids = np.append(self.fiber_ids, 0)
        self.n_ids = len(self.fiber_ids)+1
        # save bbox dict
        write_json(self.bbox_dict, './segs/'+self.name+'/bboxes-filtered.json')
        print "Fiber filtering time: %f"%(time.time()-start_time)

    def relabel(self, id_map=None, use_bboxes=False, print_labels=False):
        """
        Relabels the segmentation data such that max ID = # objects - 1 or according to supplied ID map
        """
        log = open("./logs/" + self.name +'.log', "a+")
        log.write("relabeled flag\n")
        start_time = time.time()
        print "Starting relabeling of %d objects"%(self.n_ids)
        max_id = np.max(self.seg_ids)

        # case: no need to relabel
        if max_id == self.n_ids-1:
            print "No need to relabel!"
            log.write("False\n")
            log.close()
        else:
            # check if bboxes are available
            if use_bboxes and self.bbox_dict is None:
                self.read_bboxes()
                if self.bbox_dict is None:
                    print "BBox evaluation not completed for " + self.name +". "\
                           "Will proceed to relabel volume without bboxes."
                    use_bboxes = False
            # use ID map eg: generated previously from lower res data
            if id_map is not None: 
                for i in seg_ids: # extend ID map to missing old IDs
                    if i not in id_map.tolist():
                        id_map = np.append(id_map, [i])
                assert self.n_ids == len(id_map)
                for new_id, old_id in enumerate(id_map.tolist()):
                    if print_labels: print "new ID: %d -> old ID: %d"%(new_id, old_id)
                    if use_bboxes:
                        bbox = self.bbox_dict[old_id]
                        cropped_seg = self.data[bbox[0]:bbox[3],bbox[1]:bbox[4],bbox[2]:bbox[5]]
                        change_vox = list(np.nonzero(cropped_seg==old_id))
                        change_vox = tuple([cv + bbox[c_id] for c_id, cv in enumerate(change_vox)])
                        self.data[change_vox] = new_id
                    else:
                        self.data[self.data==old_id] = new_id
                log.write("True\n")
                log.close()
                create_folder('./segs/')
                create_folder('./segs/' + self.name)
                idout = "./segs/" + self.name + "/relabeling-map.npy"
                np.save(idout, id_map)
            # relabel such that max ID = # segments - 1
            else:
                missing_ids = np.sort(np.array(list(set(range(max_id+1)).difference(set(self.seg_ids.tolist())))))
                id_map = np.arange(self.n_ids)
                for i in range(len(missing_ids)):
                    if i==len(missing_ids)-1:
                        ids_to_correct = range(missing_ids[i]+1, max_id+1)
                    else:
                        ids_to_correct = range(missing_ids[i]+1, missing_ids[i+1])
                    for j in ids_to_correct:
                        old_id = j
                        new_id = j-(i+1)
                        if print_labels: print "new ID: %d -> old ID: %d"%(new_id, old_id)
                        if use_bboxes:
                            bbox = self.bbox_dict[old_id]
                            cropped_seg = self.data[bbox[0]:bbox[3],bbox[1]:bbox[4],bbox[2]:bbox[5]]
                            change_vox = list(np.nonzero(cropped_seg==old_id))
                            change_vox = tuple([cv + bbox[c_id] for c_id, cv in enumerate(change_vox)])
                            self.data[change_vox] = new_id
                        else:
                            self.data[self.data==old_id] = new_id
                        id_map[new_id] = old_id
                # update class ID info
                self.seg_ids = np.arange(self.n_ids)
                log.write("True\n")
                log.close()
                create_folder('./segs/')
                create_folder('./segs/' + self.name)
                idout = "./segs/" + self.name + "/relabeling-map.npy"
                np.save(idout, id_map)
            # update IDs in bbox dict
            if use_bboxes:
                for new_id, old_id in enumerate(id_map.tolist()):
                    self.bbox_dict.update({new_id: self.bbox_dict.pop(old_id)})
                write_json(self.bbox_dict, './segs/'+self.name+'/bboxes-relabeled.json')

        print "Relabeling time: %f"%(time.time()-start_time)
