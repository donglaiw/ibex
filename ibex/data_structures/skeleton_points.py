import struct
import numpy as np
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from ibex.utilities.constants import *

class Joint:
    def __init__(self, iv, iz, iy, ix):
        self.iv = iv
        self.iz = iz
        self.iy = iy
        self.ix = ix

class Endpoint:
    def __init__(self, iv, iz, iy, ix, vector):
        self.iv = iv
        self.iz = iz
        self.iy = iy
        self.ix = ix
        self.vector = np.array(vector, dtype=np.float32)

class Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target

class Skeleton:
    def __init__(self, label, joints, endpoints, vectors, resolution, grid_size, edges=None):
        self.label = label
        self.grid_size = grid_size
        self.resolution = resolution
        self.joints = []
        self.endpoints = []
        self.edges = []        

        for joint in joints:
            iz = joint / (grid_size[IB_Y] * grid_size[IB_X])
            iy = (joint - iz * grid_size[IB_Y] * grid_size[IB_X]) / grid_size[IB_X]
            ix = joint % grid_size[IB_X]

            self.joints.append(Joint(joint, iz, iy, ix))

        for endpoint in endpoints:
            iz = endpoint / (grid_size[IB_Y] * grid_size[IB_X])
            iy = (endpoint - iz * grid_size[IB_Y] * grid_size[IB_X]) / grid_size[IB_X]
            ix = endpoint % grid_size[IB_X]

            vector = vectors[endpoint]

            self.endpoints.append(Endpoint(endpoint, iz, iy, ix, vector))

        if edges is not None:
            for source, target in itertools.izip(*edges):
                iz_s = source / (grid_size[IB_Y] * grid_size[IB_X])
                iy_s = (source - iz_s * grid_size[IB_Y] * grid_size[IB_X]) / grid_size[IB_X]
                ix_s = source % grid_size[IB_X]
                iz_t = target / (grid_size[IB_Y] * grid_size[IB_X])
                iy_t = (target - iz_t * grid_size[IB_Y] * grid_size[IB_X]) / grid_size[IB_X]
                ix_t = target % grid_size[IB_X]
                # note - the Edge object does not distinguish Joints and Endpoints
                # Endpoints are mapped to Joints with positive co-ordinates
                self.edges.append(Edge(Joint(source, iz_s, iy_s, ix_s), Joint(target, iz_t, iy_t, ix_t)))

    def get_edges(self):
        """Returns E x 2 x 3 ndarray of edge co-ordinates,where E is # edges"""
        n_edges = len(self.edges)
        edges = np.zeros((n_edges,2,3), dtype=np.int)
        # go through all edges
        for i in range(n_edges):
            edges[i,0,:] = [self.edges[i].source.iz, self.edges[i].source.iy, self.edges[i].source.ix]
            edges[i,1,:] = [self.edges[i].target.iz, self.edges[i].target.iy, self.edges[i].target.ix]
        return edges

    def get_nodes(self, get_ends=False):
        """Returns N x 3 bdarray of node co-ordinates, where N is # nodes"""
        n_joints = len(self.joints)
        n_nodes = len(self.joints)+len(self.endpoints)
        nodes = np.zeros((n_nodes,3), dtype=np.int)
        for i, joint in enumerate(self.joints):
            nodes[i,:] = [joint.iz, joint.iy, joint.ix]
        for i, endpoint in enumerate(self.endpoints):
            nodes[n_joints+i,:] = [endpoint.iz, endpoint.iy, endpoint.ix]
	if not get_ends:
	    return nodes
	else:
	    ends = np.ones(n_nodes)
	    ends[0:len(self.joints)] = 0
	    return nodes, ends

    def get_ends(self):
        """Returns M x 3 bdarray of node co-ordinates, where N is # endpoints"""
        n_ends = len(self.endpoints)
        ends = np.zeros((n_ends,3), dtype=np.int)
        for i, endpoint in enumerate(self.endpoints):
            ends[-1-i,:] = [endpoint.iz, endpoint.iy, endpoint.ix]
	return ends

    def get_adj(self):
        """Returns non-zero elements of adjacency matrix with nodes ordered acc to the get_nodes function"""
        n_edges = len(self.edges)
        iv_list = [joint.iv for joint in self.joints]
        iv_list.extend([ep.iv for ep in self.endpoints])
        adj = np.zeros((n_edges,2), dtype=np.int)
        for i in range(n_edges):
            adj[i,:] = np.array([iv_list.index(self.edges[i].source.iv), iv_list.index(self.edges[i].target.iv)])
        return adj

    def get_junctions(self):
        """Returns indices of junctions in node list; junctions are nodes with >2 edges"""
        n_nodes = len(self.joints)+len(self.endpoints)
        n_edges = len(self.edges)
        try:
            assert n_edges > 0
        except:
            return None
	adj = self.get_adj()
	mask = adj[:,0] != adj[:,1]
        uid, cc = np.unique(adj[mask,:], return_counts=True)
        return uid[np.where(cc>2)]

    def length(self):
        """Returns sum of all edge lengths"""
        n_edges = len(self.edges)
        sk_length = 0
        n_edges = len(self.edges)
        # go through all edges
        for i in range(n_edges):
            source = np.array([self.edges[i].source.iz,
                               self.edges[i].source.iy,
                               self.edges[i].source.ix])
            target = np.array([self.edges[i].target.iz,
                               self.edges[i].target.iy,
                               self.edges[i].target.ix])
            sk_length += np.linalg.norm(np.multiply(source-target, np.asarray(self.resolution)))
        return sk_length

    def save_image(self, write_path):
        """Saves plot of skeleton as .png file"""
        # extract ndoes and edges
        nodes = self.get_nodes()
        edges = self.get_edges()
        # plot edges and nodes
        fig = plt.figure(figsize=(16,12))
        ax = Axes3D(fig)
        ax.scatter(nodes[:,2],nodes[:,1],nodes[:,0], s=10, c='r')
        #ax.set_xlim3d(0,self.grid_size[2])
        #ax.set_ylim3d(0,self.grid_size[1])
        #ax.set_zlim3d(0,self.grid_size[0])
        for i in range(edges.shape[0]):
            ln_x = [edges[i][0][2], edges[i][1][2]]
            ln_y = [edges[i][0][1], edges[i][1][1]]
            ln_z = [edges[i][0][0], edges[i][1][0]]
            plt.plot(ln_x, ln_y, ln_z, 'b-')
        plt.savefig(write_path+'%d.png'%(self.label), bbox_inches='tight')
        plt.close()


# class Skeleton:
#     def __init__(self, label, joints, endpoints):
#         self.label = label
#         self.joints = joints
#         self.endpoints = endpoints

#     def NPoints(self):
#         return len(self.joints) + len(self.endpoints)

#     def NEndpoints(self):
#         return len(self.endpoints)

#     def NJoints(self):
#         return len(self.joints)

#     def Endpoints2Array(self):
#         nendpoints = len(self.endpoints)

#         array = np.zeros((nendpoints, 3), dtype=np.int64)
#         for ie in range(nendpoints):
#             array[ie] = self.endpoints[ie]

#         return array

#     def Joints2Array(self):
#         njoints = len(self.endpoints) + len(self.joints)

#         array = np.zeros((njoints, 3), dtype=np.int64)
#         index = 0
#         for endpoint in self.endpoints:
#             array[index] = endpoint
#             index += 1
#         for joint in self.joints:
#             array[index] = joint
#             index += 1

#         return array


#     def WorldJoints2Array(self, resolution):
#         njoints = len(self.endpoints) + len(self.joints)

#         array = np.zeros((njoints, 3), dtype=np.int64)
#         index = 0
#         for endpoint in self.endpoints:
#             array[index] = (endpoint[IB_Z] * resolution[IB_Z], endpoint[IB_Y] * resolution[IB_Y], endpoint[IB_X] * resolution[IB_X])
#             index += 1
#         for joint in self.joints:
#             array[index] = (joint[IB_Z] * resolution[IB_Z], joint[IB_Y] * resolution[IB_Y], joint[IB_X] * resolution[IB_X])
#             index += 1

#         return array


# class Skeletons:
#     def __init__(self, prefix, skeleton_algorithm, downsample_resolution, benchmark, params):
#         self.skeletons = []

#         # read in all of the skeleton points
#         if benchmark: filename = 'benchmarks/skeleton/{}-{}-{:03d}x{:03d}x{:03d}-upsample-{}-skeleton.pts'.format(prefix, skeleton_algorithm, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z], params)
#         else: filename = 'skeletons/{}/{}-{:03d}x{:03d}x{:03d}-upsample-{}-skeleton.pts'.format(prefix, skeleton_algorithm, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z], params)

#         with open(filename, 'rb') as fd:
#             zres, yres, xres, max_label, = struct.unpack('qqqq', fd.read(32))

#             for label in range(max_label):
#                 joints = []
#                 endpoints = []

#                 njoints, = struct.unpack('q', fd.read(8))
#                 for _ in range(njoints):
#                     iv, = struct.unpack('q', fd.read(8))
                    
#                     # endpoints are negative
#                     endpoint = False
#                     if (iv < 0): 
#                         iv = -1 * iv 
#                         endpoint = True

#                     iz = iv / (yres * xres)
#                     iy = (iv - iz * yres * xres) / xres
#                     ix = iv % xres

#                     if endpoint: endpoints.append((iz, iy, ix))
#                     else: joints.append((iz, iy, ix))

#                 self.skeletons.append(Skeleton(label, joints, endpoints))

#     def NSkeletons(self):
#         return len(self.skeletons)


#     def KthSkeleton(self, k):
#         return self.skeletons[k]
