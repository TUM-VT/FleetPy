# distutils: language = c++
from libcpp.string cimport string
cimport numpy as np
import numpy as np
np.import_array()
from Network cimport Network


cdef class PyNetwork:
    cdef Network*c_net  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, string node_path, string edge_path):
        """
        loads the network in the c++ class
        :param node_path: path to network/base/nodes.csv
        :type node_path: byte string! use str.encode() to converte usual python strings
        :param node_path: path to network/base/edges.csv
        :type node_path: byte string! use str.encode() to converte usual python strings
        """
        self.c_net = new Network(node_path, edge_path)

    def __dealloc__(self):
        del self.c_net

    def updateEdgeTravelTimes(self, string file_path):
        """
        updates edge travel times in the c++ class
        :param network_path: absolute path to the corresponding file "edges_td_att.csv"!
        :type network_path: byte string! use str.encode() to converte usual python strings
        """
        self.c_net.updateEdgeTravelTimes(file_path)

    def computeTravelCostsXto1(self, start_node_index, list_target_node_indices, max_time_range = None, max_targets = None):
        """
        :param start_node_index: int start node
        :param list_target_node_indices: list int targets (X)
        :param max_time_range: float; targets outside of thes range will not be reached
        :param max_targets: int; only first max_targets will be reached
        :return: list of (target_node_index, tt, dis)
        """
        #defining ctypes
        cdef int N_targets = len(list_target_node_indices)
        cdef double mr = -1.0
        if max_time_range is not None:
            mr = max_time_range
        cdef int mt = -1
        if max_targets is not None:
            mt = max_targets
        #defining arrays to pass as reference
        cdef np.ndarray[int, ndim=1, mode='c'] targets
        targets = np.zeros((N_targets,), dtype=np.int32)
        for i, x in enumerate(list_target_node_indices):
            targets[i] = x
        #tts and dis will be overwritten with c++ function
        cdef np.ndarray[double, ndim=1, mode='c'] tts
        tts = np.zeros((N_targets,), dtype=np.float)
        cdef np.ndarray[double, ndim=1, mode='c'] dis
        dis = np.zeros((N_targets,), dtype=np.float)
        #calling c++: results will be stored in tts/dis; returns number of reached targets
        cdef int reached_targets = self.c_net.computeTravelCostsXTo1py(start_node_index, N_targets, &targets[0], &targets[0], &tts[0], &dis[0], mr, mt)
        return [(targets[i], tts[i], dis[i]) for i in range(reached_targets)]

    def computeTravelCosts1toX(self, start_node_index, list_target_node_indices, max_time_range = None, max_targets = None):
        """
        :param start_node_index: int start node
        :param list_target_node_indices: list int targets (X)
        :param max_time_range: float; targets outside of thes range will not be reached
        :param max_targets: int; only first max_targets will be reached
        :return: list of (target_node_index, tt, dis)
        """
        #defining ctypes
        cdef int N_targets = len(list_target_node_indices)
        cdef double mr = -1.0
        if max_time_range is not None:
            mr = max_time_range
        cdef int mt = -1
        if max_targets is not None:
            mt = max_targets
        #defining arrays to pass as reference
        cdef np.ndarray[int, ndim=1, mode='c'] targets
        targets = np.zeros((N_targets,), dtype=np.int32)
        for i, x in enumerate(list_target_node_indices):
            targets[i] = x
        #tts and dis will be overwritten with c++ function
        cdef np.ndarray[double, ndim=1, mode='c'] tts
        tts = np.zeros((N_targets,), dtype=np.float)
        cdef np.ndarray[double, ndim=1, mode='c'] dis
        dis = np.zeros((N_targets,), dtype=np.float)
        #calling c++: results will be stored in tts/dis; returns number of reached targets
        cdef int reached_targets = self.c_net.computeTravelCosts1ToXpy(start_node_index, N_targets, &targets[0], &targets[0], &tts[0], &dis[0], mr, mt)
        return [(targets[i], tts[i], dis[i]) for i in range(reached_targets)]

    def computeTravelCosts1To1(self, start_node_index, end_node_index):
        """
        :param start_node_index: int start_node
        :param end_node_index: int end_node
        :return: tuple (tt, dis) / (-1.0, -1.0) if no route found
        """
        cdef double dis
        cdef double tt
        self.c_net.computeTravelCosts1To1py(start_node_index, end_node_index, &tt, &dis)
        return (tt, dis)

    def computeRoute1To1(self, start_node_index, end_node_index):
        cdef int route_length = self.c_net.computeRouteSize1to1(start_node_index, end_node_index)
        cdef np.ndarray[int, ndim=1, mode='c'] route
        if route_length >= 0:
            route = np.zeros((route_length,), dtype=np.int32)
            self.c_net.writeRoute(&route[0])
            return list(route)
        else:
            return []
