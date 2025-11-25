from libcpp.string cimport string

cdef extern from "Edge.h":
    pass
cdef extern from "Node.h":
    pass
cdef extern from "Edge.cpp":
    pass
cdef extern from "Node.cpp":
    pass
cdef extern from "Network.cpp":
    pass

cdef extern from "Network.h":

    cdef cppclass Network:
        Network(string, string) except +
        void updateEdgeTravelTimes(string) except +
        int computeTravelCosts1ToXpy(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis, double time_range, int max_targets) except +
        int computeTravelCostsXTo1py(int start_node_index, int number_targets, int* targets, int* reached_targets, double* reached_target_tts, double* reached_target_dis,double* reached_target_std,double* reached_target_cfv, double time_range, int max_targets,string mode) except +
        void computeTravelCosts1To1py(int start_node_index, int end_node_index, string mode, double* tt, double* dis, double* std, double* cfv) except +
        int computeRouteSize1to1(int start_node_index, int end_node_index,string mode) except +
        void writeRoute(int* output_array) except +
