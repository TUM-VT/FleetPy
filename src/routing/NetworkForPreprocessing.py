from src.routing.NetworkBasic import NetworkBasic, Node, Edge

class NetworkForPreprocessing(NetworkBasic):
    """ this network is only used in network_manipulation.py to evalute connectivity """
    def __init__(self):
        self.nodes = []     #list of all nodes in network (index == node.node_index)
        self.current_dijkstra_number = 1    #used in dijkstra-class

    def add_node(self, index, is_stop_only = False, x=0, y=0):
        node = Node(index, is_stop_only, x, y)
        if index == len(self.nodes):
            self.nodes.append(node)
        else:
            print("index not feasible with current nodes! {} {}".format(index, len(self.nodes)))
            exit()

    def add_edge(self, from_node_index, to_node_index, tt, dis):
        edge = Edge("{};{}".format(from_node_index, to_node_index), dis, tt)
        o_node = self.nodes[from_node_index]
        d_node = self.nodes[to_node_index]
        o_node.add_next_edge_to(d_node, edge)
        d_node.add_prev_edge_from(o_node, edge)