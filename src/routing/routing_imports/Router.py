import logging
LOG = logging.getLogger(__name__)

try:
    from . import PriorityQueue_python3 as PQ
except:
    try:
        import src.routing.routing_imports.PriorityQueue_python3 as PQ
    except:
        raise ImportError("couldnt import PriorityQueue_python3")

RAISE_ERROR = False

def shortest_travel_time_cost_function(travel_time, travel_distance, current_node_index):
    """ standard version of a customized_section_cost_function for computing time shortest routes"""
    return travel_time

class Router():
    """class for computing dijkstra-algorithms in various ways
    nw is a graph network -> TODO
    computes one to many
    start_node: index of start_node
    destination_nodes: list of indices of destination nodes
    mode: None: standard dijkstra | "bidirectional": bidirectional dijkstra (one after another for one to many)
    time_radius: breaks after this radius is reached and returns "no route found" ->negative travel time and travel distance
    max_steps: breaks after max_steps dijkstra steps
    max_settled_nodes: breaks after max_settled_nodes nodes are settled
    forward_flag: if False -> backwards dijkstra is performed, start_node is start of dijkstra (returned route ends with start_node)
    dijkstra_number: sets dijkstra number in nodes (this current signal, if a node has been touched by this dijkstra computation|increased by one after each dijkstra if None)
    with_arc_flags: if True -> arc_flag filtering for next arcs is used
    ch_flag: contraction hierarchy is used
    """
    def __init__(self, nw, start_node, destination_nodes = [], mode = None, time_radius = None, max_settled_targets = None, forward_flag = True, ch_flag = False, customized_section_cost_function = None):
        self.nw = nw
        self.start = start_node
        self.back_end = None
        if not forward_flag:
            self.back_end = start_node
        self.destination_nodes = {}
        for d in destination_nodes:
            self.destination_nodes[d] = True
            self.nw.nodes[d].is_target_node = True
        self.number_destinations = len(destination_nodes)
        if max_settled_targets is not None and max_settled_targets < self.number_destinations:
            self.number_destinations = max_settled_targets
        self.time_radius = time_radius
        self.forward_flag = forward_flag
        self.mode = mode
        self.ch_flag = ch_flag

        if self.ch_flag:
            self.start_hc_val = self.nw.nodes[start_node].ch_value
            self.end_hc_val = 999999999999999

        self.customized_section_cost_function = customized_section_cost_function
        if self.customized_section_cost_function == None:
            if self.ch_flag:
                print("WARNING IN ROUTER: Contraction Hierachies disabled! Only time shortest computations feasible!")
                self.ch_flag = False
            self.customized_section_cost_function = shortest_travel_time_cost_function

        self.dijkstra_number = self.nw.current_dijkstra_number + 1
        # if forward_flag:
        #     s = self.nw.nodes[self.start]
        #     if len(s.edges_from.items()) == 1:
        #         for prev in s.edges_from.keys():
        #             prev.settled = self.dijkstra_number
        #             prev.settled_back = self.dijkstra_number
        #     for d in destination_nodes:
        #         d_node = self.nw.nodes[d]
        #         if len(d_node.edges_to.items()) == 1:
        #             for pref in d_node.edges_to.keys():
        #                 pref.settled = self.dijkstra_number
        #                 pref.settled_back = self.dijkstra_number
        # else:
        #     s = self.nw.nodes[self.start]
        #     if len(s.edges_to.items()) == 1:
        #         for prev in s.edges_to.keys():
        #             prev.settled = self.dijkstra_number
        #             prev.settled_back = self.dijkstra_number
        #     for d in destination_nodes:
        #         d_node = self.nw.nodes[d]
        #         if len(d_node.edges_from.items()) == 1:
        #             for pref in d_node.edges_from.keys():
        #                 pref.settled = self.dijkstra_number
        #                 pref.settled_back = self.dijkstra_number

        self.n_settled = 0

    def compute(self, return_route = True):
        """computes routes for start -> destination_nodes
        if return_route == True:
            returns list of (route, (tt, dis))
            route: list of node indeces ([start, end] if no route found)
            tt: travel time (-1 if no route found)
            dis: distance (-1 if no route found)
        else:
            returns list of ([start, end], (tt, dis))
        computes depending on arguments set in init
        """
        if not self.mode:
            if self.forward_flag:
                self.dijkstraForward()
            elif not self.forward_flag:
                self.dijkstraBackward()
            self.nw.current_dijkstra_number += 1
            
            ret = self.createRoutes(return_route = return_route)
            
        elif self.mode == "bidirectional":
            out = self.computeBidirectional(return_route = return_route)
            self.nw.current_dijkstra_number += 1
            ret = out

        for n in self.destination_nodes.keys():
            self.nw.nodes[n].is_target_node = False

        if ret[0][1][0] < 0:
            print(ret, self.mode)
            exit()
        return ret

    def computeBidirectional(self, return_route = True):
        """computes bidirectional dijkstra from start to end sequentially for each end_node set
        if return_route == False, only tt and dis are computed"""
        sols = []
        for end in self.destination_nodes.keys():
            self.back_end = end
            if not self.ch_flag:
                common_node = self.bidirectionalDijkstra(self.start, end)
                sols.append(self.createBidirectionalRoute(common_node, end, return_route=return_route))
            else:
                common_node = self.bidirectionalContractionHierarchiesDijkstra(self.start, end)
                sols.append(self.createBidirectionalRoute(common_node, end, return_route=return_route))
            self.nw.current_dijkstra_number += 1
            self.dijkstra_number += 1
        return sols

    def createRoutes(self, return_route = True):
        """looks at solutions of standard dijkstras
        returns list of (route, (tt, dis)) for each destination
        tt, dis = -1, -1 if no route is found
        route is [start, end] if no route found or return_route == False
        """
        if self.forward_flag:
            sol = []
            for d in self.destination_nodes:
                d_node = self.nw.nodes[d]
                if not d_node.cost_index == -self.dijkstra_number:
                    sol.append( ([self.start, d], (float("inf"),float("inf"),float("inf")) ) )
                    continue
                if not return_route:
                    sol.append( ([self.start, d], d_node.cost))
                    continue
                route = []
                c_node = d_node
                while c_node.prev is not None:
                    route.append(c_node.node_index)
                    c_node = c_node.prev
                route.append(c_node.node_index)
                sol.append( (list(reversed(route)), d_node.cost))
        else:
            sol = []
            for d in self.destination_nodes:
                d_node = self.nw.nodes[d]
                if not d_node.cost_index_back == -self.dijkstra_number:
                    sol.append( ( [d, self.start], (float("inf"),float("inf"),float("inf")) ) )
                    continue
                if not return_route:
                    #print("sdf", d, self.start, d_node.cost_back)
                    sol.append( ([d, self.start], d_node.cost_back) )
                    continue
                route = []
                c_node = d_node
                while c_node.next is not None:
                    route.append(c_node.node_index)
                    c_node = c_node.next
                route.append(c_node.node_index)
                sol.append( (route, d_node.cost_back))
        return sol

    def createBidirectionalRoute(self, common_node, end, return_route = True):
        """looks at solutions of standard dijkstras
        input is common_node, where forward and backward dijkstra met
        returns  (route, (tt, dis))
        tt, dis = -1, -1 if no route is found
        route is [start, end] if no route found or return_route == False
        looks backward from common_node to start and forward from common_node to end
        """

        if self.forward_flag:
            if not common_node:
                return ([self.start, end], (float("inf"),float("inf"),float("inf"))) 
            forward_path = []
            d_node = common_node
            if not d_node.cost_index == -self.dijkstra_number:
                return ([self.start, end], (float("inf"),float("inf"),float("inf"))) 
            if return_route:    
                c_node = d_node.prev
                if c_node is not None:
                    while c_node.prev is not None:
                        forward_path.append(c_node.node_index)
                        c_node = c_node.prev
                    forward_path.append(c_node.node_index)
                    forward_path = list(reversed(forward_path))

            d_node = common_node
            if not d_node.cost_index_back == -self.dijkstra_number:
                return ([self.start, end], (float("inf"),float("inf"),float("inf"))) 
            if return_route:
                c_node = d_node
                while c_node.next is not None:
                    forward_path.append(c_node.node_index)
                    c_node = c_node.next
                forward_path.append(c_node.node_index)

            cost = ( common_node.cost[0] + common_node.cost_back[0], common_node.cost[1] + common_node.cost_back[1], common_node.cost[2] + common_node.cost_back[2] )
            if not return_route:
                forward_path = [self.start, end]
            return (forward_path, cost)

        else:
            if not common_node:
                return ([end, self.start], (float("inf"),float("inf"),float("inf"))) 
            forward_path = []
            d_node = common_node
            if not d_node.cost_index == -self.dijkstra_number:
                return ([end, self.start], (float("inf"),float("inf"),float("inf"))) 
            if return_route:    
                c_node = d_node.prev
                if c_node is not None:
                    while c_node.prev is not None:
                        forward_path.append(c_node.node_index)
                        c_node = c_node.prev
                    forward_path.append(c_node.node_index)
                    forward_path = list(reversed(forward_path))

            d_node = common_node
            if not d_node.cost_index_back == -self.dijkstra_number:
                return ([end, self.start], (float("inf"),float("inf"),float("inf"))) 
            if return_route:
                c_node = d_node
                while c_node.next is not None:
                    forward_path.append(c_node.node_index)
                    c_node = c_node.next
                forward_path.append(c_node.node_index)

            cost = ( common_node.cost[0] + common_node.cost_back[0], common_node.cost[1] + common_node.cost_back[1] )
            if not return_route:
                forward_path = [end, self.start]
            return (forward_path, cost)


    def dijkstraBackward(self):
        """ backward dijkstra
        start node is self.start and ends if all destination nodes are settled or no more route available
        """

        frontier = PQ.PriorityQueue()

        #self.main_end = list(self.destination_nodes.keys())[0]

        start_node = self.nw.nodes[self.start]
        start_node.settled_back = self.dijkstra_number #settled_back attribute is set to dijkstra number for settled nodes
        start_node.cost_index_back = -self.dijkstra_number  #if touched cost_index_attribute is set
        start_node.cost_back = (0, 0, 0)       #if touched cost_back attribute is set to current_cfv, current tt, current dis
        start_node.next = None          #pointer to next node obj in shortest path

        destinations_reached = 0
        destinations_to_reach = self.number_destinations

        frontier.addTask(start_node, 0)

        while True:
            if not frontier.hasElements():
                break
            
            current_node_obj, current_cost = frontier.popTaskPriority()
            if current_node_obj.is_target_node:
                destinations_reached += 1
                current_node_obj.settled_back = self.dijkstra_number
                if destinations_reached == destinations_to_reach:
                    break

            if self.time_radius is not None and current_cost > self.time_radius:
                break
            
            self.dijkstraStepBackwards(frontier, current_node_obj, current_cost)


    def dijkstraForward(self):
        """ forward dijkstra
        start node is self.start and ends if all destination nodes are settled or no more route available
        """

        frontier = PQ.PriorityQueue()

        destinations_reached = 0
        destinations_to_reach = self.number_destinations

        start_node = self.nw.nodes[self.start] #start_node object
        start_node.settled = self.dijkstra_number   #settled attribute is set to dijkstra number for settled nodes
        start_node.cost_index = -self.dijkstra_number   #cost_index attribute is set for touched nodes
        start_node.cost = (0, 0, 0)    #cost (cfv, tt, dis) is set for touched nodes
        start_node.prev = None      #pointer to previous node object for current fastest route

        frontier.addTask(start_node, 0)

        while True:
            if not frontier.hasElements():
                break
            
            current_node_obj, current_cost = frontier.popTaskPriority()
            if current_node_obj.is_target_node:
                destinations_reached += 1
                current_node_obj.settled = self.dijkstra_number
                if destinations_reached == destinations_to_reach:
                    break
            if self.time_radius is not None and current_cost > self.time_radius:
                break
            self.dijkstraStepForwards(frontier, current_node_obj, current_cost)

    def bidirectionalDijkstra(self, start, end, frontier_in = None):
        """ bidirectional dijkstra from start to end
        frontier_in: not implemented yet, for keeping track of Priority Queue of start for one to many routing
        """
        #sets start and end nodes depending on forward flag an initializes PQs
        if self.forward_flag:
            if not frontier_in:
                frontierForward = PQ.PriorityQueue()

                start_node = self.nw.nodes[self.start]
                start_node.settled = self.dijkstra_number
                start_node.cost_index = -self.dijkstra_number
                start_node.cost = (0, 0, 0)
                start_node.prev = None

                frontierForward.addTask(start_node, 0)
            else:
                frontierForward = frontier_in

            frontierBackward = PQ.PriorityQueue()

            end_node = self.nw.nodes[end]
            end_node.settled_back = self.dijkstra_number
            end_node.cost_index_back = -self.dijkstra_number
            end_node.cost_back = (0, 0, 0)
            end_node.next = None

            frontierBackward.addTask(end_node, 0)

        else:
            if not frontier_in:
                frontierBackward = PQ.PriorityQueue()

                start_node = self.nw.nodes[self.start]
                start_node.settled_back = self.dijkstra_number
                start_node.cost_index_back = -self.dijkstra_number
                start_node.cost_back = (0, 0, 0)
                start_node.next = None

                frontierBackward.addTask(start_node, 0)
            else:
                frontierBackward = frontier_in
                
            frontierForward = PQ.PriorityQueue()

            end_node = self.nw.nodes[end]
            end_node.settled = self.dijkstra_number
            end_node.cost_index = -self.dijkstra_number
            end_node.cost = (0, 0, 0)
            end_node.prev = None

            frontierForward.addTask(end_node, 0)

        current_forward_node_obj = None
        current_forward_cost = -1
        current_backward_node_obj = None
        current_backward_cost = -1

        common_node = None

        # alternating fowarward an backward - step depending on smaller current cost of differen PQs
        # breaks if common node is found
        # if PQ is empty cost is set very high
        while True:
            if current_forward_cost < 0:
                if frontierForward.hasElements():
                    current_forward_node_obj, current_forward_cost = frontierForward.popTaskPriority()
                else:
                    current_forward_cost = float('inf')
            if current_backward_cost < 0:
                if frontierBackward.hasElements():
                    current_backward_node_obj, current_backward_cost = frontierBackward.popTaskPriority()
                else:
                    current_backward_cost = float('inf')
            if not current_backward_node_obj and not current_forward_node_obj:
                if RAISE_ERROR:
                    prt_str = f"start {start} -> end {end} | no ch"
                    raise AssertionError(prt_str)
                else:
                    LOG.warning("routebase no route found!")
                    LOG.warning("start {} -> end {} | no ch".format(start, end))
                    return None

            if current_forward_cost < current_backward_cost:
                self.dijkstraStepForwards(frontierForward, current_forward_node_obj, current_forward_cost)
                if current_forward_node_obj.settled_back == self.dijkstra_number:
                    if not current_forward_node_obj.must_stop() or current_forward_node_obj.node_index == self.start or current_forward_node_obj.node_index == end:
                        common_node = current_forward_node_obj
                        break
                current_forward_cost = -1
                current_forward_node_obj = None
            else:
                self.dijkstraStepBackwards(frontierBackward, current_backward_node_obj, current_backward_cost)
                if current_backward_node_obj.settled == self.dijkstra_number:
                    if not current_backward_node_obj.must_stop() or current_backward_node_obj.node_index == end or current_backward_node_obj.node_index == self.start:
                        common_node = current_backward_node_obj
                        break
                current_backward_cost = -1
                current_backward_node_obj = None

        # fastest route is not necessarily through common node
        # also nodes, that have been touched by both dijkstras need to be checked
        if common_node is None or common_node.cost is None or common_node.cost_back is None:
            return None
        poss_common_nodes = [(common_node , common_node.cost[0] + common_node.cost_back[0])]
        while frontierForward.hasElements():
            x = frontierForward.popTaskPriority()
            if x[0].cost_index_back == -self.dijkstra_number:
                if not x[0].must_stop() or x[0].node_index == end or x[0].node_index == self.start:
                    poss_common_nodes.append((x[0], x[1] + x[0].cost_back[0]))
        while frontierBackward.hasElements():
            x = frontierBackward.popTaskPriority()
            if x[0].cost_index == -self.dijkstra_number:
                if not x[0].must_stop() or x[0].node_index == end or x[0].node_index == self.start:
                    poss_common_nodes.append((x[0], x[1] + x[0].cost[0]))
        common_node = min(poss_common_nodes, key = lambda x:x[1])[0]

        return common_node

    def bidirectionalContractionHierarchiesDijkstra(self, start, end, frontier_in = None):
        """ bidirectional dijkstra from start to end
        frontier_in: not implemented yet, for keeping track of Priority Queue of start for one to many routing
        """
        #sets start and end nodes depending on forward flag an initializes PQs
        if self.forward_flag:
            if not frontier_in:
                frontierForward = PQ.PriorityQueue()

                start_node = self.nw.nodes[self.start]
                start_node.settled = self.dijkstra_number
                start_node.cost_index = -self.dijkstra_number
                start_node.cost = (0, 0, 0)
                start_node.prev = None

                frontierForward.addTask(start_node, 0)
            else:
                frontierForward = frontier_in

            frontierBackward = PQ.PriorityQueue()

            end_node = self.nw.nodes[end]
            end_node.settled_back = self.dijkstra_number
            end_node.cost_index_back = -self.dijkstra_number
            end_node.cost_back = (0, 0, 0)
            end_node.next = None

            frontierBackward.addTask(end_node, 0)

        else:
            if not frontier_in:
                frontierBackward = PQ.PriorityQueue()

                start_node = self.nw.nodes[self.start]
                start_node.settled_back = self.dijkstra_number
                start_node.cost_index_back = -self.dijkstra_number
                start_node.cost_back = (0, 0, 0)
                start_node.next = None

                frontierBackward.addTask(start_node, 0)
            else:
                frontierBackward = frontier_in
                
            frontierForward = PQ.PriorityQueue()

            end_node = self.nw.nodes[end]
            end_node.settled = self.dijkstra_number
            end_node.cost_index = -self.dijkstra_number
            end_node.cost = (0, 0, 0)
            end_node.prev = None

            frontierForward.addTask(end_node, 0)

        current_forward_node_obj = None
        current_forward_cost = -1
        current_backward_node_obj = None
        current_backward_cost = -1

        common_node = None

        #different break for bidirectinal dijkstra with contraction hierarchies
        #a break is set if both dijkstras reached the cost of the current best solution found

        current_solution = (None, float('inf'))
        forward_break = False

        while True:
            if current_forward_cost < 0:
                if frontierForward.hasElements() and not forward_break:
                    current_forward_node_obj, current_forward_cost = frontierForward.popTaskPriority()
                else:
                    current_forward_cost = float('inf')
                if current_forward_cost > current_solution[1]:
                    current_forward_cost = float('inf')
                    #frontierForward = PQ.PriorityQueue()
                    forward_break = True
                    current_forward_node_obj = None
            if current_backward_cost < 0:
                if frontierBackward.hasElements():
                    current_backward_node_obj, current_backward_cost = frontierBackward.popTaskPriority()
                else:
                    current_backward_cost = float('inf')
                if current_backward_cost > current_solution[1]:
                    current_backward_cost = float('inf')
                    frontierBackward = PQ.PriorityQueue()
                    current_backward_node_obj = None
            if not current_backward_node_obj and not current_forward_node_obj:
                if current_solution[0] is None:
                    LOG.warning("routebase2 no route found!")
                    LOG.warning("start {} -> end {} | with ch".format(start, end))
                    #traceback.print_stack()
                    return None
                else:
                    #print("bidijk: ", current_solution[0].cost[0] + current_solution[0].cost_back[0], current_solution[0].node_index, current_solution[1])
                    return current_solution[0]

            if current_forward_cost < current_backward_cost:
                self.dijkstraStepForwards(frontierForward, current_forward_node_obj, current_forward_cost)
                if current_forward_node_obj.settled_back == self.dijkstra_number:
                    if not current_forward_node_obj.isFs() or current_forward_node_obj.node_index == self.start or current_forward_node_obj.node_index == end:
                        c = current_forward_node_obj.cost[0] + current_forward_node_obj.cost_back[0]
                        if c < current_solution[1]:
                            current_solution = (current_forward_node_obj, c)
                current_forward_cost = -1
                current_forward_node_obj = None
            else:
                self.dijkstraStepBackwards(frontierBackward, current_backward_node_obj, current_backward_cost)
                if current_backward_node_obj.settled == self.dijkstra_number:
                    if not current_backward_node_obj.isFs() or current_backward_node_obj.node_index == end or current_backward_node_obj.node_index == self.start:
                        c = current_backward_node_obj.cost[0] + current_backward_node_obj.cost_back[0]
                        if c < current_solution[1]:
                            current_solution = (current_backward_node_obj, c)
                current_backward_cost = -1
                current_backward_node_obj = None



    def dijkstraStepForwards(self, frontier, current_node_obj, current_cost):
        """ one dijkstra step forward
        checks all nodes at end of outgoing arcs
        arcs are filtered depending on preprocessing flags (ch/arcs)
        sets node attributes settled/cost_index/cost/prev
        """
        #print("{} {}".format(current_node_obj.id, current_cost))
        current_node_obj.settled = self.dijkstra_number
        self.n_settled += 1

        if current_node_obj.node_index != self.start and current_node_obj.must_stop():
            return

        if self.time_radius is not None and self.time_radius < current_cost:
            return

        next_nodes_and_edges = current_node_obj.get_next_node_edge_pairs(ch_flag = self.ch_flag)

        for next_node_obj, next_edge_obj in next_nodes_and_edges:
            edge_tt, edge_distance = next_edge_obj.get_tt_distance()
            new_end_cost = current_cost + self.customized_section_cost_function(edge_tt, edge_distance, next_node_obj.node_index)

            if next_node_obj.settled != self.dijkstra_number:
                if next_node_obj.cost_index != -self.dijkstra_number:
                    next_node_obj.cost = (new_end_cost, current_node_obj.cost[1] + edge_tt, current_node_obj.cost[2] + edge_distance )
                    next_node_obj.prev = current_node_obj
                    next_node_obj.cost_index = -self.dijkstra_number
                    frontier.addTask(next_node_obj, new_end_cost)
                else:
                    if next_node_obj.cost[0] > new_end_cost:
                        next_node_obj.cost = (new_end_cost, current_node_obj.cost[1] + edge_tt, current_node_obj.cost[2] + edge_distance )
                        next_node_obj.prev = current_node_obj
                        frontier.addTask(next_node_obj, new_end_cost)

    def dijkstraStepBackwards(self, frontier, current_node_obj, current_cost):
        """ one dijkstra step backward
        checks all nodes at start of incoming arcs
        arcs are filtered depending on preprocessing flags (ch/arcs)
        sets node attributes settled_back/cost_index_back/cost_back/next
        """
        current_node_obj.settled_back = self.dijkstra_number
        self.n_settled += 1

        if current_node_obj.node_index != self.back_end and current_node_obj.must_stop():
            return

        next_nodes_and_edges = current_node_obj.get_prev_node_edge_pairs(ch_flag = self.ch_flag)

        for next_node_obj, next_edge_obj in next_nodes_and_edges:
            edge_tt, edge_distance = next_edge_obj.get_tt_distance()
            new_end_cost = current_cost + self.customized_section_cost_function(edge_tt, edge_distance, next_node_obj.node_index)

            if next_node_obj.settled_back != self.dijkstra_number:
                if next_node_obj.cost_index_back != -self.dijkstra_number:
                    next_node_obj.cost_back = (new_end_cost, current_node_obj.cost_back[1] + edge_tt, current_node_obj.cost_back[2] + edge_distance )
                    next_node_obj.next = current_node_obj
                    next_node_obj.cost_index_back = -self.dijkstra_number
                    frontier.addTask(next_node_obj, new_end_cost)
                else:
                    if next_node_obj.cost_back[0] > new_end_cost:
                        next_node_obj.cost_back = (new_end_cost, current_node_obj.cost_back[1] + edge_tt, current_node_obj.cost_back[2] + edge_distance )
                        next_node_obj.next = current_node_obj
                        frontier.addTask(next_node_obj, new_end_cost)
