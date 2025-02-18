"""
Authors: Roman Engelhardt, Florian Dandl
TUM, 2020

In order to guarantee transferability of models, Network models should inherit from the ParentNetwork class
and thereby follow its conventions.

Additional conventions:
Definition of Position: (start_node_id, end_node_id, relative_pos)
    > (node_id, None, None) in case vehicle is on a node
    > relative_pos in [0.0, 1.0]
    > str-representation: start_node_id;end_node_id;relative_pos (where None is replaced by -1)
Definition of Route: [start_node_id, intermediary_node_id1, intermediary_node_id2, ... , end_node_id]
    > str-representation: node_ids separated by "-"
"""
from abc import abstractmethod, ABCMeta

INPUT_PARAMETERS_NetworkBase = {
    "doc" : "this is the base abstract network class",
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


def customized_section_cost_function(travel_time, travel_distance, current_node_index):
    """computes the customized section cost for routing

    :param travel_time: travel_time of a section
    :type travel time: float
    :param travel_distance: travel_distance of a section
    :type travel_distance: float
    :param current_node_index: index of current_node_obj in dijkstra computation to be settled
    :type current_node_index: int
    :return: travel_cost_value of section
    :rtype: float
    """
    pass 


# static global functions
# -----------------------
def return_route_str(node_index_list):
    """This method converts routes (node_index_list) to string-format

    :param node_index_list: list of node indices
    :type: list
    :return: "-" separated string
    :rtype: str
    """
    return "-".join([str(x) for x in node_index_list])


def return_node_position(node_index):
    """Conversion of node_index to position tuple

    :param node_index: id of respective node
    :type: int
    """
    return (node_index, None, None)


def return_position_str(position_tuple):
    """Conversion of position-tuple to str if entries are "None" they are set to "-1"
    """
    if position_tuple[1] is None:
        return "{};{};{}".format(position_tuple[0], -1, -1)
    else:
        return "{};{};{:.3f}".format(position_tuple[0], position_tuple[1], position_tuple[2])


def return_position_from_str(position_str):
    entries = position_str.split(";")
    first_entry = int(entries[0])
    second_entry = int(entries[1])
    if second_entry < 0:
        second_entry = None
        third_entry = None
    else:
        third_entry = float(entries[2])
    return (first_entry, second_entry, third_entry)


# TODO # consistent use of NwPos class instead of tuples in Framework
class NwPos:
    """Class to define a network position as triple."""
    # TODO # define meaningful methods -> demand, movement, zones, pricing, ...
    def __init__(self, o_node_id, d_node_id=None, rel_pos=None):
        self.o_id = o_node_id
        self.d_id = d_node_id
        self.rel_pos = rel_pos

    def __str__(self):
        if self.d_id[1] is None:
            return "{};{};{}".format(self.o_id, -1, -1)
        else:
            return "{};{};{:.3f}".format(self.o_id, self.d_id, self.rel_pos)


class Route:
    """Class to define a route, which is a sequence of node-indices."""
    # TODO # define meaningful methods -> vehicle movements
    def __init__(self, list_node_ids):
        self.route = list_node_ids


class NetworkBase(metaclass=ABCMeta):
    # static methods (call to global functions, legacy)
    # -------------------------------------------------
    @staticmethod
    def return_route_str(node_index_list):
        return return_route_str(node_index_list)

    @staticmethod
    def return_node_position(node_index):
        return return_node_position(node_index)

    @staticmethod
    def return_position_str(position_tuple):
        return return_position_str(position_tuple)

    @staticmethod
    def return_position_from_str(position_str):
        return return_position_from_str(position_str)

    # common methods
    # --------------
    @abstractmethod
    def return_node_coordinates(self, node_index):
        """ Returns the spatial coordinates of a node.

        :param node_index: id of node
        :return: (x,y) for metric systems
        """
        pass

    @abstractmethod
    def return_position_coordinates(self, position_tuple):
        """Returns the spatial coordinates of a position.

        :param position_tuple: (o_node, d_node, rel_pos) | (o_node, None, None)
        :return: (x,y) for metric systems
        """
        pass

    # TODO: Convert this method to abstract method
    def return_positions_lon_lat(self, position_tuple_list: list) -> list:
        """ Returns the longitude and latitude of list of positions

        :param position_tuple_list: List of (o_node, d_node, rel_pos) | (o_node, None, None)
        :return: List of tuple of the form (longitude, latitude)
        """
        raise NotImplementedError(f"return_positions_lon_lat method is not implemented for {type(self)}")

    def get_zones_external_route_costs(self, current_time, tmp_toll_route, park_origin=False, park_destination=False):
        # TODO #
        if self.zones is not None:
            a, toll_costs, b = \
                self.zones.get_external_route_costs(self, current_time, tmp_toll_route,
                                                    park_origin=False, park_destination=False)
        else:
            a, toll_costs, b = 0, 0, 0
        return a, toll_costs, b

    # abstract methods
    # ----------------
    @abstractmethod
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        """The network will be initialized.

        :param network_name_dir: name of the network_directory to be loaded
        :type network_name_dir: str
        :param type: determining whether the base or a pre-processed network will be used
        :type type: str
        :param scenario_time: applying travel times for a certain scenario at a given time in the scenario
        :type scenario_time: str
        :param network_dynamics_file_name: file-name of the network dynamics file
        :type network_dynamics_file_name: str
        """
        pass

    @abstractmethod
    def load_tt_file(self, scenario_time):
        """Loads new edge travel times from a file

        :param scenario_time: applying travel times for a certain scenario at a given time in the scenario
        :type scenario_time: str
        """
        pass

    @abstractmethod
    def update_network(self, simulation_time, update_state=False):
        """This method can be called during simulations to
        1) check whether a new travel time file should be loaded (deterministic networks)
        2) update travel times (dynamic networks)

        :param simulation_time: time of simulation
        :type simulation_time: float
        :return: new_tt_flag True, if new travel times found; False if not
        :rtype: bool
        """
        pass
    
    def reset_network(self, simulation_time : float):
        """ this method is used in case a module changed the travel times to future states for forecasts
        it resets the network to the travel times a stimulation_time
        :param simulation_time: current simulation time"""
        raise NotImplementedError(f"the method reset_network is not implemented for this network class")

    @abstractmethod
    def get_number_network_nodes(self):
        """This method returns a list of all street network node indices.

        :return: number of network nodes
        :rtype: int
        """
        pass

    @abstractmethod
    def get_must_stop_nodes(self):
        """ returns a list of node-indices with all nodes with a stop_only attribute """
        pass

    @abstractmethod
    def get_section_infos(self, start_node_index, end_node_index):
        """This method

        :param start_node_index: index of start_node of section
        :type start_node_index: int
        :param end_node_index: index of end_node of section
        :type end_node_index: int
        :return: (travel time, distance); if no section between nodes (None, None)
        :rtype: list
        """
        pass

    @abstractmethod
    def return_route_infos(self, route, rel_start_edge_position, start_time=0):
        """
        This method returns the information travel information along a route. The start position is given by a relative
        value on the first edge [0,1], where 0 means that the vehicle is at the first node.
        :param route: list of nodes
        :param rel_start_edge_position: float [0,1] determining the start position
        :param start_time: can be used as an offset in case the route is planned for a future time
        :return: (arrival time, distance to travel)
        """
        pass

    @abstractmethod
    def assign_route_to_network(self, route, start_time, end_time=None, number_vehicles=1):
        """This method can be used for dynamic network models in which the travel times will be derived from the
        number of vehicles/routes assigned to the network.

        :param route: list of nodes
        :type route: list
        :param start_time: start of travel, can be used as an offset in case the route is planned for a future time
        :type start_time: float
        :param end_time: optional parameter; can be used to assign a vehicle to the network for a certain time
        :type end_time: float
        :param number_vehicles: optional parameter; can be used to assign multiple vehicles at once
        :type number_vehicles: int
        """
        # TODO # think about self.routing_engine.assign_route_to_network(rq_obj, sim_time)
        # -> computation of route only if necessary
        pass

    @abstractmethod
    def return_travel_costs_1to1(self, origin_position, destination_position, customized_section_cost_function=None):
        """This method will return the travel costs of the fastest route between two nodes.

        :param origin_position: (origin_node_index, destination_node_index, relative_position) of current_edge
        :type origin_position: list
        :param destination_position: (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type destination_position: list
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: (cost_function_value, travel time, travel_distance) between the two nodes
        :rtype: tuple
        """
        pass

    @abstractmethod
    def return_travel_costs_Xto1(self, list_origin_positions, destination_position, max_routes=None,
                                 max_cost_value=None, customized_section_cost_function=None):
        """This method will return a list of tuples of origin positions and cost values of the X fastest routes between
        a list of possible origin nodes and a certain destination node. Combinations that do not fulfill all constraints
        will not be returned.

        :param list_origin_positions: list of origin_positions
                (origin_node_index, destination_node_index, relative_position) of origin edge
        :type list_origin_positions: list
        :param destination_position: (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type destination_position: list
        :param max_routes: maximal number of fastest route triples that should be returned
        :type max_routes: int/None
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution
                (max time if customized_section_cost_function == None)
        :type max_cost_value: float/None
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of (origin_position, cost_function_value, travel time, travel_distance) tuples
        :rtype: list
        """
        pass

    @abstractmethod
    def return_travel_costs_1toX(self, origin_position, list_destination_positions, max_routes=None,
                                 max_cost_value=None, customized_section_cost_function = None):
        """This method will return a list of tuples of destination node and travel time of the X fastest routes between
        a list of possible destination nodes and a certain origin node. Combinations that do not fulfill all constraints
        will not be returned.

        :param origin_position: (origin_node_index, destination_node_index, relative_position) of origin edge
        :type origin_position: list
        :param list_destination_positions: list of destination positions
                (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type list_destination_positions: list
        :param max_routes: maximal number of fastest route triples that should be returned
        :type max_routes: int/None
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution
                (max time if customized_section_cost_function == None)
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of (destination_position, cost_function_value, travel time, travel_distance) tuples
        :rtype: list
        """
        pass

    @abstractmethod
    def return_best_route_1to1(self, origin_position, destination_position, customized_section_cost_function = None):
        """This method will return the best route [list of node indices] between two nodes, where origin_position[0] and
        destination_position[1] or (destination_position[0] if destination_position[1]==None) are included

        :param origin_position: (origin_node_index, destination_node_index, relative_position) of current_edge
        :type origin_position: list
        :param destination_position: (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type destination_position: list
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of node-indices of the fastest route
        :rtype: list
        """
        pass

    @abstractmethod
    def return_best_route_Xto1(self, list_origin_positions, destination_position, max_cost_value=None,
                               customized_section_cost_function = None):
        """This method will return the best route between a list of possible origin nodes and a certain destination
        node. A best route is defined by [list of node_indices] between two nodes,
        while origin_position[0] and destination_position[1](or destination_position[0]
        if destination_position[1]==None) is included. Combinations that do not fulfill all constraints
        will not be returned.

        :param list_origin_positions: list of origin_positions
                (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :type list_origin_positions: list
        :param destination_position: (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type destination_position: list
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution
                (max time if customized_section_cost_function == None)
        :type max_cost_value: float/None
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of node-indices of the fastest route (empty, if no route is found, that fullfills the constraints)
        :rtype: list
        """
        pass

    @abstractmethod
    def return_best_route_1toX(self, origin_position, list_destination_positions, max_cost_value=None, customized_section_cost_function = None):
        """This method will return the best route between a list of possible destination nodes and a certain origin
        node. A best route is defined by [list of node_indices] between two nodes,
        while origin_position[0] and destination_position[1](or destination_position[0]
        if destination_position[1]==None) is included. Combinations that do not fulfill all constraints
        will not be returned.
        Specific to this framework: max_cost_value = None translates to max_cost_value = DEFAULT_MAX_X_SEARCH

        :param origin_position: (origin_node_index, destination_node_index, relative_position) of origin edge
        :type origin_position: list
        :param list_destination_positions: list of destination positions
                (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type list_destination_positions: list
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution
                (max time if customized_section_cost_function == None)
        :type max_cost_value: float/None
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of node-indices of the fastest route (empty, if no route is found, that fullfills the constraints)
        :rtype: list
        """
        pass

    @abstractmethod
    def return_travel_cost_matrix(self, list_positions, customized_section_cost_function = None):
        """This method will return the cost_function_value between all positions specified in list_positions

        :param list_positions: list of positions to be computed
        :type list_positions: list
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: dictionary: (o_pos,d_pos) -> (cfv, tt, dist)
        :rtype: dict
        """
        pass

    @abstractmethod
    def move_along_route(self, route, last_position, time_step, sim_vid_id=None, new_sim_time=None,
                         record_node_times=False):
        """This method computes the new position of a (vehicle) on a given route (node_index_list) from it's
        last_position (position_tuple). The first entry of route has to be the same as the first entry of last_position!

        :param route: list of node_indices of the current route
        :type route: list
        :param last_position: position_tuple of starting point
        :type last_position: list
        :param time_step: time [s] passed since last observed at last_position
        :type time_step: float
        :param sim_vid_id: id of simulation vehicle; required for simulation environments with external traffic simulator
        :type sim_vid_id: int
        :param new_sim_time: new time to coordinate simulation times
        :type new_sim_time: float
        :param record_node_times: if this flag is set False, the output list_passed_node_times will always return []
        :type record_node_times: bool
        :return: returns a tuple with
                i) new_position_tuple
                ii) driven distance
                iii) arrival_in_time_step [s]: -1 if vehicle did not reach end of route | time since beginning of time
                        step after which the vehicle reached the end of the route
                iv) list_passed_nodes: if during the time step a number of nodes were passed, these are
                v) list_passed_node_times: list of checkpoint times at the respective passed nodes
        """
        pass

    def add_travel_infos_to_database(self, travel_info_dict):
        """ this function can be used to include externally computed (e.g. multiprocessing) route travel times
        into the database if present
        :param travel_info_dict: dictionary with keys (origin_position, target_positions) -> values (cost_function_value, travel_time, travel_distance)
        """
        pass

    def return_network_bounding_box(self):
        """ Calculates the bounding box points for the whole network

        :return: a list of tuples, (longitude, latitude) for the south-west and north-east corners of the bounding box.
         """
        raise NotImplementedError("The bounding box method is not implemented for current routing class")