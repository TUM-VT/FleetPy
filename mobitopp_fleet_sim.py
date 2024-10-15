import sys
import os
import traceback

from src.APIs.mobitopp_fleet_server import MobiToppFleetServer

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
from src.misc.globals import *
import src.misc.config as config

# -------------------------------------------------------------------------------------------------------------------- #
# help functions
def read_yaml_inputs(yaml_file):
    """This function transforms the yaml_file input into a dictionary and changes keys if necessary"""
    return_dict = config.ConstantConfig(yaml_file)
    return return_dict

def routing_min_distance_cost_function(travel_time, travel_distance, current_node_index):
    """computes the customized section cost for routing (input for routing functions)

    :param travel_time: travel_time of a section
    :type travel time: float
    :param travel_distance: travel_distance of a section
    :type travel_distance: float
    :param current_node_index: index of current_node_obj in dijkstra computation to be settled
    :type current_node_index: int
    :return: travel_cost_value of section
    :rtype: float
    """
    return travel_distance

# -------------------------------------------------------------------------------------------------------------------- #
# script call
if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise IOError("Fleet simulation requires exactly one input parameter (*yaml configuration file)!")
        yaml_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, sys.argv[1]))
        print(yaml_file)
        if not os.path.isfile(yaml_file) or not yaml_file.endswith("yaml"):
            prt_str = f"Configuration file {yaml_file} not found or with invalid file ending!"
            raise IOError(prt_str)
        # read configuration and bring to standard input format
        scenario_parameter_dict = read_yaml_inputs(yaml_file)
        # initialize fleet simulation
        fs = MobiToppFleetServer(scenario_parameter_dict)
        # run simulation
        fs.run()
        print("Simulation completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        input("Press Enter to close the window...")