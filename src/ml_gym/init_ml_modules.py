# -------------------------------------------------------------------------------------------------------------------- #
# functions to add development content to dictionaries in init_modules.py


def add_simulation_environments():
    """This function adds simulation environments in the development stage as options. Additionally, some legacy names remain available for a limited amount
    of time.

    :return: dictionary of additional module options
    """
    add_sim_env_dict = {}  # str -> (module path, class name)
    # Development Simulation Environments
    return add_sim_env_dict


def add_routing_engines():
    """This function adds routing engines in the development stage as options.

    :return: dictionary of additional module options
    """
    add_re_dict = {}  # str -> (module path, class name)
    return add_re_dict


def add_request_models():
    """This function adds request/traveler models in the development stage as options. Additionally, some legacy names remain available for a limited amount
    of time.

    :return: dictionary of additional module options
    """
    add_tm_dict = {}  # str -> (module path, class name)
    return add_tm_dict


def add_fleet_control_modules():
    """This function adds fleet control models in the development stage as options.

    :return: dictionary of additional module options
    """
    add_op_dict = {}  # str -> (module path, class name)
    add_op_dict["RLRepoFleetControl"] = ("src.ml_gym.MLClasses.RLRepoFleetControl", "RLRepoFleetControl")
    return add_op_dict


def add_repositioning_modules():
    """This function adds repositioning models in the development stage as options.

    :return: dictionary of additional module options
    """
    add_repo_dict = {}  # str -> (module path, class name)
    return add_repo_dict 

def add_charging_strategy_modules():
    """This function adds charging strategy models in the development stage as options.

    :return: dictionary of additional module options
    """
    add_cs_dict = {}  # str -> (module path, class name)
    return add_cs_dict

def add_dynamic_pricing_strategy_modules():
    """This function adds dynamic pricing strategy models in the development stage as options.

    :return: dictionary of additional module options
    """
    add_dp_dict = {}  # str -> (module path, class name)
    return add_dp_dict


def add_dynamic_fleetsizing_strategy_modules():
    """This function adds dynamic fleet sizing strategy models in the development stage as options.

    :return: dictionary of additional module options
    """
    add_dfs_dict = {}  # str -> (module path, class name)
    return add_dfs_dict


def add_reservation_strategy_modules():
    """This function adds reservation strategy models in the development stage as options.

    :return: dictionary of additional module options
    """
    add_res_dict = {}  # str -> (module path, class name)
    return add_res_dict


def add_ride_pooling_batch_optimizer_modules():
    """This function adds ride pooling batch optimization models in the development stage as options.

    :return: dictionary of additional module options
    """
    add_rbo_dict = {}  # str -> (module path, class name)
    return add_rbo_dict

def add_broker_modules():
    return {}

def add_forecast_modules():
    return {}
