import importlib

# possibly load additional content from development content
try:
    dev_content = importlib.import_module("dev.misc.init_modules")
    print("Loading modules from development content.")
except ModuleNotFoundError:
    dev_content = None


# -------------------------------------------------------------------------------------------------------------------- #
# function to load class from specific module based on dictionary entries
def load_module(module_dict, module_str, module_type_str):
    tmp = module_dict.get(module_str)
    if tmp is not None:
        module_name, class_name = tmp
        sim_module = importlib.import_module(module_name)
        sim_class = getattr(sim_module, class_name)
        return sim_class
    else:
        raise IOError(f"{module_type_str} {module_str} is invalid!")

# -------------------------------------------------------------------------------------------------------------------- #
# function to get possibilties to load class from specific module
def get_src_simulation_environments():
    # FleetPy simulation environments
    sim_env_dict = {}  # str -> (module path, class name)
    sim_env_dict["BatchOfferSimulation"] = ("src.BatchOfferSimulation", "BatchOfferSimulation")
    sim_env_dict["ImmediateDecisionsSimulation"] = ("src.ImmediateDecisionsSimulation", "ImmediateDecisionsSimulation")
    sim_env_dict["BrokerDecision"] = ("src.BrokerSimulation", "BrokerDecisionSimulation")
    sim_env_dict["UserDecisionSimulation"] = ("src.BrokerSimulation", "UserDecisionSimulation")
    sim_env_dict["PreferredOperatorSimulation"] = ("src.BrokerSimulation", "PreferredOperatorSimulation")
    # add development content
    if dev_content is not None:
        dev_sim_env_dict = dev_content.add_dev_simulation_environments()
        sim_env_dict.update(dev_sim_env_dict)
    return sim_env_dict

def get_src_routing_engines():
    # FleetPy routing engine options
    re_dict = {}  # str -> (module path, class name)
    re_dict["NetworkBasic"] = ("src.routing.NetworkBasic", "NetworkBasic")
    re_dict["NetworkImmediatePreproc"] = ("src.routing.NetworkImmediatePreproc", "NetworkImmediatePreproc")
    re_dict["NetworkBasicWithStore"] = ("src.routing.NetworkBasicWithStore", "NetworkBasicWithStore")
    re_dict["NetworkPartialPreprocessed"] = ("src.routing.NetworkPartialPreprocessed", "NetworkPartialPreprocessed")
    re_dict["NetworkBasicWithStoreCpp"] = ("src.routing.NetworkBasicWithStoreCpp", "NetworkBasicWithStoreCpp")
    re_dict["NetworkBasicCpp"] = ("src.routing.NetworkBasicCpp", "NetworkBasicCpp")
    re_dict["NetworkPartialPreprocessedCpp"] = ("src.routing.NetworkPartialPreprocessedCpp", "NetworkPartialPreprocessedCpp")
    re_dict["NetworkTTMatrix"] = ("src.routing.NetworkTTMatrix", "NetworkTTMatrix")
    # add development content
    if dev_content is not None:
        dev_re_dict = dev_content.add_dev_routing_engines()
        re_dict.update(dev_re_dict)
    return re_dict

def get_src_request_modules():
    # FleetPy request model options
    rm_dict = {}  # str -> (module path, class name)
    rm_dict["BasicRequest"] = ("src.demand.TravelerModels", "BasicRequest")
    rm_dict["IndividualConstraintRequest"] = ("src.demand.TravelerModels", "IndividualConstraintRequest")
    rm_dict["PriceSensitiveIndividualConstraintRequest"] = ("src.demand.TravelerModels", "PriceSensitiveIndividualConstraintRequest")
    rm_dict["MasterRandomChoiceRequest"] = ("src.demand.TravelerModels", "MasterRandomChoiceRequest")
    rm_dict["SlaveRequest"] = ("src.demand.TravelerModels", "SlaveRequest")
    rm_dict["BasicParcelRequest"] = ("src.demand.TravelerModels", "BasicParcelRequest")
    rm_dict["SlaveParcelRequest"] = ("src.demand.TravelerModels", "SlaveParcelRequest")
    rm_dict["WaitingTimeSensitiveLinearDeclineRequest"] = ("src.demand.TravelerModels", "WaitingTimeSensitiveLinearDeclineRequest")
    rm_dict["BrokerDecisionRequest"] = ("src.demand.TravelerModels", "BrokerDecisionRequest")
    rm_dict["UserDecisionRequest"] = ("src.demand.TravelerModels", "UserDecisionRequest")
    rm_dict["PreferredOperatorRequest"] = ("src.demand.TravelerModels", "PreferredOperatorRequest")
    # add development content
    if dev_content is not None:
        dev_rm_dict = dev_content.add_request_models()
        rm_dict.update(dev_rm_dict)
    return rm_dict
    
def get_src_fleet_control_modules():
    # FleetPy fleet control options
    op_dict = {}  # str -> (module path, class name)
    op_dict["PoolingIRSOnly"] = ("src.fleetctrl.PoolingIRSOnly", "PoolingInsertionHeuristicOnly")
    op_dict["PoolingIRSAssignmentBatchOptimization"] = ("src.fleetctrl.PoolingIRSBatchOptimization", "PoolingIRSAssignmentBatchOptimization")
    op_dict["RidePoolingBatchAssignmentFleetcontrol"] = ("src.fleetctrl.RidePoolingBatchAssignmentFleetcontrol", "RidePoolingBatchAssignmentFleetcontrol")
    op_dict["BrokerExChangeCtrl"] = ("src.fleetctrl.BrokerAndExchangeFleetControl", "BrokerExChangeCtrl")
    op_dict["BrokerBaseCtrl"] = ("src.fleetctrl.BrokerAndExchangeFleetControl", "BrokerBaseCtrl")
    op_dict["BrokerDecisionCtrl"] = ("src.fleetctrl.BrokerAndExchangeFleetControl", "BrokerDecisionCtrl")
    op_dict["RPPFleetControlFullInsertion"] = ("src.fleetctrl.RPPFleetControl", "RPPFleetControlFullInsertion")
    op_dict["RPPFleetControlSingleStopInsertion"] = ("src.fleetctrl.RPPFleetControl", "RPPFleetControlSingleStopInsertion")
    op_dict["RPPFleetControlSingleStopInsertionGuided"] = ("src.fleetctrl.RPPFleetControl", "RPPFleetControlSingleStopInsertionGuided")
    # add development content
    if dev_content is not None:
        dev_op_dict = dev_content.add_fleet_control_modules()
        op_dict.update(dev_op_dict)
    return op_dict

def get_src_repositioning_strategies():
    # FleetPy repositioning options
    repo_dict = {}  # str -> (module path, class name)
    repo_dict["PavoneFC"] = ("src.fleetctrl.repositioning.PavoneHailingFC", "PavoneHailingRepositioningFC")
    repo_dict["PavoneFCV2"] = ("src.fleetctrl.repositioning.PavoneHailingFC", "PavoneHailingV2RepositioningFC")
    repo_dict["DensityFrontiers"] = ("src.fleetctrl.repositioning.FrontiersDensityBasedRepositioning", "DensityRepositioning")
    repo_dict["AlonsoMoraRepositioning"] = ("src.fleetctrl.repositioning.AlonsoMoraRepositioning", "AlonsoMoraRepositioning")
    # add development content
    if dev_content is not None:
        dev_repo_dict = dev_content.add_repositioning_modules()
        repo_dict.update(dev_repo_dict)
    return repo_dict

def get_src_charging_strategies():
    # FleetPy charging options
    # TODO # adapt charging strategy names
    cs_dict = {}  # str -> (module path, class name)
    cs_dict["Threshold_PCI"] = ("src.fleetctrl.charging.Threshold", "ChargingThresholdPublicInfrastructure")
    # add development content
    if dev_content is not None:
        dev_cs_dict = dev_content.add_charging_strategy_modules()
        cs_dict.update(dev_cs_dict)
    return cs_dict

def get_src_dynamic_pricing_strategies():
    # FleetPy dynamic pricing options
    dp_dict = {}  # str -> (module path, class name)
    dp_dict["TimeBasedDP"] = ("src.fleetctrl.pricing.TimeBasedDP", "TimeBasedDP")
    dp_dict["UtilizationBasedDP"] = ("src.fleetctrl.pricing.UtilizationBasedDP", "UtilizationBasedDP")
    # add development content
    if dev_content is not None:
        dev_dp_dict = dev_content.add_dynamic_pricing_strategy_modules()
        dp_dict.update(dev_dp_dict)
    return dp_dict

def get_src_dynamic_fleet_sizing_strategies():
    # FleetPy dynamic fleet sizing options
    dfs_dict = {}  # str -> (module path, class name)
    dfs_dict["TimeBasedFS"] = ("src.fleetctrl.fleetsizing.TimeBasedFS", "TimeBasedFS")
    dfs_dict["UtilizationBasedFS"] = ("src.fleetctrl.fleetsizing.UtilizationBasedFS", "UtilizationBasedFS")
    # add development content
    if dev_content is not None:
        dev_dfs_dict = dev_content.add_dynamic_fleetsizing_strategy_modules()
        dfs_dict.update(dev_dfs_dict)
    return dfs_dict

def get_src_reservation_strategies():
    # FleetPy reservation control strategy options
    res_dict = {}  # str -> (module path, class name)
    res_dict["RollingHorizon"] = ("src.fleetctrl.reservation.RollingHorizon", "RollingHorizonReservation")
    # add development content
    if dev_content is not None:
        dev_res_dict = dev_content.add_reservation_strategy_modules()
        res_dict.update(dev_res_dict)
    return res_dict

def get_src_ride_pooling_batch_optimizers():
    # FleetPy ride pooling optimization strategy options
    rbo_dict = {}  # str -> (module path, class name)
    rbo_dict["AlonsoMora"] = ("src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment", "AlonsoMoraAssignment")
    rbo_dict["InsertionHeuristic"] = ("src.fleetctrl.pooling.batch.InsertionHeuristic.BatchInsertionHeuristicAssignment", "BatchInsertionHeuristicAssignment")
    # add development content
    if dev_content is not None:
        dev_rbo_dict = dev_content.add_ride_pooling_batch_optimizer_modules()
        rbo_dict.update(dev_rbo_dict)
    return rbo_dict

# -------------------------------------------------------------------------------------------------------------------- #
# functions for the different modules
def load_simulation_environment(scenario_parameters):
    """This function returns the simulation environment module.

    :param scenario_parameters: scenario parameters with 'sim_env' attribute
    :return: FleetSimulation instance
    """
    sim_env_str = scenario_parameters.get("sim_env")
    # FleetPy simulation environments
    sim_env_dict = get_src_simulation_environments()
    # load simulation environment instance
    sim_env_class = load_module(sim_env_dict, sim_env_str, "Simulation environment")
    return sim_env_class(scenario_parameters)


def load_routing_engine(network_type, network_dir, network_dynamics_file_name=None):
    """ This function loads the specific network defined in the config file
    routing_engine.add_init() is not called here! (TODO!?)
    :param network_type: str network_type defined by G_NETWORK_TYPE in config
    :param network_dir: path to corresponding network folder
    :param network_dynamics_file_name: name of network dynamic file to load
    :return: routing engine obj
    """
    # FleetPy routing engine options
    re_dict = get_src_routing_engines()
    # load routing engine instance
    re_class = load_module(re_dict, network_type, "Network module")
    return re_class(network_dir, network_dynamics_file_name=network_dynamics_file_name)


def load_request_module(rq_type_string):
    """This function initiates the required fleet control module and returns the Request class, which can be used
    to generate a fleet control instance.

    :param rq_type_string: string that determines which request type should be used
    :return: Request class
    """
    # FleetPy request model options
    rm_dict = get_src_request_modules()
    # get request class
    return load_module(rm_dict, rq_type_string, "Request module")


def load_fleet_control_module(op_fleet_control_class_string):
    """This function initiates the required fleet control module and returns the FleetControl class, which can be used
    to generate a fleet control instance.

    :param op_fleet_control_class_string: string that determines which fleet control should be used
    :return: FleetControl class
    """
    # FleetPy fleet control options
    op_dict = get_src_fleet_control_modules()
    # get fleet control class
    return load_module(op_dict, op_fleet_control_class_string, "Fleet control module")


def load_repositioning_strategy(op_repo_class_string):
    """This function chooses the repositioning module that should be loaded.

    :param op_repo_class_string: string that determines which repositioning strategy will be used
    :return: Repositioning class
    """
    # FleetPy repositioning options
    repo_dict = get_src_repositioning_strategies()
    # get repositioning class
    return load_module(repo_dict, op_repo_class_string, "Repositioning module")


def load_charging_strategy(op_charging_class_string):
    """This function chooses the charging strategy module that should be loaded.

    :param op_charging_class_string: string that determines which charging strategy will be used
    :return: Charging class
    """
    # FleetPy charging options
    cs_dict = get_src_charging_strategies()
    # get charging strategy class
    return load_module(cs_dict, op_charging_class_string, "Charging strategy module")


def load_dynamic_pricing_strategy(op_pricing_class_string):
    """This function chooses the dynamic pricing strategy module that should be loaded.

    :param op_pricing_class_string:  string that determines which strategy will be used
    :return: DynamicPricing class
    """
    # FleetPy dynamic pricing options
    dp_dict = get_src_dynamic_pricing_strategies()
    # get pricing strategy class
    return load_module(dp_dict, op_pricing_class_string, "Dynamic pricing module")


def load_dynamic_fleet_sizing_strategy(op_fleetsizing_class_string):
    """This function chooses the dynamic fleetsizing strategy module that should be loaded.

    :param op_fleetsizing_class_string: string that determines which strategy will be used
    :return: DynamicFleetSizing class
    """
    # FleetPy dynamic fleet sizing options
    dfs_dict = get_src_dynamic_fleet_sizing_strategies()
    # get fleet sizing strategy class
    return load_module(dfs_dict, op_fleetsizing_class_string, "Dynamic fleet sizing module")


def load_reservation_strategy(op_reservation_class_string):
    """This function chooses the strategy to treat reservation requests that should be loaded
    :param op_reservation_class_string: string that determines the reservation strategy
    :return: Reservation class
    """
    # FleetPy reservation control strategy options
    res_dict = get_src_reservation_strategies()
    # get reservation strategy class
    return load_module(res_dict, op_reservation_class_string, "Reservation handling module")


def load_ride_pooling_batch_optimizer(op_batch_optimizer_string):
    """ this function loads the optimizer for solving the ride-pooling assignment problem
    :param op_batch_optimizer_string: string determining the optimizer
    :return: RidePoolingBatchOptimizationClass
    """
    # FleetPy ride pooling optimization strategy options
    rbo_dict = get_src_ride_pooling_batch_optimizers()
    # get ridepooling batch optimizer class
    return load_module(rbo_dict, op_batch_optimizer_string, "Ridepooling batch optimizer module")
