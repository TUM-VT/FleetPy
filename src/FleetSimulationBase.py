# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import importlib
import logging
import random
import time
import datetime
# import traceback
from abc import abstractmethod
from tqdm import tqdm
import typing as tp
from pathlib import Path
from multiprocessing import Manager

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np
# from IPython import embed

# src imports
# -----------
from src.misc.init_modules import load_fleet_control_module, load_routing_engine
from src.demand.demand import Demand, SlaveDemand
from src.simulation.Vehicles import SimulationVehicle
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.routing.NetworkBase import NetworkBase
    from src.python_plots.plot_classes import PyPlot

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
# set log level to logging.DEBUG or logging.INFO for single simulations
from src.misc.globals import *
DEFAULT_LOG_LEVEL = logging.INFO
LOG = logging.getLogger(__name__)
BUFFER_SIZE = 10
PROGRESS_LOOP = "demand"
PROGRESS_LOOP_VEHICLE_STATUS = [VRL_STATES.IDLE,VRL_STATES.CHARGING,VRL_STATES.REPOSITION]
# check for computation on LRZ cluster
if os.environ.get('SLURM_PROCID'):
    PROGRESS_LOOP = "off"
    
INPUT_PARAMETERS_FleetSimulationBase = {
    "doc" : "this is the base simulation class used for all simulations within FleetPy",
    "inherit" : None,
    "input_parameters_mandatory": [
        G_SCENARIO_NAME, G_SIM_START_TIME, G_SIM_END_TIME, G_NR_OPERATORS, G_RANDOM_SEED, G_NETWORK_NAME,
        G_DEMAND_NAME, G_RQ_FILE, G_AR_MAX_DEC_T
    ],
    "input_parameters_optional": [
        G_SIM_TIME_STEP, G_NR_CH_OPERATORS, G_SIM_REALTIME_PLOT_FLAG, "log_level", G_SIM_ROUTE_OUT_FLAG, G_SIM_REPLAY_FLAG, G_INIT_STATE_SCENARIO,
        G_FC_TYPE, G_ZONE_SYSTEM_NAME, G_FC_TR, G_FC_FNAME, G_INFRA_NAME
    ],
    "mandatory_modules": [
        G_SIM_ENV, G_NETWORK_TYPE, G_RQ_TYP1, G_OP_MODULE
    ], 
    "optional_modules": []
}


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------
def create_or_empty_dir(dirname):
    if os.path.isdir(dirname):
        "Removes all files from top"
        if(dirname == '/' or dirname == "\\"): return
        else:
            for root, dirs, files in os.walk(dirname, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception as err:
                        print(err)
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception as err:
                        print(err)
    else:
        os.makedirs(dirname)


def build_operator_attribute_dicts(parameters, n_op, prefix="op_"):
    """
    Extracts elements of parameters dict whose keys begin with prefix and generates a list of dicts.
    The values of the relevant elements of parameters must be either single values or a list of length n_op, or else
    an exception will be raised.

    :param parameters: dict (or dict-like config object) containing a superset of operator parameters
    :type parameters: dict
    :param n_op: number of operators expected
    :type n_op: int
    :param prefix: prefix by which to filter out operator parameters
    :type prefix: str
    """
    list_op_dicts = [dict() for i in range(n_op)]  # initialize list of empty dicts
    for k in [x for x in parameters if x.startswith(prefix)]:
        # if only a single value is given, use it for all operators
        if type(parameters[k]) in [str, int, float, bool, type(None), dict]:
            for di in list_op_dicts:
                di[k] = parameters[k]
        # if a list of values is given and the length matches the number of operators, use them respectively
        elif len(parameters[k]) == n_op:
            for i, op in enumerate(list_op_dicts):
                op[k] = parameters[k][i]
        elif k == G_OP_REPO_TH_DEF: # TODO # lists as inputs for op
            for di in list_op_dicts:
                di[k] = parameters[k]
        # if parameter has invalid number of values, raise exception
        else:
            raise ValueError("Number of values for parameter", k, "equals neither n_op nor 1.", type(parameters[k]))
    return list_op_dicts

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----

class FleetSimulationBase:
    def __init__(self, scenario_parameters: dict):
        self.t_init_start = time.perf_counter()
        # config
        self.scenario_name = scenario_parameters[G_SCENARIO_NAME]
        print("-"*80 + f"\nSimulation of scenario {self.scenario_name}")
        LOG.info(f"General initialization of scenario {self.scenario_name}...")
        self.dir_names = self.get_directory_dict(scenario_parameters)
        self.scenario_parameters: dict = scenario_parameters

        # check whether simulation already has been conducted -> use final_state.csv to check
        final_state_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "final_state.csv")
        if self.scenario_parameters.get("keep_old", False) and os.path.isfile(final_state_f):
            prt_str = f"Simulation {self.scenario_name} results available and keep_old flag is True!" \
                      f" Not starting the simulation!"
            print(prt_str)
            LOG.info(prt_str)
            self._started = True
            return
        else:
            self._started = False

        # general parameters
        self.start_time = self.scenario_parameters[G_SIM_START_TIME]
        self.end_time = self.scenario_parameters[G_SIM_END_TIME]
        self.time_step = self.scenario_parameters.get(G_SIM_TIME_STEP, 1)
        self.check_sim_env_spec_inputs(self.scenario_parameters)
        self.n_op = self.scenario_parameters[G_NR_OPERATORS]
        self.n_ch_op = self.scenario_parameters.get(G_NR_CH_OPERATORS, 0)
        self._manager: tp.Optional[Manager] = None
        self._shared_dict: dict = {}
        self._plot_class_instance: tp.Optional[PyPlot] = None
        self.realtime_plot_flag = self.scenario_parameters.get(G_SIM_REALTIME_PLOT_FLAG, 0)

        # build list of operator dictionaries  # TODO: this could be eliminated with a new YAML-based config system
        self.list_op_dicts: tp.Dict[str,str] = build_operator_attribute_dicts(scenario_parameters, self.n_op,
                                                                              prefix="op_")
        self.list_ch_op_dicts: tp.Dict[str,str] = build_operator_attribute_dicts(scenario_parameters, self.n_ch_op,
                                                                                 prefix="ch_op_")

        # take care of random seeds at beginning of simulations
        random.seed(self.scenario_parameters[G_RANDOM_SEED])
        np.random.seed(self.scenario_parameters[G_RANDOM_SEED])

        # empty output directory
        create_or_empty_dir(self.dir_names[G_DIR_OUTPUT])

        # write scenario config file in output directory
        self.save_scenario_inputs()

        # remove old log handlers (otherwise sequential simulations only log to first simulation)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # start new log file
        logging.VERBOSE = 5
        logging.addLevelName(logging.VERBOSE, "VERBOSE")
        logging.Logger.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
        logging.LoggerAdapter.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
        logging.verbose = lambda msg, *args, **kwargs: logging.log(logging.VERBOSE, msg, *args, **kwargs)
        if self.scenario_parameters.get("log_level", "info"):
            level_str = self.scenario_parameters["log_level"]
            if level_str == "verbose":
                log_level = logging.VERBOSE
            elif level_str == "debug":
                log_level = logging.DEBUG
            elif level_str == "info":
                log_level = logging.INFO
            elif level_str == "warning":
                log_level = logging.WARNING
            else:
                log_level = DEFAULT_LOG_LEVEL
        else:
            log_level = DEFAULT_LOG_LEVEL
            pd.set_option("mode.chained_assignment", None)
        self.log_file = os.path.join(self.dir_names[G_DIR_OUTPUT], f"00_simulation.log")
        if log_level < logging.INFO:
            streams = [logging.FileHandler(self.log_file), logging.StreamHandler()]
        else:
            print("Only minimum output to console -> see log-file")
            streams = [logging.FileHandler(self.log_file)]
        # TODO # log of subsequent simulations is saved in first simulation log
        logging.basicConfig(handlers=streams,
                            level=log_level, format='%(process)d-%(name)s-%(levelname)s-%(message)s')

        # set up output files
        self.user_stat_f = os.path.join(self.dir_names[G_DIR_OUTPUT], f"1_user-stats.csv")
        self.network_stat_f = os.path.join(self.dir_names[G_DIR_OUTPUT], f"3_network-stats.csv")
        self.pt_stat_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "4_pt_stats.csv")

        # init modules
        # ------------
        # zone system
        # TODO # after ISTTT: enable multiple zone systems
        # TODO # after ISTTT: bring init of modules in extra function (-> parallel processing)
        self.zones = None
        if self.dir_names.get(G_DIR_ZONES, None) is not None:
            if self.scenario_parameters.get(G_FC_TYPE) and self.scenario_parameters[G_FC_TYPE] == "perfect":
                from src.infra.PerfectForecastZoning import PerfectForecastZoneSystem
                self.zones = PerfectForecastZoneSystem(self.dir_names[G_DIR_ZONES], self.scenario_parameters, self.dir_names)
            else:
                from src.infra.Zoning import ZoneSystem
                self.zones = ZoneSystem(self.dir_names[G_DIR_ZONES], self.scenario_parameters, self.dir_names)

        # routing engine
        LOG.info("Initialization of network and routing engine...")
        network_type = self.scenario_parameters[G_NETWORK_TYPE]
        network_dynamics_file = self.scenario_parameters.get(G_NW_DYNAMIC_F, None)
        # TODO # check consistency of scenario inputs / another way to refactor add_init_data ?
        self.routing_engine: NetworkBase = load_routing_engine(network_type, self.dir_names[G_DIR_NETWORK],
                                                               network_dynamics_file_name=network_dynamics_file)
        if network_type == "NetworkDynamicNFDClusters":
            self.routing_engine.add_init_data(self.start_time, self.time_step,
                                              self.scenario_parameters[G_NW_DENSITY_T_BIN_SIZE],
                                              self.scenario_parameters[G_NW_DENSITY_AVG_DURATION], self.zones,
                                              self.network_stat_f)
        # public transportation module
        LOG.info("Initialization of line-based public transportation...")
        pt_type = self.scenario_parameters.get(G_PT_TYPE)
        self.gtfs_data_dir = self.dir_names.get(G_DIR_PT)
        if pt_type is None or self.gtfs_data_dir is None:
            self.pt = None
        elif pt_type == "PTMatrixCrowding":
            pt_module = importlib.import_module("src.pubtrans.PtTTMatrixCrowding")
            self.pt = pt_module.PublicTransportTravelTimeMatrixWithCrowding(self.gtfs_data_dir, self.pt_stat_f,
                                                                            self.scenario_parameters,
                                                                            self.routing_engine, self.zones)
        elif pt_type == "PtCrowding":
            pt_module = importlib.import_module("src.pubtrans.PtCrowding")
            self.pt = pt_module.PublicTransportWithCrowding(self.gtfs_data_dir, self.pt_stat_f, self.scenario_parameters,
                                                            self.routing_engine, self.zones)
        else:
            raise IOError(f"Public transport module {pt_type} not defined for current simulation environment.")

        # attribute for demand, charging and zone module
        self.demand = None
        self._load_demand_module()
        self.charging_operator_dict = {}    # dict "op" -> operator_id -> OperatorChargingInfrastructure, "pub" -> ch_op_id -> ChargingInfrastructureOperator
        self._load_charging_modules()

        # attributes for fleet controller and vehicles
        self.sim_vehicles: tp.Dict[tp.Tuple[int, int], SimulationVehicle] = {}
        self.sorted_sim_vehicle_keys: tp.List[tp.Tuple[int, int]] = sorted(self.sim_vehicles.keys())
        self.vehicle_update_order: tp.Dict[tp.Tuple[int, int], int] = {vid : 1 for vid in self.sim_vehicles.keys()} #value defines the order in whichvehicles are updated (i.e. charging first)
        self.operators: tp.List[FleetControlBase] = []
        self.op_output = {}
        self._load_fleetctr_vehicles()

        # call additional simulation environment specific init
        LOG.info("Simulation environment specific initializations...")
        self.init_blocking = True
        self.add_init(self.scenario_parameters)

        # load initial state depending on init_blocking attribute
        # HINT: it is important that this is done at the end of initialization!
        LOG.info("Creating or loading initial vehicle states...")
        np.random.seed(self.scenario_parameters[G_RANDOM_SEED])
        self.load_initial_state()
        LOG.info(f"Initialization of scenario {self.scenario_name} successful.")

        # self.routing_engine.checkNetwork()

    def _load_demand_module(self):
        """ Loads some demand modules """

        # demand module
        LOG.info("Initialization of travelers...")
        if self.scenario_parameters[G_SIM_ENV] != "MobiTopp":
            self.demand = Demand(self.scenario_parameters, self.user_stat_f, self.routing_engine, self.zones)
            self.demand.load_demand_file(self.scenario_parameters[G_SIM_START_TIME],
                                         self.scenario_parameters[G_SIM_END_TIME], self.dir_names[G_DIR_DEMAND],
                                         self.scenario_parameters[G_RQ_FILE], self.scenario_parameters[G_RANDOM_SEED],
                                         self.scenario_parameters.get(G_RQ_TYP1, None),
                                         self.scenario_parameters.get(G_RQ_TYP2, {}),
                                         self.scenario_parameters.get(G_RQ_TYP3, {}),
                                         simulation_time_step=self.time_step)
            if self.scenario_parameters.get(G_PA_RQ_FILE) is not None:
                self.demand.load_parcel_demand_file(self.scenario_parameters[G_SIM_START_TIME],
                                            self.scenario_parameters[G_SIM_END_TIME], self.dir_names[G_DIR_PARCEL_DEMAND],
                                            self.scenario_parameters[G_PA_RQ_FILE], self.scenario_parameters[G_RANDOM_SEED],
                                            self.scenario_parameters.get(G_PA_RQ_TYP1, None),
                                            self.scenario_parameters.get(G_PA_RQ_TYP2, {}),
                                            self.scenario_parameters.get(G_PA_RQ_TYP3, {}),
                                            simulation_time_step=self.time_step)
        else:
            self.demand = SlaveDemand(self.scenario_parameters, self.user_stat_f)

        if self.zones is not None:
            self.zones.register_demand_ref(self.demand)

    def _load_charging_modules(self):
        """ Loads necessary modules for charging """
        # TODO # multiple charging operators:
        #  either public charging operator or depot operator
        #  add parameter list [with extra parameter list] (e.g. list of allowed fleet operators, infrastructure data file)
        self.charging_operator_dict = {"op" : {}, "pub" : {}}
        LOG.debug("load charging infra: charging op dicts: {}".format(self.list_ch_op_dicts))
        if self.dir_names.get(G_DIR_INFRA):
            # operator depots:
            from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure
            for op_id, op_dict in enumerate(self.list_op_dicts):
                depot_f_name = op_dict.get(G_OP_DEPOT_F)
                if depot_f_name is not None:
                    depot_f = os.path.join(self.dir_names[G_DIR_INFRA], depot_f_name)
                    op_charge = OperatorChargingAndDepotInfrastructure(op_id, depot_f, op_dict, self.scenario_parameters, self.dir_names, self.routing_engine)
                    self.charging_operator_dict["op"][op_id] = op_charge
                
            # public charging
            if len(self.list_ch_op_dicts) > 0:
                from src.infra.ChargingInfrastructure import PublicChargingInfrastructureOperator
                for ch_op_id, ch_op_dict in enumerate(self.list_ch_op_dicts):
                    pub_cs_f_name = ch_op_dict.get(G_CH_OP_F)
                    if pub_cs_f_name is None:
                        raise EnvironmentError("Public charging stations file not given as input! parameter {} required!".format(G_CH_OP_F))
                    pub_cs_f = os.path.join(self.dir_names[G_DIR_INFRA], pub_cs_f_name)
                    initial_ch_events_f = None
                    if ch_op_dict.get(G_CH_OP_INIT_CH_EVENTS_F) is not None:
                        f_name = ch_op_dict.get(G_CH_OP_INIT_CH_EVENTS_F)
                        initial_ch_events_f = os.path.join(self.dir_names[G_DIR_INFRA], "charging_events", f_name)
                    ch_op = PublicChargingInfrastructureOperator(ch_op_id, pub_cs_f, ch_op_dict, self.scenario_parameters, self.dir_names, self.routing_engine, initial_charging_events_f=initial_ch_events_f)
                    self.charging_operator_dict["pub"][ch_op_id] = ch_op

    def _load_fleetctr_vehicles(self):
        """ Loads the fleet controller and vehicles """

        # simulation vehicles and fleet control modules
        LOG.info("Initialization of MoD fleets...")
        route_output_flag = self.scenario_parameters.get(G_SIM_ROUTE_OUT_FLAG, True)
        replay_flag = self.scenario_parameters.get(G_SIM_REPLAY_FLAG, False)
        veh_type_list = []
        for op_id in range(self.n_op):
            operator_attributes = self.list_op_dicts[op_id]
            operator_module_name = operator_attributes[G_OP_MODULE]
            self.op_output[op_id] = []  # shared list among vehicles
            if not operator_module_name == "LinebasedFleetControl":
                fleet_composition_dict = operator_attributes[G_OP_FLEET]
                list_vehicles = []
                vid = 0
                for veh_type, nr_veh in fleet_composition_dict.items():
                    for _ in range(nr_veh):
                        veh_type_list.append([op_id, vid, veh_type])
                        tmp_veh_obj = SimulationVehicle(op_id, vid, self.dir_names[G_DIR_VEH], veh_type,
                                                        self.routing_engine, self.demand.rq_db,
                                                        self.op_output[op_id], route_output_flag,
                                                        replay_flag)
                        list_vehicles.append(tmp_veh_obj)
                        self.sim_vehicles[(op_id, vid)] = tmp_veh_obj
                        vid += 1
                OpClass: FleetControlBase = load_fleet_control_module(operator_module_name)
                self.operators.append(OpClass(op_id, operator_attributes, list_vehicles, self.routing_engine, self.zones,
                                            self.scenario_parameters, self.dir_names, self.charging_operator_dict["op"].get(op_id, None), list(self.charging_operator_dict["pub"].values())))
            else:
                from dev.fleetctrl.LinebasedFleetControl import LinebasedFleetControl
                OpClass = LinebasedFleetControl(op_id, self.gtfs_data_dir, self.routing_engine, self.zones, self.scenario_parameters, self.dir_names, self.charging_operator_dict["op"].get(op_id, None), list(self.charging_operator_dict["pub"].values()))
                init_vids = OpClass.return_vehicles_to_initialize()
                list_vehicles = []
                for vid, veh_type in init_vids.items():
                    tmp_veh_obj = SimulationVehicle(op_id, vid, self.dir_names[G_DIR_VEH], veh_type,
                                                        self.routing_engine, self.demand.rq_db,
                                                        self.op_output[op_id], route_output_flag,
                                                        replay_flag)
                    list_vehicles.append(tmp_veh_obj)
                    veh_type_list.append([op_id, vid, veh_type])
                    self.sim_vehicles[(op_id, vid)] = tmp_veh_obj
                OpClass.continue_init(list_vehicles, self.start_time)
                self.operators.append(OpClass)
        veh_type_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "2_vehicle_types.csv")
        veh_type_df = pd.DataFrame(veh_type_list, columns=[G_V_OP_ID, G_V_VID, G_V_TYPE])
        veh_type_df.to_csv(veh_type_f, index=False)
        self.vehicle_update_order: tp.Dict[tp.Tuple[int, int], int] = {vid : 1 for vid in self.sim_vehicles.keys()}

    @staticmethod
    def get_directory_dict(scenario_parameters):
        """
        This function provides the correct paths to certain data according to the specified data directory structure.
        :param scenario_parameters: simulation input (pandas series)
        :return: dictionary with paths to the respective data directories
        """
        return get_directory_dict(scenario_parameters)

    def save_scenario_inputs(self):
        config_f = os.path.join(self.dir_names[G_DIR_OUTPUT], G_SC_INP_F)
        config = {"scenario_parameters": self.scenario_parameters, "list_operator_attributes": self.list_op_dicts,
                  "directories": self.dir_names}
        with open(config_f, "w") as fh_config:
            json.dump(config, fh_config, indent=4)

    def evaluate(self):
        """Runs standard and simulation environment specific evaluations over simulation results."""
        output_dir = self.dir_names[G_DIR_OUTPUT]
        # standard evaluation
        from src.evaluation.standard import standard_evaluation
        standard_evaluation(output_dir)
        self.add_evaluate()

    # def initialize_operators_and_vehicles(self): TODO I think this is depricated!
    #     """ this function loads and initialzie all operator classes and its vehicle objects
    #     and sets corresponding outputs"""
    #     veh_type_list = []
    #     route_output_flag = self.scenario_parameters.get(G_SIM_ROUTE_OUT_FLAG, True)
    #     replay_flag = self.scenario_parameters.get(G_SIM_REPLAY_FLAG, False)
    #     for op_id in range(self.n_op):
    #         self.op_output[op_id] = []  # shared list among vehicles
    #         operator_attributes = self.list_op_dicts[op_id]
    #         operator_module_name = operator_attributes[G_OP_MODULE]
    #         fleet_composition_dict = operator_attributes[G_OP_FLEET]
    #         list_vehicles = []
    #         vid = 0
    #         for veh_type, nr_veh in fleet_composition_dict.items():
    #             for _ in range(nr_veh):
    #                 veh_type_list.append([op_id, vid, veh_type])
    #                 tmp_veh_obj = SimulationVehicle(op_id, vid, self.dir_names[G_DIR_VEH], veh_type,
    #                                                 self.routing_engine, self.demand.rq_db,
    #                                                 self.op_output[op_id], route_output_flag,
    #                                                 replay_flag)
    #                 list_vehicles.append(tmp_veh_obj)
    #                 self.sim_vehicles[(op_id, vid)] = tmp_veh_obj
    #                 vid += 1
    #         OpClass = load_fleet_control_module(operator_module_name)
    #         self.operators.append(OpClass(op_id, operator_attributes, list_vehicles, self.routing_engine, self.zones,
    #                                     self.scenario_parameters, self.dir_names, self.cdp))
    #     veh_type_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "2_vehicle_types.csv")
    #     veh_type_df = pd.DataFrame(veh_type_list, columns=[G_V_OP_ID, G_V_VID, G_V_TYPE])
    #     veh_type_df.to_csv(veh_type_f, index=False)

    def load_initial_state(self):
        """This method initializes the simulation vehicles. It can consider an initial state file. Moreover, an
        active_vehicle files would be considered as the FleetControl already set the positions of vehicles in the depot
        and therefore the "if veh_obj.pos is None:" condition does not trigger.
        The VehiclePlans of the respective FleetControls are also adapted for blocked vehicles.

        :return: None
        """
        init_f_flag = False
        init_state_f = None
        if self.scenario_parameters.get(G_INIT_STATE_SCENARIO):
            init_state_f = os.path.join(self.dir_names[G_DIR_MAIN], "studies",
                                        self.scenario_parameters[G_STUDY_NAME], "results",
                                        str(self.scenario_parameters.get(G_INIT_STATE_SCENARIO, "None")),
                                        "final_state.csv")
            init_f_flag = True
            if not os.path.isfile(init_state_f):
                raise FileNotFoundError(f"init state variable {G_INIT_STATE_SCENARIO} given but file {init_state_f} not found!")
        set_unassigned_vid = set([(veh_obj.op_id, veh_obj.vid) for veh_obj in self.sim_vehicles.values()
                                  if veh_obj.pos is None])
        if init_f_flag:
            # set according to initial state if available
            init_state_df = pd.read_csv(init_state_f)
            init_state_df.set_index([G_V_OP_ID, G_V_VID], inplace=True)
            for sim_vid, veh_obj in self.sim_vehicles.items():
                if veh_obj.pos is None:
                    op_fleetctrl = self.operators[veh_obj.op_id]
                    init_state_info = init_state_df.loc[sim_vid]
                    if init_state_info is not None:
                        veh_obj.set_initial_state(op_fleetctrl, self.routing_engine, init_state_info,
                                                  self.scenario_parameters[G_SIM_START_TIME], self.init_blocking)
                        set_unassigned_vid.remove(sim_vid)
        if len(set_unassigned_vid) > 0:
            op_init_distributions = {}
            for op_id in range(self.n_op):
                if self.list_op_dicts[op_id].get(G_OP_INIT_VEH_DIST) is not None:   #specified random distribution
                    init_dist_df = pd.read_csv(os.path.join(self.dir_names[G_DIR_FCTRL], "initial_vehicle_distribution", self.scenario_parameters[G_NETWORK_NAME], self.list_op_dicts[op_id][G_OP_INIT_VEH_DIST]), index_col=0)
                    op_init_distributions[op_id] = init_dist_df["probability"].to_dict()
                else:   # randomly uniform
                    boarding_nodes = self.routing_engine.get_must_stop_nodes()
                    if not boarding_nodes:
                        boarding_nodes = list(range(self.routing_engine.get_number_network_nodes()))
                    op_init_distributions[op_id] = {bn : 1.0/len(boarding_nodes) for bn in boarding_nodes}
            #LOG.debug("init distributons: {}".format(op_init_distributions))
            for sim_vid in set_unassigned_vid:
                veh_obj = self.sim_vehicles[sim_vid]
                if veh_obj.pos is None:
                    op_fleetctrl = self.operators[veh_obj.op_id]
                    init_dist = op_init_distributions[veh_obj.op_id]
                    r = np.random.random()
                    s = 0.0
                    init_node = None
                    for n, prob in init_dist.items():
                        s += prob
                        if s >= r:
                            init_node = n
                            break
                    if init_node is None:
                        LOG.error(f"No init node found for random val {r} and init dist {init_dist}")
                    # randomly position all vehicles
                    init_state_info = {}
                    init_state_info[G_V_INIT_NODE] = init_node# np.random.choice(init_node)
                    init_state_info[G_V_INIT_TIME] = self.scenario_parameters[G_SIM_START_TIME]
                    init_state_info[G_V_INIT_SOC] = 0.5 * (1 + np.random.random())
                    veh_obj.set_initial_state(op_fleetctrl, self.routing_engine, init_state_info,
                                                self.scenario_parameters[G_SIM_START_TIME], self.init_blocking)

    def save_final_state(self):
        """
        Records the state at the end of the simulation; can be used as initial state for other simulations.
        """
        LOG.info("Saving final simulation state.")
        final_state_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "final_state.csv")
        sorted_sim_vehicle_keys = sorted(self.sim_vehicles.keys())
        list_vehicle_states = [self.sim_vehicles[sim_vid].return_final_state(self.end_time)
                               for sim_vid in sorted_sim_vehicle_keys]
        fs_df = pd.DataFrame(list_vehicle_states)
        fs_df.to_csv(final_state_f)

    def record_remaining_assignments(self):
        """
        This method simulates the remaining assignments at the end of the simulation in order to get them recorded
        properly. This is necessary for a consistent evaluation.
        """
        c_time = self.end_time# - self.time_step
        LOG.info("record_remaining_assignments()")
        remaining_tasks = -1
        while remaining_tasks != 0:
            self.update_sim_state_fleets(c_time - self.time_step, c_time)
            for ch_op_dict in self.charging_operator_dict.values():
                for ch_op in ch_op_dict.values():
                    ch_op.time_trigger(c_time)
            remaining_tasks = 0
            for veh_obj in self.sim_vehicles.values():
                if veh_obj.assigned_route and veh_obj.assigned_route[0].status == VRL_STATES.OUT_OF_SERVICE:
                    veh_obj.end_current_leg(c_time)
                remaining_tasks += len(veh_obj.assigned_route)
                # if len(veh_obj.assigned_route) > 0:
                    # LOG.debug(f"vid {veh_obj.vid} has remaining assignments:")
                    # LOG.debug("{}".format([str(x) for x in veh_obj.assigned_route]))
            LOG.info(f"\t time {c_time}, remaining tasks {remaining_tasks}")
            c_time += self.time_step
            self.routing_engine.update_network(c_time, update_state=False)
            if c_time > self.end_time + 2*7200:
                # # alternative 1: force end of tasks
                # LOG.warning(f"remaining assignments could not finish! Forcing end of assignments.")
                # for veh_obj in self.sim_vehicles.values():
                #     if veh_obj.assigned_route:
                #         veh_obj.end_current_leg(c_time)
                # alternative 2: just break loop
                LOG.warning(f"remaining assignments could not finish! Break Loop")
                break
        self.record_stats()

    def record_stats(self, force=True):
        """This method records the stats at the end of the simulation."""
        self.demand.save_user_stats(force)
        for op_id in range(self.n_op):
            current_buffer_size = len(self.op_output[op_id]) 
            if (current_buffer_size and force) or current_buffer_size > BUFFER_SIZE:
                op_output_f = os.path.join(self.dir_names[G_DIR_OUTPUT], f"2-{op_id}_op-stats.csv")
                if os.path.isfile(op_output_f):
                    write_mode = "a"
                    write_header = False
                else:
                    write_mode = "w"
                    write_header = True
                tmp_df = pd.DataFrame(self.op_output[op_id])
                tmp_df.to_csv(op_output_f, index=False, mode=write_mode, header=write_header)
                self.op_output[op_id].clear()
                # LOG.info(f"\t ... just wrote {current_buffer_size} entries from buffer to stats of operator {op_id}.")
                LOG.debug(f"\t ... just wrote {current_buffer_size} entries from buffer to stats of operator {op_id}.")
            self.operators[op_id].record_dynamic_fleetcontrol_output(force=force)

    def update_sim_state_fleets(self, last_time, next_time, force_update_plan=False):
        """
        This method updates the simulation vehicles, records, ends and starts tasks and returns some data that
        will be used for additional state updates (fleet control information, demand, network, ...)
        :param last_time: simulation time before the state update
        :param next_time: simulation time of the state update
        :param force_update_plan: flag that can force vehicle plan to be updated
        """
        LOG.debug(f"updating MoD state from {last_time} to {next_time}")
        #for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
        for opid_vid_tuple, veh_obj in sorted(self.sim_vehicles.items(), key=lambda x:self.vehicle_update_order[x[0]]):
            op_id, vid = opid_vid_tuple
            boarding_requests, alighting_requests, passed_VRL, dict_start_alighting =\
                veh_obj.update_veh_state(last_time, next_time)
            if veh_obj.status == VRL_STATES.CHARGING:
                self.vehicle_update_order[opid_vid_tuple] = 0
            else:
                self.vehicle_update_order[opid_vid_tuple] = 1
            for rid, boarding_time_and_pos in boarding_requests.items():
                boarding_time, boarding_pos = boarding_time_and_pos
                LOG.debug(f"rid {rid} boarding at {boarding_time} at pos {boarding_pos}")
                self.demand.record_boarding(rid, vid, op_id, boarding_time, pu_pos=boarding_pos)
                self.operators[op_id].acknowledge_boarding(rid, vid, boarding_time)
            for rid, alighting_start_time_and_pos in dict_start_alighting.items():
                # record user stats at beginning of alighting process
                alighting_start_time, alighting_pos = alighting_start_time_and_pos
                LOG.debug(f"rid {rid} deboarding at {alighting_start_time} at pos {alighting_pos}")
                self.demand.record_alighting_start(rid, vid, op_id, alighting_start_time, do_pos=alighting_pos)
            for rid, alighting_end_time in alighting_requests.items():
                # # record user stats at end of alighting process
                self.demand.user_ends_alighting(rid, vid, op_id, alighting_end_time)
                self.operators[op_id].acknowledge_alighting(rid, vid, alighting_end_time)
            # send update to operator
            if len(boarding_requests) > 0 or len(dict_start_alighting) > 0:
                self.operators[op_id].receive_status_update(vid, next_time, passed_VRL, True)
            else:
                self.operators[op_id].receive_status_update(vid, next_time, passed_VRL, force_update_plan)
        # TODO # after ISTTT: live visualization: send vehicle states (self.live_visualization_flag==True)

    def update_vehicle_routes(self, sim_time):
        """ this method can be used to recalculate routes of currently driving vehicles in case
        network travel times changed and shortest paths need to be re-set
        """
        for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
            veh_obj.update_route()

    def _rid_chooses_offer(self, rid, rq_obj, sim_time):
        """This method performs all actions that derive from a mode choice decision.

        :param rid: request id
        :param rq_obj: request object
        :param sim_time: current simulation time
        :return: chosen operator
        """
        chosen_operator = rq_obj.choose_offer(self.scenario_parameters, sim_time)
        LOG.debug(f" -> chosen operator: {chosen_operator}")
        if chosen_operator is None: # undecided
            if rq_obj.leaves_system(sim_time):
                for i, operator in enumerate(self.operators):
                    operator.user_cancels_request(rid, sim_time)
                self.demand.record_user(rid)
                del self.demand.rq_db[rid]
                try:
                    del self.demand.undecided_rq[rid]
                except KeyError:
                    # raises KeyError if request decided right away
                    pass
            else:
                self.demand.undecided_rq[rid] = rq_obj
        elif chosen_operator < 0:
            # if chosen_operator == G_MC_DEC_PV:
            #     # TODO # self.routing_engine.assign_route_to_network(rq_obj, sim_time)
            #     # TODO # computation of route only when necessary
            #     self.routing_engine.assign_route_to_network(rq_obj, sim_time)
            #     # TODO # check if following method is necessary
            #     self.demand.user_chooses_PV(rid, sim_time)
            # elif chosen_operator == G_MC_DEC_PT:
            #     pt_offer = rq_obj.return_offer(G_MC_DEC_PT)
            #     pt_start_time = sim_time + pt_offer.get(G_OFFER_ACCESS_W, 0) + pt_offer.get(G_OFFER_WAIT, 0)
            #     pt_end_time = pt_start_time + pt_offer.get(G_OFFER_DRIVE, 0)
            #     self.pt.assign_to_pt_network(pt_start_time, pt_end_time)
            #     # TODO # check if following method is necessary
            #     self.demand.user_chooses_PT(rid, sim_time)
            for i, operator in enumerate(self.operators):
                operator.user_cancels_request(rid, sim_time)
            self.demand.record_user(rid)
            del self.demand.rq_db[rid]
            try:
                del self.demand.undecided_rq[rid]
            except KeyError:
                # raises KeyError if request decided right away
                pass
        else:
            for i, operator in enumerate(self.operators):
                if i != chosen_operator:
                    operator.user_cancels_request(rid, sim_time)
                else:
                    operator.user_confirms_booking(rid, sim_time)
                    self.demand.waiting_rq[rid] = rq_obj
            try:
                del self.demand.undecided_rq[rid]
            except KeyError:
                # raises KeyError if request decided right away
                pass
        return chosen_operator

    def _check_waiting_request_cancellations(self, sim_time):
        """This method builds the interface for traveler models, where users can cancel their booking after selecting
        an operator.

        :param sim_time: current simulation time
        :return: None
        """
        for rid, rq_obj in self.demand.waiting_rq.items():
            chosen_operator = rq_obj.get_chosen_operator()
            in_vehicle = rq_obj.get_service_vehicle()
            if in_vehicle is None and chosen_operator is not None and rq_obj.cancels_booking(sim_time):
                self.operators[chosen_operator].user_cancels_request(rid, sim_time)
                self.demand.record_user(rid)
                del self.demand.rq_db[rid]
                del self.demand.waiting_rq[rid]

    def run(self, tqdm_position=0):
        self._start_realtime_plot()
        t_run_start = time.perf_counter()
        if not self._started:
            self._started = True
            if PROGRESS_LOOP == "off":
                for sim_time in range(self.start_time, self.end_time, self.time_step):
                    self.step(sim_time)
                    self._update_realtime_plots_dict(sim_time)
            elif PROGRESS_LOOP == "demand":
                # loop over time with progress bar scaling according to future demand
                all_requests = sum([len(x) for x in self.demand.future_requests.values()])
                with tqdm(total=100, position=tqdm_position) as pbar:
                    pbar.set_description(self.scenario_parameters.get(G_SCENARIO_NAME))
                    for sim_time in range(self.start_time, self.end_time, self.time_step):
                        remaining_requests = sum([len(x) for x in self.demand.future_requests.values()])
                        self.step(sim_time)
                        cur_perc = int(100 * (1 - remaining_requests/all_requests))
                        pbar.update(cur_perc - pbar.n)
                        vehicle_counts = self.count_fleet_status()
                        info_dict = {"simulation_time": sim_time,
                                     "driving": sum([vehicle_counts[x] for x in G_DRIVING_STATUS])}
                        info_dict.update({x.display_name: vehicle_counts[x] for x in PROGRESS_LOOP_VEHICLE_STATUS})
                        pbar.set_postfix(info_dict)
                        self._update_realtime_plots_dict(sim_time)
            else:
                # loop over time with progress bar scaling with time
                for sim_time in tqdm(range(self.start_time, self.end_time, self.time_step), position=tqdm_position,
                                     desc=self.scenario_parameters.get(G_SCENARIO_NAME)):
                    self.step(sim_time)
                    self._update_realtime_plots_dict(sim_time)

            # record stats
            self.record_stats()

            # save final state, record remaining travelers and vehicle tasks
            self.save_final_state()
            self.record_remaining_assignments()
            self.demand.record_remaining_users()
        t_run_end = time.perf_counter()
        # call evaluation
        self.evaluate()
        t_eval_end = time.perf_counter()
        # short report
        t_init = datetime.timedelta(seconds=int(t_run_start - self.t_init_start))
        t_sim = datetime.timedelta(seconds=int(t_run_end - t_run_start))
        t_eval = datetime.timedelta(seconds=int(t_eval_end - t_run_end))
        prt_str = f"Scenario {self.scenario_name} finished:\n" \
                  f"{'initialization':>20} : {t_init} h\n" \
                  f"{'simulation':>20} : {t_sim} h\n" \
                  f"{'evaluation':>20} : {t_eval} h\n"
        print(prt_str)
        LOG.info(prt_str)
        self._end_realtime_plot()

    def _start_realtime_plot(self):
        """ This method starts a separate process for real time python plots """
        if self.realtime_plot_flag in {1, 2}:
            if self.scenario_parameters.get(G_SIM_REALTIME_PLOT_EXTENTS, None):
                extents = self.scenario_parameters.get(G_SIM_REALTIME_PLOT_EXTENTS)
                lons, lats = extents[:2], extents[2:]
            else:
                bounding = self.routing_engine.return_network_bounding_box()
                lons, lats = list(zip(*bounding))
            if self.realtime_plot_flag == 1:
                self._manager = Manager()
                self._shared_dict = self._manager.dict()
                self._plot_class_instance = PyPlot(self.dir_names["network"], self._shared_dict, plot_extent=lons+lats)
                self._plot_class_instance.start()
            else:
                plot_dir = Path(self.dir_names["output"], "real_time_plots")
                if plot_dir.exists() is False:
                    plot_dir.mkdir()
                self._plot_class_instance = PyPlot(self.dir_names["network"], self._shared_dict,
                                                   plot_extent=lons + lats, plot_folder=str(plot_dir))

    def _end_realtime_plot(self):
        """ Closes the process for real time plots """
        if self.realtime_plot_flag == 1:
            self._shared_dict["stop"] = True
            self._plot_class_instance.join()
            self._manager.shutdown()

    def _update_realtime_plots_dict(self, sim_time):
        """ This method updates the shared dict with the realtime plot process """
        if self.realtime_plot_flag in {1, 2}:
            veh_ids = list(self.sim_vehicles.keys())
            possible_states = self.scenario_parameters.get(G_SIM_REALTIME_PLOT_VEHICLE_STATUS,
                                                           [status.value for status in VRL_STATES])
            G_VEHICLE_STATUS_DICT = VRL_STATES.G_VEHICLE_STATUS_DICT()
            possible_states = [G_VEHICLE_STATUS_DICT[x] for x in possible_states]
            veh_status = [self.sim_vehicles[veh].status for veh in veh_ids]
            veh_status = [state.display_name for state in veh_status]
            veh_positions = [self.sim_vehicles[veh].pos for veh in veh_ids]
            veh_positions = self.routing_engine.return_positions_lon_lat(veh_positions)
            df = pd.DataFrame({"status": veh_status,
                               "coordinates": veh_positions})
            self._shared_dict.update({"veh_coord_status_df": df,
                                      "possible_status": possible_states,
                                      "simulation_time": f"simulation time: {datetime.timedelta(seconds=sim_time)}"})
            if self.realtime_plot_flag == 2:
                self._plot_class_instance.save_single_plot(str(sim_time))

    def count_fleet_status(self) -> dict:
        """ This method counts the number of vehicles in each of the vehicle statuses

        :return: dictionary of vehicle state as keys and number of vehicles in those status as values
        """
        vehicles = self.sim_vehicles.values()
        count = {state: 0 for state in VRL_STATES}
        for v in vehicles:
            count[v.status] += 1
        return count

    @abstractmethod
    def step(self, sim_time):
        """This method determines the simulation flow in a time step.

        :param sim_time: new simulation time
        :return: None
        """
        LOG.warning("abstractmethod not overwritten! When defined as ABC in next commits, this will raise an error!")
        pass

    @abstractmethod
    def check_sim_env_spec_inputs(self, scenario_parameters):
        # TODO # delete? -> part of init if necessary
        LOG.warning("abstractmethod not overwritten! When defined as ABC in next commits, this will raise an error!")
        return scenario_parameters

    def add_init(self, scenario_parameters):
        for op_id, op in enumerate(self.operators):
            operator_attributes = self.list_op_dicts[op_id]
            op.add_init(operator_attributes, self.scenario_parameters)

    @abstractmethod
    def add_evaluate(self):
        LOG.warning("abstractmethod not overwritten! When defined as ABC in next commits, this will raise an error!")
        pass
