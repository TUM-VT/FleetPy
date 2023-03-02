from __future__ import annotations
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
from abc import abstractmethod, ABCMeta
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
# from IPython import embed

# src imports
# -----------
from src.simulation.Offers import Rejection, TravellerOffer
from src.simulation.Legs import VehicleRouteLeg  # ,VehicleChargeLeg

from src.fleetctrl.charging.ChargingBase import ChargingBase  # ,VehicleChargeLeg
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, RoutingTargetPlanStop
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.fleetctrl.pricing.DynamicPricingBase import DynamicPrizingBase
from src.fleetctrl.fleetsizing.DynamicFleetSizingBase import DynamicFleetSizingBase
from src.fleetctrl.reservation.ReservationBase import ReservationBase
from src.demand.TravelerModels import RequestBase

from src.misc.init_modules import load_repositioning_strategy, load_charging_strategy, \
    load_dynamic_fleet_sizing_strategy, load_dynamic_pricing_strategy, load_reservation_strategy
from src.fleetctrl.pooling.GeneralPoolingFunctions import get_assigned_rids_from_vehplan
if TYPE_CHECKING:
    from src.routing.NetworkBase import NetworkBase
    from src.simulation.Vehicles import SimulationVehicle
    from src.infra.Zoning import ZoneSystem
    from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure, PublicChargingInfrastructureOperator
    from src.simulation.StationaryProcess import ChargingProcess

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_FleetControlBase = {
    "doc" : "this class is the base class representing an MoD operator",
    "inherit" : None,
    "input_parameters_mandatory": [G_OP_VR_CTRL_F, G_OP_FLEET],
    "input_parameters_optional": [
        G_RA_SOLVER, G_RA_OPT_HOR, G_OP_MIN_WT, G_OP_MIN_WT, G_OP_MAX_WT, G_OP_MAX_DTF, G_OP_ADD_CDT, G_OP_MIN_DTW,
        G_OP_CONST_BT, G_OP_ADD_BT, G_OP_INIT_VEH_DIST
        ],
    "mandatory_modules": [],
    "optional_modules": [G_RA_RES_MOD, G_OP_CH_M, G_OP_REPO_M, G_OP_DYN_P_M, G_OP_DYN_FS_M]
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class FleetControlBase(metaclass=ABCMeta):
    def __init__(self, op_id : int, operator_attributes : Dict, list_vehicles : List[SimulationVehicle],
                 routing_engine : NetworkBase, zone_system : ZoneSystem, scenario_parameters : Dict,
                 dir_names : Dict, op_charge_depot_infra : OperatorChargingAndDepotInfrastructure=None,
                 list_pub_charging_infra: List[PublicChargingInfrastructureOperator]= []):
        """The general attributes for the fleet control module are initialized. Strategy specific attributes are
        introduced in the children classes.

        :param op_id: operator id
        :type op_id: int
        :param operator_attributes: dictionary with keys from globals and respective values
        :type operator_attributes: dict
        :param list_vehicles: simulation vehicles; their assigned plans should be instances of the VehicleRouteLeg class
        :type list_vehicles: list
        :param routing_engine: routing engine
        :type routing_engine: Network
        :param zone_system: zone system
        :type zone_system: ZoneSystem
        :param scenario_parameters: access to all scenario parameters (if necessary)
        :type scenario_parameters: dict
        :param dir_names: dictionary with references to data and output directories
        :type dir_names: dict
        :param op_charge_depot_infra: reference to a OperatorChargingAndDepotInfrastructure class (optional) (unique for each operator)
        :type OperatorChargingAndDepotInfrastructure: OperatorChargingAndDepotInfrastructure
        :param list_pub_charging_infra: list of PublicChargingInfrastructureOperator classes (optional) (accesible for all agents)
        :type list_pub_charging_infra: list of PublicChargingInfrastructureOperator
        """
        self.n_cpu = scenario_parameters["n_cpu_per_sim"]
        self.solver: str = operator_attributes.get(G_RA_SOLVER, "Gurobi")
        self.log_gurobi : bool = scenario_parameters.get(G_LOG_GUROBI, False)
        self.op_id = op_id
        self.routing_engine: NetworkBase = routing_engine
        self.zones: ZoneSystem = zone_system
        self.dir_names = dir_names
        #
        self.sim_vehicles: List[SimulationVehicle] = list_vehicles
        self.nr_vehicles = len(self.sim_vehicles)
        sim_start_time = scenario_parameters[G_SIM_START_TIME]
        self.sim_time = sim_start_time

        # dynamic output base
        # -------------------
        self.dyn_fltctrl_output_f = os.path.join(dir_names[G_DIR_OUTPUT], f"3-{self.op_id}_op-dyn_atts.csv")
        self.dyn_output_dict = {}
        self.dyn_par_keys = []

        # Vehicle Plans, Request-Assignment and Availability
        # --------------------------------------------------
        self.veh_plans : Dict[int, VehiclePlan] = {}
        for veh_obj in self.sim_vehicles:
            vid = veh_obj.vid
            self.veh_plans[vid] = VehiclePlan(veh_obj, sim_start_time, routing_engine, [])
        self.rid_to_assigned_vid = {}
        self.pos_veh_dict_time = None
        self.pos_veh_dict = {}  # pos -> list_veh
        self.vr_ctrl_f = None  # has to be set by respective children classes
        self.vid_finished_VRLs : Dict[int, List[VehicleRouteLeg]] = {}
        self.vid_with_reserved_rids : Dict[int, List[Any]] = {}  # vid -> list of reservation_rids planned for vehicle

        # PlanRequest data base (all / outside of short-term horizon)
        # -----------------------------------------------------------
        self.opt_horizon = operator_attributes.get(G_RA_OPT_HOR, 900)
        operator_attributes[G_RA_OPT_HOR] = self.opt_horizon
        res_class = load_reservation_strategy(operator_attributes.get(G_RA_RES_MOD, "RollingHorizon"))
        self.reservation_module : ReservationBase = res_class(self, operator_attributes, dir_names, solver=self.solver)
        self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_RES)
        self.rq_dict : Dict[Any, PlanRequest] = {}  # rid_struct -> PlanRequest
        self.begin_approach_buffer_time = operator_attributes.get(G_RA_RES_APP_BUF_TIME, 0)

        # required fleet control parameters -> create error if invalid entries
        # --------------------------------------------------------------------
        self.min_wait_time = operator_attributes.get(G_OP_MIN_WT, 0)
        self.max_wait_time = operator_attributes.get(G_OP_MAX_WT, LARGE_INT)
        self.max_dtf = operator_attributes.get(G_OP_MAX_DTF, None)
        self.add_cdt = operator_attributes.get(G_OP_ADD_CDT, None)
        self.max_cdt = operator_attributes.get(G_OP_MAX_CDT, None)
        self.min_dtw = operator_attributes.get(G_OP_MIN_DTW, None)
        self.const_bt = operator_attributes.get(G_OP_CONST_BT, 0)
        self.add_bt = operator_attributes.get(G_OP_ADD_BT, 0)
        # early locking/constraining of request pickup
        # --------------------------------------------
        self.early_lock_time = operator_attributes.get(G_RA_LOCK_TIME, 0)
        tmp_upd_tw_hardness = operator_attributes.get(G_RA_TW_HARD, 0)
        self.update_hard_time_windows = False
        self.update_soft_time_windows = False
        if tmp_upd_tw_hardness == 1:
            self.update_soft_time_windows = True
        elif tmp_upd_tw_hardness == 2:
            self.update_hard_time_windows = True
        self.time_window_length = operator_attributes.get(G_RA_TW_LENGTH)
        if self.update_hard_time_windows or self.update_soft_time_windows:
            if not self.time_window_length:
                raise IOError(f"Update of time windows requires {G_RA_TW_LENGTH} input!")

        # ###################################
        # Additional fleet control strategies
        # ###################################
        prt_strategy_str = f"Scenario {scenario_parameters[G_SCENARIO_NAME]}\n"
        prt_strategy_str += f"Operator {op_id} control strategy: {self.__class__.__name__}\n"
        prt_strategy_str += f"\t control function: {operator_attributes[G_OP_VR_CTRL_F]}\n"
        prt_strategy_str += f"Operator {op_id} additional strategies:\n"

        # RV and insertion heuristics
        # ---------------------------
        self.rv_heuristics = {}
        self.insertion_heuristics = {}
        rv_im_max_routes = operator_attributes.get(G_RH_I_NWS)
        if not pd.isnull(rv_im_max_routes):
            self.rv_heuristics[G_RH_I_NWS] = int(rv_im_max_routes)
        rv_res_max_routes = operator_attributes.get(G_RH_R_NWS)
        if not pd.isnull(rv_res_max_routes):
            self.rv_heuristics[G_RH_R_NWS] = int(rv_res_max_routes)
        rv_nr_veh_direction = operator_attributes.get(G_RVH_DIR)
        if not pd.isnull(rv_nr_veh_direction):
            self.rv_heuristics[G_RVH_DIR] = int(rv_nr_veh_direction)
        rv_nr_least_load = operator_attributes.get(G_RVH_LWL)
        if not pd.isnull(rv_nr_least_load):
            self.rv_heuristics[G_RVH_LWL] = rv_nr_least_load
        rv_nr_rr = operator_attributes.get(G_RVH_AM_RR)
        if not pd.isnull(rv_nr_rr):
            self.rv_heuristics[G_RVH_AM_RR] = rv_nr_rr
        rv_nr_ti = operator_attributes.get(G_RVH_AM_TI)
        if not pd.isnull(rv_nr_ti):
            self.rv_heuristics[G_RVH_AM_TI] = rv_nr_ti
        ih_keep_x_best_plans_per_veh = operator_attributes.get(G_VPI_KEEP)
        if not pd.isnull(ih_keep_x_best_plans_per_veh):
            self.insertion_heuristics[G_VPI_KEEP] = int(ih_keep_x_best_plans_per_veh)
        max_rv_con = operator_attributes.get(G_RA_MAX_VR)
        if not pd.isnull(max_rv_con):
            self.rv_heuristics[G_RA_MAX_VR] = int(max_rv_con)
        max_req_plans = operator_attributes.get(G_RA_MAX_RP)
        if not pd.isnull(max_req_plans):
            self.rv_heuristics[G_RA_MAX_RP] = int(max_req_plans)
        prt_strategy_str += f"\t RV Heuristics: {self.rv_heuristics}\n"
        prt_strategy_str += f"\t Stop-Insert Heuristics: {self.insertion_heuristics}\n"

        # charging management and strategy
        # --------------------------------
        self.min_aps_soc = operator_attributes.get(G_OP_APS_SOC, 0.1)
        # TODO # init available charging operators
        self.op_charge_depot_infra = op_charge_depot_infra
        self.list_pub_charging_infra = list_pub_charging_infra
        charging_method = operator_attributes.get(G_OP_CH_M)
        if charging_method is not None:
            ChargingClass = load_charging_strategy(charging_method)
            self.charging_strategy : ChargingBase = ChargingClass(self, operator_attributes)
            prt_strategy_str += f"\t Charging: {self.charging_strategy.__class__.__name__}\n"
            self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_CH)
        else:
            self.charging_strategy = None
            prt_strategy_str += "\t Charging: None\n"
        self._active_charging_processes: Dict[Tuple[int, str], ChargingProcess] = {}     # charging task id (ch_op_id, booking_id) -> charging process
        self._vid_to_assigned_charging_process: Dict[int, Tuple[int, str]] = {}      # vehicle id -> charging task id

        # on-street parking
        # -----------------
        if self.op_charge_depot_infra:
            self.allow_on_street_parking = scenario_parameters.get(G_INFRA_ALLOW_SP, True)
        else:
            self.allow_on_street_parking = True
        prt_strategy_str += f"\t On-Street Parking: {self.allow_on_street_parking}\n"

        # repositioning strategy
        # ----------------------
        repo_method = operator_attributes.get(G_OP_REPO_M)
        self.repo_time_step = operator_attributes.get(G_OP_REPO_TS)
        if repo_method is not None and self.repo_time_step is not None:
            RepoClass = load_repositioning_strategy(repo_method)
            self.repo : RepositioningBase = RepoClass(self, operator_attributes, dir_names)
            prt_strategy_str += f"\t Repositioning: {self.repo.__class__.__name__}\n"
            self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_REPO)
        else:
            self.repo = None
            prt_strategy_str += f"\t Repositioning: None\n"

        # pricing
        # -------
        # constant
        self.fare_per_pax = {}  # nr_pax -> total, e.g. {1:1, 2:1.8, 3:2.5}
        self.base_fare = operator_attributes.get(G_OP_FARE_B, 0)
        self.dist_fare = operator_attributes.get(G_OP_FARE_D, 0)
        self.time_fare = operator_attributes.get(G_OP_FARE_T, 0)
        self.min_fare = operator_attributes.get(G_OP_FARE_MIN, 0)
        # dynamic
        dyn_pricing_method = operator_attributes.get(G_OP_DYN_P_M)
        if dyn_pricing_method:
            DPS_class = load_dynamic_pricing_strategy(dyn_pricing_method)
            self.dyn_pricing : DynamicPrizingBase = DPS_class(self, operator_attributes)
            prt_strategy_str += f"\t Dynamic Pricing: {self.dyn_pricing.__class__.__name__}\n"
            self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_DP)
        else:
            self.dyn_pricing = None
            prt_strategy_str += f"\t Dynamic Pricing: None\n"

        # dynamic fleet sizing
        # --------------------
        dyn_fs_method = operator_attributes.get(G_OP_DYN_FS_M)
        if dyn_fs_method:
            DFS_class = load_dynamic_fleet_sizing_strategy(dyn_fs_method)
            self.dyn_fleet_sizing : DynamicFleetSizingBase = DFS_class(self, operator_attributes)
            prt_strategy_str += f"\t Dynamic Fleet Sizing: {self.dyn_fleet_sizing.__class__.__name__}\n"
            self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_DFS)
        else:
            self.dyn_fleet_sizing = None
            prt_strategy_str += f"\t Dynamic Fleet Sizing: None\n"

        # log and print summary of additional strategies
        LOG.info(prt_strategy_str)
        print(prt_strategy_str)

    def add_init(self, operator_attributes, scenario_parameters):
        """ additional init for stuff that has to be loaded (i.e. in modules) that requires full init of fleetcontrol
        """
        if self.dyn_fleet_sizing is not None:
            self.dyn_fleet_sizing.add_init(operator_attributes)

    @abstractmethod
    def receive_status_update(self, vid : int, simulation_time : int, list_finished_VRL : List[VehicleRouteLeg], force_update : bool=True):
        """This method can be used to update plans and trigger processes whenever a simulation vehicle finished some
         VehicleRouteLegs.

        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        :param list_finished_VRL: list of VehicleRouteLeg objects
        :type list_finished_VRL: list
        :param force_update: indicates if also current vehicle plan feasibilities have to be checked
        :type force_update: bool
        """
        veh_obj = self.sim_vehicles[vid]
        # the vehicle plans should be up to date from assignments of previous time steps
        if list_finished_VRL or force_update:
            LOG.debug(f"vid {vid} at time {simulation_time} recieves status update: {[str(x) for x in list_finished_VRL]}")
            LOG.debug(f"   with current vehicle plan {self.veh_plans[vid]}")
            self.veh_plans[vid].update_plan(veh_obj, simulation_time, self.routing_engine, list_finished_VRL)
            if self._vid_to_assigned_charging_process.get(vid) is not None:
                finished_charging_task_id = None
                for vrl in list_finished_VRL:
                    if vrl.stationary_process is not None:
                        finished_charging_task_id = vrl.stationary_process.id
                        break
                if finished_charging_task_id is not None:
                    assigned_id = self._vid_to_assigned_charging_process[vid]
                    if assigned_id[1] != finished_charging_task_id:
                        LOG.warning("inconsitent charging task finished! assigned : {} finished: {}".format(assigned_id, finished_charging_task_id))
                    del self._active_charging_processes[assigned_id]
                    del self._vid_to_assigned_charging_process[vid]
        upd_utility_val = self.compute_VehiclePlan_utility(simulation_time, veh_obj, self.veh_plans[vid])
        self.veh_plans[vid].set_utility(upd_utility_val)

    @abstractmethod
    def user_request(self, rq : RequestBase, simulation_time : int):
        """This method is triggered for a new incoming request. It generally generates a PlanRequest from the rq and
        adds it to the database. UserOffers can be created here but are not returned.
        These offers will be accessed by the simulation environment via the method self.get_current_offer(rid)
        Use the method "_create_user_offer" to create this dictionary!

        :param rq: request object containing all request information
        :type rq: RequestDesign
        :param simulation_time: current simulation time
        :type simulation_time: int
        """
        pass

    @abstractmethod
    def user_confirms_booking(self, rid : Any, simulation_time : int):
        """This method is used to confirm a customer booking. This can trigger some database processes.

        WHEN OVERWRITING THIS METHOD MAKE SURE TO CALL AT LEAST THE FOLLOWING LINE!

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.rq_dict[rid].set_service_accepted()

    @abstractmethod
    def user_cancels_request(self, rid : Any, simulation_time : int):
        """This method is used to confirm a customer cancellation. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        pass

    @abstractmethod
    def _create_user_offer(self, prq : PlanRequest, simulation_time : int, assigned_vehicle_plan : VehiclePlan=None,
                           offer_dict_without_plan : Dict={}) -> TravellerOffer:
        """ this method should be overwritten to create an offer for the request rid
        when overwriting this function, make sure to call the lines below depending on the offer returned

        THIS FUNCTION HAS TO BE OVERWRITTEN!

        :param prq: plan request
        :type prq: PlanRequest obj
        :param simulation_time: current simulation time
        :type simulation_time: int
        :param assigned_vehicle_plan: vehicle plan of initial solution to serve this request
        :type assigned_vehicle_plan: VehiclePlan or None
        :param offer_dict_without_plan: can be used to create an offer that is not derived from a vehicle plan
                    entries will be used to create/extend offer
        :type offer_dict_without_plan: dict
        :return: operator offer for the user
        :rtype: TravellerOffer
        """
        # TODO # new strategy: base functionality including dynamic pricing | additional attributes can be added later
        offer = None  # should be filled by overwriting method
        prq.set_service_offered(offer)  # has to be called
        raise EnvironmentError("_create_user_offer() can't be called with super()")
        return offer

    def _create_rejection(self, prq : PlanRequest, simulation_time : int) -> Rejection:
        """This method creates a TravellerOffer representing a rejection.

        :param prq: PlanRequest
        :param simulation_time: current simulation time
        :return: Rejection (child class of TravellerOffer)
        """
        offer = Rejection(prq.get_rid(), self.op_id)
        LOG.debug(f"reject customer {prq} at time {simulation_time}")
        prq.set_service_offered(offer)
        if self.repo and not prq.get_reservation_flag():
            self.repo.register_rejected_customer(prq, simulation_time)
        return offer

    def get_current_offer(self, rid : Any) -> TravellerOffer:
        """ this method returns the currently active offer for the request rid
        if a current offer is active:
            the current TravellerOffer is returned
        if the service is decline and the request didnt leave the system yet:
            a "service_declined" TravellerOffer is returned (at least offered_waiting_time is set to None in TravellerOffer init)
        if an offer is not evaluated yet:
            None is returned

        use the method "_create_user_offer" to create single user offers

        :param rid: request id
        :type rid: int
        :return: TravellerOffer or None for the request
        :rtype: TravellerOffer or None
        """
        return self.rq_dict[rid].get_current_offer()

    @abstractmethod
    def change_prq_time_constraints(self, sim_time : int, rid : Any, new_lpt : int, new_ept : int=None):
        """This method should be called when the hard time constraints of a customer should be changed.
        It changes the PlanRequest attributes. Moreover, this method called on child classes should adapt the
        PlanStops of VehiclePlans containing this PlanRequest and recheck feasibility. The VehiclePlan method
        update_prq_hard_constraints() can be used for this purpose.

        :param sim_time: current simulation time
        :param rid: request id
        :param new_lpt: new latest pickup time, None is ignored
        :param new_ept: new earliest pickup time, None is ignored
        :return: None
        """
        prq = self.rq_dict[rid]
        prq.set_new_pickup_time_constraint(new_lpt, new_ept)

    @abstractmethod
    def acknowledge_boarding(self, rid : Any, vid : int, simulation_time : int):
        """This method can trigger some database processes whenever a passenger is starting to board a vehicle.

        MAKE SURE TO CALL AT LEAST super().acknowledge_boarding(rid, vid, simulation_time) or add the following line to the
        overwritten method

        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.rq_dict[rid].set_pickup(vid, simulation_time)

    @abstractmethod
    def acknowledge_alighting(self, rid : Any, vid : int, simulation_time : int):
        """This method can trigger some database processes whenever a passenger is finishing to alight a vehicle.

        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        pass

    @abstractmethod
    def assign_vehicle_plan(self, veh_obj : SimulationVehicle, vehicle_plan : VehiclePlan, sim_time : int,
                            force_assign : bool=False, assigned_charging_task: Tuple[Tuple[str, int], ChargingProcess]=None, add_arg : Any=None):
        """ this method should be used to assign a new vehicle plan to a vehicle

        WHEN OVERWRITING THIS FUNCTION MAKE SURE TO CALL AT LEAST THE LINES BELOW (i.e. super())

        :param veh_obj: vehicle obj to assign vehicle plan to
        :type veh_obj: SimulationVehicle
        :param vehicle_plan: vehicle plan that should be assigned
        :type vehicle_plan: VehiclePlan
        :param sim_time: current simulation time in seconds
        :type sim_time: int
        :param force_assign: this parameter can be used to enforce the assignment, when a plan is (partially) locked
        :type force_assign: bool
        :param assigned_charging_task: this parameter has to be set if a new charging task is assigned to the vehicle
        :type assigned_charging_task: tuple( tuple(charging_operator_id, booking_id), corresponding charging process object )
        :param add_arg: possible additional argument if needed
        :type add_arg: not defined here
        """
        LOG.debug(f"assign to {veh_obj.vid} at time {sim_time} : {vehicle_plan}")
        vehicle_plan.update_tt_and_check_plan(veh_obj, sim_time, self.routing_engine, keep_feasible=True)
        if self._vid_to_assigned_charging_process.get(veh_obj.vid) is not None:
            veh_plan_ch_task = None
            for ps in vehicle_plan.list_plan_stops:
                if ps.get_charging_task_id() is not None:
                    veh_plan_ch_task = ps.get_charging_task_id()
            if veh_plan_ch_task is None:
                LOG.warning(f"charging task {self._vid_to_assigned_charging_process.get(veh_obj.vid)} no longer assigned! -> cancel booking!")
                assigned_veh_charge_task = self._vid_to_assigned_charging_process[veh_obj.vid]
                ch_op_id = assigned_veh_charge_task[0]
                if type(ch_op_id) == str and ch_op_id.startswith("op"):
                    self.op_charge_depot_infra.cancel_booking(sim_time, self._active_charging_processes[assigned_veh_charge_task])
                else:
                    self.list_pub_charging_infra[ch_op_id].cancel_booking(sim_time, self._active_charging_processes[assigned_veh_charge_task])
                del self._active_charging_processes[self._vid_to_assigned_charging_process[veh_obj.vid]]
                del self._vid_to_assigned_charging_process[veh_obj.vid]
        if assigned_charging_task is not None:
            self._active_charging_processes[assigned_charging_task[0]] = assigned_charging_task[1]
            self._vid_to_assigned_charging_process[veh_obj.vid] = assigned_charging_task[0]
        new_list_vrls = self._build_VRLs(vehicle_plan, veh_obj, sim_time)
        veh_obj.assign_vehicle_plan(new_list_vrls, sim_time, force_ignore_lock=force_assign)
        self.veh_plans[veh_obj.vid] = vehicle_plan
        for rid in get_assigned_rids_from_vehplan(vehicle_plan):
            pax_info = vehicle_plan.get_pax_info(rid)
            self.rq_dict[rid].set_assigned(pax_info[0], pax_info[1])
            self.rid_to_assigned_vid[rid] = veh_obj.vid

    def time_trigger(self, simulation_time : int):
        """This method is used to perform time-triggered processes. These are split into the following:
        1) the optimization of the current assignments of requests
        2) other tasks (repositioning, charging, fleetsizing, pricing)

        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        # check whether reservation requests should be considered as immediate requests
        rids_to_reveal = self.reservation_module.reveal_requests_for_online_optimization(simulation_time)
        for rid in rids_to_reveal:
            LOG.debug(f"activate {rid} with for global optimisation at time {simulation_time}!")
            self._prq_from_reservation_to_immediate(rid, simulation_time)
        self._call_time_trigger_request_batch(simulation_time)
        self._call_time_trigger_additional_tasks(simulation_time)

    @abstractmethod
    def _call_time_trigger_request_batch(self, simulation_time : int):
        """This method can be used to perform time-triggered processes, e.g. the optimization of the current
        assignments of simulation vehicles of the fleet.

        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        pass

    def compute_VehiclePlan_utility(self, simulation_time : int, veh_obj : SimulationVehicle, vehicle_plan : VehiclePlan) -> float:
        """This method computes the utility of a given plan and returns the value.

        :param simulation_time: current simulation time
        :type simulation_time: float
        :param veh_obj: vehicle object
        :type veh_obj: SimulationVehicle
        :param vehicle_plan: vehicle plan in question
        :type vehicle_plan: VehiclePlan
        :return: utility of vehicle plan
        :rtype: float
        """
        utility = self.vr_ctrl_f(simulation_time, veh_obj, vehicle_plan, self.rq_dict, self.routing_engine)
        vehicle_plan.set_utility(utility)
        return utility

    @abstractmethod
    def lock_current_vehicle_plan(self, vid : int):
        """This method performs all database actions in case the current vehicle plan becomes completely locked,
        e.g. because a charging process is appended or the vehicle becomes inactive.

        ADDITIONAL DATABASE TRIGGERS MIGHT HAVE TO BE DEFINED!

        :param vid: vehicle id
        :return: None
        """
        assigned_plan = self.veh_plans.get(vid, None)
        if assigned_plan is not None:
            for ps in assigned_plan.list_plan_stops:
                ps.set_locked(True)

    def inform_network_travel_time_update(self, simulation_time : int):
        """ this method can be used to inform the operator that new travel times are available
        at simulation time
        :param simulation_time: time the new travel times are available
        """
        LOG.warning("inform_network_travel_time_update not implemented for this operator")
        pass

    def set_soft_time_constraints(self, rid : Any, new_soft_lpt : int, new_soft_ept : int):
        """This method updates the PlanRequest soft time window constraints. These will be adapted in the PlanRequest.
        This function will raise an AttributeError if the PlanRequest is no SoftConstraintPlanRequest!
        Mind: The objective function values have to be updated later in the process flow!

        :param rid: request id
        :param new_soft_lpt: new soft latest pickup time, None is ignored
        :param new_soft_ept: new soft earliest pickup time, None is ignored
        :return: None
        """
        prq = self.rq_dict[rid]
        prq.set_soft_do_constraints(new_soft_lpt, new_soft_ept)

    def _compute_fare(self, sim_time : int, prq : PlanRequest, assigned_veh_plan : VehiclePlan=None) -> float:
        """This method can be used to compute the fare. It already considers utilization and zone surge pricing. It
        can be overwritten by subclasses if necessary.

        :param sim_time: current simulation time
        :param prq: PlanRequest
        :param assigned_veh_plan: can be utilized for fare system where actually shared trips are cheaper
        :return: fare
        """
        pax_factor = self.fare_per_pax.get(prq.nr_pax, prq.nr_pax)
        if self.dyn_pricing:
            if assigned_veh_plan is not None:
                exp_pt = assigned_veh_plan.get_pax_info(prq.get_rid_struct())[0]
            else:
                exp_pt = None
            base_fare_factor, dist_fare_factor, gen_factor =\
                self.dyn_pricing.get_elastic_price_factors(sim_time, expected_pu_time=exp_pt, o_pos=prq.o_pos,
                                                           d_pos=prq.d_pos)
        else:
            base_fare_factor, dist_fare_factor, gen_factor = 1, 1, 1
        fare = (self.base_fare * base_fare_factor + prq.init_direct_td * self.dist_fare * dist_fare_factor
                + prq.init_direct_tt * self.time_fare) * gen_factor * pax_factor
        final_fare = int(max(fare, self.min_fare))
        return final_fare

    def compute_current_fleet_utilization(self, sim_time : int) -> Tuple[float, float, int]:
        """ this method computes the current utilization of the fleet vehicles.
        only vehicles that can currently be used for passenger transport are considered as aktive vehicles
        relocation is only considered as utilized if the corresponding planstop is locked
        it also returns effectivly utilized vehicles (during a short lookahead horizon -> can be float)
            and the number of currently active vehicles
        :param sim_time: simulation_time
        :type sime_time: int
        :return: utilization [0,1], effectivly utilized vehicles (can be float!), active vehicles
        :rtype: tuple (float, float, int)
        """
        assignment_observation_horizon = 300  # TODO # global assignment observation time?
        n_active_vehicles = 0
        n_effective_utilized_vehicles = 0.0
        for veh in self.sim_vehicles:
            ass_plan = self.veh_plans[veh.vid]
            if len(ass_plan.list_plan_stops) == 0:
                n_active_vehicles += 1
                continue
            if veh.status in G_INACTIVE_STATUS:
                continue
            if ass_plan.list_plan_stops[0].is_empty() and not ass_plan.list_plan_stops[0].is_locked():
                n_active_vehicles += 1
                continue
            _, end_assignment = ass_plan.list_plan_stops[-1].get_planned_arrival_and_departure_time()
            if end_assignment - sim_time > assignment_observation_horizon:
                n_active_vehicles += 1
                n_effective_utilized_vehicles += 1.0
            else:
                frac = max(0, (end_assignment - sim_time) / assignment_observation_horizon)
                n_active_vehicles += 1.0
                n_effective_utilized_vehicles += frac 
        if n_active_vehicles > 0:
            util = n_effective_utilized_vehicles / n_active_vehicles
        else:
            util = 1.0
        #LOG.info("comute utilization: {} {} {}".format(util, n_effective_utilized_vehicles, n_active_vehicles))
        return util, n_effective_utilized_vehicles, n_active_vehicles

    @abstractmethod
    def _lock_vid_rid_pickup(self, sim_time, vid, rid):
        """This method constrains the pick-up of a rid. In the pooling case, the pick-up time is constrained to a very
        short time window. In the hailing case, the Task to serve rid is locked for the vehicle.

        WHEN OVERWRITING: MIND DATABASE UPDATES!

        :param sim_time: current simulation time
        :param vid: vehicle id
        :param rid: PlanRequest id
        :return: None
        """
        rid_info = self.veh_plans[vid].pax_info.get(rid)
        # TODO # this check should be done before calling this function, right?
        if rid_info is not None:
            planned_pu_time = rid_info[0]
            if planned_pu_time - sim_time <= self.early_lock_time:
                self.change_prq_time_constraints(sim_time, rid, planned_pu_time - 1, planned_pu_time + 1)
                prq = self.rq_dict[rid]
                prq.lock_request()
        else:
            raise AssertionError("constrain_or_lock_vid_rid_pickup(): Request pick-up is not schedule!")

    @abstractmethod
    def _prq_from_reservation_to_immediate(self, rid, sim_time):
        """This method is triggered when a reservation request becomes an immediate request.
        All database relevant methods can be triggered from here.

        :param rid: request id
        :param sim_time: current simulation time
        :return: None
        """
        pass

    def get_vid_reservation_list(self, vid):
        """This method returns the list of reservation requests that are currently assigned to a vehicle.

        :param vid: vehicle id
        :return: list of reservation rids currently assigned to a vehicle
        """
        return self.vid_with_reserved_rids.get(vid)

    def set_init_blocked(self, veh_obj : SimulationVehicle, start_time : int, routing_engine : NetworkBase, init_blocked_duration : int):
        """This method can be called by the initial vehicle state methods.

        :param init_blocked_duration: duration for which the vehicle should be blocked at the start of the simulation
        :return: None
        """
        plan_stop = RoutingTargetPlanStop(veh_obj.pos, locked=True, duration=init_blocked_duration, planstop_state=G_PLANSTOP_STATES.INACTIVE)
        self.veh_plans[veh_obj.vid].add_plan_stop(plan_stop, veh_obj, start_time, routing_engine)

    def _call_time_trigger_additional_tasks(self, sim_time):
        """This method can be used to trigger all fleet operational tasks that are not related to request assignment:
        - charging processes
        - changes to active fleet size
        - vehicle repositioning
        - dynamic pricing
        All these methods are controlled by scenario input parameters.

        :param sim_time: current simulation time
        :return: None
        """
        add_dyn_dict = {}

        # 1) Charging Processes
        # ---------------------
        if self.reservation_module:
            t0 = time.perf_counter()
            self.reservation_module.time_trigger(sim_time)
            add_dyn_dict[G_FCTRL_CT_RES] = round(time.perf_counter() - t0, 3)

        # 1) Charging Processes
        # ---------------------
        if self.charging_strategy:
            t0 = time.perf_counter()
            self.charging_strategy.time_triggered_charging_processes(sim_time)
            add_dyn_dict[G_FCTRL_CT_CH] = round(time.perf_counter() - t0, 3)

        # 2) Dynamic Fleet Sizing
        # -----------------------
        repo_activated_veh = False
        if self.dyn_fleet_sizing:
            t0 = time.perf_counter()
            change_in_fleet_size = self.dyn_fleet_sizing.check_and_change_fleet_size(sim_time)
            if change_in_fleet_size > 0:
                repo_activated_veh = True
            add_dyn_dict[G_FCTRL_CT_DFS] = round(time.perf_counter() - t0, 3)

        # 3) Repositioning
        # -------------------
        if self.repo is not None and (sim_time % self.repo_time_step == 0 or repo_activated_veh):
            t0 = time.perf_counter()
            LOG.info("Calling repositioning algorithm! (because of activated vehicles? {})".format(repo_activated_veh))
            # vehplans no longer locked, because repo called very often
            self.repo.determine_and_create_repositioning_plans(sim_time)
            add_dyn_dict[G_FCTRL_CT_REPO] = round(time.perf_counter() - t0, 3)

        # 4) Dynamic Pricing
        # ------------------
        if self.dyn_pricing is not None:
            t0 = time.perf_counter()
            self.dyn_pricing.update_current_price_factors(sim_time)
            add_dyn_dict[G_FCTRL_CT_DP] = round(time.perf_counter() - t0, 3)

        # 5) Move idle vehicles if on-street parking is not allowed
        # ---------------------------------------------------------
        if not self.allow_on_street_parking:
            self.charging_management.move_idle_vehicles_to_nearest_depots(sim_time, self)

        # record
        if add_dyn_dict:
            self._add_to_dynamic_fleetcontrol_output(sim_time, add_dyn_dict)

    def _init_dynamic_fleetcontrol_output_key(self, dict_key):
        """This method can be used to define a certain output key, such that the resulting data frame will record a '0'
        if this value was not given in the current time step.

        :param dict_key: parameter_name
        :return: None
        """
        if dict_key not in self.dyn_par_keys:
            self.dyn_par_keys.append(dict_key)

    def _get_current_dynamic_fleetcontrol_value(self, sim_time, dict_key):
        """This method can be used to get the current values of the dynamic output. This can be useful for
        separated processes (lke user requests).

        :param sim_time: current simulation time
        :param dict_key: parameter_name
        :return: parameter value at current sim_time (or None)
        """
        return self.dyn_output_dict.get(sim_time, {}).get(dict_key)

    def _add_to_dynamic_fleetcontrol_output(self, sim_time, output_dict):
        """ this method can be used to ouput time dependent fleet control attributes
        which will be stored in the file 3-{self.op_id}_op-dyn_atts.csv
        output_dict (parameter name -> value) will be added to the ouput table for the corresponding simulation time
        :param sim_time: current simulation time
        :param output_dict: dictionary parameter_name -> parameter value at current sim_time
        """
        if self.dyn_output_dict.get(sim_time):
            self.dyn_output_dict[sim_time].update(output_dict)
        else:
            self.dyn_output_dict[sim_time] = output_dict

    def record_dynamic_fleetcontrol_output(self, force=False):
        """ this method is used to store the current entries of self.dyn_output_dict in the
        file 3-{self.op_id}_op-dyn_atts.csv """
        current_buffer_size = len(self.dyn_output_dict.keys())
        if current_buffer_size > 0:
            if force or current_buffer_size > BUFFER_SIZE:
                if os.path.isfile(self.dyn_fltctrl_output_f):
                    write_mode = "a"
                    write_header = False
                else:
                    write_mode = "w"
                    write_header = True
                tmp_df_list = []
                for sim_time, entry_dict in self.dyn_output_dict.items():
                    x = {"sim_time":sim_time}
                    x.update(entry_dict)
                    for key in self.dyn_par_keys:
                        if key not in x.keys():
                            x[key] = 0.0
                    tmp_df_list.append(x)
                tmp_df = pd.DataFrame(tmp_df_list)
                unsorted_cols = tmp_df.columns
                sorted_cols = ["sim_time"] + self.dyn_par_keys.copy()
                for key in sorted(unsorted_cols):
                    if key not in sorted_cols:
                        sorted_cols.append(key)
                record_df = tmp_df[sorted_cols]
                record_df.to_csv(self.dyn_fltctrl_output_f, index=False, mode=write_mode, header=write_header)
                self.dyn_output_dict = {}
                # LOG.info(f"\t ... just wrote {current_buffer_size} entries from buffer to stats of operator {op_id}.")
                LOG.debug(f"\t ... just wrote {current_buffer_size} entries from buffer to dynamic stats of operator"
                          f" {self.op_id}.")
        # additionally save repositioning output if repositioning module is available
        if self.repo:
            self.repo.record_repo_stats()
            
    def _build_VRLs(self, vehicle_plan : VehiclePlan, veh_obj : SimulationVehicle, sim_time : int) -> List[VehicleRouteLeg]:
        """This method builds VRL for simulation vehicles from a given Plan. Since the vehicle could already have the
        VRL with the correct route, the route from veh_obj.assigned_route[0] will be used if destination positions
        are matching

        :param vehicle_plan: vehicle plan to be converted
        :param veh_obj: vehicle object to which plan is applied
        :param sim_time: current simulation time
        :return: list of VRLs according to the given plan
        """
        list_vrl = []
        c_pos = veh_obj.pos
        c_time = sim_time
        for pstop in vehicle_plan.list_plan_stops:
            # TODO: The following should be made as default behavior to delegate the specific tasks (e.g. boarding,
            #  charging etc) to the StationaryProcess class. The usage of StationaryProcess class can significantly
            #  simplify the code
            # i dont think the following lines work
            # if pstop.stationary_task is not None:
            #     list_vrl.append(self._get_veh_leg(pstop))
            #     continue
            boarding_dict = {1: [], -1: []}
            stationary_process = None
            if len(pstop.get_list_boarding_rids()) > 0 or len(pstop.get_list_alighting_rids()) > 0:
                boarding = True
                for rid in pstop.get_list_boarding_rids():
                    boarding_dict[1].append(self.rq_dict[rid])
                for rid in pstop.get_list_alighting_rids():
                    boarding_dict[-1].append(self.rq_dict[rid])
            else:
                boarding = False
            if pstop.get_charging_power() > 0:
                charging = True
                stationary_process = self._active_charging_processes[pstop.get_charging_task_id()]
            else:
                charging = False
            #if pstop.get_departure_time(0) > LARGE_INT:
            if pstop.get_state() == G_PLANSTOP_STATES.INACTIVE:
                inactive = True
            else:
                inactive = False
            if pstop.is_locked_end():
                reservation = True
            else:
                reservation = False
            if pstop.get_departure_time(0) != 0:
                planned_stop = True
                repo_target = False
            else:
                planned_stop = False
                repo_target = True
            if c_pos != pstop.get_pos():
                # driving vrl
                if boarding:
                    status = VRL_STATES.ROUTE
                elif charging:
                    status = VRL_STATES.TO_CHARGE
                elif inactive:
                    status = VRL_STATES.TO_DEPOT
                elif reservation:
                    status = VRL_STATES.TO_RESERVATION
                else:
                    # repositioning
                    status = VRL_STATES.REPOSITION
                # use empty boarding dict for this VRL, but do not overwrite boarding_dict!
                if pstop.get_earliest_start_time() - c_time > self.begin_approach_buffer_time \
                        and pstop.get_earliest_start_time() - c_time - self.routing_engine.return_travel_costs_1to1(c_pos, pstop.get_pos())[1] > self.begin_approach_buffer_time:
                    # wait at current postion until target can still be reached within self.begin_approach_buffer_time
                    list_vrl.append(VehicleRouteLeg(status, pstop.get_pos(), {1: [], -1: []}, locked=pstop.is_locked(), earliest_start_time=pstop.get_earliest_start_time() - self.begin_approach_buffer_time - self.routing_engine.return_travel_costs_1to1(c_pos, pstop.get_pos())[1]))
                else:
                    list_vrl.append(VehicleRouteLeg(status, pstop.get_pos(), {1: [], -1: []}, locked=pstop.is_locked()))
                c_pos = pstop.get_pos()
                _, c_time = pstop.get_planned_arrival_and_departure_time()
            # stop vrl
            if boarding and charging:
                status = VRL_STATES.BOARDING_WITH_CHARGING
            elif boarding:
                status = VRL_STATES.BOARDING
            elif charging:
                status = VRL_STATES.CHARGING
            elif inactive:
                status = VRL_STATES.OUT_OF_SERVICE
            elif planned_stop:
                status = VRL_STATES.PLANNED_STOP
            elif repo_target:
                status = VRL_STATES.REPO_TARGET
            else:
                # TODO # after ISTTT: add other states if necessary; for now assume vehicle idles
                status = VRL_STATES.IDLE
            if status != VRL_STATES.IDLE:
                dur, edep = pstop.get_duration_and_earliest_departure()
                earliest_start_time = pstop.get_earliest_start_time()
                #LOG.debug("vrl earliest departure: {} {}".format(dur, edep))
                if edep is not None:
                    LOG.warning("absolute earliest departure not implementen in build VRL!")
                    departure_time = edep
                else:
                    departure_time = -LARGE_INT
                if dur is not None:
                    stop_duration = dur
                else:
                    stop_duration = 0
                _, c_time = pstop.get_planned_arrival_and_departure_time()
                list_vrl.append(VehicleRouteLeg(status, pstop.get_pos(), boarding_dict, pstop.get_charging_power(),
                                                duration=stop_duration, earliest_start_time=earliest_start_time, earliest_end_time=departure_time,
                                                locked=pstop.is_locked(), stationary_process=stationary_process))
        return list_vrl
