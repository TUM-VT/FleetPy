# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.fleetctrl.SemiOnDemandBatchAssignmentFleetcontrol import PTStation, PtLine, SemiOnDemandBatchAssignmentFleetcontrol
from src.misc.globals import *
from typing import Dict, List, TYPE_CHECKING

# additional module imports (> requirements)
# ------------------------------------------
import numpy as np
import pandas as pd
import shapely
import time
import pyproj
import geopandas as gpd

# src imports
# -----------
from src.simulation.Offers import TravellerOffer
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, PlanStop
from src.fleetctrl.planning.PlanRequest import PlanRequest


# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
LOG = logging.getLogger(__name__)
LARGE_INT = 100000
# TOL = 0.1

from src.routing.NetworkBase import NetworkBase

if TYPE_CHECKING:
    from src.simulation.Vehicles import SimulationVehicle


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------
def create_stations(columns):
    return PTStation(columns["station_id"], columns["network_node_index"])


INPUT_PARAMETERS_RidePoolingBatchAssignmentFleetcontrol = {
    "doc": """Semi-on-Demand Hybrid Route Batch assignment fleet control (by Max Ng in Dec 2023)
        reference RidePoolingBatchAssignmentFleetcontrol and LinebasedFleetControl
        ride pooling optimisation is called after every optimisation_time_step and offers are created in the time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldnt be assigned in the first try, it will be considered again in the next opt-step with this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time with an interval of the size of this parameter""",
    "inherit": "RidePoolingBatchOptimizationFleetControlBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}




class SoDZonalBatchAssignmentFleetcontrol(SemiOnDemandBatchAssignmentFleetcontrol):
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                 dir_names, op_charge_depot_infra=None, list_pub_charging_infra=[]):
        """Combined fleet control for semi-on-demand flexible & fixed route
        Reference LinebasedFleetControl for more information on the solely fixed-route implementation.

        ride pooling optimisation is called after every optimisation_time_step and offers are created in the time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldnt be assigned in the first try, it will be considered again in the next opt-step with this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time with an interval of the size of this parameter


        :param op_id: operator id
        :type op_id: int
        :param operator_attributes: dictionary with keys from globals and respective values
        :type operator_attributes: dict
        :param list_vehicles: simulation vehicles; their assigned plans should be instances of the VehicleRouteLeg class
        :type list_vehicles: list
        :param routing_engine: routing engine
        :type routing_engine: Network
        :param scenario_parameters: access to all scenario parameters (if necessary)
        :type scenario_parameters: dict
        :param op_charge_depot_infra: reference to a OperatorChargingAndDepotInfrastructure class (optional) (unique for each operator)
        :type op_charge_depot_infra: OperatorChargingAndDepotInfrastructure
        :param list_pub_charging_infra: list of PublicChargingInfrastructureOperator classes (optional) (accesible for all agents)
        :type list_pub_charging_infra: list of PublicChargingInfrastructureOperator
        """

        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names, op_charge_depot_infra, list_pub_charging_infra)

        self.list_veh_in_terminus = [] # list of vehicles in terminus, for zonal control,
        # to update in receive_status_update, to check in time trigger

        # load demand distribution
        demand_dist_f = os.path.join(self.pt_data_dir, scenario_parameters.get(G_PT_DEMAND_DIST, 0))
        self.demand_dist_df = pd.read_csv(demand_dist_f)

        # zonal control state dataset
        self.zonal_control_state = {}


    def receive_status_update(self, vid, simulation_time, list_finished_VRL, force_update=True):
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
        super().receive_status_update(vid, simulation_time, list_finished_VRL, force_update)

        # check which vehicles are in terminus
        line_id = next(iter(self.PT_lines))
        terminus_id = self.PT_lines[line_id].terminus_id
        terminus_node = self.station_dict[terminus_id].street_network_node_id
        for vrl in list_finished_VRL:
            if vrl.destination_pos == terminus_node:
                self.list_veh_in_terminus.append(vid)

                # TODO: check whether this matches the schedule; if not, raise an error

    def _call_time_trigger_request_batch(self, simulation_time):
        """ this function first triggers the upper level batch optimisation
        based on the optimisation solution offers to newly assigned requests are created in the second step with following logic:
        declined requests will receive an empty dict
        unassigned requests with a new assignment try in the next opt-step dont get an answer
        new assigned request will receive a non empty offer-dict

        a retry is only made, if "user_max_wait_time_2" is given

        every request as to answer to an (even empty) offer to be deleted from the system!

        :param simulation_time: current time in simulation
        :return: dictionary rid -> offer for each unassigned request, that will recieve an answer. (offer: dictionary with plan specific entries; empty if no offer can be made)
        :rtype: dict
        """
        # TODO: remove super() and alter the veh list to only flexible route for flexible demand; only fixed vehicles for fixed demand
        RidePoolingBatchOptimizationFleetControlBase._call_time_trigger_request_batch(simulation_time)

        # embed()
        # Zonal: make use of max_wait_time_2 to put decision on hold (in case new zonal vehicle plans assigned)
        rid_to_offers = {}
        if self.sim_time % self.optimisation_time_step == 0:
            new_unassigned_requests_2 = {}
            # rids to be assigned in first try
            for rid in self.unassigned_requests_1.keys():
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                prq = self.rq_dict[rid]
                if assigned_vid is None:
                    if self.max_wait_time_2 is not None and self.max_wait_time_2 > 0:  # retry with new waiting time constraint (no offer returned)
                        new_unassigned_requests_2[rid] = 1
                        self.RPBO_Module.delRequest(rid)
                        _, earliest_pu, _ = prq.get_o_stop_info()
                        new_latest_pu = earliest_pu + self.max_wait_time_2
                        self.change_prq_time_constraints(simulation_time, rid, new_latest_pu)
                        self.RPBO_Module.add_new_request(rid, prq)
                    else:  # no retry, rid declined
                        self._create_user_offer(prq, simulation_time)
                else:
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            for rid in self.unassigned_requests_2.keys():  # check second try rids
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                if assigned_vid is None:  # decline
                    prq = self.rq_dict[rid]
                    _, earliest_pu, _ = prq.get_o_stop_info()
                    new_latest_pu = earliest_pu + self.max_wait_time_2

                    if self.sim_time > new_latest_pu: # if max_wait_time_2 is exceeded, decline
                        self._create_user_offer(self.rq_dict[rid], simulation_time)
                    else: # otherwise, add it back to the list (to check in next iteration)
                        new_unassigned_requests_2[rid] = 1
                else:
                    prq = self.rq_dict[rid]
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            self.unassigned_requests_1 = {}
            self.unassigned_requests_2 = new_unassigned_requests_2  # retry rids
            LOG.debug("end of opt:")
            LOG.debug("unassigned_requests_2 {}".format(self.unassigned_requests_2))
            LOG.debug("offers: {}".format(rid_to_offers))

        # Prepare dataset for zonal control
        # ---------------------------------
        self.zonal_control_state["simulation_time"] = simulation_time
        self.zonal_control_state["unassigned_requests_no"] = len(self.unassigned_requests_2)
        self.zonal_control_state["demand_forecast"] = self.get_closest_demand_forecast_time(simulation_time)


    def get_closest_demand_forecast_time(self, simulation_time) -> float:
        """
        This function returns the closest demand forecast (from self.demand_dist_df) to the current simulation time
        :param simulation_time: current simulation time
        :type simulation_time: float
        :return: closest demand forecast
        :rtype: float
        """

        # self.demand_dist_df is a dataframe with columns "seconds" and "count", every 900s
        return self.demand_dist_df["count"].iloc[np.argmin(np.abs(self.demand_dist_df["seconds"] - simulation_time))]
