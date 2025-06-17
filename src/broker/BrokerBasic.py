# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import typing as tp

# additional module imports (> requirements)
# ------------------------------------------


# src imports
# -----------
from src.broker.BrokerBase import BrokerBase
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.pt.PTControlBase import PTControlBase
    from src.demand.demand import Demand
    from src.routing.NetworkBase import NetworkBase
    from src.demand.TravelerModels import RequestBase
    from src.simulation.Legs import VehicleRouteLeg

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_BrokerBasic = {
    "doc" : "this class is the basic broker class, it only forwards the requests to the amod operators",
    "inherit" : BrokerBase,
    "input_parameters_mandatory": ["n_amod_op", "amod_operators"],
    "input_parameters_optional": ["pt_operator", "demand", "routing_engine", "scenario_parameters"],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class BrokerBasic(BrokerBase):
    def __init__(
        self, 
        n_amod_op: int, 
        amod_operators: tp.List['FleetControlBase'],
        pt_operator: 'PTControlBase' = None,
        demand: 'Demand' = None, 
        routing_engine: 'NetworkBase' = None,
        scenario_parameters: dict = None
    ):
        """
        The general attributes for the broker are initialized.

        Args:
            n_amod_op (int): number of AMoD operators
            amod_operators (tp.List['FleetControlBase']): list of AMoD operators
            pt_operator (PTControlBase): PT operator
            demand (Demand): demand object
            routing_engine (NetworkBase): routing engine
            scenario_parameters (dict): scenario parameters
        """
        super().__init__(n_amod_op, amod_operators, pt_operator, demand, routing_engine, scenario_parameters)

    def inform_network_travel_time_update(self, sim_time: int):
        """This method informs the broker that the network travel times have been updated.
        This information is forwarded to the amod operators.
        """
        for op_id in range(self.n_amod_op):
            self.amod_operators[op_id].inform_network_travel_time_update(sim_time)
    
    def inform_request(
        self,
        rid: int,
        rq_obj: 'RequestBase',
        sim_time: int
    ):
        """This method informs the broker that a new request has been made.
        This information is forwarded to the amod operators.
        """
        for op_id in range(self.n_amod_op):
            LOG.debug(f"Request {rid}: To operator {op_id} ...")
            self.amod_operators[op_id].user_request(rq_obj, sim_time)    

    def collect_offers(self, rid: int, sim_time: int = None) -> tp.Dict[int, 'RequestBase']:
        """This method collects the offers from the amod operators.
        The return value is a list of tuples, where each tuple contains the operator id, the offer, and the simulation time.
        """
        amod_offers = {}
        for op_id in range(self.n_amod_op):
            amod_offer = self.amod_operators[op_id].get_current_offer(rid)
            LOG.debug(f"amod offer {amod_offer}")
            if amod_offer is not None:
                amod_offers[op_id] = amod_offer
        return amod_offers

    def inform_user_booking(
        self, 
        rid: int,
        rq_obj: 'RequestBase',
        sim_time: int,
        chosen_operator: int,
    ) -> tp.List[tuple[int, 'RequestBase']]:
        """This method informs the broker that the user has booked a trip.
        """
        amod_confirmed_rids = []
        for i, operator in enumerate(self.amod_operators):
            if i != chosen_operator:
                operator.user_cancels_request(rid, sim_time)
            else:
                operator.user_confirms_booking(rid, sim_time)
                amod_confirmed_rids.append((rid, rq_obj))
        return amod_confirmed_rids

    def inform_user_leaving_system(
        self,
        rid: int,
        sim_time: int
    ):
        """This method informs the broker that the user is leaving the system.
        """
        for _, operator in enumerate(self.amod_operators):
            operator.user_cancels_request(rid, sim_time)

    def inform_waiting_request_cancellations(
        self,
        chosen_operator: int,
        rid: int,
        sim_time: int
    ):
        """This method informs the operators that the waiting requests have been cancelled.
        """
        self.amod_operators[chosen_operator].user_cancels_request(rid, sim_time)
    
    def acknowledge_user_boarding(
        self,
        op_id: int,
        rid: int,
        vid: int,
        boarding_time: int
    ):
        """This method acknowledges the user boarding.
        """
        self.amod_operators[op_id].acknowledge_boarding(rid, vid, boarding_time)

    def acknowledge_user_alighting(
        self,
        op_id: int,
        rid: int,
        vid: int,
        alighting_time: int,
    ):
        """This method acknowledges the user alighting.
        """
        self.amod_operators[op_id].acknowledge_alighting(rid, vid, alighting_time)

    def receive_status_update(
        self, 
        op_id: int,
        vid: int,
        sim_time: int,
        passed_VRL: tp.List['VehicleRouteLeg'], 
        force_update_plan: bool,
    ):
        """This method receives the status update of the vehicles.
        """
        self.amod_operators[op_id].receive_status_update(vid, sim_time, passed_VRL, force_update_plan)