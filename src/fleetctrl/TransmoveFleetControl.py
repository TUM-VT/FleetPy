import logging
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.simulation.Offers import TravellerOffer
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000

INPUT_PARAMETERS_TransmoveFleetControl = {
    "doc": """
        The fleet control module for the transmove project. 
        Ride pooling optimisation is called after every optimisation_time_step and offers are created in the time_trigger function.
        All requests that are not assigned to a vehicle in the first try will be created a hypothetical offer.
        """,
    "inherit": "RidePoolingBatchOptimizationFleetControlBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class TransmoveFleetControl(RidePoolingBatchOptimizationFleetControlBase):
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                 dir_names, op_charge_depot_infra=None, list_pub_charging_infra=[]):
        """
        The fleet control module for the transmove project.
        Ride pooling optimisation is called after every optimisation_time_step and offers are created in the time_trigger function.
        All requests that are not assigned to a vehicle in the first try will be created a hypothetical offer.

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
        :param dirnames: directories for output and input
        :type dirnames: dict
        :param op_charge_depot_infra: reference to a OperatorChargingAndDepotInfrastructure class (optional) (unique for each operator)
        :type OperatorChargingAndDepotInfrastructure: OperatorChargingAndDepotInfrastructure
        :param list_pub_charging_infra: list of PublicChargingInfrastructureOperator classes (optional) (accesible for all agents)
        :type list_pub_charging_infra: list of PublicChargingInfrastructureOperator
        """
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names=dir_names, op_charge_depot_infra=op_charge_depot_infra,
                         list_pub_charging_infra=list_pub_charging_infra)
        self.hypothetical_alighting_requests: dict = {}
        self.incoming_requests: dict = {}

        self.max_wt = scenario_parameters[G_OP_MAX_WT]
        self.bt = scenario_parameters[G_OP_CONST_BT]
        self.max_dtf = scenario_parameters[G_OP_MAX_DTF]

    def user_request(self, rq, sim_time):
        """This method is triggered for a new incoming request. It generally adds the rq to the database. It has to
        return an offer to the user. An empty dictionary means no offer is made!

        :param rq: request object containing all request information
        :type rq: RequestDesign
        :param sim_time: current simulation time
        :type sim_time: float
        :return: offer
        :rtype: dict
        """
        self.sim_time = sim_time
        super().user_request(rq, sim_time)
        self.incoming_requests[rq.get_rid_struct()] = 1
        return {}

    def _call_time_trigger_request_batch(self, simulation_time):
        """ this function first triggers the upper level batch optimisation
        based on the optimisation solution offers to newly assigned requests are created in the second step with following logic:
        declined requests will recieve an empty dict
        unassigned requests with a new assignment try in the next opt-step dont get an answer
        new assigned request will recieve a non empty offer-dict

        a retry is only made, if "user_max_wait_time_2" is given

        every request as to answer to an (even empty) offer to be deleted from the system!

        :param simulation_time: current time in simulation
        :return: dictionary rid -> offer for each unassigned request, that will recieve an answer. (offer: dictionary with plan specific entries; empty if no offer can be made)
        :rtype: dict
        """
        super()._call_time_trigger_request_batch(simulation_time)
        if self.sim_time % self.optimisation_time_step == 0:
            # rids to be assigned in first try
            for rid in self.incoming_requests.keys():
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                prq = self.rq_dict[rid]
                if assigned_vid is None:  # request not assigned to a vehicle
                    # create hypothetical offer for the request
                    self._create_hypothetical_offer(prq)
                    # delete it from self.rq_dict and RPBO_Module
                    # try:
                    #     del self.rq_dict[rid]
                    # except KeyError:
                    #     pass
                    self.RPBO_Module.delete_request(rid)
                else:
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
                    self.user_confirms_booking(rid=rid, simulation_time=simulation_time)
            # initialize incoming requests
            self.incoming_requests = {}
            LOG.debug("end of opt:")
            LOG.debug(f"hypothetical alighting requests: {self.hypothetical_alighting_requests}")

    def _create_hypothetical_offer(self, prq):
        """This method creates a hypothetical offer for the request and adds the offer information into the self.demand.rq_db[rid].
        The hypothetical offer consists of three parts:
        1. the maximum customer waiting time
        2. the direct travel time
        3. the maximum detour time

        :param rq: request object containing all request information
        :type rq: PlanRequest
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        rid = prq.get_rid_struct()
        rq_time = prq.rq_time
        direct_tt = prq.init_direct_tt
        pu_time = rq_time + self.max_wt
        do_time = pu_time + self.bt + direct_tt * (1 + self.max_dtf / 100)
        fare = 10000
        # offer = {
        #     "t_access": 0,
        #     "t_egress": 0,
        #     "t_wait": self.max_wt,
        #     "pu_time": pu_time,
        #     "do_time": do_time,
        #     "service_vid": -1,
        # }
        extendInfo = {
            "service_opid": 0,
            "service_vid": -1,
            "pu_time": pu_time,
            "do_time": do_time,
        }
        offer = TravellerOffer(rid, self.op_id, self.max_wt, do_time - pu_time, fare, additional_parameters=extendInfo)
        prq.offer = offer
        # G_PRQS_IN_VEH or G_PRQS_LOCKED, which one is better?
        prq.status = G_PRQS_IN_VEH
        prq.set_reservation_flag(False)
        self.hypothetical_alighting_requests[rid] = offer




