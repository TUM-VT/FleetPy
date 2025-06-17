# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
from datetime import datetime, timedelta
import typing as tp
import pandas as pd
# additional module imports (> requirements)
# ------------------------------------------


# src imports
# -----------
from src.broker.BrokerBasic import BrokerBasic
from src.simulation.Offers import MultimodalOffer
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.fleetctrl.planning.PlanRequest import PlanRequest
    from src.pt.PTControlBase import PTControlBase
    from src.demand.demand import Demand
    from src.routing.NetworkBase import NetworkBase
    from src.demand.TravelerModels import RequestBase, BasicMultimodalRequest
    from src.simulation.Offers import TravellerOffer, PTOffer

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_PTBroker = {
    "doc" : "this class represents a broker platform which handles multi-modal requests",
    "inherit" : BrokerBasic,
    "input_parameters_mandatory": ["n_amod_op", "amod_operators", "pt_operator", "demand", "routing_engine", "scenario_parameters"],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PTBroker(BrokerBasic):
    def __init__(
        self, 
        n_amod_op: int, 
        amod_operators: tp.List['FleetControlBase'], 
        pt_operator: 'PTControlBase', 
        demand: 'Demand', 
        routing_engine: 'NetworkBase',
        scenario_parameters: dict
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

        self.pt_operator_id: int = self.pt_operator.pt_operator_id
        
        self.sim_start_datetime: datetime = None
        self._set_sim_start_datetime(scenario_parameters.get(G_SIM_START_DATE))
    
    def inform_request(
        self,
        rid: int,
        rq_obj: 'RequestBase',
        sim_time: int,
    ):
        """This method informs the broker that a new request has been made. 
        Based on the request modal state, the broker will create the appropriate sub-requests 
        and inform the operators.

        Args:
            rid (int): parent request id
            rq_obj (RequestBase): request object
            sim_time (int): simulation time
        """
        parent_modal_state: RQ_MODAL_STATE = rq_obj.get_modal_state()

        # 1. query the PT operator for the PT travel costs when request modal state value is greater than MONOMODAL (0)
        if parent_modal_state.value > RQ_MODAL_STATE.MONOMODAL.value:
            _ = self._inform_pt_sub_request(
                                            rq_obj, 
                                            RQ_SUB_TRIP_ID.PT.value, 
                                            rq_obj.get_origin_node(), 
                                            rq_obj.get_destination_node(), 
                                            rq_obj.earliest_start_time, 
                                            parent_modal_state,
                                            )
        
        # 2.1 pure AMoD request or PT request
        if parent_modal_state == RQ_MODAL_STATE.MONOMODAL or parent_modal_state == RQ_MODAL_STATE.PT:
            self._process_inform_monomodal_request(rid, rq_obj, sim_time, parent_modal_state)
        
        # 2.2 AMoD as firstmile request
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTMILE:
            self._process_inform_firstmile_request(rq_obj, sim_time, parent_modal_state)
        
        # 2.3 AMoD as lastmile request
        elif parent_modal_state == RQ_MODAL_STATE.LASTMILE:
            self._process_inform_lastmile_request(rid, rq_obj, sim_time, parent_modal_state)
        
        # 2.4 AMoD as firstlastmile request
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            self._process_inform_firstlastmile_request(rid, rq_obj, sim_time, parent_modal_state)
        
        else:
            raise ValueError(f"Invalid modal state: {parent_modal_state}")

    def collect_offers(
        self, 
        rid: int,
        sim_time: int,
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method collects the offers from the operators.

        Args:
            rid (int): parent request id
            sim_time (int): simulation time

        Returns:
            tp.Dict[int, 'TravellerOffer']: a dictionary of offers from the operators
        """
        # get parent request modal state
        parent_rq_obj: 'RequestBase' = self.demand[rid]
        parent_modal_state: RQ_MODAL_STATE = parent_rq_obj.get_modal_state()
        offers: tp.Dict[int, 'TravellerOffer'] = {}
        LOG.debug(f"Collecting offers for request {rid} with modal state {parent_modal_state}")

        # 1. collect PT offers for multimodal requests
        if parent_modal_state.value > RQ_MODAL_STATE.MONOMODAL.value:
            pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.PT.value}"
            pt_offer = self.pt_operator.get_current_offer(pt_rid_struct)
            LOG.debug(f"pt offer {pt_offer}")
            
            if pt_offer is not None and not pt_offer.service_declined():
                offers[self.pt_operator_id] = pt_offer
                # register the pt offer in the sub-request
                self.demand[pt_rid_struct].receive_offer(self.pt_operator_id, pt_offer, None)

        # 2.1 collect AMoD offers for MONOMODAL and PT requests
        if parent_modal_state == RQ_MODAL_STATE.MONOMODAL or parent_modal_state == RQ_MODAL_STATE.PT:
            offers = self._process_collect_monomodal_offers(rid, parent_modal_state, offers)

        # 2.2 collect AMoD offers for FIRSTMILE requests
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTMILE:
            offers = self._process_collect_firstmile_offers(rid, parent_rq_obj, parent_modal_state, offers)

        # 2.3 collect AMoD offers for LASTMILE requests
        elif parent_modal_state == RQ_MODAL_STATE.LASTMILE:
            offers = self._process_collect_lastmile_offers(rid, parent_modal_state, offers)

        # 2.4 collect AMoD offers for FIRSTLASTMILE requests
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            offers = self._process_collect_firstlastmile_offers_3phases(rid, parent_rq_obj, parent_modal_state, offers, sim_time)
        
        else:
            raise ValueError(f"Invalid modal state: {parent_modal_state}")

        return offers

    def inform_user_booking(
        self,
        rid: int,
        rq_obj: 'RequestBase',
        sim_time: int,
        chosen_operator: tp.Union[int, tuple],
    ) -> tp.List[tuple[int, 'RequestBase']]:
        """This method informs the broker that the user has booked a trip.
        """
        amod_confirmed_rids = []

        parent_modal_state: RQ_MODAL_STATE = rq_obj.get_modal_state()

        # 1. Pure PT offer has been selected
        if chosen_operator == self.pt_operator_id:
            amod_confirmed_rids.append((rid, rq_obj))

            # inform all AMoD operators that the request is cancelled
            self.inform_user_leaving_system(rid, sim_time)

            # inform PT operator that the request is confirmed
            pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.PT.value}"
            pt_sub_rq_obj: 'BasicMultimodalRequest' = self.demand[pt_rid_struct]
            self.pt_operator.user_confirms_booking(pt_sub_rq_obj, None) 
            
        # 2. AMoD involved offer has been selected
        else:
            if parent_modal_state == RQ_MODAL_STATE.MONOMODAL or parent_modal_state == RQ_MODAL_STATE.PT:
                for i, operator in enumerate(self.amod_operators):
                    if i != chosen_operator:  # Non-multimodal requests: the chosen operator has the data type int
                        operator.user_cancels_request(rid, sim_time)
                    else:
                        operator.user_confirms_booking(rid, sim_time)
                        amod_confirmed_rids.append((rid, rq_obj))
            
            elif parent_modal_state.value > RQ_MODAL_STATE.MONOMODAL.value and parent_modal_state.value < RQ_MODAL_STATE.PT.value:
                for operator_id, sub_trip_id in chosen_operator:
                    if operator_id == self.pt_operator_id:
                        # inform the pt operator that the request is confirmed
                        pt_rid_struct: str = f"{rid}_{sub_trip_id}"
                        pt_sub_rq_obj: 'BasicMultimodalRequest' = self.demand[pt_rid_struct]

                        if parent_modal_state == RQ_MODAL_STATE.LASTMILE:
                            previous_amod_operator_id = None  # no previous amod operator
                        else:  # firstmile or firstlastmile
                            previous_amod_operator_id: int = chosen_operator[0][0]  # the first amod operator
                        
                        self.pt_operator.user_confirms_booking(pt_sub_rq_obj, previous_amod_operator_id)
                      
                    else:
                        # inform the amod operator that the request is confirmed
                        amod_rid_struct: str = f"{rid}_{sub_trip_id}"
                        for i, operator in enumerate(self.amod_operators):
                            if i != operator_id: 
                                operator.user_cancels_request(amod_rid_struct, sim_time)
                            else:
                                operator.user_confirms_booking(amod_rid_struct, sim_time)

                amod_confirmed_rids.append((rid, rq_obj))
            
            else:
                raise ValueError(f"Invalid modal state: {parent_modal_state}")

        return amod_confirmed_rids

    def inform_user_leaving_system(
        self,
        rid: int,
        sim_time: int
    ):
        """This method informs the broker that the user is leaving the system.
        """
        rq_obj: 'RequestBase' = self.demand[rid]
        parent_modal_state: RQ_MODAL_STATE = rq_obj.get_modal_state()
        
        if parent_modal_state == RQ_MODAL_STATE.MONOMODAL or parent_modal_state == RQ_MODAL_STATE.PT:
            for _, operator in enumerate(self.amod_operators):
                operator.user_cancels_request(rid, sim_time)
        
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTMILE:
            fm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_AMOD.value}"
            for _, operator in enumerate(self.amod_operators):
                operator.user_cancels_request(fm_amod_rid_struct, sim_time)
        
        elif parent_modal_state == RQ_MODAL_STATE.LASTMILE:
            lm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.LM_AMOD.value}"
            for _, operator in enumerate(self.amod_operators):
                operator.user_cancels_request(lm_amod_rid_struct, sim_time)
        
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
            flm_amod_rid_struct_1: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_1.value}"
            for _, operator in enumerate(self.amod_operators):
                operator.user_cancels_request(flm_amod_rid_struct_0, sim_time)
                operator.user_cancels_request(flm_amod_rid_struct_1, sim_time)
        
        else:
            raise ValueError(f"Invalid modal state: {parent_modal_state}")

    def inform_waiting_request_cancellations(
        self,
        chosen_operator: int,
        rid: int,
        sim_time: int
    ):
        """This method informs the operators that the waiting requests have been cancelled.
        """
        rq_obj: 'RequestBase' = self.demand[rid]
        parent_modal_state: RQ_MODAL_STATE = rq_obj.get_modal_state()

        if chosen_operator == self.pt_operator_id:
            return
        
        if parent_modal_state == RQ_MODAL_STATE.MONOMODAL or parent_modal_state == RQ_MODAL_STATE.PT:
            self.amod_operators[chosen_operator].user_cancels_request(rid, sim_time)
        
        elif parent_modal_state.value > RQ_MODAL_STATE.MONOMODAL.value and parent_modal_state.value < RQ_MODAL_STATE.PT.value:
            for operator_id, sub_trip_id in chosen_operator:
                if operator_id == self.pt_operator_id:
                    continue
                amod_rid_struct: str = f"{rid}_{sub_trip_id}"
                operator_id = int(operator_id)
                self.amod_operators[operator_id].user_cancels_request(amod_rid_struct, sim_time)
        
        else:
            raise ValueError(f"Invalid modal state: {parent_modal_state}")

    def query_street_node_pt_travel_costs_1to1(
        self, 
        o_node: int, 
        d_node: int, 
        est: int, 
        max_transfers: int=-1
    ) -> tp.Optional[tp.Tuple[int, int, int, int, tp.Dict[str, tp.Any]]]:
        """This method queries the pt travel costs between two street nodes at a given datetime.
        The pt station ids will be the closest pt station ids to the street nodes.

        Args:
            o_node (int): The origin street node id.
            d_node (int): The destination street node id.
            est (int): The earliest start time of the request at the origin street node in seconds.
            max_transfers (int): The maximum number of transfers allowed in the journey, -1 for no limit.
        
        Returns:
            tp.Optional[tp.Tuple[int, int, int, int, tp.Dict[str, tp.Any]]]:
                Returns a tuple containing:
                (source_pt_station_id, t_source_walking, target_pt_station_id, t_target_walking, pt_journey_plan_dict)
                if a public transport journey plan is found.
                Returns None if no public transport journey plan is available.
        """
        source_pt_station_id, t_source_walk = self.pt_operator.find_closest_pt_station(o_node)
        target_pt_station_id, t_target_walk = self.pt_operator.find_closest_pt_station(d_node)
        source_station_departure_seconds: int = est + t_source_walk
        source_station_departure_datetime: datetime = self._get_current_datetime(source_station_departure_seconds)
        pt_journey_plan_dict: tp.Union[tp.Dict[str, tp.Any], None] = self.pt_operator.return_fastest_pt_journey_1to1(
                                                                                source_pt_station_id, 
                                                                                target_pt_station_id, 
                                                                                source_station_departure_datetime,
                                                                                max_transfers,
                                                                                detailed=False,
                                                                                )
        if pt_journey_plan_dict is None:
            return None
        else:
            return source_pt_station_id, t_source_walk, target_pt_station_id, t_target_walk, pt_journey_plan_dict

    def _set_sim_start_datetime(self, sim_start_date: str):
        """This method sets the simulation start date.
        Converts the date string (format YYYYMMDD) to a datetime object.
        """
        if type(sim_start_date) is not str:
            sim_start_date = str(int(sim_start_date))
        self.sim_start_datetime = datetime.strptime(sim_start_date, "%Y%m%d")

    def _get_current_datetime(self, sim_time_in_seconds: int) -> datetime:
        """This method returns the current datetime.
        """
        return self.sim_start_datetime + timedelta(seconds=int(sim_time_in_seconds))
    
    def _inform_amod_sub_request(
        self, 
        rq_obj: 'RequestBase', 
        sub_trip_id: int, 
        mod_o_node: int, 
        mod_d_node: int, 
        mod_start_time: int,
        parent_modal_state: RQ_MODAL_STATE, 
        op_id: int, 
        sim_time: int
    ):
        """
        This method informs the AMoD operators that a new sub-request has been made.

        Args:
            rq_obj (RequestBase): the parent request object
            sub_trip_id (int): the sub-trip id
            mod_o_node (int): the origin node of the sub-request
            mod_d_node (int): the destination node of the sub-request
            mod_start_time (int): the start time of the sub-request
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        amod_sub_rq_obj: 'RequestBase' = self.demand.create_sub_requests(
                                                                        rq_obj,
                                                                        sub_trip_id, 
                                                                        mod_o_node,
                                                                        mod_d_node,
                                                                        mod_start_time, 
                                                                        parent_modal_state
                                                                        )
        LOG.debug(f"AMoD sub-request {amod_sub_rq_obj.get_rid_struct()} with modal state {parent_modal_state}: To operator {op_id} ...")
        self.amod_operators[op_id].user_request(amod_sub_rq_obj, sim_time)

    def _inform_pt_sub_request(
        self, 
        rq_obj: 'RequestBase', 
        sub_trip_id: int, 
        mod_o_node: int, 
        mod_d_node: int, 
        mod_start_time: int,
        parent_modal_state: RQ_MODAL_STATE,
        previous_amod_operator_id: int = None,
    ) -> tp.Optional[int]:
        """
        This method informs the PT operator that a new sub-request has been made.

        Args:
            rq_obj (RequestBase): the parent request object
            sub_trip_id (int): the sub_trip id
            mod_o_node (int): the origin node of the sub-request
            mod_d_node (int): the destination node of the sub-request
            mod_start_time (int): the start time of the sub-request
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
            previous_amod_operator_id (int): the id of the previous amod operator

        Returns:
            t_d_node_arrival (tp.Optional[int]):
                the pt arrival time of the sub-request at the destination street node 
                or None if the pt travel costs are not available
        """
        pt_sub_rq_obj: 'RequestBase' = self.demand.create_sub_requests(
                                                                    rq_obj, 
                                                                    sub_trip_id, 
                                                                    mod_o_node, 
                                                                    mod_d_node, 
                                                                    mod_start_time, 
                                                                    parent_modal_state,
                                                                    )
        LOG.debug(f"PT sub-request {pt_sub_rq_obj.get_rid_struct()} with modal state {parent_modal_state}: To PT operator {self.pt_operator_id} ...")
        
        costs_info = self.query_street_node_pt_travel_costs_1to1(
                                                                pt_sub_rq_obj.get_origin_node(), 
                                                                pt_sub_rq_obj.get_destination_node(), 
                                                                pt_sub_rq_obj.earliest_start_time,
                                                                pt_sub_rq_obj.get_max_transfers(),
                                                                )
        
        if costs_info is not None:
            source_pt_station_id, t_source_walk, target_pt_station_id, t_target_walk, pt_journey_plan_dict = costs_info
        else:
            source_pt_station_id = None
            t_source_walk = None
            target_pt_station_id = None
            t_target_walk = None
            pt_journey_plan_dict = None
            LOG.debug(f"PT sub-request {pt_sub_rq_obj.get_rid_struct()} with modal state {parent_modal_state}: No offer has been found!")
        
        pt_rid_struct: str =  pt_sub_rq_obj.get_rid_struct()

        self.pt_operator.record_pt_offer_db(
                                        rid_struct = pt_rid_struct,
                                        operator_id = self.pt_operator_id,
                                        source_station_id = source_pt_station_id,
                                        target_station_id = target_pt_station_id,
                                        source_walking_time = t_source_walk,
                                        target_walking_time = t_target_walk,
                                        pt_journey_plan_dict = pt_journey_plan_dict,
                                        previous_amod_operator_id = previous_amod_operator_id,
                                        )
        if pt_journey_plan_dict is not None:
            t_d_node_arrival: int = self.pt_operator.get_current_offer(pt_rid_struct, previous_amod_operator_id).destination_node_arrival_time
            return t_d_node_arrival
        else:
            return None
        
    def _process_inform_monomodal_request(
        self,
        rid: int,
        rq_obj: 'RequestBase',
        sim_time: int,
        parent_modal_state: RQ_MODAL_STATE,
    ):
        """This method processes the new monomodal request.

        Args:
            rid (int): the request id
            rq_obj (RequestBase): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        for op_id in range(self.n_amod_op):
            LOG.debug(f"AMoD Request {rid} with modal state {parent_modal_state}: To operator {op_id} ...")
            self.amod_operators[op_id].user_request(rq_obj, sim_time)

    def _process_inform_firstmile_request(
        self,
        rq_obj: 'BasicMultimodalRequest',
        sim_time: int,
        parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTMILE,
    ):
        """This method processes the new firstmile request.

        Args:
            rq_obj (BasicMultimodalRequest): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        # get the transfer station id and its closest pt station
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node, _ = self.pt_operator.find_closest_street_node(transfer_station_ids[0])
        # create sub-request for AMoD
        for op_id in range(self.n_amod_op):
            self._inform_amod_sub_request(
                                        rq_obj, 
                                        RQ_SUB_TRIP_ID.FM_AMOD.value, 
                                        rq_obj.get_origin_node(), 
                                        transfer_street_node, 
                                        rq_obj.earliest_start_time, 
                                        parent_modal_state,
                                        op_id,
                                        sim_time,
                                        )
        # PT sub-request can be created later when the AMoD operator has accepted the request and predicted the arrival time at the pt station

    def _process_inform_lastmile_request(
        self,
        rid: int,
        rq_obj: 'BasicMultimodalRequest',
        sim_time: int,
        parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.LASTMILE,
    ):
        """This method processes the new lastmile request.

        Args:
            rid (int): the request id
            rq_obj (BasicMultimodalRequest): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        # get the transfer station id and its closest pt station
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node, _ = self.pt_operator.find_closest_street_node(transfer_station_ids[0])
        # create sub-request for PT
        lm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(
                                                                    rq_obj, 
                                                                    RQ_SUB_TRIP_ID.LM_PT.value, 
                                                                    rq_obj.get_origin_node(),
                                                                    transfer_street_node, 
                                                                    rq_obj.earliest_start_time,
                                                                    parent_modal_state,
                                                                    )
        if lm_pt_arrival is not None:
            # create sub-request for AMoD
            for op_id in range(self.n_amod_op):
                self._inform_amod_sub_request(
                                            rq_obj,
                                            RQ_SUB_TRIP_ID.LM_AMOD.value,
                                            transfer_street_node,
                                            rq_obj.get_destination_node(), 
                                            lm_pt_arrival,
                                            parent_modal_state,
                                            op_id,
                                            sim_time,
                                            )
        else:
            LOG.info(f"PT offer is not available for sub_request {rid}_{RQ_SUB_TRIP_ID.LM_PT.value}")

    def _process_inform_firstlastmile_request(
        self,
        rid: int,
        rq_obj: 'BasicMultimodalRequest',
        sim_time: int,
        parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTLASTMILE,
    ):
        """This method processes the new firstlastmile request.

        Args:
            rid (int): the request id
            rq_obj (BasicMultimodalRequest): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        # get the transfer station ids and their closest pt stations
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node_0, _ = self.pt_operator.find_closest_street_node(transfer_station_ids[0])
        transfer_street_node_1, _ = self.pt_operator.find_closest_street_node(transfer_station_ids[1])
        
        # create sub-request for AMoD
        for op_id in range(self.n_amod_op):
            self._inform_amod_sub_request(
                                        rq_obj,
                                        RQ_SUB_TRIP_ID.FLM_AMOD_0.value, 
                                        rq_obj.get_origin_node(),
                                        transfer_street_node_0, 
                                        rq_obj.earliest_start_time,
                                        parent_modal_state,
                                        op_id,
                                        sim_time,
                                        )
            flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
            flm_amod_sub_rq_obj_0: 'BasicMultimodalRequest' = self.demand[flm_amod_rid_struct_0]
            # create sub-request for PT
            # estimate the dropoff time of the amod sub-request
            estimated_amod_dropoff_time: int = self._estimate_amod_dropoff_time(op_id, flm_amod_sub_rq_obj_0, "latest")
            # estimate the earliest start time of the pt sub-request
            flm_est_pt_mod: int = estimated_amod_dropoff_time + self.amod_operators[op_id].const_bt
            # create the pt sub-request
            flm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(
                                                                        rq_obj,
                                                                        RQ_SUB_TRIP_ID.FLM_PT.value, 
                                                                        transfer_street_node_0,
                                                                        transfer_street_node_1, 
                                                                        flm_est_pt_mod,
                                                                        parent_modal_state,
                                                                        op_id,
                                                                        )
            # create sub-request for the same AMoD operator
            if flm_pt_arrival is None:
                raise ValueError(f"PT offer is not available for sub_request {rid}_{RQ_SUB_TRIP_ID.FLM_PT.value}")
            else:
                self._inform_amod_sub_request(
                                            rq_obj,
                                            RQ_SUB_TRIP_ID.FLM_AMOD_1.value, 
                                            transfer_street_node_1,
                                            rq_obj.get_destination_node(), 
                                            flm_pt_arrival,
                                            parent_modal_state,
                                            op_id,
                                            sim_time,
                                            )
    
    def _process_collect_monomodal_offers(
        self,
        rid: int,
        parent_modal_state: RQ_MODAL_STATE,
        offers: tp.Dict[int, 'TravellerOffer'],
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of monomodal offers.
        """
        for amod_op_id in range(self.n_amod_op):
            amod_offer = self.amod_operators[amod_op_id].get_current_offer(rid)
            LOG.debug(f"Collecting amod offer for request {rid} with modal state {parent_modal_state} from operator {amod_op_id}: {amod_offer}")
            if amod_offer is not None and not amod_offer.service_declined():
                offers[amod_op_id] = amod_offer
        return offers
    
    def _process_collect_firstmile_offers(
        self,
        rid: int,
        parent_rq_obj: 'BasicMultimodalRequest',
        parent_modal_state: RQ_MODAL_STATE,
        offers: tp.Dict[int, 'TravellerOffer'],
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstmile offers.
        """
        fm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_AMOD.value}"
        fm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_PT.value}"

        # get the transfer station id and its closest pt station
        transfer_station_ids: tp.List[str] = parent_rq_obj.get_transfer_station_ids()
        transfer_street_node, _ = self.pt_operator.find_closest_street_node(transfer_station_ids[0])

        # get firstmile amod offer
        for amod_op_id in range(self.n_amod_op):
            fm_amod_offer = self.amod_operators[amod_op_id].get_current_offer(fm_amod_rid_struct)
            LOG.debug(f"Collecting fm_amod offer for request {fm_amod_rid_struct} from operator {amod_op_id}: {fm_amod_offer}")

            if fm_amod_offer is not None and not fm_amod_offer.service_declined():
                # register the amod offer in the sub-request
                self.demand[fm_amod_rid_struct].receive_offer(amod_op_id, fm_amod_offer, None)
                fm_est_pt_mod: int = self._determine_est_pt_mod(
                                                                parent_rq_obj,
                                                                amod_op_id,
                                                                fm_amod_offer,
                                                                )
                # query firstmile pt travel costs
                fm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(
                                                                            parent_rq_obj,
                                                                            RQ_SUB_TRIP_ID.FM_PT.value,
                                                                            transfer_street_node,
                                                                            parent_rq_obj.get_destination_node(),
                                                                            fm_est_pt_mod,
                                                                            parent_modal_state,
                                                                            amod_op_id,
                                                                            )
                
                if fm_pt_arrival is not None:
                    # get pt offer
                    fm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(fm_pt_rid_struct, amod_op_id)
                    LOG.debug(f"Collecting fm_pt offer for request {fm_pt_rid_struct} from operator {self.pt_operator_id}: {fm_pt_offer}")
                    
                    # register the pt offer in the sub-request
                    self.demand[fm_pt_rid_struct].receive_offer(self.pt_operator_id, fm_pt_offer, None)

                    # update the amod offer with the latest dropoff time
                    new_t_do_latest: int = self._determine_amod_latest_dropoff_time(
                                                                                parent_rq_obj,
                                                                                fm_amod_offer,
                                                                                fm_pt_offer.get(G_OFFER_WAIT),
                                                                                )
                    sub_prq_obj: 'PlanRequest' = self.amod_operators[amod_op_id].rq_dict[fm_amod_rid_struct]
                    sub_prq_obj.set_new_dropoff_time_constraint(new_t_do_latest)

                    # create multimodal offer
                    sub_trip_offers: tp.Dict[int, TravellerOffer] = {}
                    sub_trip_offers[RQ_SUB_TRIP_ID.FM_AMOD.value] = fm_amod_offer
                    sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value] = fm_pt_offer
                    multimodal_offer: 'MultimodalOffer' = self._create_multimodal_offer(rid, sub_trip_offers, parent_modal_state)
                    offers[multimodal_offer.operator_id] = multimodal_offer
                else:
                    LOG.info(f"PT offer is not available for sub_request {fm_pt_rid_struct}")
            else:
                LOG.info(f"AMoD offer is not available for sub_request {fm_amod_rid_struct}")
        return offers

    def _process_collect_lastmile_offers(
        self,
        rid: int,
        parent_modal_state: RQ_MODAL_STATE,
        offers: tp.Dict[int, 'TravellerOffer'],
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of lastmile offers.
        """
        # get lastmile pt offer
        lm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.LM_PT.value}"
        lm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(lm_pt_rid_struct)
        LOG.debug(f"Collecting lm_pt offer for request {lm_pt_rid_struct} from PT operator {self.pt_operator_id}: {lm_pt_offer}")
        
        if lm_pt_offer is not None and not lm_pt_offer.service_declined():
            # register the pt offer in the sub-request
            self.demand[lm_pt_rid_struct].receive_offer(self.pt_operator_id, lm_pt_offer, None)
            lm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.LM_AMOD.value}"
            for amod_op_id in range(self.n_amod_op):
                # get lastmile amod offer
                lm_amod_offer = self.amod_operators[amod_op_id].get_current_offer(lm_amod_rid_struct)
                LOG.debug(f"Collecting lm_amod offer for request {lm_amod_rid_struct} from operator {amod_op_id}: {lm_amod_offer}")
                
                if lm_amod_offer is not None and not lm_amod_offer.service_declined():
                    # register the amod offer in the sub-request
                    self.demand[lm_amod_rid_struct].receive_offer(amod_op_id, lm_amod_offer, None)
                    
                    # create multimodal offer
                    sub_trip_offers: tp.Dict[int, TravellerOffer] = {}
                    sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value] = lm_pt_offer
                    sub_trip_offers[RQ_SUB_TRIP_ID.LM_AMOD.value] = lm_amod_offer
                    multimodal_offer: 'MultimodalOffer' = self._create_multimodal_offer(rid, sub_trip_offers, parent_modal_state)
                    offers[multimodal_offer.operator_id] = multimodal_offer
                else:
                    LOG.info(f"AMoD offer is not available for sub_request {lm_amod_rid_struct}")
        else:
            LOG.info(f"PT offer is not available for sub_request {lm_pt_rid_struct}")
        return offers

    def _process_collect_firstlastmile_offers(
        self,
        rid: int,
        parent_rq_obj: 'BasicMultimodalRequest',
        parent_modal_state: RQ_MODAL_STATE,
        offers: tp.Dict[int, 'TravellerOffer'],
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstlastmile offers.
        """
        # get the firstmile amod offer
        flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
        flm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_PT.value}"
        flm_amod_rid_struct_1: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_1.value}"


        for amod_op_id in range(self.n_amod_op):
            # get the firstmile amod offer
            flm_amod_offer_0: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_0)
            LOG.debug(f"Collecting flm_amod_0 offer for request {flm_amod_rid_struct_0} from operator {amod_op_id}: {flm_amod_offer_0}")
            
            if flm_amod_offer_0 is not None and not flm_amod_offer_0.service_declined():
                # register the amod offer in the sub-request
                self.demand[flm_amod_rid_struct_0].receive_offer(amod_op_id, flm_amod_offer_0, None)
                
                # get the pt offer
                flm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(flm_pt_rid_struct, amod_op_id)
                LOG.debug(f"Collecting flm_pt offer for request {flm_pt_rid_struct} from operator {self.pt_operator_id}: {flm_pt_offer}")
                
                if flm_pt_offer is not None and not flm_pt_offer.service_declined():
                    # register the pt offer in the sub-request
                    self.demand[flm_pt_rid_struct].receive_offer(self.pt_operator_id, flm_pt_offer, None)
                    
                    # get the lastmile amod offer
                    flm_amod_offer_1: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_1)
                    LOG.debug(f"Collecting flm_amod_1 offer for request {flm_amod_rid_struct_1} from operator {amod_op_id}: {flm_amod_offer_1}")
                    
                    if flm_amod_offer_1 is not None and not flm_amod_offer_1.service_declined():
                        # register the amod offer in the sub-request
                        self.demand[flm_amod_rid_struct_1].receive_offer(amod_op_id, flm_amod_offer_1, None)

                        # update the first amod sub plan request with the latest dropoff time
                        new_t_do_latest: int = self._determine_amod_latest_dropoff_time(
                                                                                    parent_rq_obj,
                                                                                    flm_amod_offer_0,
                                                                                    flm_pt_offer.get(G_OFFER_WAIT),
                                                                                    )
                        sub_prq_obj: 'PlanRequest' = self.amod_operators[amod_op_id].rq_dict[flm_amod_rid_struct_0]
                        sub_prq_obj.set_new_dropoff_time_constraint(new_t_do_latest)
                        
                        # create multimodal offer
                        sub_trip_offers: tp.Dict[int, TravellerOffer] = {}
                        sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_0.value] = flm_amod_offer_0
                        sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value] = flm_pt_offer
                        sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_1.value] = flm_amod_offer_1
                        multimodal_offer: 'MultimodalOffer' = self._create_multimodal_offer(rid, sub_trip_offers, parent_modal_state)
                        offers[multimodal_offer.operator_id] = multimodal_offer
                    else:
                        LOG.info(f"AMoD offer is not available for sub_request {flm_amod_rid_struct_1}") 
                else:
                    LOG.info(f"PT offer is not available for sub_request {flm_pt_rid_struct}")
            else:
                LOG.info(f"AMoD offer is not available for sub_request {flm_amod_rid_struct_0}")

        return offers
    
    def _process_collect_firstlastmile_offers_3phases(
        self,
        rid: int,
        parent_rq_obj: 'BasicMultimodalRequest',
        parent_modal_state: RQ_MODAL_STATE,
        offers: tp.Dict[int, 'TravellerOffer'],
        sim_time: int,
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstlastmile offers using 3-phases approach.
        """
        # get rid struct for all sections
        flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
        flm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_PT.value}"
        flm_amod_rid_struct_1: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_1.value}"

        for amod_op_id in range(self.n_amod_op):
            # 1. Phase 1
            # 1.1 collect all offers
            flm_amod_offer_0_p1: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_0)
            LOG.debug(f"Collecting flm_amod_0 offer for request {flm_amod_rid_struct_0} from operator {amod_op_id}: {flm_amod_offer_0_p1} in 1st phase.")
            flm_pt_offer_p1: 'TravellerOffer' = self.pt_operator.get_current_offer(flm_pt_rid_struct, amod_op_id)
            LOG.debug(f"Collecting flm_pt offer for request {flm_pt_rid_struct} from operator {self.pt_operator_id}: {flm_pt_offer_p1} in 1st phase.")
            flm_amod_offer_1_p1: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_1)
            LOG.debug(f"Collecting flm_amod_1 offer for request {flm_amod_rid_struct_1} from operator {amod_op_id}: {flm_amod_offer_1_p1} in 1st phase.")

            # 1.2 determine the case
            # 1.2.1 case 1: no amod offer for firstmile, direct rejection
            if flm_amod_offer_0_p1 is None or flm_amod_offer_0_p1.service_declined():
                LOG.info(f"AMoD offer is not available for sub_request {flm_amod_rid_struct_0}, falling into case 1 of the 1st phase, skipping to next operator.")
                continue
            
            # 1.2.2 case 4: all offers are available, record the offer and proceed to 2nd phase
            if flm_pt_offer_p1 is not None and not flm_pt_offer_p1.service_declined() and flm_amod_offer_1_p1 is not None and not flm_amod_offer_1_p1.service_declined():
                LOG.info(f"All offers are available for request {rid}, falling into case 4 of the 1st phase, creating multimodal offer and proceeding to 2nd phase.")
                # create multimodal offer
                sub_trip_offers: tp.Dict[int, TravellerOffer] = {}
                sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_0.value] = flm_amod_offer_0_p1
                sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value] = flm_pt_offer_p1
                sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_1.value] = flm_amod_offer_1_p1
                multimodal_offer_p1: 'MultimodalOffer' = self._create_multimodal_offer(rid, sub_trip_offers, parent_modal_state)
            else:
                LOG.info(f"PT offer or LastMile AMoD offer is not available for request {rid}, falling into case 2 or 3 of the 1st phase, proceeding to 2nd phase.")
                multimodal_offer_p1 = None

            # 2. Phase 2
            # 2.0 cancel lastmile amod sub-request
            self.amod_operators[amod_op_id].user_cancels_request(flm_amod_rid_struct_1, sim_time)

            # 2.1 set new latest dropoff time for firstmile amod sub-request based on the firstmile amod offer in phase 1
            new_t_do_latest: int = self._determine_amod_latest_dropoff_time(
                                                                            parent_rq_obj,
                                                                            flm_amod_offer_0_p1,
                                                                            flm_pt_offer_p1.get(G_OFFER_WAIT),
                                                                            )
            sub_prq_obj: 'PlanRequest' = self.amod_operators[amod_op_id].rq_dict[flm_amod_rid_struct_0]
            sub_prq_obj.set_new_dropoff_time_constraint(new_t_do_latest)

            # 2.2 re-query the pt offer
            # 2.2.1 get the transfer station ids and their closest pt stations
            transfer_station_ids: tp.List[str] = parent_rq_obj.get_transfer_station_ids()
            transfer_street_node_0, _ = self.pt_operator.find_closest_street_node(transfer_station_ids[0])
            transfer_street_node_1, _ = self.pt_operator.find_closest_street_node(transfer_station_ids[1])

            # 2.2.2 determine the earliest start time of the pt sub-request
            flm_est_pt_mod: int = new_t_do_latest + self.amod_operators[amod_op_id].const_bt
            # 2.2.3 create the pt sub-request and get the pt arrival time
            flm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(
                                                                        parent_rq_obj,
                                                                        RQ_SUB_TRIP_ID.FLM_PT.value, 
                                                                        transfer_street_node_0,
                                                                        transfer_street_node_1, 
                                                                        flm_est_pt_mod,
                                                                        parent_modal_state,
                                                                        amod_op_id,
                                                                        )
            flm_pt_offer_p2: 'TravellerOffer' = self.pt_operator.get_current_offer(flm_pt_rid_struct, amod_op_id)
            if flm_pt_arrival is None:
                LOG.info(f"PT offer is not available for sub_request {flm_pt_rid_struct}, falling into case 2 of the 2nd phase, creating rejection.")
                multimodal_offer_p2 = None
            else:
                # 2.3 re-query the lastmile amod offer
                self._inform_amod_sub_request(
                                            parent_rq_obj,
                                            RQ_SUB_TRIP_ID.FLM_AMOD_1.value, 
                                            transfer_street_node_1,
                                            parent_rq_obj.get_destination_node(),
                                            flm_pt_arrival,
                                            parent_modal_state,
                                            amod_op_id,
                                            sim_time,
                                            )
                flm_amod_offer_1_p2: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_1)
                if flm_amod_offer_1_p2 is None or flm_amod_offer_1_p2.service_declined():
                    LOG.info(f"AMoD offer is not available for sub_request {flm_amod_rid_struct_1}, falling into case 3 of the 2nd phase, creating rejection.")
                    multimodal_offer_p2 = None
                else:
                    # 2.4 create multimodal offer
                    sub_trip_offers: tp.Dict[int, TravellerOffer] = {}
                    sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_0.value] = flm_amod_offer_0_p1
                    sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value] = flm_pt_offer_p2
                    sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_1.value] = flm_amod_offer_1_p2
                    multimodal_offer_p2: 'MultimodalOffer' = self._create_multimodal_offer(rid, sub_trip_offers, parent_modal_state)
                    LOG.info(f"Created multimodal offer in 2nd phase: {multimodal_offer_p2}")
            
            # 3. Phase 3
            # 3.1 check if the offer from phase 2 is better than the offer from phase 1
            comparison_results: int = self._compare_two_multimodal_offers(multimodal_offer_p1, multimodal_offer_p2)
            if comparison_results == 2:
                LOG.info(f"Offer from phase 2 is better than the offer from phase 1, accepting the offer from phase 2.")
                print(f"Offer from phase 2 is better than the offer from phase 1, accepting the offer from phase 2.")
                offers[multimodal_offer_p2.operator_id] = multimodal_offer_p2
            
            elif comparison_results == 1:
                LOG.info(f"Offer from phase 1 is better than the offer from phase 2, accepting the offer from phase 1, rolling back all changes in phase 2.")
                print(f"Offer from phase 1 is better than the offer from phase 2, accepting the offer from phase 1, rolling back all changes in phase 2.")
                # 3.2 rollback all changes in phase 2
                self.amod_operators[amod_op_id].user_cancels_request(flm_amod_rid_struct_1, sim_time)
                self._process_inform_firstlastmile_request(rid, parent_rq_obj, sim_time, parent_modal_state)
                offers = self._process_collect_firstlastmile_offers(rid, parent_rq_obj, parent_modal_state, offers)

            elif comparison_results == 0:
                LOG.info(f"Both offers from phase 1 and phase 2 are not available, skipping to next operator.")
                continue
            else:
                raise ValueError(f"Invalid comparison results: {comparison_results}")
        return offers

    def _get_amod_dropoff_time_range(
        self,
        amod_op_id: int,
        sub_rq_obj: 'RequestBase',
    ) -> tp.Tuple[tp.Optional[int], tp.Optional[int]]:
        """This method returns the earliest and latest amod dropoff time.
           Dropoff time is defined as the time when the user starts to leave the vehicle.
        """
        sub_rq_rid_struct: str = sub_rq_obj.get_rid_struct()
        sub_prq_obj: 'PlanRequest' = self.amod_operators[amod_op_id].rq_dict.get(sub_rq_rid_struct, None)
        if sub_prq_obj is None:
            return None, None
        else:
            return sub_prq_obj.t_do_earliest, sub_prq_obj.t_do_latest
    
    def _estimate_amod_dropoff_time(
        self,
        amod_op_id: int,
        sub_rq_obj: 'BasicMultimodalRequest',
        estimation_type: str = "latest",
    ) -> tp.Optional[int]:
        """This method estimates the dropoff time of an amod sub-request.

        Args:
            amod_op_id (int): the id of the amod operator
            sub_rq_obj (BasicMultimodalRequest): the sub-request object
            estimation_type (str): the type of the estimation

        Returns:
            int: the dropoff time of the sub-request
        """
        # get amod dropoff time range
        earliest_dropoff_time, latest_dropoff_time = self._get_amod_dropoff_time_range(amod_op_id, sub_rq_obj)
        if earliest_dropoff_time is None or latest_dropoff_time is None:
            return None
        else:
            if estimation_type == "latest":
                return latest_dropoff_time
            elif estimation_type == "earliest":
                return earliest_dropoff_time
            else:
                raise ValueError(f"Invalid estimation type: {estimation_type}")
    
    def _create_multimodal_offer(
        self,
        rid: int,
        sub_trip_offers: tp.Dict[int, 'TravellerOffer'],
        rq_modal_state: RQ_MODAL_STATE
    ) -> 'MultimodalOffer':
        """This method merges the amod and pt offers into a multimodal offer.
        """
        return MultimodalOffer(rid, sub_trip_offers, rq_modal_state)

    def _determine_est_pt_mod(
        self, 
        rq_obj: 'RequestBase',
        amod_op_id: int,
        amod_offer: 'TravellerOffer'
    ) -> int:
        """This method determines the earliest start time for the pt sub-request.
        """
        t_est_pt_mod: int = rq_obj.earliest_start_time + amod_offer.get(G_OFFER_WAIT) + amod_offer.get(G_OFFER_DRIVE) + self.amod_operators[amod_op_id].const_bt
        return t_est_pt_mod
    
    def _determine_amod_latest_dropoff_time(
        self, 
        rq_obj: 'RequestBase',
        amod_offer: 'TravellerOffer',
        pt_waiting_time: int,
    ) -> tp.Optional[int]:
        """This method determines the latest dropoff time for the amod sub-request.
        """
        t_do_latest: int = rq_obj.earliest_start_time + amod_offer.get(G_OFFER_WAIT) + amod_offer.get(G_OFFER_DRIVE) + pt_waiting_time
        return t_do_latest
    
    def _compare_two_multimodal_offers(
        self,
        offer_1: 'MultimodalOffer',
        offer_2: 'MultimodalOffer',
    ) -> int:
        """This method compares two multimodal offers based on availability and arrival time at destination node.
        If offer_1 and offer_2 are both None, it returns 0.
        If offer_1 is None and offer_2 is not None, it returns 2.
        If offer_1 is not None and offer_2 is None, it returns 1.
        If offer_1 is better than offer_2, it returns 1.
        If offer_1 is worse than offer_2, it returns 2.
        If offer_1 and offer_2 are equally good, it returns 2.
        """
        if offer_1 is None and offer_2 is None:
            return 0
        elif offer_1 is None and offer_2 is not None:
            return 2
        elif offer_1 is not None and offer_2 is None:
            return 1
        else:
            duration_p1: int = offer_1.get(G_OFFER_DURATION)
            duration_p2: int = offer_2.get(G_OFFER_DURATION)
            if duration_p1 < duration_p2:
                return 1
            elif duration_p1 > duration_p2:
                return 2
            else:
                return 2