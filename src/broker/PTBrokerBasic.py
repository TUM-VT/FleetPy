# TODO：
# - Adjust PT waiting time based on dynamic GTFS data (e.g., delays), and then adjust FM and LM offers accordingly.
# - Support multiple AMoD operators for firstlastmile requests.

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
from src.simulation.Offers import IntermodalOffer
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.fleetctrl.planning.PlanRequest import PlanRequest
    from src.ptctrl.PTControlBase import PTControlBase
    from src.demand.demand import Demand
    from src.routing.road.NetworkBase import NetworkBase
    from src.demand.TravelerModels import RequestBase, BasicIntermodalRequest
    from src.simulation.Offers import TravellerOffer, PTOffer

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_PTBroker = {
    "doc" : "this class represents a broker platform which handles intermodal requests",
    "inherit" : BrokerBasic,
    "input_parameters_mandatory": ["n_amod_op", "amod_operators", "pt_operator", "demand", "routing_engine", "scenario_parameters"],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PTBrokerBasic(BrokerBasic):
    def __init__(
        self, 
        n_amod_op: int, 
        amod_operators: tp.List['FleetControlBase'], 
        pt_operator: 'PTControlBase', 
        demand: 'Demand', 
        routing_engine: 'NetworkBase',
        scenario_parameters: dict,
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
        super().__init__(n_amod_op, amod_operators)

        self.demand: Demand = demand
        self.routing_engine: NetworkBase = routing_engine
        self.pt_operator: PTControlBase = pt_operator

        self.pt_operator_id: int = self.pt_operator.pt_operator_id
        self.scenario_parameters: dict = scenario_parameters
        
        # set whether to always query pure PT offers
        self.always_query_pt: bool = self.scenario_parameters.get(G_BROKER_ALWAYS_QUERY_PT, False)
        
        # set simulation start date for Raptor routing
        self.sim_start_datetime: datetime = None
        self._set_sim_start_datetime(self.scenario_parameters.get(G_PT_SIM_START_DATE, None))

        # method for finding transfer stations
        self.transfer_search_method: str = self.scenario_parameters.get(G_BROKER_TRANSFER_SEARCH_METHOD, "closest")
        # read necessary files based on the transfer search method
        # default method: closest transfer station search.
        if self.transfer_search_method == "closest":
            # load the street-station transfers: used for finding closest station to a street node, or vice versa
            try:
                self.street_station_transfers_fp_df = self._load_street_station_transfers_from_gtfs(self.pt_operator.gtfs_dir)
            except FileNotFoundError:
                LOG.error("PTBroker: street_station_transfers_fp.txt file not found in the GTFS directory, which is required for finding closest transfer stations!")
                raise FileNotFoundError("PTBroker: street_station_transfers_fp.txt file not found in the GTFS directory, which is required for finding closest transfer stations!")


    def inform_request(self, rid: int, rq_obj: 'RequestBase', sim_time: int):
        """This method informs the broker that a new request has been made. 
        Based on the request modal state, the broker will create the appropriate sub-requests 
        and inform the operators.

        Args:
            rid (int): parent request id
            rq_obj (RequestBase): request object
            sim_time (int): simulation time
        """
        parent_modal_state: RQ_MODAL_STATE = rq_obj.get_modal_state()
        LOG.debug(f"inform request: {rid} at sim time {sim_time} with modal state {parent_modal_state}; query pure PT offer: {self.always_query_pt}")

        if self.always_query_pt:
            # 1. query the PT operator for the pure PT travel costs
            _ = self._inform_pt_sub_request(rq_obj, RQ_SUB_TRIP_ID.PT.value, rq_obj.get_origin_node(), rq_obj.get_destination_node(), rq_obj.earliest_start_time, parent_modal_state)
        
        # 2.1 pure AMoD request or PT request
        if parent_modal_state == RQ_MODAL_STATE.MONOMODAL or parent_modal_state == RQ_MODAL_STATE.PT:
            self._process_inform_monomodal_request(rid, rq_obj, sim_time, parent_modal_state)
        
        # 2.2 AMoD as firstmile request
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTMILE:
            self._process_inform_firstmile_request(rid, rq_obj, sim_time, parent_modal_state)
        
        # 2.3 AMoD as lastmile request
        elif parent_modal_state == RQ_MODAL_STATE.LASTMILE:
            self._process_inform_lastmile_request(rid, rq_obj, sim_time, parent_modal_state)
        
        # 2.4 AMoD as firstlastmile request
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            self._process_inform_firstlastmile_request(rid, rq_obj, sim_time, parent_modal_state)
        
        else:
            raise ValueError(f"Invalid modal state: {parent_modal_state}")
        
    def collect_offers(self, rid: int, sim_time: int) -> tp.Dict[int, 'TravellerOffer']:
        """This method collects the offers from the operators.

        Args:
            rid (int): parent request id
            sim_time (int): simulation time
        Returns:
            tp.Dict[int, TravellerOffer]: a dictionary of offers from the operators
        """
        # get parent request modal state
        parent_rq_obj: RequestBase = self.demand[rid]
        parent_modal_state: RQ_MODAL_STATE = parent_rq_obj.get_modal_state()
        offers: tp.Dict[int, TravellerOffer] = {}
        LOG.debug(f"Collecting offers for request {rid} with modal state {parent_modal_state}")

        # 1. collect PT offers for multimodal requests
        if parent_modal_state.value > RQ_MODAL_STATE.MONOMODAL.value or self.always_query_pt:
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
            offers = self._process_collect_firstlastmile_offers(rid, parent_rq_obj, parent_modal_state, offers, sim_time)
        
        else:
            raise ValueError(f"Invalid modal state: {parent_modal_state}")

        return offers
    
    def inform_user_booking(self, rid: int, rq_obj: 'RequestBase', sim_time: int, chosen_operator: tp.Union[int, tuple]) -> tp.List[tuple[int, 'RequestBase']]:
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
            pt_sub_rq_obj: BasicIntermodalRequest = self.demand[pt_rid_struct]
            self.pt_operator.user_confirms_booking(pt_sub_rq_obj, None)   
        # 2. AMoD involved offer has been selected
        else:
            # non-intermodal offer has been selected
            if parent_modal_state == RQ_MODAL_STATE.MONOMODAL or parent_modal_state == RQ_MODAL_STATE.PT:
                for i, operator in enumerate(self.amod_operators):
                    if i != chosen_operator:  # Non-intermodal requests: the chosen operator has the data type int
                        operator.user_cancels_request(rid, sim_time)
                    else:
                        operator.user_confirms_booking(rid, sim_time)
                        amod_confirmed_rids.append((rid, rq_obj))
            # intermodal offer has been selected
            elif parent_modal_state.value > RQ_MODAL_STATE.MONOMODAL.value and parent_modal_state.value < RQ_MODAL_STATE.PT.value:
                # chosen_operator has the data type tuple: ((operator_id, sub_trip_id), ...)
                for operator_id, sub_trip_id in chosen_operator:
                    if operator_id == self.pt_operator_id:
                        # inform the pt operator that the request is confirmed
                        pt_rid_struct: str = f"{rid}_{sub_trip_id}"
                        pt_sub_rq_obj: BasicIntermodalRequest = self.demand[pt_rid_struct]

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
    
    def inform_user_leaving_system(self, rid: int, sim_time: int):
        """This method informs the broker that the user is leaving the system.
        """
        rq_obj: RequestBase = self.demand[rid]
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
                try:
                    operator.user_cancels_request(lm_amod_rid_struct, sim_time)
                except KeyError:
                    # LM AMoD sub-request may not be created if no PT offer is available
                    LOG.info(f"LM AMoD sub-request {lm_amod_rid_struct} not found when user leaves system, possibly no PT offer available so the LM sub-request was not created.")
        
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
            flm_amod_rid_struct_1: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_1.value}"
            for _, operator in enumerate(self.amod_operators):
                operator.user_cancels_request(flm_amod_rid_struct_0, sim_time)
                try:
                    operator.user_cancels_request(flm_amod_rid_struct_1, sim_time)
                except KeyError:
                    # LM AMoD sub-request may not be created if no PT offer is available
                    LOG.info(f"LM AMoD sub-request {flm_amod_rid_struct_1} not found when user leaves system, possibly no FM or PT offer available so the LM sub-request was not created.")

        else:
            raise ValueError(f"Invalid modal state: {parent_modal_state}")
        
    def inform_waiting_request_cancellations(self, chosen_operator: int, rid: int, sim_time: int):
        """This method informs the operators that the waiting requests have been cancelled.
        """
        rq_obj: RequestBase = self.demand[rid]
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
        
    def _process_inform_monomodal_request(self, rid: int, rq_obj: 'RequestBase', sim_time: int, parent_modal_state: RQ_MODAL_STATE,):
        """This method processes the new monomodal request.

        Args:
            rid (int): the request id
            rq_obj ('RequestBase'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        for op_id in range(self.n_amod_op):
            LOG.debug(f"AMoD Request {rid} with modal state {parent_modal_state}: To operator {op_id} ...")
            self.amod_operators[op_id].user_request(rq_obj, sim_time)

    def _process_inform_firstmile_request(self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int, parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTMILE):
        """This method processes the new firstmile request. 
        In this stage, only the first-mile AMoD sub-request is created first; the PT sub-request will be created after receiving the AMoD offer.

        Args:
            rid (int): the request id
            rq_obj ('BasicIntermodalRequest'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        pass

    def _process_inform_lastmile_request(self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int, parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.LASTMILE):
        """This method processes the new lastmile request.
        First, the PT sub-request is created. If the PT offer is available, then the last-mile AMoD sub-request is created.

        Args:
            rid (int): the request id
            rq_obj ('BasicIntermodalRequest'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        pass

    def _process_inform_firstlastmile_request(self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int, parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTLASTMILE):
        """This method processes the new firstlastmile request.
        In this stage, only the first-mile AMoD sub-request is created first; the PT and last-mile AMoD sub-requests will be created after receiving the first-mile AMoD offer.

        Args:
            rid (int): the request id
            rq_obj ('BasicIntermodalRequest'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        pass        

    def _inform_amod_sub_request(
        self, rq_obj: 'RequestBase', sub_trip_id: int, leg_o_node: int, leg_d_node: int, leg_start_time: int,
        parent_modal_state: RQ_MODAL_STATE, op_id: int, sim_time: int
    ):
        """
        This method informs the AMoD operators that a new sub-request has been made.

        Args:
            rq_obj ('RequestBase'): the parent request object
            sub_trip_id (int): the sub-trip id
            leg_o_node (int): the origin node of the sub-request
            leg_d_node (int): the destination node of the sub-request
            leg_start_time (int): the start time of the sub-request
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        amod_sub_rq_obj: RequestBase = self.demand.create_sub_requests(rq_obj, sub_trip_id, leg_o_node, leg_d_node, leg_start_time, parent_modal_state)
        LOG.debug(f"AMoD sub-request {amod_sub_rq_obj.get_rid_struct()} with modal state {parent_modal_state}: To operator {op_id} ...")

        self.amod_operators[op_id].user_request(amod_sub_rq_obj, sim_time)
        
    def _inform_pt_sub_request(
        self, rq_obj: 'RequestBase', sub_trip_id: int, leg_o_node: int, leg_d_node: int, leg_start_time: int,
        parent_modal_state: RQ_MODAL_STATE, firstmile_amod_operator_id: int = None
    ) -> tp.Optional[int]:
        """
        This method informs the PT operator that a new sub-request has been made.

        Args:
            rq_obj (RequestBase): the parent request object
            sub_trip_id (int): the sub_trip id
            leg_o_node (int): the origin street node of the sub-request
            leg_d_node (int): the destination street node of the sub-request
            leg_start_time (int): the start time [s] of the sub-request at the origin street node
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
            firstmile_amod_operator_id (int): the id of the firstmile amod operator, only used for FM and FLM requests
        Returns:
            t_d_node_arrival (tp.Optional[int]):
                the pt arrival time of the sub-request at the destination street node 
                or None if the pt travel costs are not available
        """
        pt_sub_rq_obj: RequestBase = self.demand.create_sub_requests(rq_obj, sub_trip_id, leg_o_node, leg_d_node, leg_start_time, parent_modal_state)
        LOG.debug(f"PT sub-request {pt_sub_rq_obj.get_rid_struct()} with modal state {parent_modal_state}: To PT operator {self.pt_operator_id} ...")
        
        costs_info = self._query_street_node_pt_travel_costs_1to1(
                                                                pt_sub_rq_obj.get_origin_node(), 
                                                                pt_sub_rq_obj.get_destination_node(), 
                                                                pt_sub_rq_obj.earliest_start_time,
                                                                pt_sub_rq_obj.get_max_transfers(),  # the request type should be BasicIntermodalRequest
                                                                )
        
        if costs_info is not None:
            source_pt_station_id, t_source_walk, target_pt_station_id, t_target_walk, pt_journey_plan_dict = costs_info
            LOG.debug(f"PT sub-request {pt_sub_rq_obj.get_rid_struct()} with modal state {parent_modal_state}: Found offer with source_pt_station_id {source_pt_station_id}, t_source_walk {t_source_walk}, target_pt_station_id {target_pt_station_id}, t_target_walk {t_target_walk}, pt_journey_plan_dict {pt_journey_plan_dict}")
        else:
            source_pt_station_id = None
            t_source_walk = None
            target_pt_station_id = None
            t_target_walk = None
            pt_journey_plan_dict = None
            LOG.debug(f"PT sub-request {pt_sub_rq_obj.get_rid_struct()} with modal state {parent_modal_state}: No PT offer has been found!")
        
        pt_rid_struct: str =  pt_sub_rq_obj.get_rid_struct()

        self.pt_operator.create_and_record_pt_offer_db(
                                                    rid_struct = pt_rid_struct,
                                                    operator_id = self.pt_operator_id,
                                                    source_station_id = source_pt_station_id,
                                                    target_station_id = target_pt_station_id,
                                                    source_walking_time = t_source_walk,
                                                    target_walking_time = t_target_walk,
                                                    pt_journey_plan_dict = pt_journey_plan_dict,
                                                    firstmile_amod_operator_id = firstmile_amod_operator_id,
                                                    )
        if pt_journey_plan_dict is not None:
            t_d_node_arrival: int = self.pt_operator.get_current_offer(pt_rid_struct, firstmile_amod_operator_id).destination_node_arrival_time  # Offer type: PTOffer
            return t_d_node_arrival
        else:
            return None

    def _process_collect_monomodal_offers(self, rid: int, parent_modal_state: RQ_MODAL_STATE, offers: tp.Dict[int, 'TravellerOffer']) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of monomodal offers.

        Args:
            rid (int): the request id
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
            offers (tp.Dict[int, TravellerOffer]): the current offers dictionary
        Returns:
            tp.Dict[int, TravellerOffer]: the updated offers dictionary
        """
        for amod_op_id in range(self.n_amod_op):
            amod_offer = self.amod_operators[amod_op_id].get_current_offer(rid)
            LOG.debug(f"Collecting amod offer for request {rid} with modal state {parent_modal_state} from operator {amod_op_id}: {amod_offer}")
            if amod_offer is not None and not amod_offer.service_declined():
                offers[amod_op_id] = amod_offer
        return offers
    
    def _process_collect_firstmile_offers(
        self, rid: int, parent_rq_obj: 'BasicIntermodalRequest', parent_modal_state: RQ_MODAL_STATE, 
        offers: tp.Dict[int, 'TravellerOffer']
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstmile offers and try to optimize the waiting time of the PT leg.
        """
        # get rid struct for all sections
        pass
    
    def _process_collect_lastmile_offers(self, rid: int, parent_modal_state: RQ_MODAL_STATE, offers: tp.Dict[int, 'TravellerOffer']) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of LM offers.
        """
        pass
    
    def _process_collect_firstlastmile_offers(
        self, rid: int, parent_rq_obj: 'BasicIntermodalRequest', parent_modal_state: RQ_MODAL_STATE, 
        offers: tp.Dict[int, 'TravellerOffer'], sim_time: int
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstlastmile offers.
        """
        pass
    
    def _set_sim_start_datetime(self, sim_start_date: str):
        """This method sets the simulation start date.
        Converts the date string (format YYYYMMDD) to a datetime object.

        Args:
            sim_start_date (str): the simulation start date in format YYYYMMDD
        """
        if sim_start_date is None:
            LOG.error("PTBrokerTPCS: Simulation start date for PT routing not provided in scenario parameters!")
            raise ValueError("PTBrokerTPCS: Simulation start date for PT routing not provided in scenario parameters!")
        
        if type(sim_start_date) is not str:
            sim_start_date = str(int(sim_start_date))
        self.sim_start_datetime = datetime.strptime(sim_start_date, "%Y%m%d")

    def _get_current_datetime(self, sim_time_in_seconds: int) -> datetime:
        """This method returns the current datetime based on the simulation time in seconds.
        """
        return self.sim_start_datetime + timedelta(seconds=int(sim_time_in_seconds))

    def _load_street_station_transfers_from_gtfs(self, gtfs_dir: str) -> pd.DataFrame:
        """This method loads the FleetPy-specific street station transfers file.

        Args:
            gtfs_dir (str): The directory containing the GTFS data of the operator.
        Returns:
            pd.DataFrame: The transfer data between the street nodes and the PT stations.
        """
        dtypes = {
            'node_id': 'int',
            'closest_station_id': 'str',
            'street_station_transfer_time': 'int',
        }
        return pd.read_csv(os.path.join(gtfs_dir, "street_station_transfers_fp.txt"), dtype=dtypes)
    
    def _query_street_node_pt_travel_costs_1to1(
        self, o_node: int, d_node: int, est: int, 
        max_transfers: int = 999, detailed: bool = False
    ) -> tp.Optional[tp.Tuple[int, int, int, int, tp.Dict[str, tp.Any]]]:
        """This method queries the pt travel costs between two street nodes at a given datetime.
        The pt station ids will be the closest pt station ids to the street nodes.

        Args:
            o_node (int): The origin street node id.
            d_node (int): The destination street node id.
            est (int): The earliest start time of the request at the origin street node in seconds.
            max_transfers (int): The maximum number of transfers allowed in the journey, 999 for no limit.
            detailed (bool): Whether to return detailed journey information. Defaults to False.
        Returns:
            tp.Optional[tp.Tuple[int, int, int, int, tp.Dict[str, tp.Any]]]:
                Returns a tuple containing:
                (source_pt_station_id, t_source_walking, target_pt_station_id, t_target_walking, pt_journey_plan_dict)
                if a public transport journey plan is found.
                Returns None if no public transport journey plan is available.
        """
        # find pt transfer stations
        source_pt_station_id, t_source_walk = self._find_transfer_info(o_node, "street2pt")
        target_pt_station_id, t_target_walk = self._find_transfer_info(d_node, "street2pt")

        source_station_departure_seconds: int = est + t_source_walk
        source_station_departure_datetime: datetime = self._get_current_datetime(source_station_departure_seconds)
        LOG.debug(f"Query PT travel costs: {o_node} -> {d_node} (stations: {source_pt_station_id} -> {target_pt_station_id}) at {est} (station departure: {source_station_departure_datetime})")
        pt_journey_plan_dict: tp.Union[tp.Dict[str, tp.Any], None] = self.pt_operator.return_fastest_pt_journey_1to1(
                                                                                                                    source_pt_station_id, target_pt_station_id, 
                                                                                                                    source_station_departure_datetime,
                                                                                                                    max_transfers, detailed,
                                                                                                                    )
        if pt_journey_plan_dict is None:
            return None
        else:
            return source_pt_station_id, t_source_walk, target_pt_station_id, t_target_walk, pt_journey_plan_dict
        
    def _find_transfer_info(self, node_id: tp.Union[int, str], direction: str) -> tp.Tuple[str, int]:
        """This method finds the transfer possibility between pt station and street node.

        Args:
            node_id (tp.Union[int, str]): The street node id.
            direction (str): "pt2street" or "street2pt"
        Returns:
            tp.Tuple[str, int]: The transfer node id and the walking time.
        """
        if self.transfer_search_method == "closest":  # closest station or street node
            if direction == "street2pt":  # find closest pt station from street node
                street_node_id: int = int(node_id)
                # TODO: allow multiple closest stations?
                street_station_transfer = self.street_station_transfers_fp_df[self.street_station_transfers_fp_df["node_id"] == street_node_id]
                if street_station_transfer.empty:
                    raise ValueError(f"Street node id {street_node_id} not found in the street station transfers file")
                closest_station_id: str = street_station_transfer["closest_station_id"].iloc[0]
                walking_time: int = street_station_transfer["street_station_transfer_time"].iloc[0]
                return closest_station_id, walking_time
            elif direction == "pt2street":  # find closest street node from pt station
                pt_station_id: str = str(node_id)
                street_station_transfers = self.street_station_transfers_fp_df[self.street_station_transfers_fp_df["closest_station_id"] == pt_station_id]
                if street_station_transfers.empty:
                    raise ValueError(f"PT station id {pt_station_id} not found in the street station transfers file")
                # find the record with the minimum street_station_transfer_time
                # TODO: if multiple exist, return all?
                min_transfer = street_station_transfers.loc[street_station_transfers["street_station_transfer_time"].idxmin()]
                closest_street_node_id: int = min_transfer["node_id"]
                walking_time: int = min_transfer["street_station_transfer_time"]
                return closest_street_node_id, walking_time
            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'pt2street' or 'street2pt'.")
        else:
            LOG.debug(f"PTBrokerTPCS: Transfer search method '{self.transfer_search_method}' not implemented. Using 'closest' instead.")
            raise NotImplementedError(f"PTBrokerTPCS: Transfer search method '{self.transfer_search_method}' not implemented.")
        
    def _create_intermodal_offer(self, rid: int, sub_trip_offers: tp.Dict[int, 'TravellerOffer'], rq_modal_state: RQ_MODAL_STATE) -> 'IntermodalOffer':
        """This method merges the amod and pt offers into an intermodal offer.
        """
        return IntermodalOffer(rid, sub_trip_offers, rq_modal_state)