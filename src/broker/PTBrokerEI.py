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
from src.broker.PTBrokerBasic import PTBrokerBasic
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
    "inherit" : PTBrokerBasic,
    "input_parameters_mandatory": ["n_amod_op", "amod_operators", "pt_operator", "demand", "routing_engine", "scenario_parameters"],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PTBrokerEI(PTBrokerBasic):
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
        super().__init__(n_amod_op, amod_operators, pt_operator, demand, routing_engine, scenario_parameters)

        # set MaaS detour time estimation parameter
        self.maas_detour_time_factor: float = self.scenario_parameters.get(G_BROKER_MAAS_DETOUR_TIME_FACTOR , 100)  

    def _process_inform_firstmile_request(self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int, parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTMILE):
        """This method processes the new firstmile request.

        Args:
            rid (int): the request id
            rq_obj ('BasicIntermodalRequest'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        # get the transfer station id and its closest pt station
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")

        # Make the estimation first, then based on the estimation, create sub-requests

        # create sub-request for AMoD
        for op_id in range(self.n_amod_op):
            self._inform_amod_sub_request(rq_obj, RQ_SUB_TRIP_ID.FM_AMOD.value, rq_obj.get_origin_node(), transfer_street_node, rq_obj.earliest_start_time, parent_modal_state, op_id, sim_time)
            fm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_AMOD.value}"
            fm_amod_sub_rq_obj: BasicIntermodalRequest = self.demand[fm_amod_rid_struct]
            # create sub-request for PT
            estimated_amod_dropoff_time: int = self._estimate_amod_dropoff_time(op_id, fm_amod_sub_rq_obj)
            # estimate the earliest start time of the pt sub-request
            fm_est_pt_mod: int = estimated_amod_dropoff_time + self.amod_operators[op_id].const_bt
            # create the pt sub-request
            _ = self._inform_pt_sub_request(rq_obj, RQ_SUB_TRIP_ID.FM_PT.value, transfer_street_node, rq_obj.get_destination_node(), fm_est_pt_mod, parent_modal_state, op_id)

    def _process_inform_lastmile_request(self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int, parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.LASTMILE):
        """This method processes the new lastmile request.

        Args:
            rid (int): the request id
            rq_obj ('BasicIntermodalRequest'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        # get the transfer station id and its closest pt station
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")
        # create sub-request for PT
        lm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(rq_obj, RQ_SUB_TRIP_ID.LM_PT.value, rq_obj.get_origin_node(), transfer_street_node, rq_obj.earliest_start_time, parent_modal_state)

        if lm_pt_arrival is not None:
            # create sub-request for AMoD
            for op_id in range(self.n_amod_op):
                self._inform_amod_sub_request(rq_obj, RQ_SUB_TRIP_ID.LM_AMOD.value, transfer_street_node, rq_obj.get_destination_node(), lm_pt_arrival, parent_modal_state, op_id, sim_time)
        else:
            LOG.info(f"PT offer is not available for sub_request {rid}_{RQ_SUB_TRIP_ID.LM_PT.value}, so the lastmile AMoD sub-request will not be created.")

    def _process_inform_firstlastmile_request(self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int, parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTLASTMILE):
        """This method processes the new firstlastmile request.

        Args:
            rid (int): the request id
            rq_obj ('BasicIntermodalRequest'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        # get the transfer station ids and their closest pt stations
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node_0, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")
        transfer_street_node_1, _ = self._find_transfer_info(transfer_station_ids[1], "pt2street")
        
        # create sub-request for AMoD
        for op_id in range(self.n_amod_op):
            # firstmile AMoD sub-request
            self._inform_amod_sub_request(rq_obj, RQ_SUB_TRIP_ID.FLM_AMOD_0.value, rq_obj.get_origin_node(), transfer_street_node_0, rq_obj.earliest_start_time, parent_modal_state, op_id, sim_time)
            flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
            flm_amod_sub_rq_obj_0: BasicIntermodalRequest = self.demand[flm_amod_rid_struct_0]
            # create sub-request for PT
            # estimate the dropoff time of the amod sub-request
            estimated_amod_dropoff_time: int = self._estimate_amod_dropoff_time(op_id, flm_amod_sub_rq_obj_0)
            # estimate the earliest start time of the pt sub-request
            flm_est_pt_mod: int = estimated_amod_dropoff_time + self.amod_operators[op_id].const_bt
            # create the pt sub-request
            flm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(rq_obj, RQ_SUB_TRIP_ID.FLM_PT.value, transfer_street_node_0,transfer_street_node_1, flm_est_pt_mod,parent_modal_state,op_id)

            # create sub-request for the same AMoD operator
            if flm_pt_arrival is None:
                raise ValueError(f"PT offer is not available for sub_request {rid}_{RQ_SUB_TRIP_ID.FLM_PT.value}, skipping to the next AMoD operator.")
            else:
                # last mile AMoD sub-request
                self._inform_amod_sub_request(rq_obj, RQ_SUB_TRIP_ID.FLM_AMOD_1.value, transfer_street_node_1, rq_obj.get_destination_node(), flm_pt_arrival, parent_modal_state, op_id, sim_time)
    
    def _process_collect_firstmile_offers(
        self, rid: int, parent_rq_obj: 'BasicIntermodalRequest', parent_modal_state: RQ_MODAL_STATE, offers: tp.Dict[int, 'TravellerOffer'],
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstmile offers.
        """
        # get rid struct for all sections
        fm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_AMOD.value}"
        fm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_PT.value}"

        for amod_op_id in range(self.n_amod_op):
            # TODO: if there are multiple AMoD operators, the PT offer can be overwritten here!!!
            # 1. collect FM AMoD offer
            fm_amod_offer: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(fm_amod_rid_struct)
            LOG.debug(f"Collecting fm_amod offer for request {fm_amod_rid_struct} from operator {amod_op_id}: {fm_amod_offer}.")
            # check if FM AMoD offer is available
            if fm_amod_offer is None or fm_amod_offer.service_declined():
                LOG.info(f"FM AMoD offer is not available for sub_request {fm_amod_rid_struct}, skipping to next AMoD operator.")
                continue
            # register the FM AMoD offer in the sub-request
            self.demand[fm_amod_rid_struct].receive_offer(amod_op_id, fm_amod_offer, None)

            # 2. collect FM PT offer
            fm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(fm_pt_rid_struct, amod_op_id)
            LOG.debug(f"Collecting fm_pt offer for request {fm_pt_rid_struct} from operator {self.pt_operator_id}: {fm_pt_offer}.")
            # check if PT offer is available
            if fm_pt_offer is None or fm_pt_offer.service_declined():
                LOG.info(f"PT offer is not available for sub_request {fm_pt_rid_struct}, skipping to next AMoD operator.")
                continue
            # register the PT offer in the sub-request
            self.demand[fm_pt_rid_struct].receive_offer(self.pt_operator_id, fm_pt_offer, None)

            # 3. create intermodal offer
            sub_trip_offers: tp.Dict[int, TravellerOffer] = {}
            sub_trip_offers[RQ_SUB_TRIP_ID.FM_AMOD.value] = fm_amod_offer
            sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value] = fm_pt_offer
            intermodal_offer: 'IntermodalOffer' = self._create_intermodal_offer(rid, sub_trip_offers, parent_modal_state)
            LOG.info(f"Created intermodal offer for request {rid}: {intermodal_offer}")

            # for this communication strategy, the FM AMoD DO time does not need to be updated.

            # 4. register the intermodal offer
            offers[intermodal_offer.operator_id] = intermodal_offer
        return offers
    
    def _process_collect_lastmile_offers(self, rid: int, parent_modal_state: RQ_MODAL_STATE, offers: tp.Dict[int, 'TravellerOffer']) -> tp.Dict[int, 'TravellerOffer']:
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
                    
                    # create intermodal offer
                    sub_trip_offers: tp.Dict[int, 'TravellerOffer'] = {}
                    sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value] = lm_pt_offer
                    sub_trip_offers[RQ_SUB_TRIP_ID.LM_AMOD.value] = lm_amod_offer
                    intermodal_offer: 'IntermodalOffer' = self._create_intermodal_offer(rid, sub_trip_offers, parent_modal_state)
                    offers[intermodal_offer.operator_id] = intermodal_offer
                else:
                    LOG.info(f"AMoD offer is not available for sub_request {lm_amod_rid_struct}")
        else:
            LOG.info(f"PT offer is not available for sub_request {lm_pt_rid_struct}")
        return offers
    
    def _process_collect_firstlastmile_offers(
        self, rid: int, parent_rq_obj: 'BasicIntermodalRequest', parent_modal_state: RQ_MODAL_STATE, offers: tp.Dict[int, 'TravellerOffer'], sim_time: int
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstlastmile offers.
        """
        # get rid struct for all sections
        flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
        flm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_PT.value}"
        flm_amod_rid_struct_1: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_1.value}"

        for amod_op_id in range(self.n_amod_op):
            # 1. collect FLM AMoD offer 0
            flm_amod_offer_0: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_0)
            LOG.debug(f"Collecting flm_amod_0 offer for request {flm_amod_rid_struct_0} from operator {amod_op_id}: {flm_amod_offer_0}")
            # check if FLM AMoD offer 0 is available
            if flm_amod_offer_0 is None or flm_amod_offer_0.service_declined():
                LOG.info(f"AMoD offer is not available for sub_request {flm_amod_rid_struct_0}, skipping to next AMoD operator.")
                continue
            self.demand[flm_amod_rid_struct_0].receive_offer(amod_op_id, flm_amod_offer_0, None)

            # 2. collect FLM PT offer
            flm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(flm_pt_rid_struct, amod_op_id)
            LOG.debug(f"Collecting flm_pt offer for request {flm_pt_rid_struct} from operator {self.pt_operator_id}: {flm_pt_offer}")
            # check if PT offer is available
            if flm_pt_offer is None or flm_pt_offer.service_declined():
                LOG.info(f"PT offer is not available for sub_request {flm_pt_rid_struct}, skipping to next AMoD operator.")
                continue
            self.demand[flm_pt_rid_struct].receive_offer(self.pt_operator_id, flm_pt_offer, None)

            # 3. collect FLM AMoD offer 1
            flm_amod_offer_1: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_1)
            LOG.debug(f"Collecting flm_amod_1 offer for request {flm_amod_rid_struct_1} from operator {amod_op_id}: {flm_amod_offer_1}")
            # check if FLM AMoD offer 1 is available
            if flm_amod_offer_1 is None or flm_amod_offer_1.service_declined():
                LOG.info(f"AMoD offer is not available for sub_request {flm_amod_rid_struct_1}, skipping to next AMoD operator.")
                continue
            self.demand[flm_amod_rid_struct_1].receive_offer(amod_op_id, flm_amod_offer_1, None)

            # 4. create intermodal offer
            sub_trip_offers: tp.Dict[int, 'TravellerOffer'] = {}
            sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_0.value] = flm_amod_offer_0
            sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value] = flm_pt_offer
            sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_1.value] = flm_amod_offer_1
            intermodal_offer: 'IntermodalOffer' = self._create_intermodal_offer(rid, sub_trip_offers, parent_modal_state)
            LOG.info(f"Created intermodal offer for request {rid}: {intermodal_offer}")
            
            # for this communication strategy, the FLM AMoD DO time does not need to be updated.

            # 5. register the intermodal offer
            offers[intermodal_offer.operator_id] = intermodal_offer
        return offers
        
    def _estimate_amod_dropoff_time(self, amod_op_id: int, sub_rq_obj: 'BasicIntermodalRequest') -> tp.Optional[int]:
        """This method estimates the dropoff time of an amod sub-request.
        This time point marks the start of alighting the FM amod vehicle.

        Args:
            amod_op_id (int): the id of the amod operator
            sub_rq_obj (BasicIntermodalRequest): the sub-request object
        Returns:
            int: the dropoff time of the sub-request
        """
        sub_rq_rid_struct: str = sub_rq_obj.get_rid_struct()
        sub_prq_obj: PlanRequest = self.amod_operators[amod_op_id].rq_dict.get(sub_rq_rid_struct, None)

        prq_direct_tt = sub_prq_obj.init_direct_tt
        amod_boarding_time = self.amod_operators[amod_op_id].const_bt
        prq_pu_latest = sub_prq_obj.t_pu_latest
        maas_estimated_prq_max_trip_time = (100 + self.maas_detour_time_factor) * (prq_direct_tt + amod_boarding_time) / 100

        maas_estimated_latest_dropoff_time: int = prq_pu_latest + int(maas_estimated_prq_max_trip_time)
        return maas_estimated_latest_dropoff_time

    def acknowledge_user_alighting(self, op_id: int, rid_struct: str, vid: int, alighting_time: int):
        """Override to check if FM passenger can catch their PT connection.

        After FM AMoD alighting completes, check if the alighting time is still
        within the PT offer's origin_node_latest_arrival_time. If not, cancel
        subsequent offers and mark the request as uncatchable.

        Args:
            op_id (int): the AMoD operator id
            rid_struct (str): the request id struct (e.g., "123_1" for sub-request)
            vid (int): the vehicle id
            alighting_time (int): the simulation time when alighting completes
        """
        # Call parent implementation first
        super().acknowledge_user_alighting(op_id, rid_struct, vid, alighting_time)

        # Check if this is a FM or FLM first-leg AMoD sub-request
        rid_struct_str = str(rid_struct)
        if "_" in rid_struct_str:
            parts = rid_struct_str.rsplit("_", 1)
            parent_rid = int(parts[0])
            sub_trip_id = int(parts[1])

            # Check FM case: FM_AMOD alighting completed
            if sub_trip_id == RQ_SUB_TRIP_ID.FM_AMOD.value:
                self._check_fm_pt_catchability(parent_rid, sub_trip_id, alighting_time, RQ_SUB_TRIP_ID.FM_PT.value, op_id)
            # Check FLM case: FLM_AMOD_0 alighting completed
            elif sub_trip_id == RQ_SUB_TRIP_ID.FLM_AMOD_0.value:
                self._check_fm_pt_catchability(parent_rid, sub_trip_id, alighting_time, RQ_SUB_TRIP_ID.FLM_PT.value, op_id)

    def _check_fm_pt_catchability(self, parent_rid: int, _amod_sub_trip_id: int, alighting_time: int, pt_sub_trip_id: int, amod_op_id: int):
        """Check if passenger can catch their PT connection after FM AMoD alighting.

        Args:
            parent_rid (int): the parent request id
            _amod_sub_trip_id (int): the sub-trip id of the completed AMoD leg (unused, kept for API consistency)
            alighting_time (int): the simulation time when alighting completes
            pt_sub_trip_id (int): the sub-trip id of the PT leg to check
            amod_op_id (int): the AMoD operator id that served the FM leg
        """
        # Get PT offer
        pt_rid_struct = f"{parent_rid}_{pt_sub_trip_id}"

        # Get the PT offer for the specific AMoD operator that served the FM leg
        pt_offer: 'PTOffer' = self.pt_operator.get_current_offer(pt_rid_struct, amod_op_id)

        if pt_offer is None or pt_offer.service_declined():
            LOG.debug(f"No PT offer found for {pt_rid_struct}, skipping catchability check")
            return

        # Check catchability: alighting_time vs origin_node_latest_arrival_time
        origin_node_latest_arrival_time = pt_offer.origin_node_latest_arrival_time

        if alighting_time > origin_node_latest_arrival_time:
            LOG.warning(f"Request {parent_rid}: PT uncatchable! "
                       f"Alighting time {alighting_time} > PT latest arrival {origin_node_latest_arrival_time} "
                       f"(delay: {alighting_time - origin_node_latest_arrival_time}s)")
            self._handle_uncatchable_pt(parent_rid, alighting_time, amod_op_id)
        else:
            LOG.debug(f"Request {parent_rid}: PT catchable. "
                     f"Alighting time {alighting_time} <= PT latest arrival {origin_node_latest_arrival_time} "
                     f"(buffer: {origin_node_latest_arrival_time - alighting_time}s)")

    def _handle_uncatchable_pt(self, parent_rid: int, sim_time: int, amod_op_id: int):
        """Handle the case when passenger cannot catch their PT connection.

        This method:
        1. Marks the parent request as uncatchable
        2. Cancels subsequent sub-requests (PT and any LM AMoD)

        Args:
            parent_rid (int): the parent request id
            sim_time (int): the current simulation time
            amod_op_id (int): the AMoD operator id that served the FM leg
        """
        parent_rq_obj: 'BasicIntermodalRequest' = self.demand[parent_rid]
        parent_modal_state: RQ_MODAL_STATE = parent_rq_obj.get_modal_state()

        # Mark the request as uncatchable
        parent_rq_obj.set_uncatchable_pt(True)
        LOG.info(f"Request {parent_rid} marked as uncatchable_pt")

        # Cancel subsequent sub-requests based on modal state
        if parent_modal_state == RQ_MODAL_STATE.FIRSTMILE:
            # For FM: cancel PT sub-request
            pt_rid_struct = f"{parent_rid}_{RQ_SUB_TRIP_ID.FM_PT.value}"
            try:
                self.pt_operator.user_cancels_request(pt_rid_struct, sim_time, amod_op_id)
                LOG.info(f"Cancelled PT sub-request {pt_rid_struct} due to uncatchable PT")
            except (KeyError, AttributeError) as e:
                LOG.debug(f"Could not cancel PT sub-request {pt_rid_struct}: {e}")

        elif parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            # For FLM: cancel PT and last-mile AMoD sub-requests
            pt_rid_struct = f"{parent_rid}_{RQ_SUB_TRIP_ID.FLM_PT.value}"
            lm_amod_rid_struct = f"{parent_rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_1.value}"

            # Cancel PT sub-request
            try:
                self.pt_operator.user_cancels_request(pt_rid_struct, sim_time, amod_op_id)
                LOG.info(f"Cancelled PT sub-request {pt_rid_struct} due to uncatchable PT")
            except (KeyError, AttributeError) as e:
                LOG.debug(f"Could not cancel PT sub-request {pt_rid_struct}: {e}")

            # Cancel last-mile AMoD sub-request
            for op in self.amod_operators:
                try:
                    op.user_cancels_request(lm_amod_rid_struct, sim_time)
                    LOG.info(f"Cancelled LM AMoD sub-request {lm_amod_rid_struct} due to uncatchable PT")
                except KeyError:
                    LOG.debug(f"LM AMoD sub-request {lm_amod_rid_struct} not found for operator, may not exist")