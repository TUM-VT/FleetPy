# TODO：
# - Adjust PT waiting time based on dynamic GTFS data (e.g., delays), and then adjust FM and LM offers accordingly.
# - Support multiple AMoD operators for firstlastmile requests.

# -------------------------------------------------------------------------------------------------------------------- #
# PTBroker: Collaborative Coordination Strategy
#
# Simulates a future scenario with autonomous DRT and tight MaaS–DRT integration:
# - MaaS queries DRT for FM and immediately receives a predicted dropoff time (actual offer, not estimated).
# - MaaS books PT and LM DRT based on that predicted dropoff time.
# - PT returns the user's expected waiting time at the boarding station; MaaS feeds this back to DRT.
# - DRT dynamically adjusts the user's latest dropoff deadline, giving DRT more flexibility for
#   ride-pooling while still ensuring the user catches the PT vehicle.
# - MaaS can also constrain LM DRT waiting time, minimizing wait at the destination station.
# Result: higher service rate and shorter travel times through real-time coordination between MaaS and DRT.
#
# NOTE: This code has only been tested and applied in the ImmediateDecisionsSimulation environment
#       combined with the PoolingIRSOnly fleet controller.
# -------------------------------------------------------------------------------------------------------------------- #

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
class PTBroker(PTBrokerBasic):
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

    def _inform_amod_sub_request(
        self, rq_obj: 'RequestBase', sub_trip_id: int, leg_o_node: int, leg_d_node: int, leg_start_time: int,
        parent_modal_state: RQ_MODAL_STATE, op_id: int, sim_time: int
    ):
        """Overrides PTBrokerBasic to support customizable max_wait_time for last-mile AMoD pickups."""
        amod_sub_rq_obj: 'RequestBase' = self.demand.create_sub_requests(rq_obj, sub_trip_id, (leg_o_node, None, None), (leg_d_node, None, None), leg_start_time, parent_modal_state)
        LOG.debug(f"AMoD sub-request {amod_sub_rq_obj.get_rid_struct()} with modal state {parent_modal_state}: To operator {op_id} ...")

        # get customizable wait time for last mile AMoD pickups
        if parent_modal_state == RQ_MODAL_STATE.LASTMILE or (parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE and sub_trip_id == RQ_SUB_TRIP_ID.FLM_AMOD_1.value):
            max_wait_time: tp.Optional[int] = rq_obj.get_lastmile_max_wait_time()
        else:
            max_wait_time: tp.Optional[int] = None

        self.amod_operators[op_id].user_request(amod_sub_rq_obj, sim_time, max_wait_time=max_wait_time)

    def _process_inform_firstmile_request(self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int, parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTMILE):
        """This method processes the new firstmile request. 
        In this stage, only the first-mile AMoD sub-request is created first; the PT sub-request will be created after receiving the AMoD offer.

        Args:
            rid (int): the request id
            rq_obj ('BasicIntermodalRequest'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        # get the transfer station id and its closest pt station
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")

        # create sub-request for AMoD
        for op_id in range(self.n_amod_op):
            self._inform_amod_sub_request(rq_obj, RQ_SUB_TRIP_ID.FM_AMOD.value, rq_obj.get_origin_node(), transfer_street_node, rq_obj.earliest_start_time, parent_modal_state, op_id, sim_time)

    def _process_inform_lastmile_request(self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int, parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.LASTMILE):
        """This method processes the new lastmile request.
        First, the PT sub-request is created. If the PT offer is available, then the last-mile AMoD sub-request is created.

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
        In this stage, only the first-mile AMoD sub-request is created first; the PT and last-mile AMoD sub-requests will be created after receiving the first-mile AMoD offer.

        Args:
            rid (int): the request id
            rq_obj ('BasicIntermodalRequest'): the request object
            sim_time (int): the simulation time
            parent_modal_state (RQ_MODAL_STATE): the parent modal state
        """
        # get the transfer station ids and their closest pt stations
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node_0, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")
        
        # create FM sub-request for AMoD
        for op_id in range(self.n_amod_op):
            # firstmile AMoD sub-request
            self._inform_amod_sub_request(rq_obj, RQ_SUB_TRIP_ID.FLM_AMOD_0.value, rq_obj.get_origin_node(), transfer_street_node_0, rq_obj.earliest_start_time, parent_modal_state, op_id, sim_time)         
    
    def _process_collect_firstmile_offers(
        self, rid: int, parent_rq_obj: 'BasicIntermodalRequest', parent_modal_state: RQ_MODAL_STATE, 
        offers: tp.Dict[int, 'TravellerOffer']
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstmile offers and try to optimize the waiting time of the PT leg.
        """
        # get rid struct for all sections
        fm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_AMOD.value}"
        fm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_PT.value}"

        for amod_op_id in range(self.n_amod_op):
            # collect FM offers
            fm_amod_offer: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(fm_amod_rid_struct)
            LOG.debug(f"Collecting fm_amod offer for request {fm_amod_rid_struct} from operator {amod_op_id}: {fm_amod_offer}.")

            # check if FM AMoD offer is available
            if fm_amod_offer is None or fm_amod_offer.service_declined():
                LOG.info(f"FM AMoD offer is not available for sub_request {fm_amod_rid_struct}, skipping to next AMoD operator.")
                continue
            # register the FM AMoD offer in the sub-request
            self.demand[fm_amod_rid_struct].receive_offer(amod_op_id, fm_amod_offer, None)

            # create PT sub-request and inform PT operator
            transfer_station_ids: tp.List[str] = parent_rq_obj.get_transfer_station_ids()
            transfer_street_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")
            # determine the earliest start time of the PT sub-request based on the FM AMoD offer
            fm_est_pt_mod: int = self._determine_est_pt_mod(parent_rq_obj,amod_op_id, fm_amod_offer)
            # inform PT operator
            fm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(
                                                                            parent_rq_obj,
                                                                            RQ_SUB_TRIP_ID.FM_PT.value,
                                                                            transfer_street_node,
                                                                            parent_rq_obj.get_destination_node(), 
                                                                            fm_est_pt_mod,
                                                                            parent_modal_state,
                                                                            amod_op_id,
                                                                            )
            fm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(fm_pt_rid_struct, amod_op_id)
            # check if PT offer is available
            if fm_pt_arrival is None or fm_pt_offer is None or fm_pt_offer.service_declined():
                LOG.info(f"PT offer is not available for sub_request {fm_pt_rid_struct}, skipping to next AMoD operator.")
                continue
            # register the PT offer in the sub-request
            self.demand[fm_pt_rid_struct].receive_offer(self.pt_operator_id, fm_pt_offer, None)

            # create intermodal offer
            sub_trip_offers: tp.Dict[int, TravellerOffer] = {}
            sub_trip_offers[RQ_SUB_TRIP_ID.FM_AMOD.value] = fm_amod_offer
            sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value] = fm_pt_offer
            intermodal_offer: 'IntermodalOffer' = self._create_intermodal_offer(rid, sub_trip_offers, parent_modal_state)
            LOG.info(f"Created intermodal offer for request {rid}: {intermodal_offer}")

            # update FM latest dropoff time based on the PT offer
            sub_prq_obj: 'PlanRequest' = self.amod_operators[amod_op_id].rq_dict[fm_amod_rid_struct]
            old_t_do_latest: int = sub_prq_obj.t_do_latest
            new_t_do_latest: int = self._determine_amod_latest_dropoff_time(parent_rq_obj, fm_amod_offer, fm_pt_offer.get(G_OFFER_WAIT), old_t_do_latest)
            sub_prq_obj.set_new_dropoff_time_constraint(new_t_do_latest)

            # add intermodal offer to offers dictionary
            offers[intermodal_offer.operator_id] = intermodal_offer

        return offers
    
    def _process_collect_lastmile_offers(self, rid: int, parent_modal_state: RQ_MODAL_STATE, offers: tp.Dict[int, 'TravellerOffer']) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of LM offers.
        """
        # get LM PT offer
        lm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.LM_PT.value}"
        lm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(lm_pt_rid_struct)
        LOG.debug(f"Collecting lm_pt offer for request {lm_pt_rid_struct} from PT operator {self.pt_operator_id}: {lm_pt_offer}")
        
        if lm_pt_offer is not None and not lm_pt_offer.service_declined():
            # register the PT offer in the sub-request
            self.demand[lm_pt_rid_struct].receive_offer(self.pt_operator_id, lm_pt_offer, None)

            # get LM AMoD offers
            lm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.LM_AMOD.value}"
            for amod_op_id in range(self.n_amod_op):
                lm_amod_offer = self.amod_operators[amod_op_id].get_current_offer(lm_amod_rid_struct)
                LOG.debug(f"Collecting lm_amod offer for request {lm_amod_rid_struct} from operator {amod_op_id}: {lm_amod_offer}")
                
                if lm_amod_offer is not None and not lm_amod_offer.service_declined():
                    # register the LM AMoD offer in the sub-request
                    self.demand[lm_amod_rid_struct].receive_offer(amod_op_id, lm_amod_offer, None)
                    
                    # create intermodal offer
                    sub_trip_offers: tp.Dict[int, 'TravellerOffer'] = {}
                    sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value] = lm_pt_offer
                    sub_trip_offers[RQ_SUB_TRIP_ID.LM_AMOD.value] = lm_amod_offer
                    intermodal_offer: 'IntermodalOffer' = self._create_intermodal_offer(rid, sub_trip_offers, parent_modal_state)
                    offers[intermodal_offer.operator_id] = intermodal_offer
                else:
                    LOG.info(f"AMoD offer is not available for sub_request {lm_amod_rid_struct}, skipping to next AMoD operator.")
        else:
            LOG.info(f"PT offer is not available for sub_request {lm_pt_rid_struct}")
        return offers
    
    def _process_collect_firstlastmile_offers(
        self, rid: int, parent_rq_obj: 'BasicIntermodalRequest', parent_modal_state: RQ_MODAL_STATE, 
        offers: tp.Dict[int, 'TravellerOffer'], sim_time: int
    ) -> tp.Dict[int, 'TravellerOffer']:
        """This method processes the collection of firstlastmile offers.
        """
        # get rid struct for all sections
        flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
        flm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_PT.value}"
        flm_amod_rid_struct_1: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_1.value}"

        for amod_op_id in range(self.n_amod_op):
            # collect FM AMoD offer
            flm_amod_offer_0: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_0)
            LOG.debug(f"Collecting flm_amod_0 offer for request {flm_amod_rid_struct_0} from operator {amod_op_id}: {flm_amod_offer_0}.")

            if flm_amod_offer_0 is None or flm_amod_offer_0.service_declined():
                LOG.info(f"FM AMoD offer is not available for sub_request {flm_amod_rid_struct_0}, skipping to next AMoD operator.")
                continue
            # register the FM AMoD offer in the sub-request
            self.demand[flm_amod_rid_struct_0].receive_offer(amod_op_id, flm_amod_offer_0, None)

            # create PT sub-request and inform PT operator
            # get the transfer station ids and their closest pt stations
            transfer_station_ids: tp.List[str] = parent_rq_obj.get_transfer_station_ids()
            transfer_street_node_0, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")
            transfer_street_node_1, _ = self._find_transfer_info(transfer_station_ids[1], "pt2street")
            # determine the earliest start time of the PT sub-request based on the FM AMoD offer
            flm_est_pt_mod: int = self._determine_est_pt_mod(parent_rq_obj, amod_op_id, flm_amod_offer_0)
            # inform PT operator
            flm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(
                                                                            parent_rq_obj,
                                                                            RQ_SUB_TRIP_ID.FLM_PT.value,
                                                                            transfer_street_node_0,
                                                                            transfer_street_node_1,
                                                                            flm_est_pt_mod,
                                                                            parent_modal_state,
                                                                            amod_op_id,
                                                                            )
            flm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(flm_pt_rid_struct, amod_op_id)
            # check if PT offer is available
            if flm_pt_arrival is None or flm_pt_offer is None or flm_pt_offer.service_declined():
                LOG.info(f"PT offer is not available for sub_request {flm_pt_rid_struct}, skipping to next AMoD operator.")
                continue
            # register the PT offer in the sub-request
            self.demand[flm_pt_rid_struct].receive_offer(self.pt_operator_id, flm_pt_offer, None)

            # create LM AMoD sub-request and inform AMoD operator
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
            flm_amod_offer_1: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_1)
            # check if LM AMoD offer is available
            if flm_amod_offer_1 is None or flm_amod_offer_1.service_declined():
                LOG.info(f"LM AMoD offer is not available for sub_request {flm_amod_rid_struct_1}, skipping to next AMoD operator.")
                continue
            # register the LM AMoD offer in the sub-request
            self.demand[flm_amod_rid_struct_1].receive_offer(amod_op_id, flm_amod_offer_1, None)

            # create intermodal offer
            sub_trip_offers: tp.Dict[int, 'TravellerOffer'] = {}
            sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_0.value] = flm_amod_offer_0
            sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value] = flm_pt_offer
            sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_1.value] = flm_amod_offer_1
            intermodal_offer: 'IntermodalOffer' = self._create_intermodal_offer(rid, sub_trip_offers, parent_modal_state)
            LOG.info(f"Created intermodal offer for request {rid}: {intermodal_offer}")

            # update FM latest dropoff time based on the PT offer
            sub_prq_obj: 'PlanRequest' = self.amod_operators[amod_op_id].rq_dict[flm_amod_rid_struct_0]
            old_t_do_latest: int = sub_prq_obj.t_do_latest
            new_t_do_latest: int = self._determine_amod_latest_dropoff_time(parent_rq_obj, flm_amod_offer_0, flm_pt_offer.get(G_OFFER_WAIT), old_t_do_latest)
            sub_prq_obj.set_new_dropoff_time_constraint(new_t_do_latest)

            # add intermodal offer to offers dictionary
            offers[intermodal_offer.operator_id] = intermodal_offer

        return offers
        
    def _determine_est_pt_mod(self, rq_obj: 'RequestBase', amod_op_id: int, amod_offer: 'TravellerOffer') -> int:
        """This method determines the earliest start time for the pt sub-request.
        """
        t_est_pt_mod: int = rq_obj.earliest_start_time + amod_offer.get(G_OFFER_WAIT) + amod_offer.get(G_OFFER_DRIVE) + self.amod_operators[amod_op_id].const_bt
        return t_est_pt_mod
    
    def _determine_amod_latest_dropoff_time(self, rq_obj: 'RequestBase', amod_offer: 'TravellerOffer', pt_waiting_time: int, old_t_do_latest: int) -> tp.Optional[int]:
        """This method determines the latest dropoff time for the amod sub-request.
        """
        t_do_latest: int = rq_obj.earliest_start_time + amod_offer.get(G_OFFER_WAIT) + amod_offer.get(G_OFFER_DRIVE) + pt_waiting_time
        # add latest dropoff time constraint check
        if t_do_latest > old_t_do_latest:
            t_do_latest = old_t_do_latest
        return t_do_latest