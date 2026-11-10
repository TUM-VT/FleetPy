# TODO：
# - Adjust PT waiting time based on dynamic GTFS data (e.g., delays), and then adjust FM and LM offers accordingly.
# - Support multiple AMoD operators for firstlastmile requests.

# -------------------------------------------------------------------------------------------------------------------- #
# PTBrokerBasic: Base Intermodal Broker
#
# Base class for all PTBroker variants. Provides the shared infrastructure for intermodal request handling:
# - MONOMODAL / PT: Forward the request directly to the AMoD operator or PT operator.
# - FIRSTMILE (FM): Select a PT boarding transfer stop, create FM_AMOD sub-request
#   (origin → PT boarding stop), and create PT sub-request (boarding stop → destination).
# - LASTMILE (LM): Select a PT alighting transfer stop, create PT sub-request
#   (origin → PT alighting stop), and create LM_AMOD sub-request (PT alighting stop → destination).
# - FIRSTLASTMILE (FLM): Select boarding and alighting transfer stops, create FLM_AMOD_0 sub-request
#   (origin → PT boarding stop), PT sub-request (boarding → alighting), and FLM_AMOD_1 sub-request
#   (PT alighting stop → destination).
# Subclasses (PTBroker, PTBrokerEI, PTBrokerPAYG) override specific methods to implement different
# coordination strategies between MaaS and DRT.
#
# NOTE: This code has only been tested and applied in the ImmediateDecisionsSimulation environment
#       combined with the PoolingIRSOnly fleet controller.
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from math import floor
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
        
        self._amod_proposals = {}
        self._pt_requests = {}
        self._candidate_build_records = {}
        self._candidates = {}
        self._selected_candidates = {}
        self._confirmed_amod = {}
        
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


    def inform_request(self, rid, rq_obj, sim_time):
        """Query only the modes explicitly requested by this parent."""
        if rid in self._selected_candidates:
            return
        if rid in self._amod_proposals or rid in self._pt_requests:
            self._cleanup_proposals(rid, sim_time)
        self._amod_proposals[rid] = {}
        self._pt_requests[rid] = {}
        self._candidate_build_records[rid] = {}
        self._candidates[rid] = {}
        rq_obj.offer.clear()
        rq_obj.candidate_modes.clear()
        rq_obj.candidate_complete.clear()
        rq_obj.candidate_durations.clear()
        modes = rq_obj.get_query_modes()
        # Validate station references before creating any operator proposal.
        for mode in modes:
            for station in rq_obj.get_transfer_station_ids_for_mode(mode):
                self._find_transfer_info(station, "pt2street")
        handlers = {
            RQ_MODAL_STATE.MONOMODAL: self._process_inform_monomodal_request,
            RQ_MODAL_STATE.FIRSTMILE: self._process_inform_firstmile_request,
            RQ_MODAL_STATE.LASTMILE: self._process_inform_lastmile_request,
            RQ_MODAL_STATE.FIRSTLASTMILE: self._process_inform_firstlastmile_request,
        }
        for mode in modes:
            if mode == RQ_MODAL_STATE.PT:
                self._inform_pt_sub_request(
                    rq_obj, RQ_SUB_TRIP_ID.PT.value, rq_obj.get_origin_node(),
                    rq_obj.get_destination_node(), rq_obj.earliest_start_time, mode,
                )
            else:
                handlers[mode](rid, rq_obj, sim_time, mode)

    def _is_complete_candidate(self, mode):
        return True

    def collect_offers(self, rid, sim_time):
        parent = self.demand[rid]
        result = {}
        for mode in parent.get_query_modes():
            raw = {}
            if mode == RQ_MODAL_STATE.PT:
                offer = self.pt_operator.get_current_offer(f"{rid}_{RQ_SUB_TRIP_ID.PT.value}")
                if offer is not None and not offer.service_declined():
                    raw[self.pt_operator_id] = offer
            elif mode == RQ_MODAL_STATE.MONOMODAL:
                raw = self._process_collect_monomodal_offers(rid, mode, raw)
            elif mode == RQ_MODAL_STATE.FIRSTMILE:
                raw = self._process_collect_firstmile_offers(rid, parent, mode, raw)
            elif mode == RQ_MODAL_STATE.LASTMILE:
                raw = self._process_collect_lastmile_offers(rid, mode, raw)
            elif mode == RQ_MODAL_STATE.FIRSTLASTMILE:
                raw = self._process_collect_firstlastmile_offers(rid, parent, mode, raw, sim_time)
            for original_key, offer in raw.items():
                if offer is None or offer.service_declined():
                    continue
                key = (mode.value, original_key)
                candidate = self._candidate_build_records[rid].get((mode, original_key))
                if candidate is None:
                    candidate = self._build_candidate(rid, mode, original_key, offer)
                self._candidates[rid][key] = candidate
                parent.register_candidate(key, mode, candidate['is_complete'], candidate['expected_duration'])
                result[key] = offer
        return result

    def _build_candidate(self, rid, mode, original_key, offer):
        complete = self._is_complete_candidate(mode)
        if isinstance(offer, IntermodalOffer):
            sub_offers = offer.get_sub_trip_offers()
        elif mode == RQ_MODAL_STATE.MONOMODAL:
            sub_offers = {RQ_SUB_TRIP_ID.AMOD.value: offer}
        elif mode == RQ_MODAL_STATE.PT:
            sub_offers = {RQ_SUB_TRIP_ID.PT.value: offer}
        else:
            first_leg = {RQ_MODAL_STATE.FIRSTMILE: RQ_SUB_TRIP_ID.FM_AMOD,
                         RQ_MODAL_STATE.LASTMILE: RQ_SUB_TRIP_ID.LM_PT,
                         RQ_MODAL_STATE.FIRSTLASTMILE: RQ_SUB_TRIP_ID.FLM_AMOD_0}[mode]
            sub_offers = {first_leg.value: offer}
        firstmile_op = None
        for leg in (RQ_SUB_TRIP_ID.FM_AMOD.value, RQ_SUB_TRIP_ID.FLM_AMOD_0.value):
            if leg in sub_offers:
                firstmile_op = sub_offers[leg].operator_id
        children, amod, pt = {}, [], []
        for leg, sub_offer in sub_offers.items():
            child_id = rid if leg == RQ_SUB_TRIP_ID.AMOD.value else f"{rid}_{leg}"
            if sub_offer.operator_id == self.pt_operator_id:
                pt_key = (child_id, firstmile_op)
                child = deepcopy(self._pt_requests[rid][pt_key])
                pt.append(pt_key)
            else:
                binding = (sub_offer.operator_id, child_id)
                child = deepcopy(self._amod_proposals[rid][binding])
                amod.append(binding)
            if child_id != rid:
                child.offer = {sub_offer.operator_id: sub_offer}
                children[child_id] = child
        expected = None
        if complete:
            if isinstance(offer, IntermodalOffer):
                expected = offer.get(G_IM_OFFER_DURATION)
            elif mode == RQ_MODAL_STATE.MONOMODAL:
                expected = offer.get(G_OFFER_WAIT) + offer.get(G_OFFER_DRIVE)
            elif mode == RQ_MODAL_STATE.PT:
                pt_offer = next(sub for sub in sub_offers.values() if sub.operator_id == self.pt_operator_id)
                expected = pt_offer.destination_node_arrival_time - self.demand[rid].rq_time
        return dict(mode=mode, offer=offer, original_key=original_key,
                    child_requests=children, proposal_bindings=tuple(amod),
                    pt_bindings=tuple(pt), is_complete=complete, expected_duration=expected)

    def resolve_selected_candidate(self, rid, chosen_key):
        try:
            return self._candidates[rid][chosen_key]
        except KeyError as exc:
            raise ValueError(f"Unknown PTBroker candidate {chosen_key!r} for request {rid}") from exc

    def prepare_candidate_booking(self, rid, rq_obj, sim_time, chosen_key):
        candidate = self.resolve_selected_candidate(rid, chosen_key)
        if candidate['offer'].service_declined():
            raise ValueError(f"Cannot confirm declined candidate {chosen_key!r}")
        if rid in self._selected_candidates:
            raise ValueError(f"Request {rid} already has an accepted candidate")
        for pt_key in candidate['pt_bindings']:
            pt_offer = self.pt_operator.get_current_offer(*pt_key)
            if pt_offer is None or pt_offer.service_declined():
                raise ValueError(f"Missing feasible PT proposal {pt_key!r}")
        self._cleanup_proposals(rid, sim_time, candidate)
        for child_id, child in candidate['child_requests'].items():
            child.selected_mode = candidate['mode']
            child.selected_candidate_id = chosen_key
            child.chosen_operator_id = next(iter(child.offer))
            child.fare = next(iter(child.offer.values())).get(G_OFFER_FARE, 0)
            self.demand.rq_db[child_id] = child
        rq_obj.selected_mode = candidate['mode']
        rq_obj.selected_candidate_id = chosen_key
        rq_obj.chosen_operator_id = chosen_key
        rq_obj.modal_state = candidate['mode']
        rq_obj.fare = candidate['offer'].get(G_OFFER_FARE, 0)
        self._selected_candidates[rid] = candidate
        self._candidates[rid] = {chosen_key: candidate}
        self._candidate_build_records.pop(rid, None)
        return candidate

    def _confirm_amod_binding(self, rid, op_id, request_id, sim_time):
        # Reservation routes can predate the selected child snapshot in demand.
        # Refresh supported controllers after binding the child and before confirmation.
        if request_id != rid:
            refresh = getattr(self.amod_operators[op_id], 'refresh_reservation_proposal', None)
            if refresh is not None:
                refresh(request_id, sim_time)
        self.amod_operators[op_id].user_confirms_booking(request_id, sim_time)
        self._confirmed_amod.setdefault(rid, set()).add((op_id, request_id))

    def _cancel_amod_binding(self, rid, op_id, request_id, sim_time):
        proposals = self._amod_proposals.get(rid, {})
        if (op_id, request_id) in proposals:
            self.amod_operators[op_id].user_cancels_request(request_id, sim_time)
            del proposals[(op_id, request_id)]
            self._confirmed_amod.get(rid, set()).discard((op_id, request_id))

    def _cancel_pt_binding(self, rid, child_id, firstmile_op, sim_time, *, missed_connection=False):
        requests = self._pt_requests.get(rid, {})
        if (child_id, firstmile_op) in requests:
            if not self.pt_operator.user_cancels_request(
                child_id, sim_time, firstmile_op, missed_connection=missed_connection,
            ):
                return False
            del requests[(child_id, firstmile_op)]
            return True
        return False

    def _bind_amod_child(self, rid, op_id, request_id, offer):
        child = deepcopy(self._amod_proposals[rid][(op_id, request_id)])
        parent = self.demand[rid]
        child.offer = {op_id: offer}
        child.selected_mode = parent.selected_mode
        child.selected_candidate_id = parent.selected_candidate_id
        child.chosen_operator_id = op_id
        child.fare = offer.get(G_OFFER_FARE, 0)
        self.demand.rq_db[request_id] = child
        return child

    def confirm_candidate_pt(self, candidate):
        for child_id, firstmile_op in candidate['pt_bindings']:
            child = self.demand[child_id]
            self.pt_operator.user_confirms_booking(child, firstmile_op)

    def inform_user_booking(self, rid, rq_obj, sim_time, chosen_operator):
        candidate = self.prepare_candidate_booking(rid, rq_obj, sim_time, chosen_operator)
        for op_id, request_id in candidate['proposal_bindings']:
            self._confirm_amod_binding(rid, op_id, request_id, sim_time)
        self.confirm_candidate_pt(candidate)
        for operator in self.amod_operators:
            if hasattr(operator, 'flm_excluded_vid'):
                operator.flm_excluded_vid.pop(rid, None)
        return [(rid, rq_obj)]

    def _cleanup_proposals(self, rid, sim_time, selected=None):
        keep_amod = set(selected['proposal_bindings']) if selected else set()
        keep_pt = set(selected['pt_bindings']) if selected else set()
        keep_children = set(selected['child_requests']) if selected else set()
        keep_children.update(
            child_id for child_id, child in self.demand.rq_db.items()
            if isinstance(child_id, str) and child_id.startswith(f"{rid}_")
            and child.pt_leg_state is not None
        )
        proposals = self._amod_proposals.get(rid, {})
        for binding in list(proposals):
            if binding not in keep_amod:
                op_id, request_id = binding
                self._cancel_amod_binding(rid, op_id, request_id, sim_time)
        pt_requests = self._pt_requests.get(rid, {})
        for binding in list(pt_requests):
            if binding not in keep_pt:
                self._cancel_pt_binding(rid, binding[0], binding[1], sim_time)
                child = self.demand.rq_db.get(binding[0])
                if child is not None and child.pt_leg_state is not None:
                    # Keep accepted/cancelled PT execution rows for output and audit.
                    keep_children.add(binding[0])
        for child_id in list(self.demand.rq_db):
            if isinstance(child_id, str) and child_id.startswith(f"{rid}_") and child_id not in keep_children:
                del self.demand.rq_db[child_id]
        if selected is None or selected['mode'] != RQ_MODAL_STATE.FIRSTLASTMILE:
            for operator in self.amod_operators:
                if hasattr(operator, 'flm_excluded_vid'):
                    operator.flm_excluded_vid.pop(rid, None)

    def inform_user_leaving_system(self, rid, sim_time):
        self._cleanup_proposals(rid, sim_time)
        if self._pt_requests.get(rid):
            # A refused cancellation must not discard its active execution binding.
            return False
        self._amod_proposals.pop(rid, None)
        self._pt_requests.pop(rid, None)
        self._candidate_build_records.pop(rid, None)
        self._candidates.pop(rid, None)
        self._selected_candidates.pop(rid, None)
        self._confirmed_amod.pop(rid, None)
        return True

    def inform_waiting_request_cancellations(self, chosen_operator, rid, sim_time):
        return self.inform_user_leaving_system(rid, sim_time)

    def acknowledge_user_alighting(self, op_id, rid, vid, alighting_time):
        super().acknowledge_user_alighting(op_id, rid, vid, alighting_time)
        parent_rid = int(str(rid).rsplit("_", 1)[0])
        self._amod_proposals.get(parent_rid, {}).pop((op_id, rid), None)
        self._confirmed_amod.get(parent_rid, set()).discard((op_id, rid))

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
            self._amod_proposals[rid][(op_id, rid)] = rq_obj

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

    def _create_sub_request(self, rq_obj, sub_trip_id, leg_o_node, leg_d_node, leg_start_time, mode):
        child = self.demand.create_sub_requests(
            rq_obj, sub_trip_id, (leg_o_node, None, None), (leg_d_node, None, None), leg_start_time, mode,
        )
        child.offer = {}
        child.chosen_operator_id = None
        child.selected_candidate_id = None
        child.selected_mode = None
        child.modal_state = mode
        child.candidate_modes = {}
        child.candidate_complete = {}
        child.candidate_durations = {}
        child.pu_time = child.do_time = child.service_opid = child.service_vid = None
        child.pu_pos = child.do_pos = child.leave_system_time = None
        child.t_access = child.t_egress = child.fare = None
        child.payg_interrupted = False
        child.uncatchable_pt = False
        child.pt_leg_state = None
        child.pt_cancel_time = None
        child.pt_cancel_reason = None
        return child

    def _inform_amod_sub_request(
        self, rq_obj, sub_trip_id, leg_o_node, leg_d_node, leg_start_time,
        parent_modal_state, op_id, sim_time, max_wait_time=None,
    ):
        child = self._create_sub_request(rq_obj, sub_trip_id, leg_o_node, leg_d_node, leg_start_time, parent_modal_state)
        self.amod_operators[op_id].user_request(child, sim_time, max_wait_time=max_wait_time)
        self._amod_proposals.setdefault(rq_obj.get_rid(), {})[(op_id, child.get_rid_struct())] = child

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
        pt_sub_rq_obj: RequestBase = self._create_sub_request(rq_obj, sub_trip_id, leg_o_node, leg_d_node, leg_start_time, parent_modal_state)
        self._pt_requests.setdefault(rq_obj.get_rid(), {})[(pt_sub_rq_obj.get_rid_struct(), firstmile_amod_operator_id)] = pt_sub_rq_obj
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

    def _get_current_datetime(self, sim_time_in_seconds: float) -> datetime:
        """Represent readiness without moving it to another integer-second interval.

        Datetime has microsecond precision. Keep a positive fraction positive so
        the router's integer-second ceiling cannot admit an already departed trip.
        The caller restores the original float after the query.
        """
        whole_seconds = floor(sim_time_in_seconds)
        fraction = sim_time_in_seconds - whole_seconds
        microseconds = min(999999, max(1, floor(fraction * 1000000))) if fraction else 0
        return self.sim_start_datetime + timedelta(seconds=whole_seconds, microseconds=microseconds)

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
        self, o_node: int, d_node: int, est: float,
        max_transfers: int = 999, detailed: bool = False
    ) -> tp.Optional[tp.Tuple[int, int, int, int, tp.Dict[str, tp.Any]]]:
        """This method queries the pt travel costs between two street nodes at a given datetime.
        The pt station ids will be the closest pt station ids to the street nodes.

        Args:
            o_node (int): The origin street node id.
            d_node (int): The destination street node id.
            est (float): The earliest start time of the request at the origin street node in seconds.
            max_transfers (int): Non-negative maximum PT transfers; 0 permits direct journeys only. Defaults to 999.
            detailed (bool): Whether to return detailed journey information. Defaults to False.
        Returns:
            tp.Optional[tp.Tuple[int, int, int, int, tp.Dict[str, tp.Any]]]:
                Returns a tuple containing:
                (source_pt_station_id, t_source_walking, target_pt_station_id, t_target_walking, pt_journey_plan_dict)
                if a public transport journey plan is found.
                Returns None if no public transport journey plan is available.
                Journey and step timestamps use the simulation time axis. The router's
                day metadata remains relative to the query date. Readiness, waiting,
                and total duration retain the original float precision.
        """
        # find pt transfer stations
        source_pt_station_id, t_source_walk = self._find_transfer_info(o_node, "street2pt")
        target_pt_station_id, t_target_walk = self._find_transfer_info(d_node, "street2pt")

        source_station_departure_seconds = est + t_source_walk
        source_station_departure_datetime: datetime = self._get_current_datetime(source_station_departure_seconds)
        LOG.debug(f"Query PT travel costs: {o_node} -> {d_node} (stations: {source_pt_station_id} -> {target_pt_station_id}) at {est} (station departure: {source_station_departure_datetime})")
        pt_journey_plan_dict: tp.Union[tp.Dict[str, tp.Any], None] = self.pt_operator.return_fastest_pt_journey_1to1(
                                                                                                                    source_pt_station_id, target_pt_station_id, 
                                                                                                                    source_station_departure_datetime,
                                                                                                                    max_transfers, detailed,
                                                                                                                    )
        if pt_journey_plan_dict is None:
            return None

        # The router counts seconds from the actual query date's midnight.
        # Street-to-station walking can move that date into the next simulation day.
        query_day_offset = (
            source_station_departure_datetime.date() - self.sim_start_datetime.date()
        ).days * 86400
        pt_journey_plan_dict = dict(pt_journey_plan_dict)
        for field in (G_PT_OFFER_SOURCE_STATION_DEPARTURE_TIME, G_PT_OFFER_TARGET_STATION_ARRIVAL_TIME):
            pt_journey_plan_dict[field] += query_day_offset
        pt_journey_plan_dict[G_PT_OFFER_STEPS] = [
            {
                **step,
                "departure_time": step["departure_time"] + query_day_offset,
                "arrival_time": step["arrival_time"] + query_day_offset,
            }
            for step in pt_journey_plan_dict[G_PT_OFFER_STEPS]
        ]
        # Compare fractions within one second instead of subtracting large global
        # timestamps, which would lose the sub-microsecond correction.
        readiness_error = source_station_departure_datetime.microsecond / 1000000 - (
            source_station_departure_seconds - floor(source_station_departure_seconds)
        )
        pt_journey_plan_dict[G_PT_OFFER_SOURCE_STATION_DEPARTURE_TIME] = source_station_departure_seconds
        if readiness_error:
            pt_journey_plan_dict[G_PT_OFFER_SOURCE_WAITING_TIME] += readiness_error
            pt_journey_plan_dict[G_PT_OFFER_DURATION] += readiness_error
        return source_pt_station_id, t_source_walk, target_pt_station_id, t_target_walk, pt_journey_plan_dict
        
    def _find_transfer_info(self, node_id: tp.Union[int, str], direction: str) -> tp.Tuple[str, int]:
        """This method finds the transfer possibility between pt station and street node.

        The mapping DataFrame is read-only during operation. Assigning a replacement
        DataFrame, including after a reload, invalidates cached successful lookups.
        First lookups retain the DataFrame's row and minimum-tie ordering.

        Args:
            node_id (tp.Union[int, str]): The street node id.
            direction (str): "pt2street" or "street2pt"
        Returns:
            tp.Tuple[str, int]: The transfer node id and the walking time.
        """
        if self.transfer_search_method == "closest":  # closest station or street node
            if direction == "street2pt":
                street_node_id = int(node_id)
                cache_key = (direction, street_node_id)
            elif direction == "pt2street":
                pt_station_id = str(node_id)
                cache_key = (direction, pt_station_id)
            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'pt2street' or 'street2pt'.")
            transfers = self.street_station_transfers_fp_df
            cache_state = getattr(self, "_street_station_cache", None)
            if cache_state is None or cache_state[0] is not transfers:
                cache_state = (transfers, {})
                self._street_station_cache = cache_state
            cache = cache_state[1]
            if cache_key in cache:
                return cache[cache_key]

            if direction == "street2pt":  # find closest pt station from street node
                # TODO: allow multiple closest stations?
                street_station_transfer = transfers[transfers["node_id"] == street_node_id]
                if street_station_transfer.empty:
                    raise ValueError(f"Street node id {street_node_id} not found in the street station transfers file")
                closest_station_id: str = street_station_transfer["closest_station_id"].iloc[0]
                walking_time: int = street_station_transfer["street_station_transfer_time"].iloc[0]
                result = closest_station_id, walking_time
            elif direction == "pt2street":  # find closest street node from pt station
                street_station_transfers = transfers[transfers["closest_station_id"] == pt_station_id]
                if street_station_transfers.empty:
                    raise ValueError(f"PT station id {pt_station_id} not found in the street station transfers file")
                # find the record with the minimum street_station_transfer_time
                # TODO: if multiple exist, return all?
                min_transfer = street_station_transfers.loc[street_station_transfers["street_station_transfer_time"].idxmin()]
                closest_street_node_id: int = min_transfer["node_id"]
                walking_time: int = min_transfer["street_station_transfer_time"]
                result = closest_street_node_id, walking_time
            # Non-unique DataFrame indices can return Series from .loc. Preserve
            # that existing behavior without sharing mutable results in the cache.
            if all(pd.api.types.is_scalar(value) for value in result):
                cache[cache_key] = result
            return result
        else:
            LOG.debug(f"PTBrokerTPCS: Transfer search method '{self.transfer_search_method}' not implemented. Using 'closest' instead.")
            raise NotImplementedError(f"PTBrokerTPCS: Transfer search method '{self.transfer_search_method}' not implemented.")
        
    def _create_intermodal_offer(self, rid: int, sub_trip_offers: tp.Dict[int, 'TravellerOffer'], rq_modal_state: RQ_MODAL_STATE) -> 'IntermodalOffer':
        """This method merges the amod and pt offers into an intermodal offer.
        """
        lastmile_id = {
            RQ_MODAL_STATE.LASTMILE: RQ_SUB_TRIP_ID.LM_AMOD.value,
            RQ_MODAL_STATE.FIRSTLASTMILE: RQ_SUB_TRIP_ID.FLM_AMOD_1.value,
        }.get(rq_modal_state)
        lastmile_rq_time = None
        if lastmile_id is not None:
            lastmile_offer = sub_trip_offers[lastmile_id]
            lastmile_rq_time = self._amod_proposals[rid][
                (lastmile_offer.operator_id, lastmile_offer.traveler_id)].rq_time
        offer = IntermodalOffer(rid, sub_trip_offers, rq_modal_state, self.demand[rid].rq_time, lastmile_rq_time)
        candidate = self._build_candidate(rid, rq_modal_state, offer.operator_id, offer)
        self._candidate_build_records[rid][(rq_modal_state, offer.operator_id)] = candidate
        return offer
