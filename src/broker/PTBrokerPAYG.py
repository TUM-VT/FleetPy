# -------------------------------------------------------------------------------------------------------------------- #
# PTBrokerPAYG: Plan-As-You-Go (PAYG) Broker Strategy
#
# This broker simulates traveler behavior without a MaaS platform — travelers plan their trip step by step,
# querying the next leg only after completing the current one:
# - FIRSTMILE (FM): At request time, create only FM_AMOD (origin → PT boarding stop).
#   After AMoD alighting, query PT in real-time and create the PT sub-request.
# - LASTMILE (LM): At request time, query PT (origin → PT alighting stop).
#   After PT alighting, request LM_AMOD in real-time (PT alighting stop → destination).
# - FIRSTLASTMILE (FLM): At request time, create only FLM_AMOD_0 (origin → PT boarding stop).
#   After AMoD alighting, query PT in real-time. After PT alighting, request FLM_AMOD_1 in real-time.
# If any step fails to find an available service, the trip is marked as interrupted.
# Unlike PTBrokerBasic (TPCS), no pre-planned combined offer is presented to the user upfront.
#
# NOTE: This code has only been tested and applied in the ImmediateDecisionsSimulation environment
#       combined with the PoolingIRSOnly fleet controller.
# -------------------------------------------------------------------------------------------------------------------- #

# standard distribution imports
import logging
import typing as tp

# src imports
from src.broker.PTBrokerBasic import PTBrokerBasic
from src.simulation.Offers import TravellerOffer

if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.ptctrl.PTControlBase import PTControlBase
    from src.demand.demand import Demand
    from src.routing.road.NetworkBase import NetworkBase
    from src.demand.TravelerModels import RequestBase, BasicIntermodalRequest
    from src.simulation.Offers import PTOffer

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
from src.misc.globals import *

LOG = logging.getLogger(__name__)


INPUT_PARAMETERS_PTBrokerPAYG = {
    "doc": "Plan-As-You-Go broker: simulates travelers planning trips step by step without MaaS platform integration",
    "inherit": PTBrokerBasic,
    "input_parameters_mandatory": ["n_amod_op", "amod_operators", "pt_operator", "demand", "routing_engine", "scenario_parameters"],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class PTBrokerPAYG(PTBrokerBasic):
    """
    Plan-As-You-Go (PAYG) broker strategy.

    Unlike typical PTBroker which plans the entire intermodal trip upfront, PAYG simulates
    travelers who plan only the next step of their journey:

    - FM/FLM requests: Only create FM_AMOD at request time; PT is queried after AMoD alighting
    - LM requests: Query PT at request time; LM_AMOD is requested after PT alighting
    - FLM requests: FM_AMOD -> (alighting) -> PT query -> (PT arrival) -> LM_AMOD request

    If any step fails to find an available service, the trip is marked as interrupted.
    """

    def __init__(
        self,
        n_amod_op: int,
        amod_operators: tp.List['FleetControlBase'],
        pt_operator: 'PTControlBase',
        demand: 'Demand',
        routing_engine: 'NetworkBase',
        scenario_parameters: dict,
    ):
        super().__init__(n_amod_op, amod_operators, pt_operator, demand, routing_engine, scenario_parameters)

        # PAYG-specific state tracking
        self.payg_trip_states: tp.Dict[int, PAYG_TRIP_STATE] = {}  # rid -> state

        # Pending PT arrivals: rid -> (pt_arrival_time, pt_alighting_node)
        # Used to trigger LM_AMOD requests when PT arrives
        self.pending_pt_arrivals: tp.Dict[int, tp.Tuple[int, int]] = {}

    # ============================================================================================================== #
    # Request Processing Methods
    # ============================================================================================================== #

    def _process_inform_firstmile_request(
        self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int,
        parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTMILE
    ):
        """Process FM request: Only create FM_AMOD sub-request.
        PT will be queried after FM_AMOD alighting.
        """
        # Get transfer station
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")

        # Create FM_AMOD sub-request for each operator
        for op_id in range(self.n_amod_op):
            self._inform_amod_sub_request(
                rq_obj, RQ_SUB_TRIP_ID.FM_AMOD.value,
                rq_obj.get_origin_node(), transfer_street_node,
                rq_obj.earliest_start_time, parent_modal_state, op_id, sim_time
            )

        # Initialize PAYG state
        self.payg_trip_states[rid] = PAYG_TRIP_STATE.PENDING
        LOG.debug(f"PAYG FM request {rid}: Created FM_AMOD sub-request, PT will be queried after alighting")

    def _process_inform_lastmile_request(
        self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int,
        parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.LASTMILE
    ):
        """Process LM request: Query PT first. LM_AMOD will be scheduled after user confirms booking.
        """
        # Get transfer station
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")

        # Query PT immediately
        lm_pt_arrival: tp.Optional[int] = self._inform_pt_sub_request(
            rq_obj, RQ_SUB_TRIP_ID.LM_PT.value,
            rq_obj.get_origin_node(), transfer_street_node,
            rq_obj.earliest_start_time, parent_modal_state
        )

        if lm_pt_arrival is not None:
            # PT offer available - state will be updated when user confirms booking
            self.payg_trip_states[rid] = PAYG_TRIP_STATE.PENDING
            LOG.debug(f"PAYG LM request {rid}: PT offer available, waiting for user booking confirmation")
        else:
            # No PT available - mark as interrupted
            self._mark_trip_interrupted(rid, PAYG_TRIP_STATE.INTERRUPTED_NO_PT, sim_time)
            LOG.info(f"PAYG LM request {rid}: No PT available, trip interrupted")

    def _process_inform_firstlastmile_request(
        self, rid: int, rq_obj: 'BasicIntermodalRequest', sim_time: int,
        parent_modal_state: RQ_MODAL_STATE = RQ_MODAL_STATE.FIRSTLASTMILE
    ):
        """Process FLM request: Only create FLM_AMOD_0 sub-request.
        PT and LM_AMOD will be created after respective alighting events.
        """
        # Get first transfer station
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()
        transfer_street_node_0, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")

        # Create FLM_AMOD_0 sub-request for each operator
        for op_id in range(self.n_amod_op):
            self._inform_amod_sub_request(
                rq_obj, RQ_SUB_TRIP_ID.FLM_AMOD_0.value,
                rq_obj.get_origin_node(), transfer_street_node_0,
                rq_obj.earliest_start_time, parent_modal_state, op_id, sim_time
            )

        # Initialize PAYG state
        self.payg_trip_states[rid] = PAYG_TRIP_STATE.PENDING
        LOG.debug(f"PAYG FLM request {rid}: Created FLM_AMOD_0 sub-request, PT will be queried after alighting")

    # ============================================================================================================== #
    # Offer Collection Methods
    # ============================================================================================================== #

    def collect_offers(self, rid: int, sim_time: int) -> tp.Dict[int, 'TravellerOffer']:
        """Collect offers, processing any pending PT arrivals first."""
        # Process pending PT arrivals before collecting offers
        self._process_pending_pt_arrivals(sim_time)

        # Call parent implementation
        return super().collect_offers(rid, sim_time)

    def _process_collect_firstmile_offers(
        self, rid: int, parent_rq_obj: 'BasicIntermodalRequest', parent_modal_state: RQ_MODAL_STATE,
        offers: tp.Dict[int, 'TravellerOffer']
    ) -> tp.Dict[int, 'TravellerOffer']:
        """Collect FM offers: Only return FM_AMOD offers (PT is not yet queried in PAYG mode)."""
        fm_amod_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.FM_AMOD.value}"

        for amod_op_id in range(self.n_amod_op):
            fm_amod_offer: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(fm_amod_rid_struct)
            LOG.debug(f"PAYG collecting FM_AMOD offer for {fm_amod_rid_struct} from operator {amod_op_id}: {fm_amod_offer}")

            if fm_amod_offer is not None and not fm_amod_offer.service_declined():
                # Register offer
                self.demand[fm_amod_rid_struct].receive_offer(amod_op_id, fm_amod_offer, None)
                # Return single-leg offer (not IntermodalOffer since PT is unknown)
                offers[amod_op_id] = fm_amod_offer

        return offers

    def _process_collect_lastmile_offers(
        self, rid: int, parent_modal_state: RQ_MODAL_STATE,
        offers: tp.Dict[int, 'TravellerOffer']
    ) -> tp.Dict[int, 'TravellerOffer']:
        """Collect LM offers: Return PT offer only (LM_AMOD not yet requested in PAYG mode)."""
        lm_pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.LM_PT.value}"
        lm_pt_offer: 'TravellerOffer' = self.pt_operator.get_current_offer(lm_pt_rid_struct)
        LOG.debug(f"PAYG collecting LM_PT offer for {lm_pt_rid_struct}: {lm_pt_offer}")

        if lm_pt_offer is not None and not lm_pt_offer.service_declined():
            self.demand[lm_pt_rid_struct].receive_offer(self.pt_operator_id, lm_pt_offer, None)
            # Return PT offer only
            offers[self.pt_operator_id] = lm_pt_offer

        return offers

    def _process_collect_firstlastmile_offers(
        self, rid: int, parent_rq_obj: 'BasicIntermodalRequest', parent_modal_state: RQ_MODAL_STATE,
        offers: tp.Dict[int, 'TravellerOffer'], sim_time: int
    ) -> tp.Dict[int, 'TravellerOffer']:
        """Collect FLM offers: Only return FLM_AMOD_0 offers (PT and LM not yet queried in PAYG mode)."""
        flm_amod_rid_struct_0: str = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"

        for amod_op_id in range(self.n_amod_op):
            flm_amod_offer_0: 'TravellerOffer' = self.amod_operators[amod_op_id].get_current_offer(flm_amod_rid_struct_0)
            LOG.debug(f"PAYG collecting FLM_AMOD_0 offer for {flm_amod_rid_struct_0} from operator {amod_op_id}: {flm_amod_offer_0}")

            if flm_amod_offer_0 is not None and not flm_amod_offer_0.service_declined():
                self.demand[flm_amod_rid_struct_0].receive_offer(amod_op_id, flm_amod_offer_0, None)
                # Return single-leg offer
                offers[amod_op_id] = flm_amod_offer_0

        return offers

    # ============================================================================================================== #
    # Booking Methods
    # ============================================================================================================== #

    def inform_user_booking(self, rid: int, rq_obj: 'RequestBase', sim_time: int, chosen_operator: tp.Union[int, tuple]) -> tp.List[tuple[int, 'RequestBase']]:
        """Handle user booking for PAYG mode."""
        amod_confirmed_rids = []
        parent_modal_state: RQ_MODAL_STATE = rq_obj.get_modal_state()

        # Check if this is a pure PT offer selection (not LM PT offer in PAYG mode)
        # For LM requests in PAYG, chosen_operator == pt_operator_id means LM_PT offer, not pure PT
        if chosen_operator == self.pt_operator_id and parent_modal_state != RQ_MODAL_STATE.LASTMILE:
            amod_confirmed_rids.append((rid, rq_obj))
            # inform all AMoD operators that the request is cancelled
            self.inform_user_leaving_system(rid, sim_time)
            pt_rid_struct: str = f"{rid}_{RQ_SUB_TRIP_ID.PT.value}"
            pt_sub_rq_obj = self.demand[pt_rid_struct]
            self.pt_operator.user_confirms_booking(pt_sub_rq_obj, None)
            return amod_confirmed_rids

        # PAYG mode: Handle single-leg bookings for FM/FLM
        if parent_modal_state == RQ_MODAL_STATE.MONOMODAL or parent_modal_state == RQ_MODAL_STATE.PT:
            # Standard monomodal handling
            for i, operator in enumerate(self.amod_operators):
                if i != chosen_operator:
                    operator.user_cancels_request(rid, sim_time)
                else:
                    operator.user_confirms_booking(rid, sim_time)
                    amod_confirmed_rids.append((rid, rq_obj))

        elif parent_modal_state == RQ_MODAL_STATE.FIRSTMILE:
            # PAYG FM: Book only FM_AMOD, PT will be queried later
            fm_amod_rid_struct = f"{rid}_{RQ_SUB_TRIP_ID.FM_AMOD.value}"
            for i, operator in enumerate(self.amod_operators):
                if i != chosen_operator:
                    operator.user_cancels_request(fm_amod_rid_struct, sim_time)
                else:
                    operator.user_confirms_booking(fm_amod_rid_struct, sim_time)
            self.payg_trip_states[rid] = PAYG_TRIP_STATE.FM_AMOD_BOOKED
            amod_confirmed_rids.append((rid, rq_obj))
            LOG.debug(f"PAYG FM booking {rid}: FM_AMOD booked with operator {chosen_operator}")

        elif parent_modal_state == RQ_MODAL_STATE.LASTMILE:
            # PAYG LM: Book PT, schedule LM_AMOD request for after PT arrival
            lm_pt_rid_struct = f"{rid}_{RQ_SUB_TRIP_ID.LM_PT.value}"
            lm_pt_sub_rq_obj = self.demand[lm_pt_rid_struct]
            self.pt_operator.user_confirms_booking(lm_pt_sub_rq_obj, None)

            # Get PT offer to retrieve arrival time and transfer node
            lm_pt_offer: 'PTOffer' = self.pt_operator.get_current_offer(lm_pt_rid_struct)
            if lm_pt_offer is not None and not lm_pt_offer.service_declined():
                pt_arrival_time = lm_pt_offer.destination_node_arrival_time
                # Get transfer street node from request
                transfer_station_ids = rq_obj.get_transfer_station_ids()
                transfer_street_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")
                # Schedule LM_AMOD request for PT arrival
                self.pending_pt_arrivals[rid] = (pt_arrival_time, transfer_street_node)
                LOG.debug(f"PAYG LM booking {rid}: PT booked, LM_AMOD scheduled at {pt_arrival_time}")

            self.payg_trip_states[rid] = PAYG_TRIP_STATE.PT_BOOKED
            amod_confirmed_rids.append((rid, rq_obj))
            LOG.debug(f"PAYG LM booking {rid}: PT booked")

        elif parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            # PAYG FLM: Book only FLM_AMOD_0, PT and LM will be handled later
            flm_amod_rid_struct_0 = f"{rid}_{RQ_SUB_TRIP_ID.FLM_AMOD_0.value}"
            for i, operator in enumerate(self.amod_operators):
                if i != chosen_operator:
                    operator.user_cancels_request(flm_amod_rid_struct_0, sim_time)
                else:
                    operator.user_confirms_booking(flm_amod_rid_struct_0, sim_time)
            self.payg_trip_states[rid] = PAYG_TRIP_STATE.FM_AMOD_BOOKED
            amod_confirmed_rids.append((rid, rq_obj))
            LOG.debug(f"PAYG FLM booking {rid}: FLM_AMOD_0 booked with operator {chosen_operator}")

        else:
            raise ValueError(f"Invalid modal state: {parent_modal_state}")

        return amod_confirmed_rids

    # ============================================================================================================== #
    # Alighting Event Handlers
    # ============================================================================================================== #

    def acknowledge_user_alighting(self, op_id: int, rid_struct: str, vid: int, alighting_time: int):
        """Handle AMoD alighting events - trigger next step in PAYG flow."""
        # Call parent implementation first
        super().acknowledge_user_alighting(op_id, rid_struct, vid, alighting_time)

        # Parse rid_struct
        rid_struct_str = str(rid_struct)
        if "_" not in rid_struct_str:
            return  # Not a sub-request

        parts = rid_struct_str.rsplit("_", 1)
        parent_rid = int(parts[0])
        sub_trip_id = int(parts[1])

        parent_rq_obj: 'BasicIntermodalRequest' = self.demand[parent_rid]
        parent_modal_state: RQ_MODAL_STATE = parent_rq_obj.get_modal_state()

        # Get alighting node from parent request's transfer station info
        # (sub-request is already deleted from rq_db by user_ends_alighting)
        transfer_station_ids: tp.List[str] = parent_rq_obj.get_transfer_station_ids()

        # FM_AMOD alighting -> Query PT
        if sub_trip_id == RQ_SUB_TRIP_ID.FM_AMOD.value:
            # FM_AMOD destination is the first transfer station
            alighting_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")
            self._handle_fm_amod_alighting(parent_rid, parent_rq_obj, op_id, alighting_time, parent_modal_state, alighting_node)

        # FLM_AMOD_0 alighting -> Query PT
        elif sub_trip_id == RQ_SUB_TRIP_ID.FLM_AMOD_0.value:
            # FLM_AMOD_0 destination is the first transfer station
            alighting_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")
            self._handle_flm_amod_0_alighting(parent_rid, parent_rq_obj, op_id, alighting_time, parent_modal_state, alighting_node)

        # LM_AMOD alighting -> Trip completed
        elif sub_trip_id == RQ_SUB_TRIP_ID.LM_AMOD.value:
            self._handle_trip_completed(parent_rid, alighting_time)

        # FLM_AMOD_1 alighting -> Trip completed
        elif sub_trip_id == RQ_SUB_TRIP_ID.FLM_AMOD_1.value:
            self._handle_trip_completed(parent_rid, alighting_time)

    def _handle_fm_amod_alighting(
        self, rid: int, rq_obj: 'BasicIntermodalRequest', amod_op_id: int,
        alighting_time: int, parent_modal_state: RQ_MODAL_STATE, alighting_node: int
    ):
        """Handle FM_AMOD alighting: Query PT in real-time."""
        LOG.debug(f"PAYG FM alighting: rid={rid}, time={alighting_time}, node={alighting_node}")

        # Query PT from current location/time to destination
        pt_arrival = self._inform_pt_sub_request(
            rq_obj, RQ_SUB_TRIP_ID.FM_PT.value,
            alighting_node, rq_obj.get_destination_node(),
            alighting_time, parent_modal_state, amod_op_id
        )

        if pt_arrival is None:
            # No PT available
            self._mark_trip_interrupted(rid, PAYG_TRIP_STATE.INTERRUPTED_NO_PT, alighting_time)
            LOG.info(f"PAYG FM {rid}: No PT available after alighting, trip interrupted")
            return

        # Get PT offer and record it on sub-request
        pt_rid_struct = f"{rid}_{RQ_SUB_TRIP_ID.FM_PT.value}"
        pt_sub_rq_obj = self.demand[pt_rid_struct]
        # Clear inherited AMOD offers from deepcopy of parent request
        pt_sub_rq_obj.offer = {}
        pt_offer = self.pt_operator.get_current_offer(pt_rid_struct, amod_op_id)
        if pt_offer is not None:
            pt_sub_rq_obj.receive_offer(self.pt_operator_id, pt_offer, None)

        # Auto-confirm PT booking (user is already at the station)
        self.pt_operator.user_confirms_booking(pt_sub_rq_obj, amod_op_id)

        self.payg_trip_states[rid] = PAYG_TRIP_STATE.PT_BOOKED
        LOG.info(f"PAYG FM {rid}: PT booked after alighting, arrival at {pt_arrival}")

    def _handle_flm_amod_0_alighting(
        self, rid: int, rq_obj: 'BasicIntermodalRequest', amod_op_id: int,
        alighting_time: int, parent_modal_state: RQ_MODAL_STATE, alighting_node: int
    ):
        """Handle FLM_AMOD_0 alighting: Query PT in real-time."""
        LOG.debug(f"PAYG FLM_AMOD_0 alighting: rid={rid}, time={alighting_time}, node={alighting_node}")

        # Get transfer stations
        transfer_station_ids: tp.List[str] = rq_obj.get_transfer_station_ids()

        # Get second transfer station for PT destination
        transfer_street_node_1, _ = self._find_transfer_info(transfer_station_ids[1], "pt2street")

        # Query PT from current location/time
        pt_arrival = self._inform_pt_sub_request(
            rq_obj, RQ_SUB_TRIP_ID.FLM_PT.value,
            alighting_node, transfer_street_node_1,
            alighting_time, parent_modal_state, amod_op_id
        )

        if pt_arrival is None:
            # No PT available
            self._mark_trip_interrupted(rid, PAYG_TRIP_STATE.INTERRUPTED_NO_PT, alighting_time)
            LOG.info(f"PAYG FLM {rid}: No PT available after FM alighting, trip interrupted")
            return

        # Get PT offer and record it on sub-request
        pt_rid_struct = f"{rid}_{RQ_SUB_TRIP_ID.FLM_PT.value}"
        pt_sub_rq_obj = self.demand[pt_rid_struct]
        # Clear inherited AMOD offers from deepcopy of parent request
        pt_sub_rq_obj.offer = {}
        pt_offer = self.pt_operator.get_current_offer(pt_rid_struct, amod_op_id)
        if pt_offer is not None:
            pt_sub_rq_obj.receive_offer(self.pt_operator_id, pt_offer, None)

        # Auto-confirm PT booking
        self.pt_operator.user_confirms_booking(pt_sub_rq_obj, amod_op_id)

        # Schedule LM_AMOD request for PT arrival
        self.pending_pt_arrivals[rid] = (pt_arrival, transfer_street_node_1)
        self.payg_trip_states[rid] = PAYG_TRIP_STATE.PT_BOOKED
        LOG.info(f"PAYG FLM {rid}: PT booked, LM_AMOD will be requested at {pt_arrival}")

    # ============================================================================================================== #
    # PT Arrival Processing
    # ============================================================================================================== #

    def _process_pending_pt_arrivals(self, sim_time: int):
        """Process pending PT arrivals and trigger LM_AMOD requests."""
        completed_arrivals = []

        for rid, (pt_arrival_time, alighting_node) in self.pending_pt_arrivals.items():
            if sim_time >= pt_arrival_time:
                # PT has arrived, request LM_AMOD
                self._handle_pt_alighting(rid, pt_arrival_time, alighting_node)
                completed_arrivals.append(rid)

        # Remove processed arrivals
        for rid in completed_arrivals:
            del self.pending_pt_arrivals[rid]

    def _handle_pt_alighting(self, rid: int, alighting_time: int, alighting_node: int):
        """Handle PT alighting: Request LM_AMOD in real-time."""
        LOG.debug(f"PAYG PT alighting: rid={rid}, time={alighting_time}")

        parent_rq_obj: 'BasicIntermodalRequest' = self.demand[rid]
        parent_modal_state: RQ_MODAL_STATE = parent_rq_obj.get_modal_state()

        # Determine sub-trip ID based on modal state
        if parent_modal_state == RQ_MODAL_STATE.LASTMILE:
            lm_sub_trip_id = RQ_SUB_TRIP_ID.LM_AMOD.value
        elif parent_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            lm_sub_trip_id = RQ_SUB_TRIP_ID.FLM_AMOD_1.value
        else:
            LOG.warning(f"PAYG PT alighting for unexpected modal state: {parent_modal_state}")
            return

        # Create LM_AMOD sub-request
        for op_id in range(self.n_amod_op):
            self._inform_amod_sub_request(
                parent_rq_obj, lm_sub_trip_id,
                alighting_node, parent_rq_obj.get_destination_node(),
                alighting_time, parent_modal_state, op_id, alighting_time
            )

        # Get offers and select best one
        # TODO: Could implement user choice here instead of auto-selecting best offer
        lm_amod_rid_struct = f"{rid}_{lm_sub_trip_id}"
        best_offer = None
        best_op_id = None

        for amod_op_id in range(self.n_amod_op):
            offer = self.amod_operators[amod_op_id].get_current_offer(lm_amod_rid_struct)
            if offer is not None and not offer.service_declined():
                if best_offer is None or offer.offered_waiting_time < best_offer.offered_waiting_time:
                    best_offer = offer
                    best_op_id = amod_op_id

        if best_offer is None:
            # No AMoD available
            self._mark_trip_interrupted(rid, PAYG_TRIP_STATE.INTERRUPTED_NO_LM_AMOD, alighting_time)
            LOG.info(f"PAYG {rid}: No LM_AMOD available after PT alighting, trip interrupted")
            return

        # Auto-confirm best LM_AMOD offer
        for op_id in range(self.n_amod_op):
            if op_id != best_op_id:
                self.amod_operators[op_id].user_cancels_request(lm_amod_rid_struct, alighting_time)
            else:
                self.amod_operators[op_id].user_confirms_booking(lm_amod_rid_struct, alighting_time)
                self.demand[lm_amod_rid_struct].receive_offer(op_id, best_offer, None)

        self.payg_trip_states[rid] = PAYG_TRIP_STATE.LM_AMOD_BOOKED
        LOG.info(f"PAYG {rid}: LM_AMOD booked with operator {best_op_id}, wait time {best_offer.offered_waiting_time}s")

    # ============================================================================================================== #
    # Trip State Management
    # ============================================================================================================== #

    def _handle_trip_completed(self, rid: int, completion_time: int):
        """Mark trip as completed."""
        self.payg_trip_states[rid] = PAYG_TRIP_STATE.COMPLETED
        LOG.info(f"PAYG trip {rid} completed at {completion_time}")

    def _mark_trip_interrupted(self, rid: int, interrupt_state: PAYG_TRIP_STATE, interrupt_time: int):
        """Mark trip as interrupted and record the state."""
        self.payg_trip_states[rid] = interrupt_state

        # Set interrupted flag on parent request
        parent_rq_obj: 'BasicIntermodalRequest' = self.demand[rid]
        if hasattr(parent_rq_obj, 'set_payg_interrupted'):
            parent_rq_obj.set_payg_interrupted(True, interrupt_state.value, interrupt_time)

        LOG.warning(f"PAYG trip {rid} interrupted: {interrupt_state.name} at time {interrupt_time}")

    # ============================================================================================================== #
    # User Leaving System
    # ============================================================================================================== #

    def inform_user_leaving_system(self, rid: int, sim_time: int):
        """Handle user leaving system - cancel any pending requests."""
        # Remove from pending PT arrivals if present
        if rid in self.pending_pt_arrivals:
            del self.pending_pt_arrivals[rid]

        # Call parent implementation
        super().inform_user_leaving_system(rid, sim_time)
