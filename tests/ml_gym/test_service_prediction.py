"""Unit tests for Service Prediction FleetState collection components."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from queue import Queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.BatchOfferSimulation import BatchOfferSimulation
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.VehiclePlan import PlanStop
from src.fleetctrl.RidePoolingBatchAssignmentFleetcontrol import (
    RidePoolingBatchAssignmentFleetcontrol,
)
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment import (
    AlonsoMoraAssignment,
)
from src.fleetctrl.pooling.batch.InsertionHeuristic.BatchInsertionHeuristicAssignment import (
    BatchInsertionHeuristicAssignment,
)
from src.ImmediateDecisionsSimulation import ImmediateDecisionsSimulation
from src.misc.globals import VRL_STATES
from src.ml_gym.Actors import AbstractActor
from src.ml_gym.Observers import AbstractObserver
from src.ml_gym.Observers.fleet_state_observers import FieldGetters, FleetStateObserver
from src.ml_gym.hooks_manager import Events, HookManager
from src.ml_gym.Actors.writers import JSONWriter
from src.ml_gym.FleetPyMLInterface import FleetPyMLInterface


class FakePlanRequest:
    """Small PlanRequest substitute for FleetState collection tests."""

    def __init__(self, request_time=360):
        self.rq_time = request_time
        self.nr_pax = 2
        self.o_pos = (1, 2, 0.0)
        self.d_pos = (3, 4, 0.0)
        self.t_pu_earliest = request_time
        self.t_pu_latest = request_time + 600
        self.t_do_latest = request_time + 1800
        self.init_direct_tt = 420
        self.init_direct_td = 2500.0
        self.max_trip_time = 900


class FakePassenger:
    """Passenger request with the two fields used by Fleet State getters."""

    def __init__(self, rid, nr_pax=1):
        self.rid = rid
        self.nr_pax = nr_pax

    def get_rid_struct(self):
        return self.rid


class ScalarLike:
    """Small NumPy-scalar substitute used to test JSON normalisation."""

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeFleetControl(FleetControlBase):
    """Small FleetControl substitute used without a full simulation setup."""

    def _not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    _call_time_trigger_request_batch = _not_implemented
    _create_user_offer = _not_implemented
    _lock_vid_rid_pickup = _not_implemented
    _prq_from_reservation_to_immediate = _not_implemented
    acknowledge_alighting = _not_implemented
    acknowledge_boarding = _not_implemented
    assign_vehicle_plan = _not_implemented
    change_prq_time_constraints = _not_implemented
    lock_current_vehicle_plan = _not_implemented
    receive_status_update = _not_implemented
    user_cancels_request = _not_implemented
    user_confirms_booking = _not_implemented
    user_request = _not_implemented

    def __init__(self, requests):
        self.op_id = 0
        self.sim_time = 360
        self.rq_dict = requests
        self.nr_vehicles = 1
        self.sim_vehicles = {0: SimpleNamespace(vid=0)}
        self.veh_plans = {0: SimpleNamespace(list_plan_stops=[])}
        self.dir_names = {}


class BasicObserver(AbstractObserver):
    """Observer that accepts and ignores optional keyword arguments."""

    def observe(self, fleetpy_module, **kwargs):
        return {"basic_observer_ran": True}


class ContextObserver(AbstractObserver):
    """Observer that exposes forwarded context for the compatibility test."""

    def observe(self, fleetpy_module, **kwargs):
        return {"kwargs": kwargs}


class RecordingActor(AbstractActor):
    """Actor that records Hook output without changing the supplied module."""

    def __init__(self):
        self.observations = []

    def compute_action(self, observation, process_id) -> None:
        self.observations.append(observation)


class ServicePredictionCollectionTest(unittest.TestCase):
    """Exercise FleetState collection contracts without result fixtures."""

    def setUp(self):
        self.request_id = (42, 1)
        self.normalised_request_id = str(self.request_id)
        self.plan_request = FakePlanRequest()
        self.fleet_control = FakeFleetControl({self.request_id: self.plan_request})

    def _observation(self):
        observer = FleetStateObserver(
            detail_level="custom",
            custom_fields={"veh": ["vid"], "leg": [], "stop": []},
        )
        context = {
            "sim_time": 360,
            "new_request_ids": [self.request_id],
            "optimization_request_ids": [self.request_id],
            "prediction_request_ids": [self.request_id],
        }
        return observer.observe(self.fleet_control, **context)

    @staticmethod
    def _observe_one_vehicle(
        vehicle,
        *,
        veh_fields,
        leg_fields=None,
        stop_fields=None,
        plan_stops=None,
        sim_time=360,
    ):
        """Encode one realistic vehicle/plan pair and return its table state."""
        fleet_control = SimpleNamespace(
            op_id=0,
            nr_vehicles=1,
            sim_vehicles={0: vehicle},
            veh_plans={
                0: SimpleNamespace(list_plan_stops=list(plan_stops or []))
            },
        )
        observer = FleetStateObserver(
            "custom",
            {
                "veh": list(veh_fields),
                "leg": list(leg_fields or []),
                "stop": list(stop_fields or []),
            },
        )
        return observer.observe(
            fleet_control,
            sim_time=sim_time,
            new_request_ids=[],
            optimization_request_ids=[],
            prediction_request_ids=[],
        )["fleet_state"]

    @staticmethod
    def _vehicle_row(state):
        return dict(zip(state["columns"], state["vehicles"][0]))

    def test_hook_forwards_kwargs_to_observers(self):
        """A hook forwards optional keyword arguments to its observers."""
        hook_manager = HookManager()
        actor = RecordingActor()
        event = Events.OBSERVE_FLEET_STATE_BEFORE_IMMEDIATE_REQUEST_SUBMISSION
        hook_manager.add_observer(event, BasicObserver())
        hook_manager.add_observer(event, ContextObserver())
        hook_manager.add_actor(event, actor)

        hook_manager.trigger(
            event,
            self.fleet_control,
            new_request_ids=[self.request_id],
            optimization_request_ids=[self.request_id],
            prediction_request_ids=[self.request_id],
            context_value=360,
        )

        self.assertEqual(len(actor.observations), 1)
        observation = actor.observations[0]
        self.assertTrue(observation["basic_observer_ran"])
        self.assertEqual(observation["kwargs"]["new_request_ids"], [self.request_id])
        self.assertEqual(
            observation["kwargs"]["optimization_request_ids"], [self.request_id]
        )
        self.assertEqual(
            observation["kwargs"]["prediction_request_ids"], [self.request_id]
        )
        self.assertEqual(observation["kwargs"]["context_value"], 360)
        self.assertNotIn("worker_id", observation["kwargs"])

    def test_observer_requires_explicit_sim_time(self):
        """FleetState time always comes from the triggering simulation event."""
        observer = FleetStateObserver(
            detail_level="custom",
            custom_fields={"veh": ["vid"], "leg": [], "stop": []},
        )
        with self.assertRaises(TypeError):
            observer.observe(
                self.fleet_control,
                new_request_ids=[self.request_id],
                optimization_request_ids=[self.request_id],
                prediction_request_ids=[self.request_id],
            )

    def test_observer_normalises_all_top_level_request_id_lists(self):
        """All request contexts use one stable representation for later joins."""
        fields = {"veh": ["vid"], "leg": [], "stop": []}
        observation = FleetStateObserver("custom", fields).observe(
            self.fleet_control,
            sim_time=360,
            new_request_ids=[self.request_id],
            optimization_request_ids=[ScalarLike(17)],
            prediction_request_ids=[ScalarLike(18)],
        )
        self.assertEqual(
            observation["new_request_ids"], [self.normalised_request_id]
        )
        self.assertEqual(observation["optimization_request_ids"], ["17"])
        self.assertEqual(observation["prediction_request_ids"], ["18"])
        self.assertNotIn("request_ids", observation)

    def test_observer_uses_passenger_count_and_snapshot_distance(self):
        """Vehicle totals reflect people and distance at the observation time."""
        vehicle = SimpleNamespace(
            status=VRL_STATES.IDLE,
            pax=[FakePassenger(1, nr_pax=2), FakePassenger(2, nr_pax=1)],
            assigned_route=[],
            cl_start_time=None,
            cumulative_distance=100.0,
            cl_driven_distance=12.5,
        )
        fleet_control = SimpleNamespace(
            op_id=0,
            nr_vehicles=1,
            sim_vehicles={0: vehicle},
            veh_plans={0: SimpleNamespace(list_plan_stops=[])},
        )
        observer = FleetStateObserver(
            "custom",
            {"veh": ["n_pax", "cumulative_distance"], "leg": [], "stop": []},
        )

        observation = observer.observe(
            fleet_control,
            sim_time=360,
            new_request_ids=[],
            optimization_request_ids=[],
            prediction_request_ids=[],
        )

        self.assertEqual(observation["fleet_state"]["vehicles"], [[3, 112.5]])

    def test_boarding_pickup_only_includes_all_boarders_and_nr_pax(self):
        """Active pickup exposes the non-interruptible stop's completed load."""
        continuing = FakePassenger("continuing", nr_pax=2)
        boarder = FakePassenger((7, 1), nr_pax=3)
        current_leg = SimpleNamespace(rq_dict={1: [boarder], -1: []})
        vehicle = SimpleNamespace(
            status=VRL_STATES.BOARDING,
            cl_start_time=350,
            assigned_route=[current_leg],
            # FleetPy adds boarders to pax when the BOARDING leg starts.
            pax=[continuing, boarder],
        )

        state = self._observe_one_vehicle(
            vehicle, veh_fields=["n_pax", "pax_rids"]
        )
        row = self._vehicle_row(state)

        self.assertEqual(row["n_pax"], 5)
        self.assertEqual(row["pax_rids"], ["continuing", "(7, 1)"])

    def test_boarding_dropoff_only_excludes_all_alighters(self):
        """Active dropoff exposes the load after alighters have left."""
        continuing = FakePassenger("continuing", nr_pax=2)
        alighter = FakePassenger((8, 1), nr_pax=3)
        current_leg = SimpleNamespace(rq_dict={1: [], -1: [alighter]})
        vehicle = SimpleNamespace(
            status=VRL_STATES.BOARDING,
            cl_start_time=350,
            assigned_route=[current_leg],
            # FleetPy removes alighters only when the BOARDING leg ends.
            pax=[continuing, alighter],
        )

        state = self._observe_one_vehicle(
            vehicle, veh_fields=["n_pax", "pax_rids"]
        )
        row = self._vehicle_row(state)

        self.assertEqual(row["n_pax"], 2)
        self.assertEqual(row["pax_rids"], ["continuing"])

    def test_boarding_same_stop_exchange_uses_completed_load_and_string_rids(self):
        """A zero-net same-stop exchange is not mistaken for no activity."""
        alighters = [FakePassenger((1, 1)), FakePassenger((2, 1))]
        boarders = [FakePassenger((3, 1)), FakePassenger((4, 1))]
        current_leg = SimpleNamespace(
            status=VRL_STATES.BOARDING,
            destination_pos=(2985, None, None),
            duration=10,
            earliest_start_time=350,
            earliest_end_time=360,
            locked=True,
            rq_dict={1: boarders, -1: alighters},
        )
        plan_stop = SimpleNamespace(
            get_pos=lambda: (2985, None, None),
            get_state=lambda: SimpleNamespace(name="MIXED"),
            get_list_boarding_rids=lambda: [(3, 1), (4, 1)],
            get_list_alighting_rids=lambda: [(1, 1), (2, 1)],
            get_planned_arrival_and_departure_time=lambda: (350, 360),
            get_duration_and_earliest_departure=lambda: (10, 360),
            get_earliest_start_time=lambda: 350,
            is_locked=lambda: True,
            get_change_nr_pax=lambda: 0,
        )
        vehicle = SimpleNamespace(
            status=VRL_STATES.BOARDING,
            cl_start_time=350,
            assigned_route=[current_leg],
            # At leg start, both departing and newly boarding requests coexist.
            pax=alighters + boarders,
        )

        state = self._observe_one_vehicle(
            vehicle,
            veh_fields=["n_pax", "pax_rids"],
            leg_fields=["boarding_rids", "alighting_rids"],
            stop_fields=["boarding_rids", "alighting_rids", "change_nr_pax"],
            plan_stops=[plan_stop],
        )
        row = self._vehicle_row(state)
        route = row["assigned_route"]
        stops = row["plan_stops"]

        self.assertEqual(row["n_pax"], 2)
        self.assertEqual(row["pax_rids"], ["(3, 1)", "(4, 1)"])
        self.assertEqual(route, [[['(3, 1)', '(4, 1)'], ['(1, 1)', '(2, 1)']]])
        self.assertEqual(
            stops,
            [[['(3, 1)', '(4, 1)'], ['(1, 1)', '(2, 1)'], 0]],
        )

    def test_observer_derives_started_and_normalises_leg_scalars(self):
        """Only the active first leg is started and scalar times remain numeric."""
        legs = [
            SimpleNamespace(started=False, earliest_start_time=ScalarLike(360)),
            SimpleNamespace(started=True, earliest_start_time=ScalarLike(420)),
        ]
        vehicle = SimpleNamespace(vid=0, assigned_route=legs, cl_start_time=360)
        fleet_control = SimpleNamespace(
            op_id=0,
            nr_vehicles=1,
            sim_vehicles={0: vehicle},
            veh_plans={0: SimpleNamespace(list_plan_stops=[])},
        )
        observer = FleetStateObserver(
            "custom",
            {"veh": ["vid"], "leg": ["started", "earliest_start_time"], "stop": []},
        )

        observation = observer.observe(
            fleet_control,
            sim_time=360,
            new_request_ids=[],
            optimization_request_ids=[],
            prediction_request_ids=[],
        )
        state = observation["fleet_state"]
        route = state["vehicles"][0][state["columns"].index("assigned_route")]

        self.assertEqual(route, [[True, 360], [False, 420]])
        json.dumps(observation)

    def test_zero_departure_time_is_not_treated_as_missing(self):
        """A planned departure at simulation time zero remains observable."""
        plan_stop = SimpleNamespace(
            get_planned_arrival_and_departure_time=lambda: (None, 0),
        )
        fleet_control = SimpleNamespace(
            op_id=0,
            nr_vehicles=1,
            sim_vehicles={0: SimpleNamespace(vid=0)},
            veh_plans={0: SimpleNamespace(list_plan_stops=[plan_stop])},
        )
        observer = FleetStateObserver(
            "custom",
            {"veh": ["vid"], "leg": [], "stop": ["remaining_time_to_departure"]},
        )

        observation = observer.observe(
            fleet_control,
            sim_time=0,
            new_request_ids=[],
            optimization_request_ids=[],
            prediction_request_ids=[],
        )
        state = observation["fleet_state"]
        stops = state["vehicles"][0][state["columns"].index("plan_stops")]

        self.assertEqual(stops, [[0]])

    def test_missing_times_become_null_but_meaningful_negatives_survive(self):
        """Known absolute-time sentinels become null without erasing real negatives."""
        built_stop_leg = SimpleNamespace(
            status=VRL_STATES.ROUTE,
            destination_pos=(2, None, None),
            duration=None,
            # FleetControlBase._build_VRLs propagates these two missing values
            # from an unconstrained PlanStop.
            earliest_start_time=-1,
            earliest_end_time=-100000000,
            locked=False,
            rq_dict={},
        )
        default_leg = SimpleNamespace(
            status=VRL_STATES.ROUTE,
            destination_pos=(3, None, None),
            duration=None,
            earliest_start_time=-1000,
            earliest_end_time=-1000,
            locked=False,
            rq_dict={},
        )
        zero_time_leg = SimpleNamespace(
            status=VRL_STATES.ROUTE,
            destination_pos=(4, None, None),
            duration=None,
            earliest_start_time=0,
            earliest_end_time=0,
            locked=False,
            rq_dict={},
        )
        plan_stop = SimpleNamespace(
            get_pos=lambda: (2, None, None),
            get_state=lambda: SimpleNamespace(name="MIXED"),
            get_list_boarding_rids=lambda: [],
            get_list_alighting_rids=lambda: [],
            get_planned_arrival_and_departure_time=lambda: (None, 350),
            get_duration_and_earliest_departure=lambda: (None, None),
            get_earliest_start_time=lambda: -1,
            is_locked=lambda: False,
            get_change_nr_pax=lambda: 0,
        )
        vehicle = SimpleNamespace(
            status=VRL_STATES.BLOCKED_INIT,
            cl_start_time=None,
            cl_start_pos=None,
            assigned_route=[built_stop_leg, default_leg, zero_time_leg],
        )

        state = self._observe_one_vehicle(
            vehicle,
            veh_fields=["status_value", "cl_start_time", "cl_start_pos"],
            leg_fields=["duration", "earliest_start_time", "earliest_end_time"],
            stop_fields=[
                "planned_arrival_time",
                "remaining_time_to_departure",
                "duration",
                "earliest_departure",
                "earliest_start_time",
            ],
            plan_stops=[plan_stop],
            sim_time=360,
        )
        row = self._vehicle_row(state)
        decoded = json.loads(json.dumps(row, allow_nan=False))

        self.assertEqual(decoded["status_value"], -1)
        self.assertIsNone(decoded["cl_start_time"])
        self.assertIsNone(decoded["cl_start_pos"])
        self.assertEqual(
            decoded["assigned_route"],
            [[None, None, None], [None, None, None], [None, 0, 0]],
        )
        self.assertEqual(decoded["plan_stops"], [[None, -10, None, None, None]])

    def test_real_vrl_builder_sentinels_become_null(self):
        """The actual FleetControl VRL builder's missing values never escape."""
        controller = object.__new__(FakeFleetControl)
        controller.rq_dict = {}
        controller._active_charging_processes = {}
        controller.begin_approach_buffer_time = 0
        controller.routing_engine = SimpleNamespace(
            return_travel_costs_1to1=lambda origin, destination: (0, 1, 1)
        )

        vehicle = SimpleNamespace(pos=(1, None, None))
        unconstrained_stop = PlanStop((2, None, None), duration=1)
        unconstrained_legs = controller._build_VRLs(
            SimpleNamespace(list_plan_stops=[unconstrained_stop]),
            vehicle,
            sim_time=0,
        )
        self.assertEqual(
            [
                (leg.earliest_start_time, leg.earliest_end_time)
                for leg in unconstrained_legs
            ],
            [(-1000, -1000), (-1, -100000000)],
        )

        zero_time_stop = PlanStop(
            (1, None, None),
            duration=1,
            earliest_start_time=0,
            earliest_end_time=0,
        )
        with patch("src.fleetctrl.FleetControlBase.LOG.warning"):
            zero_time_legs = controller._build_VRLs(
                SimpleNamespace(list_plan_stops=[zero_time_stop]),
                vehicle,
                sim_time=0,
            )
        self.assertEqual(
            [
                (leg.earliest_start_time, leg.earliest_end_time)
                for leg in zero_time_legs
            ],
            [(0, 0)],
        )

        observed_vehicle = SimpleNamespace(
            vid=0,
            assigned_route=unconstrained_legs + zero_time_legs,
        )
        fleet_control = SimpleNamespace(
            op_id=0,
            nr_vehicles=1,
            sim_vehicles={0: observed_vehicle},
            veh_plans={0: SimpleNamespace(list_plan_stops=[])},
        )
        observation = FleetStateObserver(
            "custom",
            {
                "veh": ["vid"],
                "leg": ["earliest_start_time", "earliest_end_time"],
                "stop": [],
            },
        ).observe(
            fleet_control,
            sim_time=0,
            new_request_ids=[],
            optimization_request_ids=[],
            prediction_request_ids=[],
        )
        state = observation["fleet_state"]
        route = state["vehicles"][0][state["columns"].index("assigned_route")]
        self.assertEqual(route, [[None, None], [None, None], [0, 0]])

    def test_max_schema_excludes_charging_and_unreliable_route_fields(self):
        """FleetState stays focused on passenger-service observation fields."""
        removed_vehicle_fields = {"soc", "battery_size", "range", "cl_start_soc"}
        removed_leg_fields = {"power", "route_len"}
        removed_stop_fields = {"charging_power", "charging_task_id"}

        self.assertTrue(removed_vehicle_fields.isdisjoint(FieldGetters.VEH))
        self.assertTrue(removed_leg_fields.isdisjoint(FieldGetters.LEG))
        self.assertTrue(removed_stop_fields.isdisjoint(FieldGetters.STOP))

    def test_observer_rebuilds_getter_cache_after_multiprocessing_pickle(self):
        """Spawn workers can deserialize the configured Fleet State schema."""
        observer = FleetStateObserver(
            detail_level="custom",
            custom_fields={"veh": ["vid"], "leg": [], "stop": []},
        )

        restored_observer = pickle.loads(pickle.dumps(observer))
        observation = restored_observer.observe(
            self.fleet_control,
            sim_time=360,
            new_request_ids=[self.request_id],
            optimization_request_ids=[self.request_id],
            prediction_request_ids=[self.request_id],
        )

        self.assertEqual(observation["fleet_state"]["vehicles"], [[0]])

    def test_observer_uses_one_mode_independent_structure(self):
        """Simulation mode remains scenario metadata rather than row data."""
        observation = self._observation()

        self.assertEqual(
            observation["new_request_ids"], [self.normalised_request_id]
        )
        self.assertEqual(
            observation["optimization_request_ids"], [self.normalised_request_id]
        )
        self.assertEqual(
            observation["prediction_request_ids"], [self.normalised_request_id]
        )
        self.assertEqual(observation["sim_time"], 360)
        self.assertNotIn("simulation_mode", observation)
        self.assertNotIn("request_time", observation)
        self.assertNotIn("snapshot_id", observation)
        self.assertNotIn("schema_version", observation)
        self.assertNotIn("run_id", observation)
        self.assertNotIn("worker_id", observation)
        self.assertNotIn("online_request_context", observation)
        self.assertEqual(observation["fleet_state"]["vehicles"], [[0]])

    def test_immediate_observer_accepts_request_before_controller_registration(self):
        """A simulation-level request event must not depend on FleetControl.rq_dict."""
        self.fleet_control.rq_dict = {}
        observer = FleetStateObserver(
            detail_level="custom",
            custom_fields={"veh": ["vid"], "leg": [], "stop": []},
        )

        observation = observer.observe(
            self.fleet_control,
            sim_time=360,
            new_request_ids=[self.request_id],
            optimization_request_ids=[self.request_id],
            prediction_request_ids=[self.request_id],
        )

        self.assertEqual(
            observation["new_request_ids"], [self.normalised_request_id]
        )
        self.assertEqual(
            observation["optimization_request_ids"], [self.normalised_request_id]
        )
        self.assertEqual(
            observation["prediction_request_ids"], [self.normalised_request_id]
        )
        self.assertEqual(observation["sim_time"], 360)
        self.assertNotIn("online_request_context", observation)

    def test_immediate_simulation_triggers_hook_before_informing_broker(self):
        """Every operator observes the raw request before FleetControl processing starts."""
        events = []
        request = SimpleNamespace(rq_time=300)
        operators = [
            SimpleNamespace(time_trigger=lambda sim_time: None),
            SimpleNamespace(time_trigger=lambda sim_time: None),
        ]
        simulation = SimpleNamespace(
            time_step=60,
            start_time=0,
            operators=operators,
            charging_operator_dict={},
            routing_engine=SimpleNamespace(update_network=lambda sim_time: False),
            demand=SimpleNamespace(
                get_undecided_travelers=lambda sim_time: [],
                get_new_travelers=lambda sim_time, since: [(42, request)],
            ),
            hook_manager=SimpleNamespace(
                trigger=lambda event, operator, **context: events.append(
                    ("hook", event, operator, context)
                )
            ),
            broker=SimpleNamespace(
                inform_request=lambda rid, rq_obj, sim_time: events.append(("inform", rid)),
                collect_offers=lambda rid: {},
            ),
            update_sim_state_fleets=lambda last_time, sim_time: None,
            _rid_chooses_offer=lambda rid, rq_obj, sim_time: None,
            _check_waiting_request_cancellations=lambda sim_time: None,
            record_stats=lambda: None,
        )

        ImmediateDecisionsSimulation.step(simulation, 360)

        self.assertEqual([event[0] for event in events], ["hook", "hook", "inform"])
        self.assertEqual(
            events[0][1], Events.OBSERVE_FLEET_STATE_BEFORE_IMMEDIATE_REQUEST_SUBMISSION
        )
        self.assertIs(events[0][2], operators[0])
        self.assertIs(events[1][2], operators[1])
        self.assertEqual(events[0][3]["new_request_ids"], [42])
        self.assertEqual(events[0][3]["optimization_request_ids"], [42])
        self.assertEqual(events[0][3]["prediction_request_ids"], [42])
        self.assertNotIn("request_ids", events[0][3])
        self.assertNotIn("request_time", events[0][3])
        self.assertEqual(events[0][3]["sim_time"], 360)

    def test_immediate_same_step_requests_get_sequential_pre_submission_states(self):
        """The second request sees the first one's change, but neither sees itself."""
        timeline = []
        accepted_request_ids = []
        requests = [(42, SimpleNamespace()), (43, SimpleNamespace())]

        def trigger(event, operator, **context):
            timeline.append(
                (
                    "hook",
                    context,
                    list(accepted_request_ids),
                )
            )

        def inform_request(rid, rq_obj, sim_time):
            timeline.append(("inform", rid))
            accepted_request_ids.append(rid)

        operator = SimpleNamespace(
            time_trigger=lambda sim_time: timeline.append(("time_trigger", sim_time))
        )
        simulation = SimpleNamespace(
            time_step=60,
            start_time=0,
            operators=[operator],
            charging_operator_dict={},
            routing_engine=SimpleNamespace(update_network=lambda sim_time: False),
            demand=SimpleNamespace(
                get_undecided_travelers=lambda sim_time: [],
                get_new_travelers=lambda sim_time, since: requests,
            ),
            hook_manager=SimpleNamespace(trigger=trigger),
            broker=SimpleNamespace(
                inform_request=inform_request,
                collect_offers=lambda rid: {},
            ),
            update_sim_state_fleets=lambda last_time, sim_time: timeline.append(
                ("fleet_update", last_time, sim_time)
            ),
            _rid_chooses_offer=lambda rid, rq_obj, sim_time: None,
            _check_waiting_request_cancellations=lambda sim_time: timeline.append(
                ("cancellations", sim_time)
            ),
            record_stats=lambda: None,
        )

        ImmediateDecisionsSimulation.step(simulation, 360)

        hooks = [entry for entry in timeline if entry[0] == "hook"]
        self.assertEqual(timeline[0], ("fleet_update", 300, 360))
        self.assertEqual(hooks[0][1]["new_request_ids"], [42])
        self.assertEqual(hooks[0][1]["optimization_request_ids"], [42])
        self.assertEqual(hooks[0][1]["prediction_request_ids"], [42])
        self.assertEqual(hooks[0][2], [])
        self.assertEqual(hooks[1][1]["new_request_ids"], [43])
        self.assertEqual(hooks[1][1]["optimization_request_ids"], [43])
        self.assertEqual(hooks[1][1]["prediction_request_ids"], [43])
        self.assertEqual(hooks[1][2], [42])
        self.assertLess(timeline.index(hooks[0]), timeline.index(("inform", 42)))
        self.assertLess(timeline.index(("inform", 42)), timeline.index(hooks[1]))

    def test_batch_records_every_step_and_separates_arrival_from_optimization(self):
        """Delayed Batch optimization keeps arrival and candidate contexts distinct."""
        timeline = []
        hook_calls = []
        submitted_request_ids = []
        requests_by_time = {
            60: [(10, SimpleNamespace()), (11, SimpleNamespace())],
            120: [],
            180: [],
        }

        class FakeBatchOperator:
            def get_optimization_request_ids(self, sim_time):
                # Request 9 accepted an earlier offer but is still mutable in
                # Alonso-Mora; only 10 and 11 await this batch's offers.
                return [9, 10, 11] if sim_time == 120 else []

            def get_prediction_request_ids(self, sim_time):
                return [10, 11] if sim_time == 120 else []

            def time_trigger(self, sim_time):
                timeline.append((sim_time, "time_trigger"))

        operator = FakeBatchOperator()

        def inform_request(rid, rq_obj, sim_time):
            submitted_request_ids.append(rid)
            timeline.append((sim_time, f"inform:{rid}"))

        def trigger(event, observed_operator, **context):
            timeline.append((context["sim_time"], "hook"))
            hook_calls.append(
                (event, observed_operator, context, list(submitted_request_ids))
            )

        simulation = SimpleNamespace(
            time_step=60,
            start_time=0,
            operators=[operator],
            charging_operator_dict={},
            routing_engine=SimpleNamespace(update_network=lambda sim_time: False),
            demand=SimpleNamespace(
                get_new_travelers=lambda sim_time, since: requests_by_time[sim_time],
                get_undecided_travelers=lambda sim_time: [],
            ),
            hook_manager=SimpleNamespace(trigger=trigger),
            broker=SimpleNamespace(
                inform_request=inform_request,
                collect_offers=lambda rid: {},
            ),
            update_sim_state_fleets=lambda last_time, sim_time: timeline.append(
                (sim_time, "fleet_update")
            ),
            _check_waiting_request_cancellations=lambda sim_time: timeline.append(
                (sim_time, "cancellations")
            ),
            _rid_chooses_offer=lambda rid, rq_obj, sim_time: None,
            record_stats=lambda: None,
        )

        for sim_time in (60, 120, 180):
            BatchOfferSimulation.step(simulation, sim_time)

        self.assertEqual(len(hook_calls), 3)
        self.assertTrue(
            all(
                call[0] == Events.OBSERVE_FLEET_STATE_BEFORE_BATCH_TIME_TRIGGER
                for call in hook_calls
            )
        )
        self.assertTrue(all(call[1] is operator for call in hook_calls))

        contexts = [call[2] for call in hook_calls]
        self.assertEqual(contexts[0]["new_request_ids"], [10, 11])
        self.assertEqual(contexts[0]["optimization_request_ids"], [])
        self.assertEqual(contexts[0]["prediction_request_ids"], [])
        self.assertEqual(contexts[1]["new_request_ids"], [])
        self.assertEqual(contexts[1]["optimization_request_ids"], [9, 10, 11])
        self.assertEqual(contexts[1]["prediction_request_ids"], [10, 11])
        self.assertEqual(contexts[2]["new_request_ids"], [])
        self.assertEqual(contexts[2]["optimization_request_ids"], [])
        self.assertEqual(contexts[2]["prediction_request_ids"], [])
        self.assertEqual([call[3] for call in hook_calls], [[10, 11]] * 3)

        self.assertEqual(
            [label for time, label in timeline if time == 60],
            [
                "fleet_update",
                "inform:10",
                "inform:11",
                "cancellations",
                "hook",
                "time_trigger",
            ],
        )
        self.assertEqual(
            [label for time, label in timeline if time == 120],
            ["fleet_update", "cancellations", "hook", "time_trigger"],
        )

    def test_batch_request_accessors_follow_optimizer_and_offer_semantics(self):
        """Optimizer context and pending-offer targets remain distinct."""
        optimizer = object.__new__(BatchInsertionHeuristicAssignment)
        optimizer.unassigned_requests = {10: object(), 11: object(), 12: object()}
        # A false-y marker is still present; None means excluded, matching the
        # insertion optimizer's actual filter.
        optimizer.rid_to_consider_for_global_optimisation = {
            10: 1,
            11: 0,
            12: None,
        }

        controller = object.__new__(RidePoolingBatchAssignmentFleetcontrol)
        controller.optimisation_time_step = 120
        controller.RPBO_Module = optimizer
        controller.unassigned_requests_1 = {10: 1, 11: 1}
        controller.unassigned_requests_2 = {11: 1, 13: 1}

        self.assertEqual(controller.get_optimization_request_ids(60), [])
        self.assertEqual(controller.get_optimization_request_ids(120), [10, 11])
        self.assertEqual(controller.get_prediction_request_ids(60), [])
        self.assertEqual(
            controller.get_prediction_request_ids(120), [10, 11, 13]
        )

    def test_alonso_mora_context_includes_mutable_assigned_but_not_locked_requests(self):
        """AM context includes reassignable bookings and omits vehicle-locked users."""
        optimizer = object.__new__(AlonsoMoraAssignment)
        optimizer.active_requests = {
            "accepted_not_boarded": object(),
            "awaiting_offer": object(),
            "on_board": object(),
            "excluded": object(),
        }
        optimizer.rid_to_consider_for_global_optimisation = {
            "accepted_not_boarded": 1,
            "awaiting_offer": 1,
            "on_board": 1,
            "excluded": None,
        }
        optimizer.rid_to_mutually_exclusive_cluster_id = {}
        optimizer.r2v_locked = {"on_board": 0}
        # Incremental tree rebuilds are deliberately narrower than the full
        # mutable request context and must not drive the public list.
        optimizer.requests_to_compute = {"awaiting_offer": 1}

        self.assertEqual(
            optimizer.get_optimization_request_ids(),
            ["accepted_not_boarded", "awaiting_offer"],
        )

    def test_json_writer_appends_every_immediate_observation(self):
        """Same-time Immediate events remain separate JSON Lines records."""
        observation = self._observation()
        second_request_id = (42, 2)
        second_normalised_id = str(second_request_id)
        self.fleet_control.rq_dict[second_request_id] = FakePlanRequest()
        second_observation = FleetStateObserver(
            detail_level="custom",
            custom_fields={"veh": ["vid"], "leg": [], "stop": []},
        ).observe(
            self.fleet_control,
            sim_time=360,
            new_request_ids=[second_request_id],
            optimization_request_ids=[second_request_id],
            prediction_request_ids=[second_request_id],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "fleet_states.jsonl"
            writer = JSONWriter(output_file)

            writer._act(observation, self.fleet_control, hook_id=0)
            writer._act(observation, self.fleet_control, hook_id=0)
            writer._act(second_observation, self.fleet_control, hook_id=0)

            records = [
                json.loads(line)
                for line in output_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 3)
            self.assertEqual(
                [record["new_request_ids"] for record in records],
                [
                    [self.normalised_request_id],
                    [self.normalised_request_id],
                    [second_normalised_id],
                ],
            )

    def test_json_writer_stores_all_batch_request_contexts_in_one_record(self):
        """Arrival, optimizer, and prediction-target ids keep their meanings."""
        second_request_id = (42, 2)
        second_normalised_id = str(second_request_id)
        self.fleet_control.rq_dict[second_request_id] = FakePlanRequest(request_time=360)
        observer = FleetStateObserver(
            detail_level="custom",
            custom_fields={"veh": ["vid"], "leg": [], "stop": []},
        )
        observation = observer.observe(
            self.fleet_control,
            sim_time=360,
            new_request_ids=[second_request_id],
            optimization_request_ids=[self.request_id],
            prediction_request_ids=[second_request_id],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "fleet_states.jsonl"
            writer = JSONWriter(output_file)
            writer._act(observation, self.fleet_control, hook_id=0)

            records = [
                json.loads(line)
                for line in output_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0]["new_request_ids"],
                [second_normalised_id],
            )
            self.assertEqual(
                records[0]["optimization_request_ids"],
                [self.normalised_request_id],
            )
            self.assertEqual(
                records[0]["prediction_request_ids"],
                [second_normalised_id],
            )

    def test_json_writer_rejects_unknown_objects_and_non_finite_numbers(self):
        """Writer failures propagate instead of silently producing permissive JSON."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "fleet_states.jsonl"
            writer = JSONWriter(output_file)

            with self.assertRaises(TypeError):
                writer.compute_action({"unknown": object()}, process_id=None)
            self.assertEqual(output_file.read_text(encoding="utf-8"), "")

            with self.assertRaises(ValueError):
                writer.compute_action({"invalid": float("nan")}, process_id=None)
            self.assertEqual(output_file.read_text(encoding="utf-8"), "")

    def test_json_writer_distinguishes_operators_by_op_id(self):
        """Multiple operators share one file without requiring separate mode files."""
        second_fleet_control = FakeFleetControl({self.request_id: FakePlanRequest()})
        second_fleet_control.op_id = 1
        observer = FleetStateObserver(
            detail_level="custom",
            custom_fields={"veh": ["vid"], "leg": [], "stop": []},
        )
        observations = [
            observer.observe(
                fleet_control,
                sim_time=360,
                new_request_ids=[self.request_id],
                optimization_request_ids=[self.request_id],
                prediction_request_ids=[self.request_id],
            )
            for fleet_control in (self.fleet_control, second_fleet_control)
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "fleet_states.jsonl"
            writer = JSONWriter(output_file)
            for observation in observations:
                writer.compute_action(observation, process_id=None)

            records = [
                json.loads(line)
                for line in output_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["op_id"] for record in records], [0, 1])


class FleetPyMLInterfaceScenarioTest(unittest.TestCase):
    """Verify that nr_parallel schedules distinct scenario configurations."""

    @staticmethod
    def _scenarios():
        return [
            {"scenario_name": "scenario_a"},
            {"scenario_name": "scenario_b"},
            {"scenario_name": "scenario_c"},
        ]

    def test_single_mapping_and_scenario_sequence_share_one_internal_shape(self):
        single_interface = FleetPyMLInterface(self._scenarios()[0])
        batch_interface = FleetPyMLInterface(self._scenarios())

        self.assertEqual(len(single_interface.scenario_parameters_list), 1)
        self.assertEqual(len(batch_interface.scenario_parameters_list), 3)

    def test_duplicate_scenario_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "scenario_name must be unique"):
            FleetPyMLInterface(
                [{"scenario_name": "duplicate"}, {"scenario_name": "duplicate"}],
                nr_parallel=2,
            )

    def test_sequential_mode_runs_every_scenario_row(self):
        completed = []

        class FakeSimulation:
            def __init__(self, scenario_name):
                self.scenario_name = scenario_name

            def run(self):
                completed.append(self.scenario_name)

        def load_simulation(parameters, hook_manager):
            return FakeSimulation(parameters["scenario_name"])

        interface = FleetPyMLInterface(self._scenarios(), nr_parallel=1)
        with patch(
            "src.ml_gym.FleetPyMLInterface.load_simulation_environment",
            side_effect=load_simulation,
        ):
            interface.run()

        self.assertEqual(completed, ["scenario_a", "scenario_b", "scenario_c"])

    def test_sequential_collection_failure_propagates_to_the_caller(self):
        """A collection error must make the public run entry point fail."""
        class FailingSimulation:
            def run(self):
                raise RuntimeError("fleet-state collection failed")

        interface = FleetPyMLInterface(self._scenarios()[0], nr_parallel=1)
        with patch(
            "src.ml_gym.FleetPyMLInterface.load_simulation_environment",
            return_value=FailingSimulation(),
        ), self.assertRaisesRegex(RuntimeError, "fleet-state collection failed"):
            interface.run()

    def test_parallel_mode_runs_distinct_scenarios_up_to_the_limit(self):
        completed = []

        class FakeSimulation:
            def __init__(self, scenario_name):
                self.scenario_name = scenario_name

            def run(self, process_id):
                completed.append((self.scenario_name, process_id))

        class ImmediateProcess:
            def __init__(self, target, args):
                self.target = target
                self.args = args
                self.exitcode = None

            def start(self):
                self.target(*self.args)
                self.exitcode = 0

            def is_alive(self):
                return False

            def join(self):
                pass

        def load_simulation(parameters, hook_manager, process_id):
            return FakeSimulation(parameters["scenario_name"])

        with patch(
            "src.ml_gym.FleetPyMLInterface.mp.Queue",
            side_effect=Queue,
        ), patch(
            "src.ml_gym.FleetPyMLInterface.mp.Process",
            ImmediateProcess,
        ), patch(
            "src.ml_gym.FleetPyMLInterface.load_simulation_environment",
            side_effect=load_simulation,
        ):
            interface = FleetPyMLInterface(self._scenarios(), nr_parallel=2)
            interface.run()

        self.assertEqual(
            completed,
            [("scenario_a", 0), ("scenario_b", 1), ("scenario_c", 2)],
        )

    def test_finished_worker_slot_is_refilled_without_a_batch_barrier(self):
        active = set()
        starts = []

        class UnevenProcess:
            def __init__(self, target, args):
                self.scenario_index = args[2]
                self.remaining_alive_checks = 2 if self.scenario_index == 0 else 0
                self.exitcode = 0

            def start(self):
                active.add(self.scenario_index)
                starts.append((self.scenario_index, set(active)))

            def is_alive(self):
                if self.remaining_alive_checks > 0:
                    self.remaining_alive_checks -= 1
                    return True
                return False

            def join(self):
                active.remove(self.scenario_index)

        with patch(
            "src.ml_gym.FleetPyMLInterface.mp.Queue",
            side_effect=Queue,
        ), patch(
            "src.ml_gym.FleetPyMLInterface.mp.Process",
            UnevenProcess,
        ), patch(
            "src.ml_gym.FleetPyMLInterface.time.sleep",
        ):
            FleetPyMLInterface(self._scenarios(), nr_parallel=2).run()

        # Scenario C starts while the longer scenario A is still active. A
        # fixed two-scenario batch would wait for A before starting C.
        self.assertIn((2, {0, 2}), starts)


if __name__ == "__main__":
    unittest.main()
