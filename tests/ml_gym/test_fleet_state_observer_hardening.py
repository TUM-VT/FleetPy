import unittest
from types import SimpleNamespace

from src.misc.globals import VRL_STATES
from src.ml_gym.Observers.fleet_state_observers import FleetStateObserver


class FakeRequest:
    def __init__(self, rid, nr_pax):
        self._rid = rid
        self.nr_pax = nr_pax

    def get_rid_struct(self):
        return self._rid


class FleetStateObserverHardeningTest(unittest.TestCase):
    @staticmethod
    def _value(state, row, field):
        return row[state["columns"].index(field)]

    def test_boarding_snapshot_uses_completed_leg_passengers_and_string_rids(self):
        alighter = FakeRequest(0, 3)
        boarder_1 = FakeRequest(1, 1)
        boarder_2 = FakeRequest(2, 1)
        active_leg = SimpleNamespace(
            destination_pos=(0, None, None),
            earliest_start_time=-1000,
            earliest_end_time=-1000,
            rq_dict={1: [boarder_1, boarder_2], -1: [alighter]},
        )
        vehicle = SimpleNamespace(
            vid=0,
            status=VRL_STATES.BOARDING,
            pos=(0, None, None),
            pax=[alighter, boarder_1, boarder_2],
            cl_start_time=0,
            assigned_route=[active_leg],
        )
        plan_stop = SimpleNamespace(
            get_pos=lambda: (0, None, None),
            get_list_boarding_rids=lambda: [(1, 0), 2],
            get_list_alighting_rids=lambda: [0],
            get_planned_arrival_and_departure_time=lambda: (0, 0),
            get_earliest_start_time=lambda: -1,
        )
        fleet_control = SimpleNamespace(
            op_id=0,
            nr_vehicles=1,
            sim_vehicles={0: vehicle},
            veh_plans={0: SimpleNamespace(list_plan_stops=[plan_stop])},
        )
        observer = FleetStateObserver(
            "custom",
            {
                "veh": ["vid", "pos", "n_pax", "pax_rids", "cl_start_time"],
                "leg": [
                    "destination_pos",
                    "earliest_start_time",
                    "earliest_end_time",
                    "boarding_rids",
                    "alighting_rids",
                ],
                "stop": [
                    "pos",
                    "boarding_rids",
                    "alighting_rids",
                    "planned_arrival_time",
                    "planned_departure_time",
                    "remaining_time_to_departure",
                    "earliest_start_time",
                ],
            },
        )

        observation = observer.observe(
            fleet_control,
            sim_time=0,
            new_request_ids=[(10, 1)],
            optimization_request_ids=[(10, 1), 11],
            prediction_request_ids=[11],
        )
        state = observation["fleet_state"]
        vehicle_row = state["vehicles"][0]

        self.assertEqual(observation["new_request_ids"], ["(10, 1)"])
        self.assertEqual(observation["optimization_request_ids"], ["(10, 1)", "11"])
        self.assertEqual(observation["prediction_request_ids"], ["11"])
        self.assertEqual(self._value(state, vehicle_row, "pos"), [0, None, None])
        self.assertEqual(self._value(state, vehicle_row, "n_pax"), 2)
        self.assertEqual(self._value(state, vehicle_row, "pax_rids"), ["1", "2"])
        self.assertEqual(self._value(state, vehicle_row, "cl_start_time"), 0)

        route = self._value(state, vehicle_row, "assigned_route")
        self.assertEqual(
            route[0],
            [[0, None, None], None, None, ["1", "2"], ["0"]],
        )
        stops = self._value(state, vehicle_row, "plan_stops")
        self.assertEqual(
            stops[0],
            [
                [0, None, None],
                ["(1, 0)", "2"],
                ["0"],
                0,
                0,
                0,
                None,
            ],
        )

    def test_negative_status_value_is_not_treated_as_missing(self):
        vehicle = SimpleNamespace(
            status=VRL_STATES.BLOCKED_INIT,
            pax=[],
        )
        fleet_control = SimpleNamespace(
            op_id=0,
            nr_vehicles=1,
            sim_vehicles={0: vehicle},
            veh_plans={0: SimpleNamespace(list_plan_stops=[])},
        )
        observer = FleetStateObserver(
            "custom",
            {"veh": ["status_value", "n_pax"], "leg": [], "stop": []},
        )

        observation = observer.observe(
            fleet_control,
            sim_time=0,
            new_request_ids=[],
            optimization_request_ids=[],
            prediction_request_ids=[],
        )

        self.assertEqual(observation["fleet_state"]["vehicles"], [[-1, 0]])


if __name__ == "__main__":
    unittest.main()
