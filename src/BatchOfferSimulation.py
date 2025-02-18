# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import importlib
import os
import json

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
from src.FleetSimulationBase import FleetSimulationBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------


# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
INPUT_PARAMETERS_BatchOfferSimulation = {
    "doc" :     """
    customers request trips from a single ride-pooling operator continously in time.
    offers are only created after the optimisation step of the operator and fetched from the time_trigger function.
    """,
    "inherit" : "FleetSimulationBase",
    "input_parameters_mandatory": [
    ],
    "input_parameters_optional": [
    ],
    "mandatory_modules": [
    ], 
    "optional_modules": []
}

class BatchOfferSimulation(FleetSimulationBase):
    """
    customers request trips from a single ride-pooling operator continously in time.
    offers are only created after the optimisation step of the operator and fetched from the time_trigger function
    """
    def add_init(self, scenario_parameters):
        """
        Simulation specific additional init.
        :param scenario_parameters: row of pandas data-frame; entries are saved as x["key"]
        """
        super().add_init(scenario_parameters)

    def step(self, sim_time):
        """This method determines the simulation flow in a time step.
            # 1) update fleets and network
            # 2) get new travelers, add to undecided request
            # 3) make request (without immediate response) to operators
            # 4) periodically for waiting requests: run decision process -> possibly leave system (cancellation)
            # 5) call time trigger -> offer to all undecided assigned requests
            # 6) sequential processes for each undecided request: user-decision
            # 7) trigger charging ops

        :param sim_time: new simulation time
        :return: None
        """
        # 1)
        self.update_sim_state_fleets(sim_time - self.time_step, sim_time)
        new_travel_times = self.routing_engine.update_network(sim_time)
        if new_travel_times:
            for op_id in range(self.n_op):
                self.operators[op_id].inform_network_travel_time_update(sim_time)
        # 2)
        last_time = sim_time - self.time_step
        if last_time < self.start_time:
            last_time = None
        list_new_traveler_rid_obj = self.demand.get_new_travelers(sim_time, since=last_time)

        # 3)
        for rid, rq_obj in list_new_traveler_rid_obj:
            for op_id in range(self.n_op):
                LOG.debug(f"Request {rid}: To operator {op_id} ...")
                self.operators[op_id].user_request(rq_obj, sim_time)

        # 4)
        self._check_waiting_request_cancellations(sim_time)

        # 5)
        for op_id, op_obj in enumerate(self.operators):
            # here offers are created in batch assignment
            op_obj.time_trigger(sim_time)

        # 6)
        for rid, rq_obj in self.demand.get_undecided_travelers(sim_time):
            for op_id in range(self.n_op):
                amod_offer = self.operators[op_id].get_current_offer(rid)
                LOG.debug(f"amod offer {amod_offer}")
                if amod_offer is not None:
                    rq_obj.receive_offer(op_id, amod_offer, sim_time)
            self._rid_chooses_offer(rid, rq_obj, sim_time)
            
        # 7)
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(sim_time)

        self.record_stats()
