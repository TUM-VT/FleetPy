# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
import datetime
from tqdm import tqdm

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
from src.BatchOfferSimulation import BatchOfferSimulation

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
PROGRESS_LOOP = "off"
PROGRESS_LOOP_VEHICLE_STATUS = [VRL_STATES.IDLE, VRL_STATES.CHARGING, VRL_STATES.REPOSITION]

# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------


# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
INPUT_PARAMETERS_RLBatchOfferSimulation = {
    "doc": """
    this class wraps the BatchOfferSimulation class for usage in a reinforcement learning environment.
    compared to the BatchOfferSimulation, information for the reinforcement learning agent is extracted.
    """,
    "inherit": "BatchOfferSimulation",
    "input_parameters_mandatory": [
    ],
    "input_parameters_optional": [
    ],
    "mandatory_modules": [
    ],
    "optional_modules": []
}


class RLBatchOfferSimulation(BatchOfferSimulation):
    """
    this class wraps the BatchOfferSimulation class for usage in a reinforcement learning environment.
    compared to the BatchOfferSimulation, information for the reinforcement learning agent is extracted.    """

    def step(self, sim_time: int, rl_action: int | None = None):
        """This method determines the simulation flow in a time step.
            # 1) update fleets and network
            # 2) get new travelers, add to undecided request
            # 3) make request (without immediate response) to operators
            # 4) periodically for waiting requests: run decision process -> possibly leave system (cancellation)
            # 5) call time trigger -> offer to all undecided assigned requests
            # 6) sequential processes for each undecided request: user-decision
            # 7) trigger charging ops

        :param sim_time: new simulation time
        :param rl_action: action from RL agent
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
            if rl_action is None:
                op_obj.time_trigger(sim_time)
            else:
                rl_var = op_obj.time_trigger(sim_time, rl_action=rl_action)

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

        if rl_action is not None:
            return rl_var

    def run(self, tqdm_position=0, rl_init=False):
        self._start_realtime_plot()
        t_run_start = time.perf_counter()
        if not self._started:
            self._started = True
            if PROGRESS_LOOP == "off":
                for sim_time in range(self.start_time, self.end_time, self.time_step):
                    if rl_init:  # end if it's for RL initialization
                        return
                    self.step(sim_time)
                    self._update_realtime_plots_dict(sim_time)
            elif PROGRESS_LOOP == "demand":
                # loop over time with progress bar scaling according to future demand
                all_requests = sum([len(x) for x in self.demand.future_requests.values()])
                with tqdm(total=100, position=tqdm_position) as pbar:
                    pbar.set_description(self.scenario_parameters.get(G_SCENARIO_NAME))
                    for sim_time in range(self.start_time, self.end_time, self.time_step):
                        remaining_requests = sum([len(x) for x in self.demand.future_requests.values()])
                        if rl_init:  # end if it's for RL initialization
                            return
                        self.step(sim_time)
                        cur_perc = int(100 * (1 - remaining_requests / all_requests))
                        pbar.update(cur_perc - pbar.n)
                        vehicle_counts = self.count_fleet_status()
                        info_dict = {"simulation_time": sim_time,
                                     "driving": sum([vehicle_counts[x] for x in G_DRIVING_STATUS])}
                        info_dict.update({x.display_name: vehicle_counts[x] for x in PROGRESS_LOOP_VEHICLE_STATUS})
                        pbar.set_postfix(info_dict)
                        self._update_realtime_plots_dict(sim_time)
            else:
                # loop over time with progress bar scaling with time
                for sim_time in tqdm(range(self.start_time, self.end_time, self.time_step), position=tqdm_position,
                                     desc=self.scenario_parameters.get(G_SCENARIO_NAME)):
                    self.step(sim_time)
                    self._update_realtime_plots_dict(sim_time)

            # record stats
            self.record_stats()

            # save final state, record remaining travelers and vehicle tasks
            self.save_final_state()
            self.record_remaining_assignments()
            self.demand.record_remaining_users()
        if self.skip_output:
            return

        t_run_end = time.perf_counter()
        # call evaluation
        self.evaluate()
        t_eval_end = time.perf_counter()
        # short report
        t_init = datetime.timedelta(seconds=int(t_run_start - self.t_init_start))
        t_sim = datetime.timedelta(seconds=int(t_run_end - t_run_start))
        t_eval = datetime.timedelta(seconds=int(t_eval_end - t_run_end))
        prt_str = f"Scenario {self.scenario_name} finished:\n" \
                  f"{'initialization':>20} : {t_init} h\n" \
                  f"{'simulation':>20} : {t_sim} h\n" \
                  f"{'evaluation':>20} : {t_eval} h\n"
        print(prt_str)
        LOG.info(prt_str)
        self._end_realtime_plot()

    def record_stats(self, force=True):
        """This method records the stats at the end of the simulation."""
        super().record_stats(force=force)

        if self.skip_output:
            return

        output_dir = self.dir_names[G_DIR_OUTPUT]
        outputfile = os.path.join(output_dir, "3-0_RL_action_time.csv")
        for op_id, op_obj in enumerate(self.operators):
            op_obj.output_assigned_zone_time(outputfile)
        outputfile = os.path.join(output_dir, "3-0_n_SAV_zone.csv")
        for op_id, op_obj in enumerate(self.operators):
            op_obj.output_no_sav_zone_assigned_time(outputfile)
        outputfile = os.path.join(output_dir, "3-0_RL_state_time.csv")
        for op_id, op_obj in enumerate(self.operators):
            op_obj.output_state_df(outputfile)
