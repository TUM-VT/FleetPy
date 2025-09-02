# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import random

from typing import TYPE_CHECKING

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
from src.FleetSimulationBase import FleetSimulationBase
from src.simulation.Vehicles import ExternallyControlledVehicle
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.VehiclePlan import VehiclePlan

if TYPE_CHECKING:
    from src.demand.TravelerModels import SlaveRequest

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

class MATSimSimulationClass(FleetSimulationBase):
    """
    A class to handle the coupling with MATSim.
    """
    def __init__(self, scenario_parameters):
        super().__init__(scenario_parameters)

        self.fs_time = self.scenario_parameters[G_SIM_START_TIME]
        
        self._registered_boardings = {} # dict rid -> 1
        self._current_alightings = {} # dict vid -> rid -> 1
        
    @staticmethod
    def get_directory_dict(scenario_parameters, list_operator_dicts):
        dirs = get_directory_dict(scenario_parameters, list_operator_dicts)
        iteration = scenario_parameters.get("matsim_iteration",0)
        dirs[G_DIR_OUTPUT] = os.path.join(dirs[G_DIR_OUTPUT], "matsim", str(iteration))
        os.makedirs(dirs[G_DIR_OUTPUT], exist_ok=True)
        return dirs

    def add_init(self, scenario_parameters):
        """
        Simulation specific additional init.
        :param scenario_parameters: row of pandas data-frame; entries are saved as x["key"]
        """
        super().add_init(scenario_parameters)

    def add_evaluate(self):
        """Runs standard and simulation environment specific evaluations over simulation results."""
        super().add_evaluate()
    
    def _load_fleetctr_vehicles(self):
        
        # set number of vehicles to 0 in fleet composition (those will be initialized in MATSim; super() must still run to load operators)
        old_fleet_composition = {}
        for op_id in range(self.n_op):
            operator_attributes = self.list_op_dicts[op_id]
            fleet_composition = operator_attributes[G_OP_FLEET]
            old_fleet_composition[op_id] = fleet_composition.copy()
            for veh_type, nr_veh in fleet_composition.items():
                fleet_composition[veh_type] = 0
            self.list_op_dicts[op_id][G_OP_FLEET] = fleet_composition
        
        super()._load_fleetctr_vehicles()
        
        #reset configuration of fleet composition
        for op_id in range(self.n_op):
            self.list_op_dicts[op_id][G_OP_FLEET] = old_fleet_composition[op_id]
    
    def add_vehicle(self, operator_id, max_pax, init_node ):
        route_output_flag = self.scenario_parameters.get(G_SIM_ROUTE_OUT_FLAG, True)
        replay_flag = self.scenario_parameters.get(G_SIM_REPLAY_FLAG, False)
        veh_attributes_dict = {
            G_VTYPE_NAME: "matsim_vehicle",
            G_VTYPE_MAX_PAX: max_pax,
            G_VTYPE_MAX_PARCELS: 0,
            G_VTYPE_FIX_COST: 0,
            G_VTYPE_DIST_COST: 0,
            G_VTYPE_BATTERY_SIZE: 9999999999999,
            G_VTYPE_RANGE: 99999999999999,
            "soc_per_m": 1/99999999999999*1000
        }
        veh_id = len(self.operators[operator_id].sim_vehicles)
        new_veh = ExternallyControlledVehicle(operator_id, veh_id, veh_attributes_dict, 
                                    self.routing_engine, self.demand.rq_db,
                                    self.op_output[operator_id], route_output_flag,
                                    replay_flag)
                            
        init_state_info = {}
        init_state_info[G_V_INIT_NODE] = init_node# np.random.choice(init_node)
        init_state_info[G_V_INIT_TIME] = self.scenario_parameters[G_SIM_START_TIME]
        init_state_info[G_V_INIT_SOC] = 1.0
        new_veh.set_initial_state(self.operators[operator_id], self.routing_engine, init_state_info,
                                    self.scenario_parameters[G_SIM_START_TIME], self.init_blocking)
        
        self.sim_vehicles[(operator_id, veh_id)] = new_veh
        self.operators[operator_id].sim_vehicles.append(new_veh)
        self.operators[operator_id].veh_plans[veh_id] = VehiclePlan(new_veh, self.fs_time, self.routing_engine, [])
        return veh_id
    
    def add_request(self, request_series):
        """
        Add a request to the simulation.
        :param request_series: pandas series with request information
        """
        rq_obj = self.demand.add_request(request_series, self.routing_engine, self.fs_time)
        self.broker.inform_request(rq_obj.rid, rq_obj, self.fs_time)
            
    def update_veh_state(self, sim_time, vid, op_id, veh_pos, rids_picked_up, rids_dropped_off, status, earliest_diverge_pos, earliest_diverge_time, finished_leg_ids,
                         current_pick_up, current_drop_off):
        """
        Update the vehicle state in the simulation.
        :param veh_id: Vehicle ID
        :param veh_pos: Vehicle position
        :param rids_picked_up: List of requests that have been picked up since last update
        :param rids_dropped_off: List of requests that have been dropped off since last update
        :param status: Vehicle status
        :param earliest_diverge_pos: Earliest diverge position
        :param earliest_diverge_time: Earliest diverge time
        :param finished_leg_ids: list of leg ids that the vehicle finished since the last update
        :param rids_picked_up: List of requests that are currently being picked up
        :param rids_dropped_off: List of requests that are currently being dropped off
        """
        LOG.debug(f"update_veh_state op_id {op_id} vid {vid} pos {veh_pos} pu {rids_picked_up} do {rids_dropped_off} status {status} edp {earliest_diverge_pos} edt {earliest_diverge_time} fl {finished_leg_ids} cpu {current_pick_up} cdo {current_drop_off}")
        veh_obj: ExternallyControlledVehicle = self.sim_vehicles[(op_id, vid)]
        for rid in current_pick_up:# rids_picked_up:
            if self._registered_boardings.get(rid) is None:
                self.demand.record_boarding(rid, vid, op_id, sim_time, pu_pos=veh_pos)
                self.broker.acknowledge_user_boarding(op_id, rid, vid, sim_time)
                self._registered_boardings[rid] = 1
        for rid in current_drop_off:
            if self._current_alightings.get(vid, {}).get(rid) is None:
                self.demand.record_alighting_start(rid, vid, op_id, sim_time, do_pos=veh_pos)
                try:
                    self._current_alightings[vid][rid] = 1
                except KeyError:
                    self._current_alightings[vid] = {rid: 1}
        for rid in list(self._current_alightings.get(vid, {}).keys()):
            if rid not in current_drop_off:
                # alighting process finished
                self.demand.user_ends_alighting(rid, vid, op_id, sim_time)
                self.broker.acknowledge_user_alighting(op_id, rid, vid, sim_time)
                try:
                    del self._registered_boardings[rid]
                except KeyError:
                    pass
                try:
                    del self._current_alightings[vid][rid]
                except KeyError:
                    pass
        # for rid in rids_dropped_off:
        #     self.demand.record_alighting_start(rid, vid, op_id, sim_time, do_pos=veh_pos) # TODO -> start_time/end_time of alighting process not really defined
        #     # # record user stats at end of alighting process
        #     self.demand.user_ends_alighting(rid, vid, op_id, sim_time)
        #     self.broker.acknowledge_user_alighting(op_id, rid, vid, sim_time)
        #     try:
        #         del self._registered_boardings[rid]
        #     except KeyError:
        #         pass
        done_VRLS = veh_obj.update_state(sim_time, veh_pos, rids_picked_up, rids_dropped_off, status,
                             earliest_diverge_pos, earliest_diverge_time, finished_leg_ids, current_pick_up, current_drop_off)
        # send update to operator
        if len(rids_picked_up) > 0 or len(rids_dropped_off) > 0:
            self.broker.receive_status_update(op_id, vid, sim_time, done_VRLS, True)
        else:
            self.broker.receive_status_update(op_id, vid, sim_time, done_VRLS, True) # TODO force update plan
    
    def step(self, sim_time):
        """
        Perform a simulation step.
        :param sim_time: Simulation time
        """
        self.fs_time = sim_time
            
        # 5)
        for op_id, op_obj in enumerate(self.operators):
            # here offers are created in batch assignment
            op_obj.time_trigger(sim_time)

        # 6)
        for rid, rq_obj in self.demand.get_undecided_travelers(sim_time):
            amod_offers = self.broker.collect_offers(rid)
            for op_id, amod_offer in amod_offers.items():
                rq_obj.receive_offer(op_id, amod_offer, sim_time)
            self._rid_chooses_offer(rid, rq_obj, sim_time)
            
        # 7)
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(sim_time)

        self.record_stats()
        
    def _rid_chooses_offer(self, rid, rq_obj: "SlaveRequest", sim_time):
        """This method performs all actions that derive from a mode choice decision.

        :param rid: request id
        :param rq_obj: request object
        :param sim_time: current simulation time
        :return: chosen operator
        """
        # The op_id/chosen_operator should be clearly defined: 
        # 0-n are amod operators, -1 is the rejection, -2 is the pt operator, None is undecided
        chosen_operator = rq_obj.set_chosen_offer(sim_time)
        LOG.debug(f" -> chosen operator: {chosen_operator}")
        if chosen_operator is None: # undecided
            if rq_obj.leaves_system(sim_time):
                self._user_leaves_system(rid, sim_time)
            else:
                self.demand.undecided_rq[rid] = rq_obj
        elif chosen_operator == -1:
            self._user_leaves_system(rid, sim_time)
        else:
            amode_confirmed_rids = self.broker.inform_user_booking(rid, rq_obj, sim_time, chosen_operator)
            for rid, rq_obj in amode_confirmed_rids:
                self.demand.waiting_rq[rid] = rq_obj
            try:
                del self.demand.undecided_rq[rid]
            except KeyError:
                # raises KeyError if request decided right away
                pass
        return chosen_operator
        
    def terminate(self):
        """Terminate the simulation."""
        # record stats
        self.record_stats()

        # save final state, record remaining travelers and vehicle tasks
        self.save_final_state()
        self.record_remaining_assignments()
        self.demand.record_remaining_users()

        self.evaluate()
        
    def get_current_assignments(self, sim_time):
        """
        Get new assignments from the simulation.
        :return: dict of new assignments
        """
        new_assignments = {}
        for vid, veh in self.sim_vehicles.items():
            new_assignment = veh.get_new_assignment(sim_time)
            if new_assignment is not None:
                new_assignments[vid] = new_assignment
        return new_assignments