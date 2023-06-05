from __future__ import annotations
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import pandas as pd
import typing as tp

# src imports
# -----------
from src.misc.globals import *
from src.simulation.Vehicles import SimulationVehicle
LOG = logging.getLogger(__name__)

if tp.TYPE_CHECKING:
    from src.simulation.Legs import VehicleRouteLeg
    from src.fleetctrl.planning.VehiclePlan import VehiclePlan

# Simulation Vehicle class
# ------------------------
# > guarantee consistent movements in simulation and output
class FreelancerSimulationVehicle(SimulationVehicle):
    def __init__(self, operator_id, vehicle_id, vehicle_data_dir, vehicle_type, routing_engine, rq_db, op_output, record_route_flag, replay_flag, freelancer_op_id, possible_op_ids=[], operating_intervals=[], driver_id = None):
        super().__init__(operator_id, vehicle_id, vehicle_data_dir, vehicle_type, routing_engine, rq_db, op_output, record_route_flag, replay_flag)
        self.op_id = operator_id    # wird online upgedated (ohne job/operator -> freelancer-operator class) | entspricht dem momentanen aktiven operator
        self.freelancer_op_id = freelancer_op_id
        self.possible_op_ids = possible_op_ids  # -> über simulation gefixt | generell für das fzg verfügbar
        self.current_op_id_options = possible_op_ids[:]    # -> wird während simulation geupdated (für welchen betreiber darf das fzg im moment fahren?) | im momentanen simulationsschritt für das fzg verfügbar
        
        self.driver_id = driver_id
        if self.driver_id is None:
            self.driver_id = vehicle_id
        
        self.active_operating_intervals = [] # list of intervals in sim_time (s) -> alternating between activation and deactivation times
        for i in range(0, len(operating_intervals), 2):
            start_active = operating_intervals[i]
            end_active = operating_intervals[i+1]
            self.active_operating_intervals.append( (start_active, end_active) )
        LOG.debug(f"FreelancerVehicle {self} with operating hours {self.active_operating_intervals}")
        
    def end_current_leg(self, simulation_time):
        """_summary_
        to update:
            - wird das fahrzeug idle -> current_op_ids anpassen
        """
        r = super().end_current_leg(simulation_time)
        if len(self.assigned_route) == 0:
            self.op_id = self.freelancer_op_id
            self.current_op_id_options = self.possible_op_ids[:]
        return r
    
    def _append_to_output_dict(self) -> dict:
        app = {
            "driver_id" : self.driver_id
        }
        return app
    
    def check_vehicle_acceptance(self, sim_time, vehicle_plan: VehiclePlan = None) -> bool:
        """ 
        checks decicison of vehicle/driver if it accepts the vehicle plan (i.e. if its in line with operating interval)
        returns True if sim_time within an active operating interval
        """
        if len(self.active_operating_intervals) == 0:
            return True
        else:
            for s, e in self.active_operating_intervals:
                if sim_time >= s and sim_time <= e:
                    return True
        return False
    
    def assign_operator(self, op_id):
        # update operator ids (current op_ids)
        # should be called right before assign_vehicle_plan (in fleetctrl.assign_vehicle_plan)
        self.op_id = op_id
        self.current_op_id_options = [op_id]
    
    def assign_vehicle_plan(self, list_route_legs: tp.List[VehicleRouteLeg], sim_time, force_ignore_lock=False):
        # maybe all is covered by assign_operator
        return super().assign_vehicle_plan(list_route_legs, sim_time, force_ignore_lock)
    
    def set_initial_state(self, fleetctrl, routing_engine, state_dict, start_time, veh_init_blocking=True):
        # anzupassen? setzt ggf die neuen parameter aus der init
        # initialiserung der fahrzeuge am anfang alle in freelancerfleetcontrol-klasse
        # (state_dict)
        return super().set_initial_state(fleetctrl, routing_engine, state_dict, start_time, veh_init_blocking)
        