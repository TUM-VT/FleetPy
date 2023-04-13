# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging

# additional module imports (> requirements)
# ------------------------------------------
import numpy as np
from abc import abstractmethod, ABCMeta
from typing import List, Dict, Tuple, Optional

# src imports
# -----------
from src.simulation.Legs import VehicleRouteLeg
from src.simulation.Vehicles import SimulationVehicle
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.routing.NetworkBase import NetworkBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000


# =================================================================================================================== #
# ========= PLAN STOP CLASSES ======================================================================================= #
# =================================================================================================================== #

# class PlanStop:
#     def __init__(self, position, boarding_dict, max_trip_time_dict, earliest_pickup_time_dict, latest_pickup_time_dict,
#                  change_nr_pax, planned_arrival_time=None, planned_departure=None, planned_arrival_soc=None,
#                  locked=False, charging_power=0, started_at=None,
#                  existing_vcl=None, charging_unit_id=None):

class PlanStopBase(metaclass=ABCMeta):
    """ this abstract class defines all methods a PlanStop-Class has to implement
    this class corresponds to one spatiotemporal action a vehicle is planned to do during a vehicle plan
    a vehicle plan thereby consists of an temporal ordered list of PlanStops which are performed one after another
    vehicles are moving between these different plan stops.
    """
    
    @abstractmethod
    def get_pos(self) -> tuple:
        """returns network position of this plan stop
        :return: network position tuple """
        pass
    
    @abstractmethod
    def get_state(self) -> G_PLANSTOP_STATES:
        """ returns the state of the planstop 
        :return: plan stop state"""
        
    @abstractmethod
    def get_list_boarding_rids(self) -> list:
        """returns list of all request ids boarding at this plan stop
        :return: list of boarding rids"""
        
    @abstractmethod
    def get_list_alighting_rids(self) -> list:
        """ returns list of all request ids alighting at this plan stop
        :return: list of alighting rids"""
        
    @abstractmethod
    def get_charging_task_id(self) -> Tuple[int, str]:
        """ returns the id of the stationary charging process of the plan stop if present
        :return: charging task id (tuple(charging operator id, task id)); None if not present"""
    
    @abstractmethod
    def get_earliest_start_time(self) -> float:
        """ this function evaluates all time constraints and returns the
        earliest start time for the PlanStop
        :return: (float) earliest start time """
        pass
    
    @abstractmethod
    def get_latest_start_time(self, pax_infos : dict) -> float:
        """ this function evaluates all time constraints and returns the 
        latest start time of the Plan Stop.
        if maximum trip time constraints are applied, infos about boarding times are need to evaluate the
        latest drop off time constraints
        :param pax_infos: (dict) from corresponding vehicle plan rid -> list (boarding_time, deboarding time) (only boarding time needed)
        :return: (float) latest start time"""
        pass
    
    @abstractmethod
    def get_duration_and_earliest_departure(self) -> tuple:
        """ returns a tuple of planned stop duration and absolute earliest departure time at stop
        :return: (stop duration, earliest departure time) | None if not given"""
    
    @abstractmethod
    def get_started_at(self) -> float:
        """ this function returns the time this plan stop started at; None if not started by the vehicle yet
        :return: float of time or None"""
        
    @abstractmethod
    def get_change_nr_pax(self) -> int:
        """ get the change of person occupancy after this plan stop 
        :return: change number pax (difference between boarding and deboarding persons)"""
        
    @abstractmethod
    def get_change_nr_parcels(self) -> int:
        """ get the change of parcel occupancy after this plan stop 
        :return: change number parcels (difference between boarding and deboarding parcels)"""
        
    @abstractmethod
    def get_departure_time(self, start_time : float) -> float:
        """ this function returns the time the vehicle leaves the plan stop if it is started at start_time
        :param start_time: time the plan stop has been started
        :return: time vehicle is supposed to leave"""
        
    @abstractmethod
    def get_charging_power(self) -> float:
        """ returns the charging power at this plan stop 
        :return: charging power"""
        
    @abstractmethod
    def get_boarding_time_constraint_dicts(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """ returns a tuple of all boarding constraints dicts (rid -> time constraint)
        :return: dict earliest_boarding_time, latest_boarding_times, max_travel_times, latest_arrival_times"""
        
    @abstractmethod
    def get_planned_arrival_and_departure_time(self) -> Tuple[float, float]:
        """ returns time of arrival and departure planned within the plan
        :return: tuple of planned arrival time and planned departure time"""
        
    @abstractmethod
    def get_planned_arrival_and_departure_soc(self) -> Tuple[float, float]:
        """returns the planned soc when arriving at plan stop
        :return: planned soc at start and end of charging process"""
    
    @abstractmethod
    def is_locked(self) -> bool:
        """test for lock
        :return: bool True, if plan stop is locked"""
        
    @abstractmethod
    def is_locked_end(self) -> bool:
        """ ths for end lock
        :return: bool True, if plan stop is locked at end of plan stop (no insertion after this possible)"""
        
    @abstractmethod
    def is_infeasible_locked(self) -> bool:
        """ this if planstop is locked due to infeasible time constraints
        :return: True, if infeasible locked"""
    
    @abstractmethod
    def is_inactive(self) -> bool:
        """ this function evaluates if this is an inactive PlanStop (i.e. undefined duration and no tasks)
        :return: (bool) True if inactive, else False """
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        """ tests if nothing has to be done here and its just a routing target marker (i.e. reloc target)
        :return: (bool)"""
        
    @abstractmethod
    def set_locked(self, locked : bool):
        """ sets the locked state of the plan stop
        :param locked: True, if this plan stop should be locked"""
        
    @abstractmethod
    def set_infeasible_locked(self, infeasible_locked : bool):
        """ sets infeasible locked state if time constraints can no longer be fullfilled
        :param infeasible_locked: True, if infeasible locked state applied"""
        
    @abstractmethod
    def set_started_at(self, start_time : float):
        """this function sets the time when the plan stop has been started by a vehicle
        :param start_time: float; simulation time when vehicle started the plan stop"""
        
    @abstractmethod
    def set_planned_arrival_and_departure_time(self, arrival_time : float, departure_time : float):
        """ set the planned arrival and departure time at plan stop
        :param arrival_time: time of vehicle arrival
        :param departure_time: planned time of departure"""
        
    @abstractmethod
    def set_duration_and_earliest_end_time(self, duration : float=None, earliest_end_time : float=None):
        """ can be used to reset duration and earliest end time of the plan stop (ignored if None)
        :param duration: new duration of plan stop
        :param earliest_end_time: new earliest end time of plan stop"""
        
    @abstractmethod
    def set_planned_arrival_and_departure_soc(self, arrival_soc : float, departure_soc : float):
        """ set the planned soc at arrival and departure at plan stop
        :param arrival soc: soc of vehicle at arrival
        :param departure_soc: soc at end of charging process"""
        
    @abstractmethod
    def update_rid_boarding_time_constraints(self, rid, new_earliest_pickup_time : float=None, new_latest_pickup_time : float=None):
        """ this method can be used to update boarding time constraints a request in this plan stop (if given)
        :param rid: request id
        :param new_earliest_pickup_time: new earliest pick up time constraint of rid
        :param new_latest_pickup_time: new latest pick up time constraint of rid"""
        
    @abstractmethod
    def update_rid_alighting_time_constraints(self, rid, new_maxmium_travel_time : float=None, new_latest_dropoff_time : float=None):
        """ this method can be used to update alighting time constraints a request in this plan stop (if given)
        :param rid: request id
        :param new_maxmium_travel_time: new maximum travel time constraint of rid
        :param new_latest_dropoff_time: new latest dropoff time constraint of rid"""
        
    @abstractmethod
    def copy(self):
        """ this function returns the copy of a plan stop
        :return: PlanStop copy
        """
        pass

class PlanStop(PlanStopBase):
    """this class corresponds to one spatiotemporal action a vehicle is planned to do during a vehicle plan
        a vehicle plan thereby consists of an temporal ordered list of PlanStops which are performed one after another
        vehicles are moving between these different plan stops.
        this class is the most general class of plan stops"""
    def __init__(self, position, boarding_dict={}, max_trip_time_dict={}, latest_arrival_time_dict={}, earliest_pickup_time_dict={}, latest_pickup_time_dict={},
                 change_nr_pax=0, change_nr_parcels=0, earliest_start_time=None, latest_start_time=None, duration=None, earliest_end_time=None,
                 locked=False, locked_end=False, charging_power=0, planstop_state : G_PLANSTOP_STATES=G_PLANSTOP_STATES.MIXED,
                 charging_task_id: Tuple[int, str] = None, status: Optional[VRL_STATES] = None):
        """
        :param position: network position (3 tuple) of the position this PlanStops takes place (target for routing)
        :param boarding_dict: dictionary with entries +1 -> list of request ids that board the vehicle there; -1 -> list of requests that alight the vehicle there
        :param max_trip_time_dict: dictionary request_id -> maximum trip time of all requests alighting at this stop to check max trip time constraint
        :param latest_arrival_time_dict: dictionary request_id -> absolute latest arival time of all requests alighting at this stop to check latest arrival time constraint
        :param earliest_pickup_time_dict: dictionary request_id -> earliest pickup time of all requests boarding at this stop to check earliest pickup time constraint
        :param latest_pickup_time_dict: dictionary request_id -> latest pickup time of all requests boarding at this top to check latest pickup time constraint
        :param change_nr_pax: (int) change of number of passengers at this point: number people boarding - number people alighting to check capacity constraint
        :param change_nr_parcels: (int) change of number of parcels at this point: number boarding parcels - number alighting parcels to check capacity constraint
        :param earliest_start_time: (float) absolute earliest start time this plan stop is allowed to start
        :param latest_start_time: (float) absolute latest start time this plan stop is allowed to start
        :param duration: (float) minimum duration this plan stops takes at this location
        :param earliest_end_time: (float) absolute earliest time a vehicle is allowed to leave at this plan stop
        :param locked: (bool) false by default; if true this planstop can no longer be unassigned from vehicleplan and has to be fullfilled. currently only working when also all planstops before this planstop are locked, too
        :param locked_end: (bool) false by default; if true, no planstops can be added after this planstop in the assignment algorithm and it cannot be removed by the assignemnt algorithm (insertions before are possible!)
        :param charging_power: optional (float); if given the vehicle is charged with this power (TODO unit!) while at this stop
        :param planstop_state: used to characterize the planstop state (task to to there)
        :param charging_task_id: the stationary task to be performed at the plan stop
        :param status:          vehicle status while performing the current plan stop
        """
        
        self.pos = position
        self.state = planstop_state
        
        self.boarding_dict = boarding_dict  # +1: [rids] for boarding | -1: [rids] for alighting
        self.locked = locked
        self.locked_end = locked_end
        
        # charging
        self.charging_power = charging_power
        
        # parameters that define capacity constraints
        self.change_nr_pax = change_nr_pax
        self.change_nr_parcels = change_nr_parcels
        # parameters that define time constraints
        self.max_trip_time_dict = max_trip_time_dict  # deboarding rid -> max_trip_time constraint
        self.latest_arrival_time_dict = latest_arrival_time_dict    # deboarding rid -> latest_arrival_time constraint
        self.earliest_pickup_time_dict = earliest_pickup_time_dict  # boarding rid -> earliest pickup time
        self.latest_pickup_time_dict = latest_pickup_time_dict  # boarding rid -> latest pickup time
        if type(self.boarding_dict) != dict:
            raise TypeError
        if type(self.max_trip_time_dict) != dict:
            raise TypeError
        if type(self.latest_arrival_time_dict) != dict:
            raise TypeError
        if type(self.earliest_pickup_time_dict) != dict:
            raise TypeError
        if type(self.latest_pickup_time_dict) != dict:
            raise TypeError
        # constraints independent from boarding processes
        self.direct_earliest_start_time = earliest_start_time
        self.direct_latest_start_time = latest_start_time
        
        self.direct_duration = duration
        self.direct_earliest_end_time = earliest_end_time
        if duration is not None:
            x = int(self.direct_duration)
        
        # constraints   (will be computed in update travel time by evaluating the whole plan)
        self._latest_start_time = None
        self._earliest_start_time = None 
        
        # planning properties (will be set during evaluation of whole plan)
        self._planned_arrival_time = None
        self._planned_departure_time = None
        self._planned_arrival_soc = None
        self._planned_departure_soc = None

        self.started_at = None  # is only set in update_plan
        self.infeasible_locked = False

        self.charging_task_id: Tuple[int, str] = charging_task_id
        
    def get_pos(self) -> tuple:
        """returns network position of this plan stop
        :return: network position tuple """
        return self.pos
    
    def get_state(self) -> G_PLANSTOP_STATES:
        return self.state
        
    def get_list_boarding_rids(self) -> list:
        """returns list of all request ids boarding at this plan stop
        :return: list of boarding rids"""
        return self.boarding_dict.get(1, [])
        
    def get_list_alighting_rids(self) -> list:
        """ returns list of all request ids alighting at this plan stop
        :return: list of alighting rids"""
        return self.boarding_dict.get(-1, [])

    def get_charging_task_id(self) -> Tuple[int, str]:
        """ returns the id of the stationary charging process of the plan stop if present
        :return: charging task id (tuple(charging operator id, task id)); None if not present"""
        return self.charging_task_id

    def copy(self):
        """ this function returns the copy of a plan stop
        :return: PlanStop copy
        """
        cp_ps = PlanStop(self.pos, boarding_dict=self.boarding_dict.copy(), max_trip_time_dict=self.max_trip_time_dict.copy(),
                         latest_arrival_time_dict=self.latest_arrival_time_dict.copy(), earliest_pickup_time_dict=self.earliest_pickup_time_dict.copy(),
                         latest_pickup_time_dict=self.latest_pickup_time_dict.copy(), change_nr_pax=self.change_nr_pax, change_nr_parcels=self.change_nr_parcels,
                         earliest_start_time=self.direct_earliest_start_time, latest_start_time=self.direct_latest_start_time,
                         duration=self.direct_duration, earliest_end_time=self.direct_earliest_end_time, locked=self.locked, locked_end=self.locked_end,
                         charging_power=self.charging_power, charging_task_id=self.charging_task_id, planstop_state=self.state)
        cp_ps._planned_arrival_time = self._planned_arrival_time
        cp_ps._planned_departure_time = self._planned_departure_time
        cp_ps._planned_arrival_soc = self._planned_arrival_soc
        cp_ps._planned_departure_soc = self._planned_departure_soc
        cp_ps.started_at = self.started_at
        return cp_ps

    def get_earliest_start_time(self) -> float:
        """ this function evaluates all time constraints and returns the
        earliest start time for the PlanStop
        :return: (float) earliest start time """
        self._earliest_start_time = -1
        if self.direct_earliest_start_time is not None and self.direct_earliest_start_time > self._earliest_start_time:
            self._earliest_start_time = self.direct_earliest_start_time
        if len(self.earliest_pickup_time_dict.values()) > 0:
            ept = np.floor(max(self.earliest_pickup_time_dict.values()))
            if ept > self._earliest_start_time:
                self._earliest_start_time = ept
        #LOG.debug("get earliest start time: {}".format(str(self)))
        return self._earliest_start_time        

    def get_latest_start_time(self, pax_infos : dict) -> float:
        """ this function evaluates all time constraints and returns the 
        latest start time of the Plan Stop.
        if maximum trip time constraints are applied, infos about boarding times are need to evaluate the
        latest drop off time constraints
        :param pax_infos: (dict) from corresponding vehicle plan rid -> list (boarding_time, deboarding time) (only boarding time needed)
        :return: (float) latest start time"""
        self._latest_start_time = LARGE_INT
        if self.direct_latest_start_time is not None and self.direct_latest_start_time < self._latest_start_time:
            self._latest_start_time = self.direct_latest_start_time
        if len(self.latest_pickup_time_dict.values()) > 0:
            la = np.ceil(min(self.latest_pickup_time_dict.values()))
            if la < self._latest_start_time:
                self._latest_start_time = la
        if len(self.max_trip_time_dict.values()) > 0:
            la = np.ceil(min((pax_infos[rid][0] + self.max_trip_time_dict[rid] for rid in self.boarding_dict.get(-1, []))))
            if la < self._latest_start_time:
                self._latest_start_time = la
        if len(self.latest_arrival_time_dict.values()) > 0:
            la = np.ceil(min(self.latest_arrival_time_dict.values()))
            if la < self._latest_start_time:
                self._latest_start_time = la
        #LOG.debug("get latest start time: {}".format(str(self)))
        return self._latest_start_time
    
    def get_started_at(self) -> float:
        return self.started_at
    
    def get_change_nr_pax(self) -> int:
        return self.change_nr_pax
    
    def get_change_nr_parcels(self) -> int:
        return self.change_nr_parcels
    
    def get_departure_time(self, start_time: float) -> float:
        """ this function returns the time the vehicle leaves the plan stop if it is started at start_time
        :param start_time: time the plan stop has been started
        :return: time vehicle is supposed to leave"""
        departure_time = start_time
        if self.direct_duration is not None:
            departure_time = start_time + self.direct_duration
        if self.direct_earliest_end_time is not None and departure_time < self.direct_earliest_end_time:
            departure_time = self.direct_earliest_end_time
        return departure_time
    
    def get_duration_and_earliest_departure(self) -> tuple:
        return self.direct_duration, self.direct_earliest_end_time
    
    def get_charging_power(self) -> float:
        return self.charging_power
    
    def get_boarding_time_constraint_dicts(self) -> Tuple[Dict, Dict, Dict, Dict]:
        return self.earliest_pickup_time_dict, self.latest_pickup_time_dict, self.max_trip_time_dict, self.latest_arrival_time_dict

    def get_planned_arrival_and_departure_time(self) -> Tuple[float, float]:
        return self._planned_arrival_time, self._planned_departure_time
    
    def get_planned_arrival_and_departure_soc(self) -> Tuple[float, float]:
        return self._planned_arrival_soc, self._planned_departure_soc

    def is_inactive(self) -> bool:
        """ this function evaluates if this is an inactive PlanStop (i.e. undefined duration and no tasks)
        :return: (bool) True if inactive, else False """
        if self.state == G_PLANSTOP_STATES.INACTIVE or self.get_departure_time(0) > LARGE_INT:
            return True
        else:
            return False
        
    def is_locked(self) -> bool:
        return self.locked
    
    def is_locked_end(self) -> bool:
        return self.locked_end
    
    def is_infeasible_locked(self) -> bool:
        return self.infeasible_locked
    
    def set_locked(self, locked: bool):
        self.locked = locked
        
    def set_infeasible_locked(self, infeasible_locked: bool):
        self.infeasible_locked = infeasible_locked
        
    def set_started_at(self, start_time: float):
        self.started_at = start_time
        
    def set_planned_arrival_and_departure_soc(self, arrival_soc: float, departure_soc: float):
        self._planned_arrival_soc = arrival_soc
        self._planned_departure_soc = departure_soc
        
    def set_planned_arrival_and_departure_time(self, arrival_time: float, departure_time: float):
        self._planned_arrival_time = arrival_time
        self._planned_departure_time = departure_time
        
    def set_duration_and_earliest_end_time(self, duration: float = None, earliest_end_time: float = None):
        if duration is not None:
            self.direct_duration = duration
        if earliest_end_time is not None:
            self.direct_earliest_end_time = earliest_end_time
        
    def update_rid_boarding_time_constraints(self, rid, new_earliest_pickup_time: float = None, new_latest_pickup_time: float = None):
        if new_earliest_pickup_time is not None:
            self.earliest_pickup_time_dict[rid] = new_earliest_pickup_time
        if new_latest_pickup_time is not None:
            self.latest_pickup_time_dict[rid] = new_latest_pickup_time
            
    def update_rid_alighting_time_constraints(self, rid, new_maxmium_travel_time: float = None, new_latest_dropoff_time: float = None):
        if new_maxmium_travel_time is not None:
            self.max_trip_time_dict[rid] = new_maxmium_travel_time
        if new_latest_dropoff_time is not None:
            self.latest_arrival_time_dict[rid] = new_latest_dropoff_time

    def __str__(self):
        return f"PS: {self.pos} state {self.state.name} locked {self.locked} bd {self.boarding_dict} earl dep {self._earliest_start_time} latest arr " \
               f"{self._latest_start_time} eta {self._planned_arrival_time}"

    def is_empty(self) -> bool:
        """ tests if nothing has to be done here and its just a routing target marker (i.e. reloc target)
        :return: (bool)"""
        if self.change_nr_pax == 0 and len(self.boarding_dict.get(1, [])) == 0 and len(self.boarding_dict.get(-1, [])) == 0 and self.charging_power == 0: #and len(self.planned_departure) == 0
            return True
        else:
            return False

class BoardingPlanStop(PlanStop):
    """ this class can be used to generate a plan stop where only boarding processes take place """
    def __init__(self, position, boarding_dict={}, max_trip_time_dict={}, latest_arrival_time_dict={},
                 earliest_pickup_time_dict={}, latest_pickup_time_dict={}, change_nr_pax=0, change_nr_parcels=0, duration=None, locked=False):
        """
        :param position: network position (3 tuple) of the position this PlanStops takes place (target for routing)
        :param boarding_dict: dictionary with entries +1 -> list of request ids that board the vehicle there; -1 -> list of requests that alight the vehicle there
        :param max_trip_time_dict: dictionary request_id -> maximum trip time of all requests alighting at this stop to check max trip time constraint
        :param latest_arrival_time_dict: dictionary request_id -> absolute latest arival time of all requests alighting at this stop to check latest arrival time constraint
        :param earliest_pickup_time_dict: dictionary request_id -> earliest pickup time of all requests boarding at this stop to check earliest pickup time constraint
        :param latest_pickup_time_dict: dictionary request_id -> latest pickup time of all requests boarding at this top to check latest pickup time constraint
        :param change_nr_pax: (int) change of number of passengers at this point: number people boarding - number people alighting to check capacity constraint
        :param change_nr_parcels: (int) change of number of parcels at this point: number boarding parcels - number alighting parcels to check capacity constraint
        :param duration: (float) minimum duration this plan stops takes at this location
        :param locked: (bool) false by default; if true this planstop can no longer be unassigned from vehicleplan and has to be fullfilled. currently only working when also all planstops before this planstop are locked, too
        """
        super().__init__(position, boarding_dict=boarding_dict, max_trip_time_dict=max_trip_time_dict,
                         latest_arrival_time_dict=latest_arrival_time_dict, earliest_pickup_time_dict=earliest_pickup_time_dict,
                         latest_pickup_time_dict=latest_pickup_time_dict, change_nr_pax=change_nr_pax, change_nr_parcels=change_nr_parcels,
                         earliest_start_time=None, latest_start_time=None,
                         duration=duration, earliest_end_time=None, locked=locked,
                         charging_power=0, planstop_state=G_PLANSTOP_STATES.BOARDING)
        
class RoutingTargetPlanStop(PlanStop):
    """ this plan stop can be used to schedule a routing target for vehicles with the only task to drive there
        i.e repositioning"""
    def __init__(self, position, earliest_start_time=None, latest_start_time=None, duration=None, earliest_end_time=None, locked=False, locked_end=True, planstop_state=G_PLANSTOP_STATES.REPO_TARGET):
        """
        :param position: network position (3 tuple) of the position this PlanStops takes place (target for routing)
        :param earliest_start_time: (float) absolute earliest start time this plan stop is allowed to start
        :param latest_start_time: (float) absolute latest start time this plan stop is allowed to start
        :param duration: (float) minimum duration this plan stops takes at this location
        :param earliest_end_time: (float) absolute earliest time a vehicle is allowed to leave at this plan stop
        :param locked: (bool) false by default; if true this planstop can no longer be unassigned from vehicleplan and has to be fullfilled. currently only working when also all planstops before this planstop are locked, too
        :param locked_end: (bool) false by default; if true, no planstops can be added after this planstop in the assignment algorithm and it cannot be removed by the assignemnt algorithm (insertions before are possible!)
        :param planstop_state: (G_PLANSTOP_STATES) indicates the planstop state. should be in (REPO_TARGET, INACTIVE, RESERVATION)
        """
        super().__init__(position, boarding_dict={}, max_trip_time_dict={}, latest_arrival_time_dict={}, earliest_pickup_time_dict={}, latest_pickup_time_dict={},
                         change_nr_pax=0, earliest_start_time=earliest_start_time, latest_start_time=latest_start_time, duration=duration,
                         earliest_end_time=earliest_end_time, locked=locked, locked_end=locked_end, charging_power=0, planstop_state=planstop_state)

class ChargingPlanStop(PlanStop):
    """ this plan stop can be used to schedule a charging only process """
    def __init__(self, position, earliest_start_time=None, latest_start_time=None, duration=None, 
                 earliest_end_time=None, locked=False, locked_end=False, charging_power=0,
                 charging_task_id: Tuple[int, str] = None, status: Optional[VRL_STATES] = None):
        """
        :param position: network position (3 tuple) of the position this PlanStops takes place (target for routing)
        :param earliest_start_time: (float) absolute earliest start time this plan stop is allowed to start
        :param latest_start_time: (float) absolute latest start time this plan stop is allowed to start
        :param duration: (float) minimum duration this plan stops takes at this location
        :param earliest_end_time: (float) absolute earliest time a vehicle is allowed to leave at this plan stop
        :param locked: (bool) false by default; if true this planstop can no longer be unassigned from vehicleplan and has to be fullfilled. currently only working when also all planstops before this planstop are locked, too
        :param locked_end: (bool) false by default; if true, no planstops can be added after this planstop in the assignment algorithm and it cannot be removed by the assignemnt algorithm (insertions before are possible!)        
        :param charging_power: optional (float); if given the vehicle is charged with this power (TODO unit!) while at this stop
        """
        super().__init__(position, boarding_dict={}, max_trip_time_dict={}, latest_arrival_time_dict={}, 
                         earliest_pickup_time_dict={}, latest_pickup_time_dict={}, change_nr_pax=0, 
                         earliest_start_time=earliest_start_time, latest_start_time=latest_start_time, duration=duration, 
                         earliest_end_time=earliest_end_time, locked=locked, locked_end=locked_end, charging_power=charging_power, 
                         planstop_state=G_PLANSTOP_STATES.CHARGING, charging_task_id=charging_task_id, status=status)

class VehiclePlan:
    """ this class is used to plan tasks for a vehicle and evaluates feasiblity of time constraints of this plan
    a Vehicle mainly consists of two parts:
        - a vehicle this plan is assigned to, and therefore the current state of the vehicle
        - an ordered list of PlanStops defining the tasks the vehicle is supposed to perform (vehicles move from one plan stop to another)"""
    def __init__(self, veh_obj : SimulationVehicle, sim_time : float, routing_engine : NetworkBase, list_plan_stops : List[PlanStopBase], copy: bool =False, external_pax_info : dict = {}):
        """
        :param veh_obj: corresponding simulation vehicle reference
        :param sim_time: current simulation time
        :param routing_engine: reference to routing engine
        :param list_plan_stops: ordered list of plan stops to perform
        :param copy: optional; set if an init is set for creation of a copy of the plan (only for internal use)
        :param external_pax_info: optional; dictionary of allready computed pax info (only for internal use)
        """
        self.list_plan_stops = list_plan_stops
        self.utility = None
        # pax info:
        # rid -> [start_boarding, end_alighting] where start_boarding can be in past or planned
        # rid -> [start_boarding_time] in case only boarding is planned
        self.pax_info = external_pax_info
        self.vid = None
        self.feasible = None
        self.structural_feasible = True  # indicates if plan is in line with vehicle state ignoring time constraints
        if not copy:
            self.vid = veh_obj.vid
            self.feasible = self.update_tt_and_check_plan(veh_obj, sim_time, routing_engine, keep_feasible=True)

    def __str__(self):
        return "veh plan for vid {} feasible? {} : {} | pax info {}".format(self.vid, self.feasible,
                                                                            [str(x) for x in self.list_plan_stops],
                                                                            self.pax_info)

    def copy(self):
        """
        creates a copy
        """
        tmp_VehiclePlan = VehiclePlan(None, None, None, [ps.copy() for ps in self.list_plan_stops], copy=True)
        tmp_VehiclePlan.vid = self.vid
        tmp_VehiclePlan.utility = self.utility
        tmp_VehiclePlan.pax_info = self.pax_info.copy()
        tmp_VehiclePlan.feasible = True
        return tmp_VehiclePlan

    def is_feasible(self) -> bool:
        """ this method can be used to check of plan is feasible
        :return: (bool) True if feasible"""
        return self.feasible

    def is_structural_feasible(self) -> bool:
        """ indicates if stop order is feasible with current vehicles state (ignoring time constraints) 
        :return: (bool) True if structural feasible """
        return self.structural_feasible

    def get_pax_info(self, rid) -> list:
        """ this function returns passenger infos regarding planned boarding and alighting time for this plan
        :param rid: request id involved in this plan
        :return: list with maximally length 2; first entry planned boarding time; second entry planned alighting time; None if no information found"""
        return self.pax_info.get(rid)

    def get_involved_request_ids(self) -> list:
        """ get a list of all request ids that are scheduled in this plan 
        :return: list of request ids"""
        return list(self.pax_info.keys())

    def set_utility(self, utility_value : float):
        """ this method is used to set the utility (cost function value) of this plan
        :param utility_value: float of utility value"""
        self.utility = utility_value

    def get_utility(self) -> float:
        """ returns the utility value of the plan (None if not set yet)
        :return: utility value (cost function value) or None"""
        return self.utility

    def add_plan_stop(self, plan_stop : PlanStopBase, veh_obj : SimulationVehicle, sim_time : float, routing_engine : NetworkBase, return_copy : bool=False, position : tuple=None):
        """This method adds a plan stop to an existing vehicle plan. After that, it updates the plan.

        :param plan_stop: new plan stop
        :param veh_obj: simulation vehicle instance
        :param sim_time: current simulation time
        :param routing_engine: routing engine
        :param return_copy: controls whether the current plan is changed or a changed copy will be returned
        :param position: position in list_plan_stops in which the plan stop should be added
        :return: None (return_copy=False) or VehiclePlan instance (return_copy=True)
        """
        if return_copy:
            new_veh_plan = self.copy()
        else:
            new_veh_plan = self
        if position is None:
            new_veh_plan.list_plan_stops.append(plan_stop)
        else:
            new_veh_plan.list_plan_stops.insert(position, plan_stop)
        new_veh_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine, keep_feasible=True)

    def update_plan(self, veh_obj : SimulationVehicle, sim_time : float, routing_engine : NetworkBase, list_passed_VRLs : List[VehicleRouteLeg]=None, keep_time_infeasible : bool=True) -> bool:
        """This method checks whether the simulation vehicle passed some of the planned stops and removes them from the
        plan after passing. It returns the feasibility of the plan.

        :param veh_obj: vehicle object to which plan is applied
        :param sim_time: current simulation time
        :param routing_engine: reference to routing engine
        :param list_passed_VRLs: list of passed VRLs
        :param keep_time_infeasible: if True full evaluation of feasiblity even though infeasibility of time constraints have been found
        :return: is_feasible returns True if all
        """
        # 1) check if list_passed_VRLs invalidates the plan or removes some stops
        # LOG.debug("update_plan")
        self.feasible = True
        if list_passed_VRLs is None:
            list_passed_VRLs = []
        # LOG.debug(str(self))
        # LOG.debug([str(x) for x in list_passed_VRLs])
        # LOG.debug([str(x) for x in self.list_plan_stops])
        key_translator = {sub_rid[0]: sub_rid for sub_rid in self.pax_info.keys() if type(sub_rid) == tuple}
        if list_passed_VRLs and self.list_plan_stops:
            for vrl in list_passed_VRLs:
                if vrl.status in G_DRIVING_STATUS or vrl.status in G_LAZY_STATUS:
                    continue
                # if vrl.status in G_LAZY_STATUS:
                #     # waiting part should not be part of the vehicle plan
                #     continue
                # if vrl.status in G_DRIVING_STATUS or vrl.status in G_LAZY_STATUS:
                #     if vrl.destination_pos == self.list_plan_stops[0].get_pos() and self.list_plan_stops[0].is_empty():
                #         # LOG.info("jumped ps {} becouse of vrl {}".format(self.list_plan_stops[0], vrl))
                #         self.list_plan_stops = self.list_plan_stops[1:]
                #     continue
                if vrl.destination_pos == self.list_plan_stops[0].get_pos():
                    # plan infeasible as soon as other people board the vehicle
                    rid_boarded_at_stop = set([key_translator.get(rq.get_rid_struct(), rq.get_rid_struct())
                                               for rq in vrl.rq_dict.get(1, [])])
                    if not rid_boarded_at_stop == set(self.list_plan_stops[0].get_list_boarding_rids()):
                        # LOG.debug(" -> wrong boarding")
                        self.feasible = False
                        self.structural_feasible = False
                        return False
                    # other people alighting should not be possible. keep check nevertheless
                    rid_alighted_at_stop = set([key_translator.get(rq.get_rid_struct(), rq.get_rid_struct()) for rq in
                                                vrl.rq_dict.get(-1, [])])
                    if not rid_alighted_at_stop == set(self.list_plan_stops[0].get_list_alighting_rids()):
                        # LOG.debug(" -> wrong alighting")
                        self.feasible = False
                        self.structural_feasible = False
                        return False
                    # remove stop from plan
                    self.list_plan_stops = self.list_plan_stops[1:]
                else:
                    # plan infeasible as soon as anybody boarded or alighted the vehicle
                    if vrl.rq_dict.get(1) or vrl.rq_dict.get(-1):
                        # LOG.debug(" -> unplanned boarding step")
                        self.feasible = False
                        self.structural_feasible = False
                        return False
        # 2) check for current boarding processes and check if current stop should be locked
        if veh_obj.assigned_route and self.list_plan_stops:
            ca = veh_obj.assigned_route[0]
            if not ca.status in G_DRIVING_STATUS and not ca.status in G_LAZY_STATUS:
                if ca.destination_pos == self.list_plan_stops[0].get_pos():
                    rid_boarding_at_stop = set(
                        [key_translator.get(rq.get_rid_struct(), rq.get_rid_struct()) for rq in ca.rq_dict.get(1, [])])
                    if not rid_boarding_at_stop == set(self.list_plan_stops[0].get_list_boarding_rids()):
                        # LOG.debug(" -> current boarding states is wrong!")
                        self.feasible = False
                        self.structural_feasible = False
                        return False
                    rid_deboarding_at_stop = set(
                        [key_translator.get(rq.get_rid_struct(), rq.get_rid_struct()) for rq in ca.rq_dict.get(-1, [])])
                    if not rid_deboarding_at_stop == set(self.list_plan_stops[0].get_list_alighting_rids()):
                        # LOG.debug(" -> current deboarding states is wrong!")
                        self.feasible = False
                        self.structural_feasible = False
                        return False
                else:
                    # LOG.debug(" -> infeasible planned stop")
                    self.feasible = False
                    self.structural_feasible = False
                    return False

            if ca.locked and ca.destination_pos == self.list_plan_stops[0].get_pos():
                # LOG.debug(" -> LOCK!")
                self.list_plan_stops[0].set_locked(True)
                # LOG.verbose("set starting time: {}".format(veh_obj.cl_start_time))
                if not ca.status in G_DRIVING_STATUS and not ca.status in G_LAZY_STATUS:  # TODO #
                    self.list_plan_stops[0].set_started_at(veh_obj.cl_start_time)

        # 3) update planned attributes (arrival_time, arrival_soc, departure)
        # LOG.debug("after update plan:")
        # LOG.debug(str(self))
        # LOG.debug(f"currently ob: {veh_obj.pax}")
        self.feasible = self.update_tt_and_check_plan(veh_obj, sim_time, routing_engine,
                                                      keep_feasible=keep_time_infeasible)
        return self.feasible

    def return_intermediary_plan_state(self, veh_obj : SimulationVehicle, sim_time : float, routing_engine : NetworkBase, stop_index : int) -> dict:
        """ this function evalutes the future vehicle state after it would have performed the next stop_index plan stops of the vehicle plan
        and returns a dictionary specifing the vehicle state
        :param veh_obj: reference the vehicle object
        :param sim_time: simulation time
        :param routing_engine: routing engine reference
        :param stop_index: index of list plan stops of vehicle plan until the state is evaluated
        :return: dictionary specifying the future vehicle state"""
        c_pos = veh_obj.pos
        c_soc = veh_obj.soc
        c_time = sim_time
        if self.list_plan_stops[0].is_locked():  # set time at start_time of boarding process
            boarding_startet = self.list_plan_stops[0].get_started_at()
            if boarding_startet is not None:
                c_time = boarding_startet
        key_translator = {sub_rid[0]: sub_rid for sub_rid in self.pax_info.keys() if type(sub_rid) == tuple}
        c_pax = {key_translator.get(rq.get_rid_struct(), rq.get_rid_struct()): 1 for rq in veh_obj.pax}
        nr_pax = veh_obj.get_nr_pax_without_currently_boarding()  # sum([rq.nr_pax for rq in veh_obj.pax])
        nr_parcels = veh_obj.get_nr_parcels_without_currently_boarding()
        self.pax_info = {}
        for rq in veh_obj.pax:
            rid = key_translator.get(rq.get_rid_struct(), rq.get_rid_struct())
            self.pax_info[rid] = [rq.pu_time]
        # for pstop in self.list_plan_stops[:stop_index + 1]:
        for i, pstop in enumerate(self.list_plan_stops[:stop_index + 1]):
            if c_pos != pstop.get_pos():
                _, tt, tdist = routing_engine.return_travel_costs_1to1(c_pos, pstop.get_pos())
                c_pos = pstop.get_pos()
                c_time += tt
                c_soc -= veh_obj.compute_soc_consumption(tdist)
            if c_pos == pstop.get_pos():
                last_c_time = c_time
                last_c_soc = c_soc

                earliest_time = pstop.get_earliest_start_time()
                if c_time < earliest_time:
                    c_time = earliest_time
                    # LOG.debug(f"c_time 3 {c_time}")
                # update pax and check max. passenger constraint
                nr_pax += pstop.get_change_nr_pax()
                nr_parcels += pstop.get_change_nr_parcels()
                for rid in pstop.get_list_boarding_rids():
                    if self.pax_info.get(rid):
                        continue
                    self.pax_info[rid] = [c_time]
                    c_pax[rid] = 1
                for rid in pstop.get_list_alighting_rids():
                    self.pax_info[rid].append(c_time)
                    try:
                        del c_pax[rid]
                    except KeyError:
                        LOG.warning(f"update_tt_and_check_plan(): try to remove a rid that is not on board!")
                        
                # set departure time
                c_time = pstop.get_departure_time(c_time)
                pstop.set_planned_arrival_and_departure_time(last_c_time, c_time)
                # set charge
                if pstop.get_charging_power() > 0:  # TODO # is charging now in waiting included as planned here?
                    c_soc += veh_obj.compute_soc_charging(pstop.get_charging_power(), c_time - last_c_time)
                    c_soc = max(c_soc, 1.0)
                pstop.set_planned_arrival_and_departure_soc(last_c_soc, c_soc)
                    
        return {"stop_index": stop_index, "c_pos": c_pos, "c_soc": c_soc, "c_time": c_time, "c_pax": c_pax,
                "pax_info": self.pax_info.copy(), "c_nr_pax": nr_pax, "c_nr_parcels" : nr_parcels}

    def update_tt_and_check_plan(self, veh_obj : SimulationVehicle, sim_time : float, routing_engine : NetworkBase, init_plan_state : dict=None, keep_feasible : bool=False):
        """This method updates the planning properties of all PlanStops of the Plan according to the new vehicle
        position and checks if it is still feasible.

        :param veh_obj: vehicle object to which plan is applied
        :param sim_time: current simulation time
        :param routing_engine: reference to routing engine
        :param init_plan_state: {} requires "stop_index" "c_index", "c_pos", "c_soc", "c_time", "c_pax" and "pax_info"
        :param keep_feasible: useful flag to keep assigned VehiclePlans for simulations with dynamic travel times
        :return: is_feasible returns True if all
        """
        # TODO # think about update of duration of VehicleChargeLegs
        # LOG.debug(f"update tt an check plan {veh_obj} pax {veh_obj.pax} | at {sim_time} | pax info {self.pax_info}")
        is_feasible = True
        if len(self.list_plan_stops) == 0:
            self.pax_info = {}
            return is_feasible
        infeasible_index = -1  # lock all plan stops until last infeasible stop if vehplan is forced to stay feasible
        if init_plan_state is not None:
            start_stop_index = init_plan_state["stop_index"] + 1
            c_pos = init_plan_state["c_pos"]
            c_soc = init_plan_state["c_soc"]
            c_time = init_plan_state["c_time"]
            c_pax = init_plan_state["c_pax"].copy()
            c_nr_pax = init_plan_state["c_nr_pax"]
            c_nr_parcels = init_plan_state["c_nr_parcels"]
            self.pax_info = {}
            for k, v in init_plan_state["pax_info"].items():
                self.pax_info[k] = v.copy()
            # LOG.debug(f"init plan state available | c_pos {c_pos} c_pax {c_pax} pax info {self.pax_info}")
        else:
            key_translator = {sub_rid[0]: sub_rid for sub_rid in self.pax_info.keys() if type(sub_rid) == tuple}
            self.pax_info = {}
            start_stop_index = 0
            c_pos = veh_obj.pos
            c_soc = veh_obj.soc
            c_time = sim_time
            if self.list_plan_stops[0].is_locked():  # set time at start_time of boarding process
                boarding_started = self.list_plan_stops[0].get_started_at()
                if boarding_started is not None:
                    c_time = boarding_started
            c_pax = {key_translator.get(rq.get_rid_struct(), rq.get_rid_struct()): 1 for rq in veh_obj.pax}
            c_nr_pax = veh_obj.get_nr_pax_without_currently_boarding()  # sum([rq.nr_pax for rq in veh_obj.pax])
            c_nr_parcels = veh_obj.get_nr_parcels_without_currently_boarding()
            for rq in veh_obj.pax:
                # LOG.debug(f"add pax info {rq.get_rid_struct()} : {rq.pu_time}")
                rid = key_translator.get(rq.get_rid_struct(), rq.get_rid_struct())
                self.pax_info[rid] = [rq.pu_time]
            #LOG.verbose("init pax {} | {} | {}".format(c_pax, veh_obj.pax, self.pax_info))
        # LOG.debug(f"c_time 1 {c_time}")
        for i in range(start_stop_index, len(self.list_plan_stops)):
            pstop = self.list_plan_stops[i]
        #for i, pstop in enumerate(self.list_plan_stops[start_stop_index:], start=start_stop_index):
            pstop_pos = pstop.get_pos()
            if c_pos != pstop_pos:
                if not is_feasible and not keep_feasible:
                    # LOG.debug(f" -> break because infeasible | is feasible {is_feasible} keep_feasible {keep_feasible}")
                    break
                _, tt, tdist = routing_engine.return_travel_costs_1to1(c_pos, pstop_pos)
                c_pos = pstop_pos
                c_time += tt
                # LOG.debug(f"c_time 2 {c_time}")

                c_soc -= veh_obj.compute_soc_consumption(tdist)
                if c_soc < 0:
                    is_feasible = False
                    infeasible_index = i
                    # LOG.debug(" -> charging wrong")

            if c_pos == pstop_pos:

                last_c_time = c_time
                last_c_soc = c_soc

                earliest_time = pstop.get_earliest_start_time()
                if c_time < earliest_time:
                    c_time = earliest_time
                    # LOG.debug(f"c_time 3 {c_time}")
                # update pax and check max. passenger constraint
                c_nr_pax += pstop.get_change_nr_pax()
                c_nr_parcels += pstop.get_change_nr_parcels()
                #LOG.debug(f"change nr pax {pstop.change_nr_pax}")
                for rid in pstop.get_list_boarding_rids():
                    if i == 0 and self.pax_info.get(rid):
                        continue
                    self.pax_info[rid] = [c_time]
                    c_pax[rid] = 1
                for rid in pstop.get_list_alighting_rids():
                    self.pax_info[rid].append(c_time)
                    try:
                        del c_pax[rid]
                    except KeyError:
                        LOG.warning(f"update_tt_and_check_plan(): try to remove a rid that is not on board!")
                        LOG.warning(f"{self}")
                        is_feasible = False
                        infeasible_index = i
                        raise EnvironmentError
                # LOG.debug("pax info {}".format(self.pax_info))
                latest_time = pstop.get_latest_start_time(self.pax_info)
                if c_time > latest_time:
                    is_feasible = False
                    infeasible_index = i
                    # LOG.debug(f" -> arrival after latest {c_time} > {latest_time}")
                #LOG.debug(f"-> c nr {c_nr_pax} | cap {veh_obj.max_pax}")
                if c_nr_pax > veh_obj.max_pax or c_nr_parcels > veh_obj.max_parcels:
                    # LOG.debug(" -> capacity wrong")
                    is_feasible = False
                    infeasible_index = i

                c_time = pstop.get_departure_time(c_time)
                pstop.set_planned_arrival_and_departure_time(last_c_time, c_time)

                if pstop.get_charging_power() > 0:  # TODO # is charging now in waiting included as planned here?
                    c_soc += veh_obj.compute_soc_charging(pstop.get_charging_power(), c_time - last_c_time)
                    c_soc = max(c_soc, 1.0)
                pstop.set_planned_arrival_and_departure_soc(last_c_soc, c_soc)
                    
        if keep_feasible and not is_feasible:
            for i, p_stop in enumerate(self.list_plan_stops):
                if i > infeasible_index:
                    break
                # LOG.debug("LOCK because infeasible {}".format(i))
                p_stop.set_infeasible_locked(True)
        # LOG.debug(f"is feasible {is_feasible} | pax info {self.pax_info}")
        # LOG.debug("update plan and check tt {}".format(self))
        return is_feasible

    def get_dedicated_rid_list(self) -> list:
        """ returns a list of request-ids whicht are part of this vehicle plan
        :return: list of rid
        """
        return list(self.pax_info.keys())

    def update_prq_hard_constraints(self, veh_obj : SimulationVehicle, sim_time : float, routing_engine : NetworkBase, prq : PlanRequest, new_lpt : float, new_ept : float=None,
                                    keep_feasible : bool=False):
        """Adapts the earliest_pickup_time_dict and latest_pickup_time_dict of the pick-up PlanStop of a request.

        :param veh_obj: simulation vehicle
        :param sim_time: current simulation time
        :param routing_engine: routing engine
        :param prq: PlanRequest
        :param new_lpt: new latest pick-up time constraint
        :param new_ept: new earliest pick-up time constraint, not set if None
        :param keep_feasible: optional argument to add as input in update_tt_and_check_plan
        :return: feasibility of plan
        :rtype: bool
        """
        for ps in self.list_plan_stops:
            if prq.get_rid_struct() in ps.get_list_boarding_rids():
                ps.update_rid_boarding_time_constraints(new_latest_pickup_time=new_lpt, new_earliest_pickup_time=new_ept)
        return self.update_tt_and_check_plan(veh_obj, sim_time, routing_engine, keep_feasible=keep_feasible)

    def copy_and_remove_empty_planstops(self, veh_obj : SimulationVehicle, sim_time : float, routing_engine : NetworkBase):
        """ this function removes all plan stops from the vehicle plan that are empty
        i.e. are not locked and no pick-up/drop-offs are performes
        :param veh_obj: vehicle object
        :param sim_time: simulation time
        :param routing_engine: routing engine
        :return: vehicle plan without empty planstops
        :rtype: vehicleplan
        """
        new_plan = self.copy()
        tmp = []
        rm = False
        for ps in new_plan.list_plan_stops:
            if not ps.is_empty() or ps.is_locked() or ps.is_locked_end():
                tmp.append(ps)
            else:
                rm = True
        if rm:
            new_plan.list_plan_stops = tmp
            new_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
        return new_plan