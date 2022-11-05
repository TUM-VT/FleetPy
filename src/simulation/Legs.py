from __future__ import annotations

from src.misc.globals import *
import typing as tp
if tp.TYPE_CHECKING is True:
    from src.simulation.StationaryProcess import StationaryProcess
    from src.demand.TravelerModels import RequestBase


# -------------------------------------------------------------------------------------------------------------------- #
# Simulation Vehicle Route Leg class
# ----------------------------------
class VehicleRouteLeg:
    def __init__(self, status:VRL_STATES, destination_pos:tuple, rq_dict:tp.Dict[tp.Any, RequestBase], power:float=0.0, duration:float=None, route:tp.List[int]=[], locked:bool=False,
                 earliest_start_time:float=-1000, earliest_end_time:float=-1000, stationary_process:StationaryProcess=None):
        """
        This class summarizes the minimal information for a a route leg. It only reflects a complete state
        with the information of the initial state at the start of the leg.

        :param status: numbers reflecting the state of the leg (see misc/globals.py for specification)
        :param destination_pos: end of driving leg
        :param rq_dict: +/-1 -> [rq] ('+' for boarding, '-' for alighting)
        :param duration: required for non-driving legs
        :param route: list of nodes
        :param locked: locked tasks cannot be changed anymore
        :param earliest_start_time: for e.g. boarding processes
        :param earliest_end_time: earliest time for ending the process
        :param stationary_process:  The stationary process do be carried out at the stop
        """
        self.status = status
        self.rq_dict = rq_dict
        self.destination_pos = destination_pos
        self.power = power
        self.earliest_start_time = earliest_start_time
        self.earliest_end_time = earliest_end_time
        self.duration = duration
        self.route = route
        self.locked = locked
        if duration is not None:
            try:
                x = int(duration)
            except:
                raise TypeError("wrong type for duration: {}".format(duration))
        self.stationary_process: StationaryProcess = stationary_process

    def __eq__(self, other)->bool:
        """Comparison of two VehicleRouteLegs.

        :param other: other vehicle route leg
        :type other: VehicleRouteLeg
        :return: True if equal, False else
        :rtype: bool
        """
        if self.status == other.status and set(self.rq_dict.get(1,[])) == set(other.rq_dict.get(1,[]))\
                and set(self.rq_dict.get(-1,[])) == set(other.rq_dict.get(-1,[]))\
                and self.destination_pos == other.destination_pos and self.locked == other.locked:
            return True
        else:
            return False

    def __str__(self):
        return "VRL: status {} dest {} duration {} earliest start time {} locked {} bd: {}".format(self.status, self.destination_pos, self.duration, self.earliest_start_time, self.locked, {key: [rq.get_rid_struct() for rq in val] for key, val in self.rq_dict.items()})

    def additional_str_infos(self):
        return "{}".format(self.rq_dict)

    def update_lock_status(self, lock=True):
        self.locked = lock

