from __future__ import annotations
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging

# additional module imports (> requirements)
# ------------------------------------------
from abc import abstractmethod, ABCMeta
from typing import Optional, TYPE_CHECKING

# src imports
# -----------
if TYPE_CHECKING:
    from src.infra.ChargingInfrastructure import ChargingStation
    from src.simulation.Vehicles import SimulationVehicle

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)


class StationaryProcess(metaclass=ABCMeta):
    """ A StationaryProcess provides basic interfaces for the tasks planned at the PlanStop """

    @abstractmethod
    def start_task(self, sim_time) -> bool:
        """ Return True if the task started successfully

        :param sim_time:    current simulation time
        """
        pass

    @abstractmethod
    def end_task(self, sim_time):
        """ Ends the current task """
        pass

    def remaining_time_to_start(self, sim_time) -> Optional[int]:
        """ Returns the time remaining to start the task (i.e. delay). The returned time could be negative. It returns
         None if the values cannot be determined. The default value returns 0 """
        return 0

    @abstractmethod
    def remaining_duration_to_finish(self, sim_time) -> Optional[int]:
        """ Returns the estimated duration for task completion. Returns None if the process did not start """
        pass

    @abstractmethod
    def update_state(self, delta_time):
        """ Updates the state of the stationary process """
        pass


class ChargingProcess(StationaryProcess):

    def __init__(self, booking_id:str, veh_obj: SimulationVehicle, station: ChargingStation, start_time, end_time=None):
        """ The class represents the stationary process for charging. This class is also used for making a schedule
        for each socket of the charging station

        :param booking_id           Unique id for the current booking
        :param veh_obj:             vehicle object of the current charging process
        :param station:             associated charging station
        :param start_time:          estimated earliest start time of the charging process
        :param end_time:            estimated end time of the charging process
        """

        self.id: str = booking_id
        self.veh: SimulationVehicle = veh_obj
        self.start_time = start_time
        self.end_time = end_time
        # TODO: Remove the "station" attribute to avoid direct access of stationary process to charging station
        self.station: ChargingStation = station
        self.socket_id: int = int(booking_id.split("_")[1])
        self.locked: bool = False
        self._task_started = False
        
    def __str__(self) -> str:
        return f"charging process: id {self.id} vid {self.veh.vid} station {self.station.id} socked {self.socket_id} start time {self.start_time} end time {self.end_time} started {self._task_started}"

    def start_task(self, sim_time):
        """ Connects the vehicle to the socket and returns true for successful connection """
        self._task_started = self.station.start_charging_process(sim_time, self)
        assert self._task_started is True, "failed to connect to the socket"
        return self._task_started

    def end_task(self, sim_time):
        """ Ends the current task by disconnecting the vehicle """
        assert self._task_started is True
        self.station.end_charging_process(sim_time, self)

    def get_scheduled_start_end_times(self):
        return self.start_time, self.end_time

    def remaining_time_to_start(self, sim_time):
        return self.start_time - sim_time

    def remaining_duration_to_finish(self, sim_time):
        if self._task_started is None:
            return None
        else:
            return self.station.remaining_charging_time(sim_time, self.veh.vid)

    def update_state(self, delta_time):
        assert self.veh.status is VRL_STATES.CHARGING
        self.station.update_charging_state(self, delta_time)