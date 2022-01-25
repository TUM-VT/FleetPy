from src.misc.globals import *
import typing as tp
if tp.TYPE_CHECKING is True:
    from src.simulation.StationaryProcess import StationaryProcess


# -------------------------------------------------------------------------------------------------------------------- #
# Simulation Vehicle Route Leg class
# ----------------------------------
class VehicleRouteLeg:
    def __init__(self, status, destination_pos, rq_dict, power=0.0, duration=None, route=[], locked=False,
                 earliest_start_time=-1000, stationary_process=None):
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
        :param stationary_process:  The stationary process do be carried out at the stop
        """
        self.status = status
        self.rq_dict = rq_dict
        self.destination_pos = destination_pos
        self.power = power
        self.earliest_start_time = earliest_start_time
        self.duration = duration
        self.route = route
        self.locked = locked
        if duration is not None:
            try:
                x = int(duration)
            except:
                raise TypeError("wrong type for duration: {}".format(duration))
        self.stationary_process: StationaryProcess = stationary_process

    def __eq__(self, other):
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


class VehicleChargeLeg(VehicleRouteLeg):
    def __init__(self, vcd_id, charging_unit, power, earliest_start_time, desired_soc, latest_end, locked=False):
        """This class is used for stops which focus on charging. These are characterized by a desired SOC goal or a
        certain duration which they should charge for. These VRL are of status 2, in contrast to status 3
        board and charging VRL, which are characterized by the boarding process and allow charging as a "byproduct"
        during the short stopping process.

        :param charging_unit: reference to charging unit
        :param power: desired charging power (<= charging_unit.max_power)
        :param earliest_start_time: time at which charging can begin (charging unit may be occupied before that)
        :param desired_soc: target SOC for the VCL
        :param latest_end: time at which charging has to end latest (charging unit may be occupied after that)
        :param locked: this parameter determines whether this VCL can be cut short or started later
        """
        status = VRL_STATES.CHARGING
        route = []
        rq_dict = {}
        duration = None
        self.vcd_id = vcd_id
        self.started_at = None
        self.charging_unit = charging_unit
        self.desired_soc = desired_soc
        self.latest_end = latest_end
        super().__init__(status, charging_unit.get_pos(), rq_dict, power, duration, route, locked, earliest_start_time)

    def __eq__(self, other):
        """Compares a VCL with another VRL/VCL.

        :param other: other VRL, VCL
        :return: True if VCL is equal
        """
        if type(other) == VehicleChargeLeg and self.vcd_id == other.vcd_id:
            return True
        else:
            return False

    def get_vcl_id(self):
        return self.vcd_id

    def is_active(self):
        if self.started_at is None:
            return False
        else:
            return True

    def set_duration_at_start(self, sim_time, veh_obj):
        """This method is called when the vehicle starts to charge. It is used to adapt the duration of the charging
        process in case a vehicle arrives late or with a different SOC. In these cases, shorter/longer charging
        durations to reach desired_soc or max_duration are considered as long as the charging unit schedule allows it.
        If the charging schedule does constrains an extension, the maximum available time is used.

        :param sim_time: simulation time
        :param veh_obj: contains the battery and SOC information
        :return: None; self.duration is set
        """
        self.started_at = sim_time
        desired_soc_duration = veh_obj.get_charging_duration(self.power, veh_obj.soc, self.desired_soc)
        if self.latest_end and self.started_at + desired_soc_duration > self.latest_end:
                self.duration = self.latest_end - self.started_at
                if self.duration < 0:
                    self.duration = 1
        else:
            self.duration = desired_soc_duration

    def get_current_schedule_info(self):
        """This method returns the earliest start time and the latest end time for a charging leg. The charging unit
        will be blocked in between these times for this task regardless of the vehicle actually being there.

        :return: earliest_time, latest_time
        """
        return self.earliest_start_time, self.latest_end

    def adapt_schedule(self, new_earliest_start_time=None, new_latest_end=None):
        """This method adapts the earliest start time and the latest end time of a VCL. The duration of an already
        running VCL is also cut short if necessary.

        :param new_earliest_start_time: can be used to let a task start later than originally planned
        :param new_latest_end: can be used to cut a charging task short, e.g. to start another charging process
        :return: None
        """
        if new_earliest_start_time:
            self.earliest_start_time = new_earliest_start_time
        if new_latest_end:
            self.latest_end = new_latest_end
            if self.started_at and self.started_at + self.duration > self.latest_end:
                self.duration = self.latest_end - self.started_at