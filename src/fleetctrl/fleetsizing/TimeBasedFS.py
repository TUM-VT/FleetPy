from src.fleetctrl.fleetsizing.DynamicFleetSizingBase import DynamicFleetSizingBase


class TimeBasedFS(DynamicFleetSizingBase):
    def __init__(self, fleetctrl, operator_attributes):
        """Initialization of repositioning class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes)
        # TODO # TimeBasedFS.__init__()
        # load elastic fleet size file and insert them to time trigger activate and deactivate
        # beware of initialization of active/inactive vehicles! -> check old add_init()

    def check_and_change_fleet_size(self, sim_time):
        """This method checks whether fleet vehicles should be activated or deactivated.

        :param sim_time: current simulation time
        :return: net change in fleet size
        """
        # TODO # TimeBasedFS.check_and_change_fleet_size()
        # nothing has to be done here -> charging_management takes care
        pass
