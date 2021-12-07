from src.fleetctrl.fleetsizing.DynamicFleetSizingBase import DynamicFleetSizingBase


class UtilizationBasedFS(DynamicFleetSizingBase):
    def __init__(self, fleetctrl, operator_attributes):
        """Initialization of repositioning class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes)
        # TODO # UtilizationBasedFS.__init__()
        # -> check out Romans new approach for MOIA -> different branch

    def check_and_change_fleet_size(self, sim_time):
        """This method checks whether fleet vehicles should be activated or deactivated.

        :param sim_time: current simulation time
        :return: net change in fleet size
        """
        # TODO # UtilizationBasedFS.check_and_change_fleet_size()
        # -> check out Romans new approach for MOIA -> different branch
        pass
