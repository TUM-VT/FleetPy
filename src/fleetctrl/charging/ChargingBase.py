from __future__ import annotations
from abc import abstractmethod, ABC
import logging
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

from src.misc.globals import G_OP_CH_N_OFFER_P_ST_QUERY, G_OP_CH_N_STATION_QUERY
if TYPE_CHECKING:
    from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure, PublicChargingInfrastructureOperator
    from src.fleetctrl.FleetControlBase import FleetControlBase

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_ChargingBase = {
    "doc" :  """This sub-modules deals with algorithms and strategies to sent vehicles to charging stations.. """,
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class ChargingBase(ABC):
    def __init__(self, fleetctrl: FleetControlBase, operator_attributes: dict, solver="Gurobi"):
        """This sub-modules deals with algorithms and strategies to sent vehicles to charging stations.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        self.fleetctrl = fleetctrl
        self.op_charge_depot_infra = fleetctrl.op_charge_depot_infra
        self.list_pub_charging_infra = fleetctrl.list_pub_charging_infra
        self.all_charging_infra: List[PublicChargingInfrastructureOperator] = []
        if self.op_charge_depot_infra is not None:
            self.all_charging_infra.append(self.op_charge_depot_infra)
        self.all_charging_infra += self.list_pub_charging_infra[:]
        self.routing_engine = fleetctrl.routing_engine
        self.solver_key = solver
        self.n_stations_to_query = operator_attributes.get(G_OP_CH_N_STATION_QUERY, 1)
        self.n_offers_p_station = operator_attributes.get(G_OP_CH_N_OFFER_P_ST_QUERY, 1)
        self.target_soc = 1.0   #TODO
        # children classes:
        # - check of additionally required attributes from operator_attributes
        # - save these as class attributes

    @abstractmethod
    def time_triggered_charging_processes(self, sim_time):
        """This method can be used to apply a charging strategy and additionally, charge vehicles in depots if there are
        free slots.

        :param sim_time: current simulation time
        :return: None
        """
        pass
