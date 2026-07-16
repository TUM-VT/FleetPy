# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import typing as tp
import pandas as pd

# additional module imports (> requirements)
# ------------------------------------------
from BrokerBasic import BrokerBasic

# src imports
# -----------
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.demand.TravelerModels import RequestBase
    from src.simulation.Legs import VehicleRouteLeg
    from src.infra.Zoning import ZoneSystem

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_BrokerBasic = {
    "doc" : "this class is the basic broker class, it only forwards the requests to the amod operators",
    "inherit" : BrokerBasic,
    "input_parameters_mandatory": [G_ZONE_SYSTEM_NAME, G_BR_STAT_P_MAT],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----

class StaticPriceRegulatingBroker(BrokerBasic):
    """This broker class implements a static price regulating mechanism, which is based on a predefined
    OD subsidy matrix."""
    def __init__(self, n_amod_op: int, amod_operators: tp.List['FleetControlBase'], zone_system: ZoneSystem,
                 od_subsidy_csv_f: str):
        """
        The general attributes for the broker are initialized.

        Args:
            amod_operators (tp.List['FleetControlBase']): list of AMoD operators
            zone_system (ZoneSystem): zone system, which is used to map the origin and destination
                                        of the requests to the corresponding zones
            od_subsidy_csv_f (str): csv file with columns "origin_zone", "destination_zone", "subsidy",
                                        which defines the subsidy for each OD pair
        """
        super().__init__(n_amod_op, amod_operators)
        self.zone_system: ZoneSystem = zone_system
        self.subsidy_df: pd.DataFrame = pd.read_csv(od_subsidy_csv_f)
        self.subsidy_df.set_index("[origin_zone, destination_zone]", inplace=True)
        self.rq2od: tp.Dict[int, tp.Tuple[int, int]] = {}  # maps request id to origin and destination zone

    def inform_request(self, rid: int, rq_obj: RequestBase, sim_time: int):
        super().inform_request(rid, rq_obj, sim_time)
        self.rq2od[rid] = (self.zone_system.get_zone_from_pos(rq_obj.o_pos),
                           self.zone_system.get_zone_from_pos(rq_obj.d_pos))


    def collect_offers(self, rid: int) -> tp.Dict[int, 'RequestBase']:
        """This method collects the offers from the amod operators.
        The return value is a list of tuples, where each tuple contains the operator id, the offer, and the simulation time.
        """
        rq_subsidy = self.subsidy_df.loc[self.rq2od[rid], "subsidy"]
        amod_offers = {}
        for op_id in range(self.n_amod_op):
            amod_offer = self.amod_operators[op_id].get_current_offer(rid)
            amod_offer.fare -= rq_subsidy
            LOG.debug(f"amod offer {amod_offer}")
            if amod_offer is not None:
                amod_offers[op_id] = amod_offer
        return amod_offers
