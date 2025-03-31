# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
import typing as tp
from abc import abstractmethod, ABCMeta

# additional module imports (> requirements)
# ------------------------------------------


# src imports
# -----------
from src.broker.BrokerBase import BrokerBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.pt.PTControlBase import PTControlBase
    from src.demand.TravelerModels import RequestBase

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_PTBroker = {
    "doc" : "this class is the base class representing an pt broker platform",
    "inherit" : BrokerBase,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PTBroker(BrokerBase):
    def __init__(self, n_amod_op: int, amod_operators: tp.List['FleetControlBase'], pt_operator: 'PTControlBase'):
        # TODO: Test the intermodal request id and original request id
        super().__init__(n_amod_op, amod_operators)
        self.pt_operator = pt_operator
    
    def inform_user_decision(self, rid, rq_obj, sim_time, chosen_operator):
        # TODO: Implement this method with PT specific logic
        # TODO: The intermodal offer logic should be implemented before the super call
        super().inform_user_decision(rid, rq_obj, sim_time, chosen_operator)
        # TODO: For intermodal requests, make a return and let the simulation call the addRequest method to store the intermodal request
        return


    def inform_request(self, rid, rq_obj, sim_time):
        """This method informs the broker that a new request has been made.
        This information is forwarded to the operators.
        """
        # Forward the request to the operators
        super().inform_request(rid, rq_obj, sim_time)
        # TODO: Determine the direct PT journey
        # TODO: Determine and forward the intermodal journey to the operators

        # TODO: The operator should be able to determine the original and the intermodal journey by request id

    def collect_offers(self, rid, rq_obj, sim_time):
        """This method collects the offers from the operators.
        """
        # TODO: The rid here should be the intermodal request id and original request id
        # TODO: The format of the offers should be defined --> additional information for the PT journey
        super().collect_offers(rid, rq_obj, sim_time)


    def backup(self):
                    # if chosen_operator == G_MC_DEC_PV:
            #     # TODO # self.routing_engine.assign_route_to_network(rq_obj, sim_time)
            #     # TODO # computation of route only when necessary
            #     self.routing_engine.assign_route_to_network(rq_obj, sim_time)
            #     # TODO # check if following method is necessary
            #     self.demand.user_chooses_PV(rid, sim_time)
            # elif chosen_operator == G_MC_DEC_PT:
            #     pt_offer = rq_obj.return_offer(G_MC_DEC_PT)
            #     pt_start_time = sim_time + pt_offer.get(G_OFFER_ACCESS_W, 0) + pt_offer.get(G_OFFER_WAIT, 0)
            #     pt_end_time = pt_start_time + pt_offer.get(G_OFFER_DRIVE, 0)
            #     self.pt.assign_to_pt_network(pt_start_time, pt_end_time)
            #     # TODO # check if following method is necessary
        pass










        