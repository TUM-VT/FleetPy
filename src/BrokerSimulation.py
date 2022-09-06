# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import numpy as np

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
from src.FleetSimulationBase import FleetSimulationBase
from src.ImmediateDecisionsSimulation import ImmediateDecisionsSimulation

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------


# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----

INPUT_PARAMETERS_BrokerDecisionSimulation = {
    "doc" :     """
    this fleetsimulation is used in the publication
    Competition and Cooperation of Autonomous Ridepooling Services: Game-Based Simulation of a Broker Concept; Engelhardt, Malcolm, Dandl, Bogenberger (2022)
    in this case offer from 2 different fleetcontrols are collected and a broker decides which of these
    offers will be forwarded to the customer
    the customer always chooses this offer
    the BrokerDescisionFleetCtrl has to be used for operators to create all needed offer attributes for the broker deciscion
    """,
    "inherit" : "FleetSimulationBase",
    "input_parameters_mandatory": [
    ],
    "input_parameters_optional": [
    ],
    "mandatory_modules": [
    ], 
    "optional_modules": []
}

class BrokerDecisionSimulation(FleetSimulationBase):
    """
    this fleetsimulation is used in the publication
    Competition and Cooperation of Autonomous Ridepooling Services: Game-Based Simulation of a Broker Concept; Engelhardt, Malcolm, Dandl, Bogenberger (2022)
    in this case offer from 2 different fleetcontrols are collected and a broker decides which of these
    offers will be forwarded to the customer
    the customer always chooses this offer
    """
    def __init__(self, scenario_parameters):
        super().__init__(scenario_parameters)

    def add_init(self, scenario_parameters):
        """
        Simulation specific additional init.
        :param scenario_parameters: row of pandas data-frame; entries are saved as x["key"]
        """
        # Set up parallelization for operators
        if scenario_parameters[G_SLAVE_CPU] > 1:
            # test for same opt algorithms
            opt_alg = None
            same = True
            for op_atts in self.list_op_dicts:
                if opt_alg is None:
                    opt_alg = op_atts.get(G_RA_RP_BATCH_OPT, "AlonsoMora")
                else:
                    if opt_alg != op_atts.get(G_RA_RP_BATCH_OPT, "AlonsoMora"):
                        same = False
            if same:
                N_cores = scenario_parameters[G_SLAVE_CPU]
                dir_names = self.dir_names
                from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import load_parallelization_manager
                pm_class = load_parallelization_manager(opt_alg)
                Parallelisation_Manager = pm_class(N_cores, scenario_parameters, dir_names)
                for op_id, op in enumerate(self.operators):
                    try:
                        op.register_parallelization_manager(Parallelisation_Manager)
                    except:
                        LOG.warning("couldnt register parallelization for op {}".format(op_id))
            else:
                LOG.warning("different opt algorithms between operators -> not implemented to share parallelization managers -> separeted processes will be started!")
        super().add_init(scenario_parameters)

    @staticmethod
    def _is_nonempty_offer(offer):  # TODO # still needed?
        flag_count = 0
        if G_OFFER_WILLING_FLAG in offer:
            flag_count += 1
        if G_OFFER_PREFERRED_OP in offer:
            flag_count += 1
        return len(offer) > flag_count

    def step(self, sim_time):
        """
        This method determines the simulation flow in a time step.
            1) update fleets and network
            2) get new travelers, add to undecided request
            3) sequential processes for each undecided request: request -> offer -> user(broker)-decision
                1) collect offers from all operators for each rid
                2) choose the best offer based on fleetctrl attribute added to offer (additional driven km)
                3) user recieves this offer (if there is one)
                4) user decides for this single offer
            4) periodically for waiting requests: run decision process -> possibly leave system (cancellation)
            5) periodically operator: call ride pooling optimization, repositioning, charging management

        :param sim_time: new simulation time
        :return: None
        """
        # 1)
        self.update_sim_state_fleets(sim_time - self.time_step, sim_time)
        new_travel_times = self.routing_engine.update_network(sim_time)
        if new_travel_times:
            for op_id in range(self.n_op):
                self.operators[op_id].inform_network_travel_time_update(sim_time)
        # 2)
        list_undecided_travelers = list(self.demand.get_undecided_travelers(sim_time))
        last_time = sim_time - self.time_step
        if last_time < self.start_time:
            last_time = None
        list_new_traveler_rid_obj = self.demand.get_new_travelers(sim_time, since=last_time)
        # 3)
        for rid, rq_obj in list_undecided_travelers + list_new_traveler_rid_obj:
            # send all requests to all operators
            for op_id in range(self.n_op):
                LOG.debug(f"Request {rid}: To operator {op_id} ...")
                self.operators[op_id].user_request(rq_obj, sim_time)
            # get offers and choose best option (broker criteria)
            operator_offers = {}
            for op_id in range(self.n_op):
                amod_offer = self.operators[op_id].get_current_offer(rid)
                LOG.debug(f"amod offer {amod_offer}")
                if amod_offer is not None:
                    operator_offers[op_id] = amod_offer
            operator_offers = self._broker_decision(rq_obj, operator_offers)
            for op_id, amod_offer in operator_offers.items():
                if amod_offer is not None:
                    rq_obj.receive_offer(op_id, amod_offer, sim_time)
            # customer decisions
            self._rid_chooses_offer(rid, rq_obj, sim_time)
        # # 4)
        self._check_waiting_request_cancellations(sim_time)
        # 5)
        for op in self.operators:
            op.time_trigger(sim_time)
        # record at the end of each time step
        self.record_stats()

    def _broker_decision(self, rq_obj, operator_offer_dict):
        """ this method is used to simulate the broker decision
        in this version the broker takes the offer which creates fewest addition km
        to have all information in the output, this function sets the flag TODO to mark the broker chosen offer
        which will be chosen by the customer 
        :param rq_obj: traveler obj
        :param operator_offer_dict: dict op_id -> corresponding offer obj for rq_obj
        :return operator_offer_dict: dict op_id -> corresponding offer with added flag for chosen_by_broker (True/False)"""
        best_op = -1
        best_add_vmt = float("inf")
        for op_id, op_offer in operator_offer_dict.items():
            if not op_offer.service_declined():
                if op_offer[G_OFFER_ADD_VMT] < best_add_vmt:
                    best_op = op_id
                    best_add_vmt = op_offer[G_OFFER_ADD_VMT]
                elif op_offer[G_OFFER_ADD_VMT] == best_add_vmt:
                    r = np.random.randint(2)
                    if r == 0:
                        best_op = op_id
                        best_add_vmt = op_offer[G_OFFER_ADD_VMT]
        for op_id, op_offer in operator_offer_dict.items():
            if op_id == best_op:
                op_offer.extend_offer({G_OFFER_BROKER_FLAG : True})
            else:
                op_offer.extend_offer({G_OFFER_BROKER_FLAG : False})
            operator_offer_dict[op_id] = op_offer
        return operator_offer_dict

    def evaluate(self):
        """Runs standard and simulation environment specific evaluations over simulation results."""
        output_dir = self.dir_names[G_DIR_OUTPUT]
        # standard evaluation
        from src.evaluation.standard import standard_evaluation
        standard_evaluation(output_dir)

# ---------------------------------------------------------------------------------------------------------------

INPUT_PARAMETERS_UserDecisionSimulation = {
    "doc" :     """
    this fleetsimulation is used 
    Competition and Cooperation of Autonomous Ridepooling Services: Game-Based Simulation of a Broker Concept; Engelhardt, Malcolm, Dandl, Bogenberger (2022)
    in this case offer from 2 different fleetcontrols are directly forwarded to the customer and the customer
    decides on an offer based on a simple choice model
    all needed implementations are already implemented in the ImmediateDecisionSimulation class
    """,
    "inherit" : "ImmediateDecisionsSimulation",
    "input_parameters_mandatory": [
    ],
    "input_parameters_optional": [
    ],
    "mandatory_modules": [
    ], 
    "optional_modules": []
}

class UserDecisionSimulation(ImmediateDecisionsSimulation):
    """
    this fleetsimulation is used 
    Competition and Cooperation of Autonomous Ridepooling Services: Game-Based Simulation of a Broker Concept; Engelhardt, Malcolm, Dandl, Bogenberger (2022)
    in this case offer from 2 different fleetcontrols are directly forwarded to the customer and the customer
    decides on an offer based on a simple choice model
    all needed implementations are already implemented in the ImmediateDecisionSimulation class
    """

# ---------------------------------------------------------------------------------------------------------------

INPUT_PARAMETERS_PreferredOperatorSimulation = {
    "doc" :     """
    this fleetsimulation is used 
    Competition and Cooperation of Autonomous Ridepooling Services: Game-Based Simulation of a Broker Concept; Engelhardt, Malcolm, Dandl, Bogenberger (2022)
    in this scenario a "preferred_operator" attribute is set for all customers. The customers ("PreferredOperatorRequest" - class) only accepts offers from this operator
    -> serves as base scenario for independent operators, but can also track if other operator was able to create an offer
    """,
    "inherit" : "ImmediateDecisionsSimulation",
    "input_parameters_mandatory": [
    ],
    "input_parameters_optional": [
        G_MULTIOP_PREF_OP_RSEED
    ],
    "mandatory_modules": [
    ], 
    "optional_modules": []
}
    
class PreferredOperatorSimulation(ImmediateDecisionsSimulation):
    """
    this fleetsimulation is used 
    Competition and Cooperation of Autonomous Ridepooling Services: Game-Based Simulation of a Broker Concept; Engelhardt, Malcolm, Dandl, Bogenberger (2022)
    in this scenario a "preferred_operator" attribute is set for all customers. The customers ("PreferredOperatorRequest" - class) only accepts offers from this operator
    -> serves as base scenario for independent operators, but can also track if other operator was able to create an offer
    """
    def add_init(self, scenario_parameters):
        """ set preferred operator attribut in requests """
        super().add_init(scenario_parameters)
        if G_MULTIOP_PREF_OP_RSEED in scenario_parameters:
            np.random.seed(scenario_parameters[G_MULTIOP_PREF_OP_RSEED])
        p = scenario_parameters.get(G_MULTIOP_PREF_OP_PROB, [1/self.n_op for o in range(self.n_op)])
        LOG.debug(f"Assigning preferred operator to each request with operator shares {p}")
        for rid, req in self.demand:
            if getattr(req, "preferred_operator", None) is None:
                setattr(req, "preferred_operator", np.random.choice(self.n_op, p=p))
