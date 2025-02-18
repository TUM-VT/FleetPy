# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging

# additional module imports (> requirements)
# ------------------------------------------
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# src imports
# -----------
from src.misc.distributions import draw_from_distribution_dict
from src.misc.init_modules import load_request_module

# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
BUFFER_SIZE = 10
# -------------------------------------------------------------------------------------------------------------------- #


# functions
# ---------
def create_traveler(rq_row, rq_node_type_distr, zone_definition, routing_engine, simulation_time_step,
                    scenario_parameters):
    if None in rq_node_type_distr.keys():
        distribution = rq_node_type_distr[None]
    else:
        if not zone_definition:
            od_zones = None
        else:
            o_zone_index = zone_definition.get_zone_from_node(rq_row[G_RQ_ORIGIN])
            d_zone_index = zone_definition.get_zone_from_node(rq_row[G_RQ_DESTINATION])
            if o_zone_index is None or d_zone_index is None:
                od_zones = None
            else:
                od_zones = (o_zone_index, d_zone_index)
        distribution = rq_node_type_distr.get(od_zones)
    if not distribution:
        raise IOError(f"Could not find fitting request-type for traveler {rq_row}")
    if len(distribution.keys()) >= 1:
        TravelerClass = draw_from_distribution_dict(distribution)
    else:
        raise AssertionError(f"Empty distribution of Traveler classes for rq-row {rq_row}")
    return TravelerClass(rq_row, routing_engine, simulation_time_step, scenario_parameters)


# TODO # write more efficient create_traveler; just-in-time creation?
class Demand:
    def __init__(self, scenario_parameters, output_f, routing_engine=None, zone_system=None):
        self.scenario_parameters = scenario_parameters
        # prepare output
        self.output_f = output_f
        self.user_stat_buffer = []  # list of dictionaries
        # request data bases
        self.rq_db = {}  # rid > rq
        self.undecided_rq = {} # rid > rq
        self.waiting_rq = {} # rid > rq
        self.future_requests = {}
        # optional
        self.zone_definition = zone_system
        self.routing_engine = routing_engine
        # TODO # init user output parameters for correct treatment of output?

    def load_demand_file(self, start_time, end_time, rq_file_dir, rq_file_name, np_random_seed, rq_type=None,
                         rq_type_distr={}, rq_od_zone_distr={}, simulation_time_step=1):
        np.random.seed(int(1712 * np_random_seed))
        # classification of requests
        rq_node_type_distr = {}
        if rq_type:
            rq_class = load_request_module(rq_type)
            rq_node_type_distr[None] = {rq_class: 1.0}
        elif rq_type_distr:
            rq_node_type_distr[None] = {}
            for rq_type, share in rq_type_distr.items():
                rq_class = load_request_module(rq_type)
                rq_node_type_distr[None][rq_class] = share
        elif rq_od_zone_distr:
            if not self.zone_definition:
                raise IOError("Zones for different request classes are not defined!")
            for od_zone_tuple, rq_type_distr in rq_od_zone_distr.items():
                rq_node_type_distr[od_zone_tuple] = {}
                for rq_type, share in rq_type_distr.items():
                    rq_class = load_request_module(rq_type)
                    rq_node_type_distr[od_zone_tuple][rq_class] = share
        else:
            raise IOError("No valid traveler type found")
        # read input
        abs_req_f = os.path.join(rq_file_dir, rq_file_name)
        tmp_df = pd.read_csv(abs_req_f, dtype={"start": int, "end": int})
        number_rq_0 = tmp_df.shape[0]
        future_requests = tmp_df[(tmp_df[G_RQ_TIME] >= start_time) & (tmp_df[G_RQ_TIME] < end_time)]
        number_rq_1 = future_requests.shape[0]
        future_requests = future_requests[(future_requests[G_RQ_ORIGIN] != future_requests[G_RQ_DESTINATION])]
        number_rq = future_requests.shape[0]
        future_requests[G_RQ_TIME] = future_requests[G_RQ_TIME] - np.mod(future_requests[G_RQ_TIME],
                                                                         simulation_time_step)
        # define maximum decision time
        if G_RQ_LDT not in future_requests.columns:
            max_dec_time = self.scenario_parameters[G_AR_MAX_DEC_T]
            future_requests[G_RQ_LDT] = future_requests[G_RQ_TIME] + max_dec_time
        for rq_time, rq_time_df in future_requests.groupby(G_RQ_TIME):
            new_rq_dict = {}
            # TODO: note, iterrows() does not preserve dtype, so strange things happen if demand file contains mixed types. Possibly rethink how this works.
            for _, rq_row in rq_time_df.iterrows():
                rq_obj = create_traveler(rq_row, rq_node_type_distr, self.zone_definition, self.routing_engine,
                                         simulation_time_step, self.scenario_parameters)
                new_rq_dict[rq_obj.rid] = rq_obj
            if rq_time in self.future_requests:
                self.future_requests[rq_time].update(new_rq_dict)
            else:
                self.future_requests[rq_time] = new_rq_dict
        LOG.info(f"init(): {number_rq_0 - number_rq_1}/{number_rq_0}"
                 f" requests removed ({G_RQ_TIME} not in simulation time)")
        LOG.info(f"init(): {number_rq_1 - number_rq}/{number_rq_1}"
                 f" requests removed ({G_RQ_ORIGIN} == {G_RQ_DESTINATION})")
        # LOG.debug(f"self.future_requests = {self.future_requests}")

    def load_parcel_demand_file(self, start_time, end_time, parcel_rq_file_dir, parcel_rq_file_name, np_random_seed, parcel_rq_type=None,
                         parcel_rq_type_distr={}, parcel_rq_od_zone_distr={}, simulation_time_step=1):
        np.random.seed(int(1712 * np_random_seed))
        # classification of requests
        rq_node_type_distr = {}
        if parcel_rq_type:
            rq_class = load_request_module(parcel_rq_type)
            rq_node_type_distr[None] = {rq_class: 1.0}
        elif parcel_rq_type_distr:
            rq_node_type_distr[None] = {}
            for rq_type, share in parcel_rq_type_distr.items():
                rq_class = load_request_module(rq_type)
                rq_node_type_distr[None][rq_class] = share
        elif parcel_rq_od_zone_distr:
            if not self.zone_definition:
                raise IOError("Zones for different request classes are not defined!")
            for od_zone_tuple, rq_type_distr in parcel_rq_od_zone_distr.items():
                rq_node_type_distr[od_zone_tuple] = {}
                for rq_type, share in rq_type_distr.items():
                    rq_class = load_request_module(rq_type)
                    rq_node_type_distr[od_zone_tuple][rq_class] = share
        else:
            raise IOError("No valid traveler type found")
        # read input
        abs_req_f = os.path.join(parcel_rq_file_dir, parcel_rq_file_name)
        tmp_df = pd.read_csv(abs_req_f, dtype={"start": int, "end": int})
        number_rq_0 = tmp_df.shape[0]
        future_requests = tmp_df[(tmp_df[G_RQ_TIME] >= start_time) & (tmp_df[G_RQ_TIME] < end_time)]
        number_rq = future_requests.shape[0]
        future_requests[G_RQ_TIME] = future_requests[G_RQ_TIME] - np.mod(future_requests[G_RQ_TIME],
                                                                         simulation_time_step)
        # define maximum decision time
        if G_RQ_LDT not in future_requests.columns:
            max_dec_time = self.scenario_parameters[G_AR_MAX_DEC_T]
            future_requests[G_RQ_LDT] = future_requests[G_RQ_TIME] + max_dec_time
        for rq_time, rq_time_df in future_requests.groupby(G_RQ_TIME):
            new_rq_dict = {}
            # TODO: note, iterrows() does not preserve dtype, so strange things happen if demand file contains mixed types. Possibly rethink how this works.
            for _, rq_row in rq_time_df.iterrows():
                rq_obj = create_traveler(rq_row, rq_node_type_distr, self.zone_definition, self.routing_engine,
                                         simulation_time_step, self.scenario_parameters)
                new_rq_dict[rq_obj.rid] = rq_obj
            if rq_time in self.future_requests:
                self.future_requests[rq_time].update(new_rq_dict)
            else:
                self.future_requests[rq_time] = new_rq_dict
        LOG.info(f"init(): {number_rq_0 - number_rq}/{number_rq_0}"
                 f" requests removed ({G_RQ_TIME} not in simulation time)")
        # LOG.debug(f"self.future_requests = {self.future_requests}")

    def save_user_stats(self, force=True):
        current_buffer_size = len(self.user_stat_buffer)
        if (current_buffer_size and force) or current_buffer_size >= BUFFER_SIZE:
            if os.path.isfile(self.output_f):
                write_mode, write_header = "a", False
            else:
                write_mode, write_header = "w", True
            out_df = pd.DataFrame(self.user_stat_buffer)
            out_df.set_index(G_RQ_ID, inplace=True)
            out_df.to_csv(self.output_f, mode=write_mode, header=write_header)
            self.user_stat_buffer = []
            # LOG.info(f"\t ... just wrote {current_buffer_size} entries from buffer to customer output file.")
            LOG.debug(f"\t ... just wrote {current_buffer_size} entries from buffer to customer output file.")

    def record_user(self, rid):
        try:
            self.user_stat_buffer.append(self.rq_db[rid].record_data())
        except KeyError:
            LOG.warning(f"addToUserStatBuffer({rid}): user not found in database!")

    def get_new_travelers(self, simulation_time, *, since=None):
        """
        Get the new travelers as a list of (rid, Request) tuples. Also moves said requests from self.future_requests
        to self.rq_db.

        :param simulation_time: time for which to retrieve requests
        :type simulation_time: int
        :param since: (optional) time of last retrieval. If given, all times in (since, simulation_time] will be returned.
        :type since: int
        :return: list of (rid, rq) tuples
        """
        since = since if since is not None else simulation_time - 1  # default to only retrieving for current sim time
        list_new_traveler_rid_obj = []
        for t in range(since + 1, simulation_time + 1):
            rqs = self.future_requests.pop(t, {})
            for rid, rq in rqs.items():
                rq.set_direct_route_travel_infos(self.routing_engine)
                self.rq_db[rid] = rq
                self.undecided_rq[rid] = rq
                list_new_traveler_rid_obj.append((rid, rq))
        LOG.debug(f"{len(list_new_traveler_rid_obj)} new travelers join the simulation at time {simulation_time}.")
        return list_new_traveler_rid_obj

    def get_undecided_travelers(self, simulation_time):
        """This method returns the list of currently undecided requests.

        :param simulation_time: current simulation time
        :return: generator for (rid, rq) tuples
        """
        return list(self.undecided_rq.items())

    def record_boarding(self, rid, vid, op_id, simulation_time, pu_pos=None, t_access=None):
        """This method should be called whenever a customer boards a vehicle.

        :param rid: request id
        :param vid: vehicle id
        :param op_id: MoD operator id
        :param simulation_time: even if a simulation environment internally uses a finer resolution to update states,
        the output should be on the level of the simulation time step.
        :param pu_pos: the position of the pick-up. If None is given, the request origin is assumed.
        :param t_access: time needed to reach pu_pos from request-position. If None is given, this is ignored
        """
        LOG.debug(f"Traveler {rid} boards vehicle {vid} of operator {op_id} at time {simulation_time}")
        self.rq_db[rid].user_boards_vehicle(simulation_time, op_id, vid, pu_pos, t_access)
        if self.waiting_rq.get(rid):  # TODO # in case a vehicle arrives before the customer made a decision, simplest solution: customer boards
            del self.waiting_rq[rid]
        else:
            LOG.warning("waiting rq boarding warning : rid {} -> vid {} at {}".format(rid, vid, simulation_time))

    def record_alighting_start(self, rid, vid, op_id, simulation_time, do_pos=None, t_egress=None):
        """

        :param rid: request id
        :param vid: vehicle id
        :param op_id: MoD operator id
        :param simulation_time: even if a simulation environment internally uses a finer resolution to update states,
        the output should be on the level of the simulation time step.
        :param do_pos: the position of the drop-off. If None is given, the request destination is assumed.
        :param t_egress: time needed to get from drop-off position to target position. If None, this is ignored
        """
        LOG.debug(f"Traveler {rid} alights vehicle {vid} of operator {op_id} at time {simulation_time}")
        self.rq_db[rid].user_leaves_vehicle(simulation_time, do_pos, t_egress)
        self.record_user(rid)

    def user_ends_alighting(self, rid, vid, op_id, simulation_time):
        rq = self.rq_db.get(rid)
        if rq:
            del self.rq_db[rid]
        else:
            LOG.warning(f"user_ends_alighting({rid}): user not found in database!")
            
    def record_remaining_users(self):
        for rid in list(self.rq_db.keys()):
            self.record_user(rid)
        self.save_user_stats(force=True)

    def _get_all_requests(self):
        """Returns a list of (rid, Request) pairs for all requests currently in the Demand object."""
        all_requests = []
        for time in self.future_requests:
            all_requests += [(rid, req_obj) for rid, req_obj in self.future_requests[time].items()]
        all_requests += [(rid, req_obj) for rid, req_obj in self.rq_db.items()]
        return all_requests

    def __iter__(self):
        """Provides iterator over all requests in Demand as (rid, request) tuple."""
        return iter(self._get_all_requests())

    def __getitem__(self, rid):
        """Provides square-bracket indexing of Request objects (by request id)."""
        return dict(self._get_all_requests())[rid]


class SlaveDemand(Demand):
    """This class can be used when request are added from an external demand module."""
    rq_class = load_request_module("SlaveRequest")
    rq_parcel_class = load_request_module("SlaveParcelRequest")
    def add_request(self, rq_info_dict, offer_id, routing_engine, sim_time, modal_state = G_RQ_STATE_MONOMODAL):
        """ this function is used to add a new (person) request to the demand class
        :param rq_info_dict: dictionary with all information regarding the request input
        :param offer_id: used if there are different subrequests (TODO make optional? needed for moia)
        :param routing_engine: routinge engine obj
        :param sim_time: current simulation time
        :modal_state: look in globals for additional states (used to indicate first or last mile customers if needed)
        :return: request object
        """
        rq_info_dict[G_RQ_TIME] = sim_time
        if rq_info_dict.get(G_RQ_LDT) is None:
            rq_info_dict[G_RQ_LDT] = 0
        if modal_state == G_RQ_STATE_MONOMODAL:
            # original request
            rq_obj = self.rq_class(rq_info_dict, routing_engine, 1, self.scenario_parameters)
            rq_obj.set_direct_route_travel_infos(routing_engine)
        else:
            # first/last mile request
            if self.rq_db.get(rq_info_dict[G_RQ_ID]):
                parent_request = self.rq_db[rq_info_dict[G_RQ_ID]]
            else:
                parent_request = self.rq_class(rq_info_dict, routing_engine, 1, self.scenario_parameters)
            mod_o_node = rq_info_dict[G_RQ_ORIGIN]
            mod_d_node = rq_info_dict[G_RQ_DESTINATION]
            mod_start_time = rq_info_dict[G_RQ_EPT]
            rq_obj = parent_request.create_SubTripRequest(offer_id, mod_o_node, mod_d_node, mod_start_time, modal_state = modal_state)
            rq_obj.set_direct_route_travel_infos(routing_engine)
        # use rid-struct as key
        self.rq_db[rq_obj.get_rid_struct()] = rq_obj
        return rq_obj

    def add_parcel_request(self, rq_info_dict, offer_id, routing_engine, sim_time):
        """ this function is used to add a new (person) request to the demand class
        :param rq_info_dict: dictionary with all information regarding the request input
        :param offer_id: used if there are different subrequests (TODO make optional? needed for moia)
        :param routing_engine: routinge engine obj
        :param sim_time: current simulation time
        :return: request object
        """
        rq_info_dict[G_RQ_TIME] = sim_time
        if rq_info_dict.get(G_RQ_LDT) is None:
            rq_info_dict[G_RQ_LDT] = 0
        # original request
        rq_obj = self.rq_parcel_class(rq_info_dict, routing_engine, 1, self.scenario_parameters)
        rq_obj.set_direct_route_travel_infos(routing_engine)
        # use rid-struct as key
        self.rq_db[rq_obj.get_rid_struct()] = rq_obj
        return rq_obj

    def user_cancels_request(self, rid, simulation_time):
        LOG.debug(f"Traveler {rid} declines MoD offers at time {simulation_time}")
        self.record_user(rid)
        if rid in self.rq_db:
            del self.rq_db[rid]
        else:
            LOG.warning(f"user_cancels_request({rid}): user not found in database!")

    def record_boarding(self, rid, vid, op_id, simulation_time, pu_pos=None, t_access=None):
        """This method should be called whenever a customer boards a vehicle.

        :param rid: request id
        :param vid: vehicle id
        :param op_id: MoD operator id
        :param simulation_time: even if a simulation environment internally uses a finer resolution to update states,
        the output should be on the level of the simulation time step.
        :param pu_pos: the position of the pick-up. If None is given, the request origin is assumed.
        :param t_access: time needed to reach pu_pos from request-position. If None is given, this is ignored
        """
        LOG.debug(f"Traveler {rid} boards vehicle {vid} of operator {op_id} at time {simulation_time}")
        self.rq_db[rid].user_boards_vehicle(simulation_time, op_id, vid, pu_pos, t_access)

    def get_new_travelers(self, simulation_time):
        raise AssertionError("Method get_new_travelers() not accessible in slave mode.")
