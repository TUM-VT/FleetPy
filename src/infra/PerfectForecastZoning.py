# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np

# src imports
# -----------
from src.infra.Zoning import ZoneSystem
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
#from src.fleetctrl.FleetControlBase import PlanRequest # TODO # circular dependency!
# set log level to logging.DEBUG or logging.INFO for single simulations
LOG_LEVEL = logging.WARNING
LOG = logging.getLogger(__name__)


class PerfectForecastZoneSystem(ZoneSystem):
    # this class can be used like the "basic" ZoneSystem class
    # but instead of getting values from a demand forecast dabase, this class has direct access to the demand file
    # and therefore makes perfect predictions for the corresponding forecast querries
    def __init__(self, zone_network_dir, scenario_parameters, dir_names):
        tmp_scenario_parameters = scenario_parameters.copy()
        if scenario_parameters[G_FC_FNAME] is not None:
            LOG.warning("forecast file for perfact forecast given. will not be loaded!")
            del tmp_scenario_parameters[G_FC_FNAME] 
        super().__init__(zone_network_dir, tmp_scenario_parameters, dir_names)

    def _get_trip_forecasts(self, trip_type, t0, t1, aggregation_level):
        """This method returns the number of expected trip arrivals or departures inside a zone in the
        time interval [t0, t1]. The return value is created by interpolation of the forecasts in the data frame
        if necessary. The default if no values can be found for a zone should be 0.

        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :return: {}: zone -> forecast of arrivals
        :rtype: dict
        """

        if trip_type == "in":
            incoming = True
        elif trip_type == "out":
            incoming = False
        else:
            raise AssertionError("Invalid forecast column chosen!")
        #
        if aggregation_level is not None:
            raise NotImplementedError("aggregation level for perfect forecast not implemented")
        #
        return_dict = {}
        if not incoming:
            for t in range(t0, t1):
                future_rqs = self.demand.future_requests.get(t, {})
                for rq in future_rqs.values():
                    o_zone = self.get_zone_from_node(rq.o_node)
                    if o_zone >= 0:
                        try:
                            return_dict[o_zone] += 1
                        except:
                            return_dict[o_zone] = 1
        else:
            for t in range(t0, t1):
                future_rqs = self.demand.future_requests.get(t, {})
                for rq in future_rqs.values():
                    d_zone = self.get_zone_from_node(rq.d_node)
                    if d_zone >= 0:
                        try:
                            return_dict[d_zone] += 1
                        except:
                            return_dict[d_zone] = 1
        return return_dict

    def get_trip_arrival_forecasts(self, t0, t1, aggregation_level=None):
        """This method returns the number of expected trip arrivals inside a zone in the time interval [t0, t1].
        The return value is created by interpolation of the forecasts in the data frame if necessary.
        The default if no values can be found for a zone should be 0.

        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :return: {}: zone -> forecast of arrivals
        :rtype: dict
        """
        if self.in_fc_type is None:
            raise AssertionError("get_trip_arrival_forecasts() called even though no forecasts are available!")
        return self._get_trip_forecasts("in", t0, t1, aggregation_level)

    def get_trip_departure_forecasts(self, t0, t1, aggregation_level=None):
        """This method returns the number of expected trip departures inside a zone in the time interval [t0, t1].

        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :return: {}: zone -> forecast of departures
        :rtype: dict
        """
        if self.out_fc_type is None:
            raise AssertionError("get_trip_departure_forecasts() called even though no forecasts are available!")
        return self._get_trip_forecasts("out", t0, t1, aggregation_level)

    def draw_future_request_sample(self, t0, t1, request_attribute = None, attribute_value = None, scale = None): #request_type=PlanRequest # TODO # cant import PlanRequest because of circular dependency of files!
        """ this function returns exact future request attributes  [t0, t1]
        currently origin is drawn from get_trip_departure_forecasts an destination is drawn form get_trip_arrival_forecast (independently! # TODO #)
        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param request_attribute: name of the attribute of the request class. if given, only returns requests with this attribute
        :type request_attribute: str
        :param attribute_value: if and request_attribute given: only returns future requests with this attribute value
        :type attribute_value: type(request_attribute)
        :param scale: (not for this class) scales forecast distribution by this values
        :type scale: float
        :return: list of (time, origin_node, destination_node) of future requests
        :rtype: list of 3-tuples
        """ 
        future_list = []
        if request_attribute is None and attribute_value is None:
            for t in range(int(np.math.floor(t0)), int(np.math.ceil(t1))):
                future_rqs = self.demand.future_requests.get(t, {})
                for rq in future_rqs.values():
                    future_list.append( (t, rq.o_node, rq.d_node) )
        elif request_attribute is not None:
            if attribute_value is None:
                for t in range(int(np.math.floor(t0)), int(np.math.ceil(t1))):
                    future_rqs = self.demand.future_requests.get(t, {})
                    for rq in future_rqs.values():
                        if rq.__dict__.get(request_attribute) is not None:
                            future_list.append( (t, rq.o_node, rq.d_node) )
            else:
                for t in range(int(np.math.floor(t0)), int(np.math.ceil(t1))):
                    future_rqs = self.demand.future_requests.get(t, {})
                    for rq in future_rqs.values():
                        if rq.__dict__.get(request_attribute) is not None and rq.__dict__.get(request_attribute) == attribute_value:
                            future_list.append( (t, rq.o_node, rq.d_node) )
        LOG.info("perfect forecast list: {}".format(future_list))
        return future_list