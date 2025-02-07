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
from src.fleetctrl.forecast.ForecastZoneSystemBase import ForecastZoneSystemBase
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
#from src.fleetctrl.FleetControlBase import PlanRequest # TODO # circular dependency!
# set log level to logging.DEBUG or logging.INFO for single simulations
LOG_LEVEL = logging.WARNING
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_PerfectForecastZoneSystem = {
    "doc" :     """
    this class can be used like the "basic" ZoneSystem class
    but instead of getting values from a demand forecast dabase, this class has direct access to the demand file
    and therefore makes perfect predictions for the corresponding forecast querries """,
    "inherit" : "ForecastZoneSystemBase",
    "input_parameters_mandatory": [G_RA_FC_TR],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class PerfectForecastZoneSystem(ForecastZoneSystemBase):
    # this class can be used like the "basic" ZoneSystem class
    # but instead of getting values from a demand forecast dabase, this class has direct access to the demand file
    # and therefore makes perfect predictions for the corresponding forecast querries
    def __init__(self, zone_network_dir, scenario_parameters, dir_names, operator_attributes):
        if operator_attributes.get(G_RA_FC_FNAME) is not None:
            LOG.warning("forecast file for perfact forecast given. will not be loaded!")
        super().__init__(zone_network_dir, scenario_parameters, dir_names, operator_attributes)
        if self.fc_temp_resolution is None:
            self.fc_temp_resolution = operator_attributes[G_RA_FC_TR] # TODO ?

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
        if scale is not None:
            LOG.warning("scale is not used for perfect forecast and sampling!")
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
    
    def get_trip_od_forecasts(self, t0, t1, aggregation_level=None, scale=None):
        """ this function returns the number of expected trips from one zone to another int the time interval [t0, t1]
        
        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :param scale: scales forecast distributen by this value if given
        :type scale: float
        :return: {}: zone -> zone -> forecast of trips
        :rtype: dict
        """
        if scale is None:
            scale = 1.0
        return_dict = {}
        for t in range(t0, t1):
            future_rqs = self.demand.future_requests.get(t, {})
            for rq in future_rqs.values():
                o_zone = self.get_zone_from_node(rq.o_node)
                d_zone = self.get_zone_from_node(rq.d_node)
                if o_zone >= 0 and d_zone >= 0:
                    try:
                        return_dict[o_zone][d_zone] += 1 * scale
                    except KeyError:
                        try:
                            return_dict[o_zone][d_zone] = 1 * scale
                        except KeyError:
                            return_dict[o_zone] = {d_zone : 1 * scale}
        return return_dict
    
#########################################################################################################

INPUT_PARAMETERS_PerfectForecastDistributionZoneSystem= {
    "doc" :     """
        the only difference to PerfectForecastZoneSystem is that the sampling process does not return the exact future requests but a
        sample from the forecast distribution """,
    "inherit" : "PerfectForecastZoneSystem",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class PerfectForecastDistributionZoneSystem(PerfectForecastZoneSystem):
    """ the only difference to PerfectForecastZoneSystem is that the sampling process does not return the exact future requests but a
    sample from the forecast distribution
    """
    
    def __init__(self, zone_network_dir, scenario_parameters, dir_names, operator_attributes):
        super().__init__(zone_network_dir, scenario_parameters, dir_names, operator_attributes)
        self._forecast = {(s, s + self.fc_temp_resolution) : {} for s in 
                          range(0//self.fc_temp_resolution * self.fc_temp_resolution, 86400//self.fc_temp_resolution * self.fc_temp_resolution, self.fc_temp_resolution)} #TODO hard coded (but scenario parameters is epmpty!)
                            # (start_time, end_time) -> {o_zone -> {d_zone -> count}}
    
    def time_trigger(self, sim_time):
        """"
        this method is triggered at the beginning of a repositioning time step
        """
        pass
    
    def register_demand_ref(self, demand_ref):
        x = super().register_demand_ref(demand_ref)
        for time_bin in self._forecast.keys():
            for t in range(time_bin[0], time_bin[1]):
                future_rqs = self.demand.future_requests.get(t, {})
                for rq in future_rqs.values():
                    o_zone = self.get_zone_from_node(rq.o_node)
                    d_zone = self.get_zone_from_node(rq.d_node)
                    if o_zone >= 0 and d_zone >= 0:
                        try:
                            self._forecast[time_bin][o_zone][d_zone] += 1
                        except KeyError:
                            try:
                                self._forecast[time_bin][o_zone][d_zone] = 1
                            except KeyError:
                                self._forecast[time_bin][o_zone] = {d_zone : 1}
        return x
    
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
        
        if request_attribute is not None or attribute_value is not None:
            raise NotImplementedError("request_attribute and attribute_value not implemented yet for PerfectForecastDistributionZoneSystem")
        
        relevant_intervals = self._get_relevant_forcast_intervals(t0, t1)
        
        future_list = []
        
        for start_int, end_int, scale_int in relevant_intervals:
            if scale is not None:
                scale_int *= scale
            if scale_int == 0:
                continue
            if scale is None:
                scale = 1.0
            try:
                future_poisson_rates = self._forecast[(start_int, end_int)]
            except KeyError:
                LOG.warning(f"no forecast found for interval {(start_int, end_int)} | {self._forecast.keys()}")
                future_poisson_rates = {}
            for o_zone, d_zone_dict in future_poisson_rates.items():
                for d_zone, poisson_rate in d_zone_dict.items():
                    number_rqs = np.random.poisson(poisson_rate * scale_int)
                    ts = [np.random.randint(start_int, high=end_int) for _ in range(number_rqs)]
                    for t in ts:
                        o_n = self.get_random_node(o_zone, only_boarding_nodes=True)
                        d_n = self.get_random_node(d_zone, only_boarding_nodes=True)
                        if o_n >= 0 and d_n >= 0:
                            future_list.append( (t, o_n, d_n) )
                        else:
                            LOG.warning(f"no node found for zone {o_zone} or {d_zone}")
        future_list.sort(key=lambda x:x[0])

        LOG.debug("perfect forecast list: {}".format(future_list))
        return future_list
    
    def _get_relevant_forcast_intervals(self, t0, t1):
        """
        this function returns the forecast intervals in the aggregation level of self.fc_temp_resolution that are relevant for the time interval [t0, t1]
        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :return: list of forecast intervals (start, end, fraction of forecast interval in [t0, t1])
        :rtype: list of 3-tuples
        """
        if t1 <= t0:
            return []
        cur_t = t0 // self.fc_temp_resolution * self.fc_temp_resolution
        next_t = cur_t + self.fc_temp_resolution
        relevant_intervals = []
        while True:
            if cur_t >= t0 and next_t <= t1:
                relevant_intervals.append((cur_t, next_t, 1.0))
            elif cur_t < t0 < t1 < next_t:
                frac = (t1 - t0) / self.fc_temp_resolution
                relevant_intervals.append((cur_t, next_t, frac))
            elif cur_t < t0 < next_t:
                frac = (next_t - t0) / self.fc_temp_resolution
                relevant_intervals.append((cur_t, next_t, frac))
            elif cur_t < t1 < next_t:
                frac = (t1 - cur_t) / self.fc_temp_resolution
                relevant_intervals.append((cur_t, next_t, frac))
                break
            else:
                break
            cur_t = next_t
            next_t += self.fc_temp_resolution
        return relevant_intervals