from __future__ import annotations

# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
from ast import Not
import os
import logging
import typing as tp

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
from src.fleetctrl.forecast.PerfectForecastZoning import PerfectForecastDistributionZoneSystem
from src.misc.globals import *
#from src.fleetctrl.FleetControlBase import PlanRequest # TODO # circular dependency!
# set log level to logging.DEBUG or logging.INFO for single simulations

if tp.TYPE_CHECKING:
    from src.fleetctrl.planning.PlanRequest import PlanRequest

LOG_LEVEL = logging.WARNING
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_PerfectOMyopicDForecast = {
    "doc" :     """
    this class can be used like the "basic" ZoneSystem class
    but instead of getting values from a demand forecast dabase, this class has direct access to the demand file
    this class produces forecasts based on the requests from the last time step and projects the same distribution into the future
    (it looks self.fc_temp_resolution into the past and uses this forecast to produce the forecast) """,
    "inherit" : "PerfectForecastDistributionZoneSystem",
    "input_parameters_mandatory": [G_RA_FC_TR],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class PerfectOMyopicDForecast(PerfectForecastDistributionZoneSystem):
    """
    this class can be used like the "basic" ZoneSystem class
    but instead of getting values from a demand forecast dabase, this class has direct access to the demand file
    this class produces forecasts based on the requests from the last time step and projects the same distribution into the future
    (it looks self.fc_temp_resolution into the past and uses this forecast to produce the forecast)
    """
    def __init__(self, zone_network_dir, scenario_parameters, dir_names, operator_attributes):
        if operator_attributes.get(G_RA_FC_FNAME) is not None:
            LOG.warning("forecast file for forecast given. will not be loaded in this forecast system!")
        super().__init__(zone_network_dir, scenario_parameters, dir_names, operator_attributes)
        if self.fc_temp_resolution is None:
            self.fc_temp_resolution = operator_attributes[G_RA_FC_TR] # TODO ?
            
        self._past_request_ods = [] # list of (request_time, origin_zone, destination_zone)
        
    def time_trigger(self, sim_time):
        """"
        this method is triggered at the beginning of a repositioning time step
        -> here: cleaning of past request od list
        """
        super().time_trigger(sim_time)
        LOG.debug(f"trigger at {sim_time}: past rqs before {self._past_request_ods}")
        if len(self._past_request_ods) > 0 and self._past_request_ods[0][0] < sim_time - self.fc_temp_resolution:
            # remove old requests from list
            break_index = 0
            for i, entry in enumerate(self._past_request_ods):
                if entry[0] >= sim_time - self.fc_temp_resolution:
                    break_index = i
                    break
            self._past_request_ods = self._past_request_ods[break_index:]
            LOG.debug(f"past rqs after: {self._past_request_ods}")
            
    def register_new_request(self, sim_time: int, plan_request: PlanRequest):
        """ 
        This method is triggered when a new user requested a trip.
        :param sim_time: current simulation time
        :param plan_request: plan_request obj
        """
        o_node = plan_request.get_o_stop_info()[0][0]
        o_zone = self.get_zone_from_node(o_node)
        d_node = plan_request.get_d_stop_info()[0][0]
        d_zone = self.get_zone_from_node(d_node)
        self._past_request_ods.append( (sim_time, o_zone, d_zone) )
    
    def register_rejected_request(self, sim_time, plan_request):
        """ 
        This method is triggered when a new user has been rejected.
        :param sim_time: current simulation time
        :param plan_request: plan_request obj
        """
        pass

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
        raise NotImplementedError("not implemented yet")
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
        
        future_fraction = (t1 - t0)/self.fc_temp_resolution # scaling factor
        
        return_dict = {}
        if not incoming:
            for _, o_zone, _ in self._past_request_ods:
                if o_zone >= 0:
                    try:
                        return_dict[o_zone] += future_fraction
                    except:
                        return_dict[o_zone] = future_fraction
        else:
            for _, _, d_zone in self._past_request_ods:
                if d_zone >= 0:
                    try:
                        return_dict[d_zone] += future_fraction
                    except:
                        return_dict[d_zone] = future_fraction
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
        
        future_fraction = (t1 - t0)/self.fc_temp_resolution # scaling factor
        if scale is not None:
            future_fraction = future_fraction*scale
        else:
            scale = 1.0
        LOG.debug(f"call get od forecasts: {t0} {t1} {self.fc_temp_resolution}")
        d_distributions = {}
        for _, o_zone, d_zone in self._past_request_ods:
            if o_zone >= 0 and d_zone >= 0:
                try:
                    d_distributions[o_zone][d_zone] += future_fraction
                except KeyError:
                    try:
                        d_distributions[o_zone][d_zone] = future_fraction
                    except KeyError:
                        d_distributions[o_zone] = {d_zone : future_fraction}
        for o_zone, d_dict in d_distributions.items():
            s = sum(d_dict.values())
            for d_zone in d_dict.keys():
                d_dict[d_zone] /= s
                
        return_dict = {}
        for t in range(t0, t1):
            future_rqs = self.demand.future_requests.get(t, {})
            for rq in future_rqs.values():
                o_zone = self.get_zone_from_node(rq.o_node)
                d_dict = d_distributions.get(o_zone, None)
                if d_dict is None:
                    d_dict = {z : 1.0/len(self.zones) for z in range(len(self.zones))}
                for d_zone, val in d_dict.items():
                    try:
                        return_dict[o_zone][d_zone] += val * scale
                    except KeyError:
                        try:
                            return_dict[o_zone][d_zone] = val * scale
                        except KeyError:
                            return_dict[o_zone] = {d_zone : val * scale}
        return return_dict
        
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
        
        LOG.debug(f"call get od forecasts: {t0} {t1} {self.fc_temp_resolution}")
        d_distributions = {}
        for _, o_zone, d_zone in self._past_request_ods:
            if o_zone >= 0 and d_zone >= 0:
                try:
                    d_distributions[o_zone][d_zone] += 1
                except KeyError:
                    try:
                        d_distributions[o_zone][d_zone] = 1
                    except KeyError:
                        d_distributions[o_zone] = {d_zone : 1}
        for o_zone, d_dict in d_distributions.items():
            s = sum(d_dict.values())
            for d_zone in d_dict.keys():
                d_dict[d_zone] /= s
        
        relevant_intervals = self._get_relevant_forcast_intervals(t0, t1)
        
        future_list = []
        
        for start_int, end_int, scale_int in relevant_intervals:
            if scale is not None:
                scale_int *= scale
            if scale_int == 0:
                continue
            try:
                future_poisson_rates = self._forecast[(start_int, end_int)]
            except KeyError:
                LOG.warning(f"no forecast found for interval {(start_int, end_int)} | {self._forecast.keys()}")
                future_poisson_rates = {}
            for o_zone, d_zone_dict in future_poisson_rates.items():
                number = sum(d_zone_dict.values())
                d_dict = d_distributions.get(o_zone, None)
                if d_dict is None:
                    d_dict = {z : 1.0/len(self.zones) for z in range(len(self.zones))}
                
                for d_zone, d_fraction in d_dict.items():
                    poisson_rate = number * d_fraction * scale_int
                    number_rqs = np.random.poisson(poisson_rate  )
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