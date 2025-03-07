# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging
from abc import abstractmethod, ABCMeta

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

# src imports
# -----------
from src.infra.Zoning import ZoneSystem
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_ForecastZoneSystemBase = {
    "doc" : "this class is the base class for providing demand forecasts for the fleet control system",
    "inherit" : "ZoneSystem",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class ForecastZoneSystemBase(ZoneSystem):
    def __init__(self, zone_network_dir, scenario_parameters, dir_names, operator_attributes):
        super().__init__(zone_network_dir, scenario_parameters, dir_names)
        self.fc_temp_resolution = None
        self.demand = None

    def register_demand_ref(self, demand_ref):
        self.demand = demand_ref
        
    def time_trigger(self, sim_time):
        """"
        this method is triggered at the beginning of a repositioning time step
        """
        pass
        
    def register_new_request(self, sim_time, plan_request):
        """ 
        This method is triggered when a new user requested a trip.
        :param sim_time: current simulation time
        :param plan_request: plan_request obj
        """
        pass
    
    def register_rejected_request(self, sim_time, plan_request):
        """ 
        This method is triggered when a new user has been rejected.
        :param sim_time: current simulation time
        :param plan_request: plan_request obj
        """
        pass
    
    @abstractmethod
    def get_trip_arrival_forecasts(self, t0, t1, aggregation_level=None, scale = None):
        """This method returns the number of expected trip arrivals inside a zone in the time interval [t0, t1].
        The return value is created by interpolation of the forecasts in the data frame if necessary.
        The default if no values can be found for a zone should be 0.

        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :param scale: scales forecast distributen by this value if given
        :type scale: float
        :return: {}: zone -> forecast of arrivals
        :rtype: dict
        """

    @abstractmethod
    def get_trip_departure_forecasts(self, t0, t1, aggregation_level=None, scale = None):
        """This method returns the number of expected trip departures inside a zone in the time interval [t0, t1].

        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :param scale: scales forecast distributen by this value if given
        :type scale: float
        :return: {}: zone -> forecast of departures
        :rtype: dict
        """
    
    @abstractmethod
    def draw_future_request_sample(self, t0, t1, request_attribute = None, attribute_value = None, scale = None): #request_type=PlanRequest # TODO # cant import PlanRequest because of circular dependency of files!
        """ this function returns future request attributes drawn probabilistically from the forecast method for the intervall [t0, t1]
        currently origin is drawn from get_trip_departure_forecasts an destination is drawn form get_trip_arrival_forecast (independently! # TODO #)
        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param request_attribute: (not for this class) name of the attribute of the request class. if given, only returns requests with this attribute
        :type request_attribute: str
        :param attribute_value: (not for this class) if and request_attribute given: only returns future requests with this attribute value
        :type attribute_value: type(request_attribute)
        :param scale: scales forecast distribution by this values
        :type scale: float
        :return: list of (time, origin_node, destination_node) of future requests
        :rtype: list of 3-tuples
        """ 
        
    @abstractmethod
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