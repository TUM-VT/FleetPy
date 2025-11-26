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
from src.fleetctrl.forecast.ForecastZoneSystemBase import ForecastZoneSystemBase
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_ODForecastZoneSystem = {
    "doc" :     """
    this class read OD-specific forecasts from a file and provides methods to access them
    """,
    "inherit" : "ForecastZoneSystemBase",
    "input_parameters_mandatory": [G_RA_FC_FNAME],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class ODForecastZoneSystem(ForecastZoneSystemBase):
    def __init__(self, zone_network_dir, scenario_parameters, dir_names, operator_attributes):
        """this class read OD-specific forecasts from a file and provides methods to access them"""
        super().__init__(zone_network_dir, scenario_parameters, dir_names, operator_attributes)
        # read forecast files
        self.od_forecasts = {}  # time -> o_zone -> d_zone -> counts
        if operator_attributes.get(G_RA_FC_FNAME):
            fc_dir = dir_names.get(G_DIR_FC)
            self.fc_temp_resolution = int(os.path.basename(fc_dir))
            forecast_f = os.path.join(fc_dir, operator_attributes.get(G_RA_FC_FNAME))
            if os.path.isfile(forecast_f):
                forecast_df = pd.read_csv(forecast_f)
                LOG.debug(f"read forecast file {forecast_f}")
                LOG.debug(f"{forecast_df.head()}")
                for time,o_zone_id,d_zone_id,trips in zip(forecast_df["time"].values,forecast_df["o_zone_id"].values,forecast_df["d_zone_id"].values,forecast_df["trips"].values):
                    try:
                        self.od_forecasts[time][o_zone_id][d_zone_id] = trips
                    except KeyError:
                        try:
                            self.od_forecasts[time][o_zone_id] = {d_zone_id : trips}
                        except KeyError:
                            self.od_forecasts[time] = {o_zone_id : {d_zone_id : trips}}
            else:
                raise FileNotFoundError(f"Didnt find forecast file {forecast_f}")
        else:
            LOG.warning("No forecast-file given -> no forecast provided!")
        

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
        raise NotImplementedError

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
        raise NotImplementedError
    
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
        future_list = []
        fc = self.get_trip_od_forecasts(t0, t1, scale=scale)
        for o_z, d_z_dict in fc.items():
            for d_z, val in d_z_dict.items():
                n_rqs = np.random.poisson(lam=val)
                ts = [np.random.randint(t0, high=t1) for _ in range(n_rqs)]
                for t in ts:
                    o_n = self.get_random_node(o_z, only_boarding_nodes=True)
                    d_n = self.get_random_node(d_z, only_boarding_nodes=True)
                    if o_n != -1 and d_n != -1:
                        future_list.append( (t, o_n, d_n) )
                    else:
                        LOG.warning(f"Couldnt find random node for zone {o_z} or {d_z}!")
        future_list.sort(key=lambda x:x[0])
        LOG.info("forecast list: {}".format(future_list))
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
        if t0%self.fc_temp_resolution != 0 or t1%self.fc_temp_resolution != 0:
            raise NotImplementedError("Forecast queries should be in line with forecast bins! Available bins: {} | queried: {} {}".format(sorted(self.od_forecasts.keys()), t0, t1))
        if t1 - t0 != self.fc_temp_resolution:
            raise NotImplementedError("Currently only forecasts with same resolutions possible")
        fc = self.od_forecasts.get(t0, {})
        if scale is not None and scale != 1:
            ret_fc = {}
            for o_z, d_dict in fc.items():
                ret_fc[o_z] = {}
                for d_z, val in d_dict.items():
                    ret_fc[o_z][d_z] = val * scale
            return ret_fc
        LOG.debug(f"forecast: {fc}")
        return fc