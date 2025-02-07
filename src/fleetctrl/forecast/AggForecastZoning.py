# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging

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

INPUT_PARAMETERS_AggForecastZoneSystem = {
    "doc" : "this class predicts demand based on a forecast file; this file contains the expected number of origins and destinations per zone " ,
    "inherit" : "ForecastZoneSystemBase",
    "input_parameters_mandatory": [G_RA_FC_FNAME],
    "input_parameters_optional": [
        G_RA_OP_CORR_M_F
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class AggForecastZoneSystem(ForecastZoneSystemBase):
    def __init__(self, zone_network_dir, scenario_parameters, dir_names, operator_attributes):
        super().__init__(zone_network_dir, scenario_parameters, dir_names, operator_attributes)
        # reading zone-correlation matrix if available
        if operator_attributes.get(G_RA_OP_CORR_M_F):
            # load correlation matrix files; these are saved as sparse matrices by scipy module
            # important: name of squared matrix depends on linear correlation matrix
            tmp_k_f = os.path.join(self.zone_general_dir, operator_attributes[G_RA_OP_CORR_M_F])
            tmp_k2_f = tmp_k_f.replace("zone_to_zone_correlations", "zone_to_zone_squared_correlations")
            if not os.path.isfile(tmp_k_f) or not os.path.isfile(tmp_k2_f):
                raise IOError(f"Could not find zone-to-zone correlation files {tmp_k_f} or {tmp_k2_f}!")
            self.zone_corr_matrix = load_npz(tmp_k_f).todense()
            self.zone_sq_corr_matrix = load_npz(tmp_k2_f).todense()
        else:
            self.zone_corr_matrix = np.eye(len(self.zones))
            self.zone_sq_corr_matrix = np.eye(len(self.zones))
        # read forecast files
        if operator_attributes.get(G_RA_FC_FNAME) and operator_attributes.get(G_RA_FC_TYPE):
            fc_dir = dir_names.get(G_DIR_FC)
            self.fc_temp_resolution = int(os.path.basename(fc_dir))
            forecast_f = os.path.join(fc_dir, operator_attributes.get(G_RA_FC_FNAME))
            if os.path.isfile(forecast_f):
                fc_type = operator_attributes.get(G_RA_FC_TYPE)
                self.forecast_df = pd.read_csv(forecast_f)
                self.fc_times = sorted(self.forecast_df[G_ZONE_FC_T].unique())
                self.forecast_df.set_index([G_ZONE_FC_T, G_ZONE_ZID], inplace=True)
                self.in_fc_type = f"in {fc_type}"
                self.out_fc_type = f"out {fc_type}"
                if self.in_fc_type not in self.forecast_df.columns or self.out_fc_type not in self.forecast_df.columns:
                    raise IOError(f"Could not find forecast data for {fc_type} in {forecast_f}")
                drop_columns = []
                for col in self.forecast_df.columns:
                    if col != self.in_fc_type and col != self.out_fc_type:
                        drop_columns.append(col)
                self.forecast_df.drop(drop_columns, axis=1, inplace=True)
            else:
                raise IOError(f"Could not find forecast file {forecast_f}")
        else:
            self.forecast_df = None
            self.in_fc_type = None
            self.out_fc_type = None
            self.fc_times = []
            self.fc_temp_resolution = None
        
    def _get_trip_forecasts(self, trip_type, t0, t1, aggregation_level, scale = None):
        """This method returns the number of expected trip arrivals or departures inside a zone in the
        time interval [t0, t1]. The return value is created by interpolation of the forecasts in the data frame
        if necessary. The default if no values can be found for a zone should be 0.

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

        if trip_type == "in":
            col = self.in_fc_type
        elif trip_type == "out":
            col = self.out_fc_type
        else:
            raise AssertionError("Invalid forecast column chosen!")
        #
        if aggregation_level is not None:
            tmp_forecast_df = self.forecast_df.reset_index().groubpy([G_ZONE_FC_T,
                                                                      aggregation_level]).aggregate({col: sum})
        else:
            tmp_forecast_df = self.forecast_df
        #
        def _create_forecast_dict(tmp_col, row_index, tmp_return_dict, tmp_scale_factor=1.0):
            # LOG.info(f"{self.forecast_df}")
            # LOG.info(f"{tmp_forecast_df}")
            # LOG.info(f"{row_index} | {G_ZONE_FC_T}")
            try:
                tmp_df = tmp_forecast_df.xs(row_index, level=G_ZONE_FC_T)
            except:
                LOG.info("couldnt find forecast for t {}".format(row_index))
                return {}
            tmp_dict = tmp_df[tmp_col].to_dict()
            for k, v in tmp_dict.items():
                try:
                    tmp_return_dict[k] += (v * tmp_scale_factor)
                except KeyError:
                    tmp_return_dict[k] = (v * tmp_scale_factor)
            return tmp_return_dict
        #
        return_dict = {}
        # get forecast of initial time interval
        last_t0 = t0
        if t0 not in self.fc_times:
            # check whether t0 and t1 are valid times
            if t0 > self.fc_times[-1] or t1 < self.fc_times[0]:
                # use first/last forecast and scale
                if t1 > self.fc_times[0]:
                    last_t0 = self.fc_times[0]
                else:
                    last_t0 = self.fc_times[-1]
                scale_factor = (t1 - t0) / self.fc_temp_resolution
                return_dict = _create_forecast_dict(col, last_t0, return_dict, scale_factor)
                # if scale is not None:
                #     for key, val in return_dict.items():
                #         return_dict[key] = val * scale
                return return_dict
            else:
                # get forecast from t0 to next value in self.fc_times
                for i in range(len(self.fc_times)):
                    # last_t0 = self.fc_times[i]
                    # next_t0 = self.fc_times[i+1]
                    next_t0 = self.fc_times[i]
                    if next_t0 > t1:
                        if last_t0 == t0:
                            scale_factor = (t1 - t0) / self.fc_temp_resolution
                            return_dict = _create_forecast_dict(col, self.fc_times[i-1], return_dict, scale_factor)
                            return return_dict
                        break
                    if last_t0 <= t0 and t0 < next_t0:
                        scale_factor = (next_t0 - last_t0) / self.fc_temp_resolution
                        # scale down the values
                        return_dict = _create_forecast_dict(col, next_t0, return_dict, scale_factor)
                        last_t0 = next_t0
                        break
        # add forecasts of next intervals as well
        while t1 - last_t0 > self.fc_temp_resolution:
            return_dict = _create_forecast_dict(col, last_t0, return_dict)
            last_t0 += self.fc_temp_resolution
            if last_t0 not in self.fc_times:
                break
        # append rest of last interval
        if t1 != last_t0:
            scale_factor = (t1 - last_t0) / self.fc_temp_resolution
            return_dict = _create_forecast_dict(col, last_t0, return_dict, scale_factor)
            if scale is not None:
                for key, val in return_dict.items():
                    return_dict[key] = val * scale
        return return_dict
    
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
        if self.in_fc_type is None:
            raise AssertionError("get_trip_arrival_forecasts() called even though no forecasts are available!")
        return self._get_trip_forecasts("in", t0, t1, aggregation_level, scale = scale)
    
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
        if self.out_fc_type is None:
            raise AssertionError("get_trip_departure_forecasts() called even though no forecasts are available!")
        return self._get_trip_forecasts("out", t0, t1, aggregation_level, scale = scale)
    
    def get_zone_correlation_matrix(self):
        """This method returns the zone correlation matrix for a given bandwidth (see PhD thesis of Flo) for further
        details.

        :return: N_z x N_z numpy matrix, where N_z is the number of forecast zones
        """
        return self.zone_corr_matrix

    def get_squared_correlation_matrix(self):
        """This method returns the squared zone correlation matrix for a given bandwidth (see RFFR Frontiers paper of
        Arslan and Flo or PhD thesis of Flo) for further details.

        :return: N_z x N_z numpy matrix, where N_z is the number of forecast zones
        """
        return self.zone_sq_corr_matrix
    
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
        dep_fc = self.get_trip_departure_forecasts(t0, t1, scale = scale)
        arr_fc = self.get_trip_arrival_forecasts(t0, t1, scale = scale)

        N_dep = sum(dep_fc.values())
        N_arr = sum(arr_fc.values())

        if N_dep == 0 or N_arr == 0:
            return []

        dep_zones = [dep_z for dep_z, dep_val in dep_fc.items() if dep_val > 0]
        dep_prob = [dep_val/N_dep for dep_val in dep_fc.values() if dep_val > 0]
        arr_zones = [arr_z for arr_z, arr_val in arr_fc.items() if arr_val > 0]
        arr_prob = [arr_val/N_arr for arr_val in arr_fc.values() if arr_val > 0]

        future_list = []
        tc = t0
        #LOG.warning(f"draw future: dep {N_dep} arr {N_arr} from {t0} - {t1} with scale {scale}")
        while True:
            tc += np.random.exponential(scale=float(t1-t0)/N_dep)
            if tc > t1:
                break
            o_zone = np.random.choice(dep_zones, p=dep_prob)
            d_zone = np.random.choice(arr_zones, p=arr_prob)
            o_n = self.get_random_node(o_zone, only_boarding_nodes=True)
            d_n = self.get_random_node(d_zone, only_boarding_nodes=True)
            if o_n != -1 and d_n != -1:
                future_list.append( (int(tc), o_n, d_n) )
            else:
                LOG.warning(f"draw future: couldnt find nodes for {o_zone} and {d_zone}")

        return future_list