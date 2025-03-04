"""
Utilities for handling configurations.

Both YAML and legacy CSV formats supported.

Author:Patrick Malcolm
"""
import re
import numpy as np
import pandas as pd
import yaml
from yaml.parser import ParserError, ScannerError

EPS = 0.0001
CAPTURE_RE = re.compile("\d+:\d+")


def str_smart_convert(in_str):
    """
    Converts an input string to a type according to its content:
    1. float conversion
    2. int conversion of floats with |x - int(x)| < EPS (EPS = 0.000001)
    3. bool conversion of strings "True" and "False"
    4. None conversion of string "None"

    :param in_str: scenario input string
    :type in_str: str
    :return: in_str in converted format
    :rtype: depending on in_str
    """
    try:
        return_val = float(in_str)
    except ValueError:
        # non numeric input
        mapping = {"True": True, "False": False, "None": None, "nan": None, "": None}
        return_val = mapping.get(in_str, in_str)
    else:
        # numeric input
        int_return_val = int(round(return_val, 0))
        if abs(return_val - int_return_val) < EPS:
            return_val = int_return_val
    return return_val


def decode_config_str(in_string):
    """
    Decodes a string from a config file into a python object depending on its format.
    If the string is valid YAML, it will be interpreted as such. If not, the following legacy format will be used:
    - list: ";" separated inputs. Example: "12;15;8"
    - top-level list: "|" separated inputs. Example (nested list): "12;15;8|1;2;3|10;11;12"
    - dict: ":" separation of key and value, ";" separation of key, value tuples. Example: "v1:12;v2:48"
    - all string inputs (str, values of lists, keys and values of dicts):
    1. float conversion
    2. int conversion of floats with |x - int(x)| < EPS (EPS = 0.000001)
    3. bool conversion of strings "True" and "False"
    4. None conversion of string "None"

    :param in_string: scenario input string
    :type in_string: str
    :return: formatted scenario input
    :rtype: depending on in_string
    """
    if type(in_string) != str:
        if np.isnan(in_string):
            return None
        else:
            return in_string
    # one special case: dictionary with one key "number:number" -> yaml 1.1 interpretation as sexagesimal number
    if not CAPTURE_RE.fullmatch(in_string):
        # First, try to interpret string as YAML
        try:
            as_yaml = yaml.load(in_string, Loader=yaml.Loader)
        except (ParserError, ScannerError):
            pass
        else:
            if type(as_yaml) != str:
                return as_yaml
    # If YAML interpretation failed, interpret as legacy format
    if "|" in in_string:
        return [decode_config_str(s) for s in in_string.split("|")]
    elif ":" in in_string:
        pairs = [pair.split(":") for pair in in_string.split(";")]
        return {str_smart_convert(k): str_smart_convert(v) for k, v in pairs}
    elif ";" in in_string:
        return [str_smart_convert(x) for x in in_string.split(";")]
    else:
        return str_smart_convert(in_string)


class ConstantConfig(dict):
    def __init__(self, arg=None, **kwargs):
        """
        Initialize a ConstantConfig object with an optional file.
        This class is just a wrapper around a dict with convenient file loading functions.
        The + operator is supported as an alias for the update() method. When used, the two operands will be merged,
        with key-value pairs in the second operand overwriting those in the first (if present).

        :param arg: if str type, interpreted as path to configuration file, otherwise passed on to parent class (dict)
        """
        # Call the parent class __init__ method
        file_path = None
        if arg is not None and type(arg) != str:
            super().__init__(arg, **kwargs)
        else:
            file_path = arg
            super().__init__(**kwargs)
        # If a file was specified, load it based on the file extension
        if file_path is not None:
            if file_path.endswith(".csv"):
                self.update(self.read_csv(file_path))
            elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
                self.update(self.read_yaml(file_path))
            else:
                raise NotImplementedError("Config file type for " + file_path + " not supported.")

    @classmethod
    def read_csv(cls, file_path):
        """
        Generate a ConstantConfig object from a csv file.

        :param file_path: path to the csv configuration file
        :type file_path: str
        """
        cfg = cls()
        constant_series = pd.read_csv(file_path, index_col=0, comment="#").squeeze("columns")
        for k, v in constant_series.items():
            cfg[k] = decode_config_str(v)
        return cfg

    @classmethod
    def read_yaml(cls, file_path):
        """
        Generate a ConstantConfig object from a yaml file.

        :param file_path: path to the csv configuration file
        :type file_path: str
        """
        with open(file_path) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        return cls(cfg)

    def __add__(self, other):
        if not isinstance(other, dict):
            raise NotImplementedError("Can't add", other, "of type", type(other), "to ConstantConfig.")
        return type(self)({**self, **other})


class ScenarioConfig(list):
    def __init__(self, file_path=None):
        """
        Initialize a ScenarioConfig object, which is essentially just a list of ConstantConfig objects.

        :param file_path: path to scenario configuration file
        :type file_path: str
        """
        super().__init__()
        if file_path is not None:
            if file_path.endswith(".csv"):
                self += self.read_csv(file_path)
            elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
                self += self.read_yaml(file_path)
            else:
                raise NotImplementedError("Config file type for " + file_path + " not supported.")

    @classmethod
    def read_csv(cls, file_path):
        """
        Generate a ScenarioConfig object from a csv file.

        :param file_path: path to the csv scenario configuration file
        :type file_path: str
        """
        cfgs = cls()
        df = pd.read_csv(file_path, comment="#")
        for col in df.columns:
            df[col] = df[col].apply(decode_config_str)
        for i, row in df.iterrows():
            cfgs.append(ConstantConfig(row))
        return cfgs

    @classmethod
    def read_yaml(cls, file_path):
        """
        Generate a ScenarioConfig object from a yaml file.

        :param file_path: path to the csv configuration file
        :type file_path: str
        """
        with open(file_path) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        return cls(cfg)


if __name__ == "__main__":
    const_cfg = ConstantConfig("../../scenarios/TestBroker/constant_config.yaml")
    print(const_cfg)
    scenarios_cfg = ScenarioConfig("../../scenarios/TestBroker/scenarios_0.csv")
    print(scenarios_cfg)
    print("Combinations: ")
    for scenario_cfg in scenarios_cfg:
        print(const_cfg+scenario_cfg)

