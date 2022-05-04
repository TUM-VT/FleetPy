import os

from src.misc.globals import *
from src.FleetSimulationBase import INPUT_PARAMETERS_FleetSimulationBase
from src.misc.init_modules import *

MODULE_PARAM_TO_DICT_LOAD = {
    G_SIM_ENV : get_src_simulation_environments,
    G_NETWORK_TYPE : get_src_routing_engines,
    G_RQ_TYP1 : get_src_request_modules,
    G_OP_MODULE : get_src_fleet_control_modules,
    G_RA_RES_MOD : get_src_reservation_strategies,
    G_OP_CH_M : get_src_charging_strategies,
    G_OP_REPO_M : get_src_repositioning_strategies,
    G_OP_DYN_P_M : get_src_dynamic_pricing_strategies,
    G_OP_DYN_FS_M : get_src_dynamic_fleet_sizing_strategies
}

def load_module_parameters(module_dict, module_str):
    tmp = module_dict.get(module_str)
    if tmp is not None:
        module_name, class_name = tmp
        module = importlib.import_module(module_name)
        input_params_str = f"INPUT_PARAMETERS_{class_name}"
        input_param_dict = getattr(module, input_params_str)
        return input_param_dict
    else:
        raise IOError(f"{module_str} is invalid!")


class ScenarioCreator():
    def __init__(self):
        self._current_mandatory_params = INPUT_PARAMETERS_FleetSimulationBase["input_parameters_mandatory"]
        self._current_optional_params = INPUT_PARAMETERS_FleetSimulationBase["input_parameters_optional"]
        
        self._current_mandatory_modules = INPUT_PARAMETERS_FleetSimulationBase["mandatory_modules"]
        self._current_optional_modules = INPUT_PARAMETERS_FleetSimulationBase["optional_modules"]
        
        self._currently_selected_modules = {}
        
    def _load_module_params(self, module_param, module_param_value):
        print("")
        print(f"load parameters for module {module_param_value}!")
        module_dict = MODULE_PARAM_TO_DICT_LOAD[module_param]()
        input_param_dict = load_module_parameters(module_dict, module_param_value)
        
        for mandatory_param, mandatory_param_str in input_param_dict["input_parameters_mandatory"].items():
            self._current_mandatory_params[mandatory_param] = mandatory_param_str
            if self._current_optional_params.get(mandatory_param) is not None:
                del self._current_optional_params[mandatory_param]
        for optional_param, optional_param_str in input_param_dict["input_parameters_optional"].items():
            if self._current_mandatory_params.get(optional_param) is None:
                self._current_optional_params[optional_param] = optional_param_str
        for mandatory_param, mandatory_param_str in input_param_dict["mandatory_modules"].items():
            self._current_mandatory_modules[mandatory_param] = mandatory_param_str
            if self._current_optional_modules.get(mandatory_param) is not None:
                del self._current_optional_modules[mandatory_param]
        for optional_param, optional_param_str in input_param_dict["optional_modules"].items():
            if self._current_mandatory_modules.get(optional_param) is None:
                self._current_optional_modules[optional_param] = optional_param_str
                
        while len(input_param_dict["inherit"]) != 0:
            for inherit_class in input_param_dict["inherit"]:
                input_param_dict["inherit"].remove(inherit_class)
                if inherit_class.endswith("Base"): # TODO!
                    base_p = list(module_dict.values())[0][0].split(".")[:-1]
                    base_p.append(inherit_class)
                    base_p = ".".join(base_p)
                    module_dict[inherit_class] = (base_p, inherit_class)
                new_input_param_dict = load_module_parameters(module_dict, inherit_class)
                print(f" -> inherit {inherit_class} : {new_input_param_dict['doc']}")
                for o_inherit in new_input_param_dict["inherit"]:
                    input_param_dict["inherit"].append(o_inherit)
                for mandatory_param, mandatory_param_str in new_input_param_dict["input_parameters_mandatory"].items():
                    self._current_mandatory_params[mandatory_param] = mandatory_param_str
                    if self._current_optional_params.get(mandatory_param) is not None:
                        del self._current_optional_params[mandatory_param]
                for optional_param, optional_param_str in new_input_param_dict["input_parameters_optional"].items():
                    if self._current_mandatory_params.get(optional_param) is None:
                        self._current_optional_params[optional_param] = optional_param_str
                for mandatory_param, mandatory_param_str in new_input_param_dict["mandatory_modules"].items():
                    self._current_mandatory_modules[mandatory_param] = mandatory_param_str
                    if self._current_optional_modules.get(mandatory_param) is not None:
                        del self._current_optional_modules[mandatory_param]
                for optional_param, optional_param_str in new_input_param_dict["optional_modules"].items():
                    if self._current_mandatory_modules.get(optional_param) is None:
                        self._current_optional_modules[optional_param] = optional_param_str
                        
        if self._current_mandatory_modules.get(module_param) is not None:
            del self._current_mandatory_modules[module_param]
        if self._current_optional_modules.get(module_param) is not None:
            del self._current_optional_modules[module_param]
        print("==========================================================")
                
        
    def print_current_mandatory_and_optional_modules(self):
        print("Mandatory Modules:")
        print("")
        for param, desc in self._current_mandatory_modules.items():
            print(f"Parameter Value: {param}")
            print(f"Description: {desc}")
            if MODULE_PARAM_TO_DICT_LOAD.get(param) is not None:
                options = list(MODULE_PARAM_TO_DICT_LOAD[param]().keys())
            else:
                options = []
            print(f"Options: {options}")
            print("")
        print("")
        print("Optional Modules:")
        print("")
        for param, desc in self._current_optional_modules.items():
            print(f"Parameter Value: {param}")
            print(f"Description: {desc}")
            if MODULE_PARAM_TO_DICT_LOAD.get(param) is not None:
                options = list(MODULE_PARAM_TO_DICT_LOAD[param]().keys())
            else:
                options = []
            print(f"Options: {options}")
        print("==========================================================")
        
    def print_current_mandatory_and_optional_parameters(self):
        print("Mandatory Parameters:")
        print("")
        for param, desc in self._current_mandatory_params.items():
            print(f"Parameter Value: {param}")
            print(f"Description: {desc}")
            print("Options: TBD!")
            print("")
        print("")
        print("Optional Parameters:")
        print("")
        for param, desc in self._current_optional_params.items():
            print(f"Parameter Value: {param}")
            print(f"Description: {desc}")
            print("Options: TBD!")
            print("")
        print("==========================================================")
            
    def select_module(self, param, param_value):
        if self._currently_selected_modules.get(param) is not None:
            raise NotImplementedError(f"{param} allready selected. reconsidering not implemented yet")
        else:
            self._currently_selected_modules[param] = param_value
            self._load_module_params(param, param_value)
            # module_dict = MODULE_PARAM_TO_DICT_LOAD[param]()
            # input_param_dict = load_module_parameters(module_dict, param_value)
            # print(input_param_dict)
            
if __name__=="__main__":
    sc = ScenarioCreator()
    sc.print_current_mandatory_and_optional_modules()
    sc.print_current_mandatory_and_optional_parameters()
    print("_________________________________________________")
    sc.select_module("network_type", "NetworkBasic")
    sc.print_current_mandatory_and_optional_modules()
    sc.print_current_mandatory_and_optional_parameters()
    print("_________________________________________________")
    sc.select_module("op_module", "PoolingIRSOnly")
    sc.print_current_mandatory_and_optional_modules()
    sc.print_current_mandatory_and_optional_parameters()
