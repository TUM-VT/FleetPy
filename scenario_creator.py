import os
from click import option
import pandas as pd

from src.misc.globals import *
from src.FleetSimulationBase import INPUT_PARAMETERS_FleetSimulationBase
from src.misc.init_modules import *

#read md table into dataframe
INPUT_PARAMETERS_PATH = os.path.join(os.path.dirname(__file__), "Input_Parameters.md")
input_parameters = pd.read_table(INPUT_PARAMETERS_PATH, sep="|", header=0, index_col=1, skipinitialspace=True)
input_parameters = input_parameters.dropna(axis=1, how='all')
input_parameters = input_parameters.iloc[1:]
input_parameters.columns = input_parameters.columns.str.strip()
input_parameters["Parameter"] = input_parameters.index
for c in input_parameters.columns:
    input_parameters[c] = input_parameters[c].str.strip()
input_parameters.set_index("Parameter", inplace=True)

#set dictionariey
parameter_docs = input_parameters['Description'].to_dict()
parameter_defaults = input_parameters['Default Value'].dropna().to_dict()
parameter_types = input_parameters["Type"].to_dict()

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
        self._currently_selected_parameters = {}
        
    def _add_new_params_and_modules(self, input_param_dict):
        for mandatory_param in input_param_dict["input_parameters_mandatory"]:
            if not mandatory_param in self._current_mandatory_params:
                self._current_mandatory_params.append(mandatory_param)
            if mandatory_param in self._current_optional_params:
                self._current_optional_params.remove(mandatory_param)
        for optional_param in input_param_dict["input_parameters_optional"]:
            if not optional_param in self._current_mandatory_params:
                if not optional_param in self._current_optional_params:
                    self._current_optional_params.append(optional_param)
        for mandatory_module in input_param_dict["mandatory_modules"]:
            if not mandatory_module in self._current_mandatory_modules:
                self._current_mandatory_modules.append(mandatory_module)
            if mandatory_module in self._current_optional_modules:
                self._current_optional_modules.remove(mandatory_module)
        for optional_module in input_param_dict["optional_modules"]:
            if not optional_module in self._current_mandatory_modules:
                if not optional_module in self._current_optional_modules:
                    self._current_optional_modules.append(optional_module)
        
    def _load_module_params(self, module_param, module_param_value):
        print("")
        print(f"load parameters for module {module_param_value}!")
        module_dict = MODULE_PARAM_TO_DICT_LOAD[module_param]()
        input_param_dict = load_module_parameters(module_dict, module_param_value)
        
        self._add_new_params_and_modules(input_param_dict)
        
        inherit_class = input_param_dict["inherit"]    
        while inherit_class is not None:
            if inherit_class.endswith("Base"): # TODO!
                if not inherit_class.startswith("Request"):
                    base_p = list(module_dict.values())[0][0].split(".")[:-1]
                    base_p.append(inherit_class)
                    base_p = ".".join(base_p)
                    module_dict[inherit_class] = (base_p, inherit_class)
                else:
                    module_dict[inherit_class] = ("src.demand.TravelerModels", inherit_class)
                
            new_input_param_dict = load_module_parameters(module_dict, inherit_class)
            print(f" -> inherit {inherit_class} : {new_input_param_dict['doc']}")
            self._add_new_params_and_modules(new_input_param_dict) 
            inherit_class = new_input_param_dict["inherit"]
                        
        if module_param in self._current_mandatory_modules:
            self._current_mandatory_modules.remove(module_param)
        if module_param in self._current_optional_modules:
            self._current_optional_modules.remove(module_param)
        print("==========================================================")
                
        
    def print_current_mandatory_and_optional_modules(self):
        print("Mandatory Modules:")
        print("")
        for param in self._current_mandatory_modules:
            desc = parameter_docs[param]
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
        for param in self._current_optional_modules:
            desc = parameter_docs[param]
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
        for param in self._current_mandatory_params:
            desc = parameter_docs[param]
            print(f"Parameter Value: {param}")
            print(f"Description: {desc}")
            print("Options: TBD!")
            print("")
        print("")
        print("Optional Parameters:")
        print("")
        for param in self._current_optional_params:
            desc = parameter_docs[param]
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
            
    def select_param(self, param, param_value):
        if not param in self._current_mandatory_params and not param in self._current_optional_params:
            raise EnvironmentError(f"{param} not defined or does not have to be specified!")
        self._currently_selected_parameters[param] = param_value
        if param in self._current_mandatory_params:
            self._current_mandatory_params.remove(param)
        if param in self._current_optional_params:
            self._current_optional_params.remove(param)
            
    def create_filled_scenario_df(self):
        print("")
        print("___________________________________________________")
        print("")
        can_be_created = True
        if len(self._current_mandatory_modules) != 0:
            print("To be specified: {}".format(self._current_mandatory_modules))
            can_be_created = False
        if len(self._current_mandatory_params) != 0:
            print("To be specified: {}".format(self._current_mandatory_params))
            can_be_created = False
        if can_be_created:
            print("Created Scenario Table:")            
            sc_df_list = []
            for p, v in self._currently_selected_modules.items():
                sc_df_list.append( {"Parameter" : p, "Value": v})
            for p, v in self._currently_selected_parameters.items():
                sc_df_list.append( {"Parameter" : p, "Value": v})
            sc_df = pd.DataFrame(sc_df_list)
            print(sc_df)
            return sc_df
            
    def create_shell_scenario_df(self):
        print("")
        print("___________________________________________________")
        print("")
        if len(self._current_mandatory_modules) != 0:
            print("To be specified: {}".format(self._current_mandatory_modules))
            return
        if len(self._current_optional_modules) != 0:
            print("Not specified modules: {}".format(self._current_optional_modules))
            print(" -> input parameters for these modules are not included in the shell scenario table!")
        print("")
        print("Shell scenario table:")
        sc_df_list = []
        for p, v in self._currently_selected_modules.items():
            sc_df_list.append( {"Parameter" : p, "Value": v})
        for p, v in self._currently_selected_parameters.items():
            sc_df_list.append( {"Parameter" : p, "Value": v})
        for p in self._current_mandatory_params + self._current_optional_params:
            v = f"TO BE SPECIFIED: {parameter_docs[p]} | Type {parameter_types[p]}"
            if parameter_defaults.get(p) is not None:
                v += " | Default {}".format(parameter_defaults[p])
            if p in self._current_optional_params:
                v = "(OPTIONAL) " + v
            sc_df_list.append( {"Parameter" : p, "Value": v})
        sc_df = pd.DataFrame(sc_df_list)
        print(sc_df)
        return sc_df
        
            
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
    
    sc.select_module("sim_env", "ImmediateDecisionsSimulation")
    sc.select_module("rq_type", "BasicRequest")
    
    sc.create_shell_scenario_df()
    
    sc.select_param('scenario_name', "Test_Scenario")
    sc.select_param('start_time', 0)
    sc.select_param("end_time", 3600)
    sc.select_param("nr_mod_operators", 1)
    sc.select_param("random_seed", 123)
    sc.select_param("network_name", "TestNetwork")
    sc.select_param("demand_name", "TestDemand")
    sc.select_param("rq_file", "TestRqFile")
    sc.select_param("op_vr_control_func_dict", "inputForObjFunc")
    
    sc_df = sc.create_filled_scenario_df()
    sc_df.to_csv(r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\FleetPy\studies\SUMO_Grafing_test\scenarios\test_cfg.csv', index=False)
