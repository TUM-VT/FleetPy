import os
from click import option
import pandas as pd

from src.misc.globals import *
from src.FleetSimulationBase import INPUT_PARAMETERS_FleetSimulationBase
from src.misc.init_modules import *
from typing import Dict, Tuple

FLEETPY_PATH = os.path.dirname(os.path.abspath(__file__))

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

#set dictionaries
parameter_docs = input_parameters['Description'].to_dict()  # parameter name -> docu string
parameter_defaults = input_parameters['Default Value'].dropna().to_dict()   # parameter name -> default value (no entry if no default value specified)
parameter_types = input_parameters["Type"].to_dict()    # parameter name -> data type (in string form)

MODULE_PARAM_TO_DICT_LOAD = {
    G_SIM_ENV : get_src_simulation_environments,
    G_NETWORK_TYPE : get_src_routing_engines,
    G_RQ_TYP1 : get_src_request_modules,
    G_OP_MODULE : get_src_fleet_control_modules,
    G_RA_RES_MOD : get_src_reservation_strategies,
    G_OP_CH_M : get_src_charging_strategies,
    G_OP_REPO_M : get_src_repositioning_strategies,
    G_OP_DYN_P_M : get_src_dynamic_pricing_strategies,
    G_OP_DYN_FS_M : get_src_dynamic_fleet_sizing_strategies,
    G_RA_RP_BATCH_OPT: get_src_ride_pooling_batch_optimizers
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
    
def create_study_directories(study_name):
    """ creates all directories within the current directory for a new study
    :param study_name: (str) name of the new study
    """
    protected = ["src", "documentation", "data"]
    if study_name in protected:
        raise IOError("ERROR {} can't be used as study_name!".format(study_name))
    studies_folder = os.path.join(FLEETPY_PATH, "studies")
    if not os.path.isdir(studies_folder):
        print("Initializing Studies Folder {}".format(studies_folder))
        os.mkdir(studies_folder)
    study_folder = os.path.join(studies_folder, study_name)
    if os.path.isdir(study_folder):
        print("Warning: {} already existent!".format(study_name))
    else:
        os.mkdir(study_folder)
        print("creating {}".format(study_folder))
    preprocessing_folder = os.path.join(study_folder, "preprocessing")
    if not os.path.isdir(preprocessing_folder):
        os.mkdir(preprocessing_folder)
        print("creating {}".format(preprocessing_folder))
    scenarios_folder = os.path.join(study_folder, "scenarios")
    if not os.path.isdir(scenarios_folder):
        os.mkdir(scenarios_folder)
        print("creating {}".format(scenarios_folder))
    results_folder = os.path.join(study_folder, "results")
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
        print("creating {}".format(preprocessing_folder))
    evaluation_folder = os.path.join(study_folder, "evaluation")
    if not os.path.isdir(evaluation_folder):
        os.mkdir(evaluation_folder)
        print("creating {}".format(evaluation_folder))
    return scenarios_folder

class Parameter():
    def __init__(self, name, doc_string, type, default_value=None, options=None):
        """ this class collects all necessary information describing a parameter
        :param name: string name of the parameter (oder module)
        :param doc_string: string describing the parameter
        :param type: string describing the expected data type
        :param default_value: (optional) value of the parameter if not actively specified
        :param options: list possible options of the parameter (especially for modules); None, if no options are available"""
        self.name = name
        self.doc_string = doc_string
        self.type = type
        self.default_value = default_value
        self.options = options
        
    def __str__(self):
        return f"name : {self.name} | doc : {self.doc_string} | type : {self.type} | default : {self.default_value} | options : {self.options}"

class ScenarioCreator():
    def __init__(self):
        self.possible_modules = list(MODULE_PARAM_TO_DICT_LOAD.keys())
        
        self._current_mandatory_params = INPUT_PARAMETERS_FleetSimulationBase["input_parameters_mandatory"] # list of mandatory parameters that have not been selected
        self._current_optional_params = INPUT_PARAMETERS_FleetSimulationBase["input_parameters_optional"]   # list of optional parameters that have not been selected
        
        self._current_mandatory_modules = INPUT_PARAMETERS_FleetSimulationBase["mandatory_modules"] # list of mandatory modules that have not been selected
        self._current_optional_modules = INPUT_PARAMETERS_FleetSimulationBase["optional_modules"]   # list of optional modules that have not been selected
        
        self._currently_selected_modules = {}   # module_name -> currently selected value
        self._currently_selected_parameters = {}    # parameter name -> currently selected value
        
        self.parameter_dict : Dict[str, Parameter] = {}    # parameter_name -> parameter (collects all possible parameters in FleetPy)
        for module_name, module_load_fct in MODULE_PARAM_TO_DICT_LOAD.items():
            module_dict = module_load_fct()
            options = list(module_dict.keys())
            self.parameter_dict[module_name] = Parameter(module_name, parameter_docs[module_name], parameter_types[module_name], 
                                                         default_value=parameter_defaults.get(module_name), options=options)
        for parameter_name, doc in parameter_docs.items():
            if self.parameter_dict.get(parameter_name):
                continue
            self.parameter_dict[parameter_name] = Parameter(parameter_name, doc, parameter_types[parameter_name],
                                                            default_value=parameter_defaults.get(parameter_name))
        study_name_str = "this parameter specifies the name of the simulation study. all simulation scenario configurations and simulation results \
            are stored in the folder FleetPy\studies\{study_name}. If this folder does not exist, it will be create automatically!"
        self.parameter_dict["study_name"] = Parameter("study_name", study_name_str, "str")
        self._current_mandatory_params.append("study_name")
            
    def _reset_module_init(self):
        self._current_mandatory_params = INPUT_PARAMETERS_FleetSimulationBase["input_parameters_mandatory"] # list of mandatory parameters that have not been selected
        self._current_optional_params = INPUT_PARAMETERS_FleetSimulationBase["input_parameters_optional"]   # list of optional parameters that have not been selected
        self._current_mandatory_params.append("study_name")
        
        self._current_mandatory_modules = INPUT_PARAMETERS_FleetSimulationBase["mandatory_modules"] # list of mandatory modules that have not been selected
        self._current_optional_modules = INPUT_PARAMETERS_FleetSimulationBase["optional_modules"]   # list of optional modules that have not been selected

        for mod, mod_val in self._currently_selected_modules.items():
            self._load_module_params(mod, mod_val)
        
    def _add_new_params_and_modules(self, input_param_dict):
        """ this method adopts the current list of mandatory/optinal modules/parameters i.e. when a new module is loaded
        the input_parameter_dict is importet from the corresponding module file"""
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
        """ this method loads new module parameters when a module has been selected and looks through the whole inheritance tree
        :param module_param: module specification parameter
        :param module_param_value: selected module name"""
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
            try:
                new_input_param_dict = load_module_parameters(module_dict, inherit_class)
            except ModuleNotFoundError: # TODO
                if inherit_class.endswith("Base") and not inherit_class.startswith("Request"): # TODO!
                    base_p = list(module_dict.values())[0][0].split(".")[:-2]
                    base_p.append(inherit_class)
                    base_p = ".".join(base_p)
                    module_dict[inherit_class] = (base_p, inherit_class)
                    new_input_param_dict = load_module_parameters(module_dict, inherit_class)
                else:
                    raise ModuleNotFoundError(f"no module named {base_p}")
            print(f" -> inherit {inherit_class} : {new_input_param_dict['doc']}")
            self._add_new_params_and_modules(new_input_param_dict) 
            inherit_class = new_input_param_dict["inherit"]
                        
        # if module_param in self._current_mandatory_modules:
        #     self._current_mandatory_modules.remove(module_param)
        # if module_param in self._current_optional_modules:
        #     self._current_optional_modules.remove(module_param)
        print("==========================================================")
            
    def select_module(self, module_param, module_param_value):
        """ this method should be called if a value for a module is selected in the GUI
        TODO reselction currently not possible
        :param module_param: module specification parameter
        :param module_param_value: selected module name"""
        if self._currently_selected_modules.get(module_param) is not None:
            print(f"{module_param} re-selected")
            self._currently_selected_modules[module_param] = module_param_value
            self._reset_module_init()
        else:
            self._currently_selected_modules[module_param] = module_param_value
            self._load_module_params(module_param, module_param_value)
            # module_dict = MODULE_PARAM_TO_DICT_LOAD[param]()
            # input_param_dict = load_module_parameters(module_dict, param_value)
            # print(input_param_dict)
            
    def select_param(self, param, param_value):
        """ this method should be called if a value for a parameter is selected in the GUI
        :param param: parameter name
        :param param_value: selected parameter value"""
        print(f"Select {param_value} for parameter {param}")
        if not param in self._current_mandatory_params and not param in self._current_optional_params \
                and self._currently_selected_parameters.get(param) is None:
            raise EnvironmentError(f"{param} not defined or does not have to be specified!")
        self._currently_selected_parameters[param] = param_value
        # if param in self._current_mandatory_params:
        #     self._current_mandatory_params.remove(param)
        # if param in self._current_optional_params:
        #     self._current_optional_params.remove(param)
            
    def create_filled_scenario_df(self):
        """ this function creates a dataframe from all selected modules and parameters
        a dataframe is only returned if all mandatory parameters and modules have been selected
        TODO this function doesnt save the file to a csv yet
        :return: path to sc_df"""
        print("Created Scenario Table:")            
        sc_df_list = []
        for p, v in self._currently_selected_modules.items():
            sc_df_list.append( {"Parameter" : p, "Value": v})
        for p, v in self._currently_selected_parameters.items():
            if p == "study_name":
                continue
            sc_df_list.append( {"Parameter" : p, "Value": v})
        sc_df = pd.DataFrame(sc_df_list)
        study_name = self._currently_selected_parameters["study_name"]
        scenario_path = create_study_directories(study_name)
        f_p = os.path.join(scenario_path, "scenario_creator_config.csv")
        sc_df.to_csv(f_p, index = False)
        return f_p
            
    def create_shell_scenario_df(self):
        """ this function returns an empty dataframe of all parameters according to the selected modules
        it doesnt return anything in case a mandatory module is not specified yet
        :return: dataframe with columsn Parameter and Value (Value entries are filled with corresponding doc-strings of the parameters)"""
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
    
    def get_current_mandatory_and_optional_modules(self) -> Tuple[Dict[str, Parameter], Dict[str, Parameter]]:
        """ returns two dictionaries specifing currently selectable mandatory and optional parameters
        each dictionary describes the module parameter name -> Paramter object, which collects all information regarding the paramter
        :return: current mandatory module dict, current optional module dict"""
        man_module_dict = {}
        op_module_dict = {}
        for param in self._current_mandatory_modules:
            man_module_dict[param] = self.parameter_dict[param]

        for param in self._current_optional_modules:
            op_module_dict[param] = self.parameter_dict[param]

        return man_module_dict, op_module_dict
        
    def get_current_mandatory_and_optional_parameters(self) -> Tuple[Dict[str, Parameter], Dict[str, Parameter]]:
        """ returns two dictionaries specifing currently selectable mandatory and optional parameters
        each dictionary describes the parameter name -> Paramteer object, which collects all information regarding the parameter
        :return: current mandatory module dict, current optional module dict"""
        man_param_dict = {}
        op_param_dict = {}
        for param in self._current_mandatory_params:
            man_param_dict[param] = self.parameter_dict[param]

        for param in self._current_optional_params:
            op_param_dict[param] = self.parameter_dict[param]

        return man_param_dict, op_param_dict
    
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
