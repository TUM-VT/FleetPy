from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from functools import partial

from typing import TYPE_CHECKING

from click import command
if TYPE_CHECKING:
    from ScenarioCreatorGUI import ScenarioCreatorMainFrame

class ParameterSelectionPage(tk.Frame):
    def __init__(self, parent, controller : ScenarioCreatorMainFrame):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        self.id = controller.mandatory_module
        row_count=1
        col_count=1
        list_of_selected_param = []
        label = tk.Label(self, text = "Parameter Selection Page\n" + controller.mandatory_module.get(), font=controller.titlefont).grid(row=0,column=0, columnspan=4,padx=20, pady=10)

        #Feature 1
        mandatory_param = tk.Label(self, text=controller.mandatory_input_params.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count=row_count+1
        col_count=2
        man_param_dict, op_param_dict = controller.sc.get_current_mandatory_and_optional_parameters()
        parameter_to_input_variable = {}

        for parameter_name, parameter in man_param_dict.items():
            if parameter.type == "int":
                param_var = tk.IntVar()
                param_var.set(controller.default_int) #if the type is integer/float
            else:   
                param_var = tk.StringVar()
                param_var.set(controller.default_text) 
            module_view = tk.Label(self, text=parameter_name, font=controller.normalfont).grid(row=row_count,column=col_count)
            #sel_module = tk.OptionMenu(self,module, *["TBD", "TBD"])
            sel_module = tk.Entry(self, textvariable=param_var)
            callback_func = partial(self.select_param, parameter_name, sel_module)
            sel_module.bind('<Return>', callback_func)
            sel_module.config(width=80)
            sel_module.grid(row=row_count, column=(col_count+1))  
            col_count = 2
            row_count = row_count+1
            list_of_selected_param.append([parameter_name, param_var])                    

        #Feature 2
        optional_param = tk.Label(self, text=controller.optional_input_params.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count = row_count + 1

        for parameter_name, parameter in op_param_dict.items():
            if parameter.type == "int":
                param_var = tk.IntVar()
                param_var.set(controller.default_int) #if the type is integer/float
            else:   
                param_var = tk.StringVar()
                param_var.set(controller.default_text)            
            module_view = tk.Label(self, text=parameter_name, font=controller.normalfont).grid(row=row_count,column=col_count)
            sel_module = tk.Entry(self, textvariable=param_var)
            callback_func = partial(self.select_param, parameter_name, sel_module)
            sel_module.bind('<Return>', callback_func)
            sel_module.config(width=80)
            sel_module.grid(row=row_count, column=(col_count+1))
            col_count = 2
            row_count = row_count+1
            list_of_selected_param.append([parameter_name, param_var])     

        back_button = tk.Button(self, text = "Back to main",
                                command= lambda: controller.up_frame('ModuleSelectionPage')).grid(row=30, column=0, columnspan=4, padx=20, pady=10)
        
        save_button = tk.Button(self, text = "Save to csv and exit",
                                        command= lambda: controller.save_and_exit('ModuleSelectionPage',list_of_selected_param)).grid(row=31, column=0, columnspan=4, padx=20, pady=10)

    def select_param(self, param_name, module, kwargs):
        self.controller.sc.select_param(param_name, module.get())