from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from functools import partial

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ScenarioCreatorGUI import ScenarioCreatorMainFrame
    
NONSELECTION = "None"

class ModuleSelectionPage(tk.Frame):
    def __init__(self, parent, controller : ScenarioCreatorMainFrame):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        self.id = controller.mandatory_module
        row_count=1
        col_count=1
        list_of_selected_modules = []
        self.created_modules = {}
    
        page_heading = tk.Label(self, text = "Module Selection Page", font=controller.titlefont).grid(row=0,column=0, columnspan=4,padx=20, pady=10)

        #Feature 1
        mandatory_module = tk.Label(self, text=controller.mandatory_module.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count=row_count+1
        col_count=2
        man_module_dict, op_module_dict = controller.sc.get_current_mandatory_and_optional_modules()

        for module_name in self.controller.sc.possible_modules:
            module = tk.StringVar()
            module.set(module_name)
            options = []
            if man_module_dict.get(module_name) is not None:
                options = man_module_dict[module_name].options
            elif op_module_dict.get(module_name) is not None:
                options = [NONSELECTION] + op_module_dict[module_name].options
            module_view = tk.Label(self, text=module_name, font=controller.normalfont).grid(row=row_count,column=col_count)
            callback_func = partial(self.select_module, module_name)
            if len(options) > 0:
                sel_module = tk.OptionMenu(self,module, *options, command=callback_func)
            else:
                sel_module = tk.OptionMenu(self,module, 0, command=callback_func)
            sel_module.config(width=30)
            sel_module.grid(row=row_count, column=(col_count+1))  
            if len(options) == 0:
                sel_module.configure(state="disabled")
            col_count = 2
            row_count = row_count+1
            list_of_selected_modules.append([module_name, module])
            self.created_modules[module_name] = (module, sel_module, callback_func)                   

        # #Feature 2
        # optional_module = tk.Label(self, text=controller.optional_modules.get(), font=controller.normalfont).grid(row=row_count,column=1)
        # row_count = row_count + 1
        
        # for module_name, module_specifications in op_module_dict.items():

        #     if module_specifications.type == "int":
        #         module = tk.IntVar()
        #         module.set(controller.default_int)
        #     else:
        #         module = tk.StringVar()
        #         module.set(controller.default_text) 
        #     module_view = tk.Label(self, text=module_name, font=controller.normalfont).grid(row=row_count,column=col_count)
        #     callback_func = partial(self.select_module, module_name)
        #     sel_module = tk.OptionMenu(self,module, *module_specifications.options, command=callback_func)
        #     sel_module.config(width=30)
        #     sel_module.grid(row=row_count, column=(col_count+1))
        #     col_count = 2
        #     row_count = row_count+1
        #     list_of_selected_modules.append([module_name, module]) 
        #     self.created_modules[module_name] = (module, sel_module)         

        save_and_next_page = tk.Button(self, text = "Save and Next",
                                command= lambda: controller.store_input_args('ParameterSelectionPage', list_of_selected_modules)).grid(row=30, column=0, columnspan=4, padx=20, pady=10)
        
    def select_module(self, module, input):
        print(f"select {input}")
        if input != NONSELECTION:
            self.controller.sc.select_module(module, input)
            self.update_module_options()
        
    def update_module_options(self):
        print("update menu")
        man_module_dict, op_module_dict = self.controller.sc.get_current_mandatory_and_optional_modules()
        for module_name, var_mod in self.created_modules.items():
            var, module, callback_func = var_mod
            #print(module_name, var.get())
            module['menu'].delete(0, 'end')
            options = []
            if man_module_dict.get(module_name) is not None:
                options = man_module_dict[module_name].options
            elif op_module_dict.get(module_name) is not None:
                options = [NONSELECTION] + op_module_dict[module_name].options
            for choice in options:
                module['menu'].add_command(label=choice, command=tk._setit(var, choice, callback=callback_func))
            if len(options) == 0:
                module.configure(state="disabled")
            else:
                module.configure(state="active")

