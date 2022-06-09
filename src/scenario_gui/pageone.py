from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui_pages import MainFrame

class PageOne(tk.Frame):
    def __init__(self, parent, controller : MainFrame):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        self.id = controller.mandatory_module
        row_count=1
        col_count=1

        label = tk.Label(self, text = "Page One\n" + controller.mandatory_module.get(), font=controller.titlefont).grid(row=0,column=0, columnspan=4,padx=20, pady=10)

        #Feature 1
        mandatory_param = tk.Label(self, text=controller.mandatory_input_params.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count=row_count+1
        col_count=2
        man_param_dict, op_param_dict = controller.sc.get_current_mandatory_and_optional_parameters()
        parameter_to_input_variable = {}

        for parameter_name, parameter in man_param_dict.items():
            param_var = tk.StringVar()
            param_var.set(parameter.doc_string) 
            module_view = tk.Label(self, text=parameter_name, font=controller.normalfont).grid(row=row_count,column=col_count)
            #sel_module = tk.OptionMenu(self,module, *["TBD", "TBD"])
            sel_module = tk.Entry(self, textvariable=param_var)
            sel_module.config(width=80)
            sel_module.grid(row=row_count, column=(col_count+1))  
            col_count = 2
            row_count = row_count+1                     

        #Feature 2
        optional_param = tk.Label(self, text=controller.optional_input_params.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count = row_count + 1

        for parameter_name, parameter in op_param_dict.items():
            param_var = tk.StringVar()
            param_var.set(parameter.doc_string) 
            module_view = tk.Label(self, text=parameter_name, font=controller.normalfont).grid(row=row_count,column=col_count)
            sel_module = tk.Entry(self, textvariable=param_var)
            sel_module.config(width=80)
            sel_module.grid(row=row_count, column=(col_count+1))
            col_count = 2
            row_count = row_count+1      

        back_button = tk.Button(self, text = "Back to main",
                                command= lambda: controller.up_frame('StartPage')).grid(row=30, column=0, columnspan=4, padx=20, pady=10)                                
        save_button = tk.Button(self, text = "Save to csv",
                                        command= lambda: controller.up_frame('StartPage')).grid(row=31, column=0, columnspan=4, padx=20, pady=10)
