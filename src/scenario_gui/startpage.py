from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui_pages import MainFrame

class StartPage(tk.Frame):
    def __init__(self, parent, controller : MainFrame):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        self.id = controller.mandatory_module
        row_count=1
        col_count=1
        list_of_selected_modules = []
    
        page_heading = tk.Label(self, text = "Start Page", font=controller.titlefont).grid(row=0,column=0, columnspan=4,padx=20, pady=10)

        #Feature 1
        mandatory_module = tk.Label(self, text=controller.mandatory_module.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count=row_count+1
        col_count=2
        man_module_dict, op_module_dict = controller.sc.get_current_mandatory_and_optional_modules()

        for module_name, module_specifications in man_module_dict.items():
            if module_specifications.type == "int":
                module = tk.IntVar()
                module.set(controller.default_int)
            else:
                module = tk.StringVar()
                module.set(module_name) 
            module_view = tk.Label(self, text=module_name, font=controller.normalfont).grid(row=row_count,column=col_count)
            sel_module = tk.OptionMenu(self,module, *module_specifications.options)
            sel_module.config(width=30)
            sel_module.grid(row=row_count, column=(col_count+1))  
            col_count = 2
            row_count = row_count+1
            list_of_selected_modules.append([module_name, module])                   

        #Feature 2
        optional_module = tk.Label(self, text=controller.optional_modules.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count = row_count + 1
        
        for module_name, module_specifications in op_module_dict.items():

            if module_specifications.type == "int":
                module = tk.IntVar()
                module.set(controller.default_int)
            else:
                module = tk.StringVar()
                module.set(controller.default_text) 
            module_view = tk.Label(self, text=module_name, font=controller.normalfont).grid(row=row_count,column=col_count)
            sel_module = tk.OptionMenu(self,module, *module_specifications.options)
            sel_module.config(width=30)
            sel_module.grid(row=row_count, column=(col_count+1))
            col_count = 2
            row_count = row_count+1
            list_of_selected_modules.append([module_name, module])          

        save_and_next_page = tk.Button(self, text = "Save and Next",
                                command= lambda: controller.store_input_args('PageOne', list_of_selected_modules)).grid(row=30, column=0, columnspan=4, padx=20, pady=10)
