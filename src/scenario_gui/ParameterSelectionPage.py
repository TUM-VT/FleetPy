from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from functools import partial

from typing import TYPE_CHECKING

from click import command
if TYPE_CHECKING:
    from ScenarioCreatorGUI import ScenarioCreatorMainFrame
    
class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

class ParameterSelectionPage(tk.Frame):
    def __init__(self, parent, controller : ScenarioCreatorMainFrame):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        self.id = controller.mandatory_module
        row_count=1
        col_count=1
        list_of_selected_param = []
        
        self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        self.frame = tk.Frame(self.canvas, background="#ffffff")
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw",
                                  tags="self.frame")

        self.frame.bind("<Configure>", self.onFrameConfigure)
        
        label = tk.Label(self.frame, text = "Parameter Selection Page\n" + controller.mandatory_module.get(), font=controller.titlefont).grid(row=0,column=0, columnspan=4,padx=20, pady=10)

        #Feature 1
        mandatory_param = tk.Label(self.frame, text=controller.mandatory_input_params.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count=row_count+1
        col_count=2
        man_param_dict, op_param_dict = controller.sc.get_current_mandatory_and_optional_parameters()
        self.parameter_to_input_variable = {}

        for parameter_name, parameter in man_param_dict.items():
            param_var = tk.StringVar()
            # if parameter.type == "int":
            #     param_var = tk.IntVar()
            #     #param_var.set(controller.default_int) #if the type is integer/float
            # else:   
            #     param_var = tk.StringVar()
            #     #param_var.set(controller.default_text) 
            module_view = tk.Label(self.frame, text=parameter_name, font=controller.normalfont)
            module_view.grid(row=row_count,column=col_count)
            parameter_describtion = f"{parameter.doc_string}\nExpected Type : {parameter.type}"
            if parameter.default_value is not None:
                parameter_describtion += f"\nDefault Value : {parameter.default_value}"
            CreateToolTip(module_view, parameter_describtion)
            #sel_module = tk.OptionMenu(self,module, *["TBD", "TBD"])
            sel_module = tk.Entry(self.frame, textvariable=param_var)
            callback_func = partial(self.select_param, parameter_name, sel_module)
            sel_module.bind('<Return>', callback_func)
            sel_module.config(width=40)
            sel_module.grid(row=row_count, column=(col_count+1))  
            col_count = 2
            row_count = row_count+1
            list_of_selected_param.append([parameter_name, param_var])  
            self.parameter_to_input_variable[parameter_name] = param_var                  

        #Feature 2
        optional_param = tk.Label(self.frame, text=controller.optional_input_params.get(), font=controller.normalfont).grid(row=row_count,column=1)
        row_count = row_count + 1

        for parameter_name, parameter in op_param_dict.items():
            param_var = tk.StringVar()
            # if parameter.type == "int":
            #     param_var = tk.IntVar()
            #     #param_var.set(controller.default_int) #if the type is integer/float
            # else:   
            #     param_var = tk.StringVar()
            #     #param_var.set(controller.default_text)            
            module_view = tk.Label(self.frame, text=parameter_name, font=controller.normalfont)
            module_view.grid(row=row_count,column=col_count)
            parameter_describtion = f"{parameter.doc_string}\nExpected Type : {parameter.type}"
            if parameter.default_value is not None:
                parameter_describtion += f"\nDefault Value : {parameter.default_value}"
            CreateToolTip(module_view, parameter_describtion)
            sel_module = tk.Entry(self.frame, textvariable=param_var)
            callback_func = partial(self.select_param, parameter_name, sel_module)
            sel_module.bind('<Return>', callback_func)
            sel_module.config(width=40)
            sel_module.grid(row=row_count, column=(col_count+1))
            col_count = 2
            row_count = row_count+1
            list_of_selected_param.append([parameter_name, param_var])   
            self.parameter_to_input_variable[parameter_name] = param_var  

        back_button = tk.Button(self.frame, text = "Back to main",
                                command= controller.to_module_selection_page).grid(row=row_count, column=0, columnspan=4, padx=20, pady=10)
        
        save_button = tk.Button(self.frame, text = "Save to csv and exit",
                                        command= lambda: controller.save_and_exit('ModuleSelectionPage',self.parameter_to_input_variable)).grid(row=row_count+1, column=0, columnspan=4, padx=20, pady=10)

    def select_param(self, param_name, module, kwargs):
        self.controller.sc.select_param(param_name, module.get())
        
    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))