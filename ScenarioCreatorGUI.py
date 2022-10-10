import tkinter as tk
from tkinter import font as tkfont
from scenario_creator import ScenarioCreator
from src.scenario_gui.ModuleSelectionPage import ModuleSelectionPage
from src.scenario_gui.ParameterSelectionPage import ParameterSelectionPage
import csv
# build pages all together and raise the page we need over other pages

class ScenarioCreatorMainFrame(tk.Tk):
    """
    Frame object which will hold all other pages
    -acts as a controller of all the pages
    """
    def __init__(self, *args,  **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.sc = ScenarioCreator()

        self.titlefont = tkfont.Font(family = 'Verdana', size = 12,
                                        weight = "bold", slant='roman')
        self.normalfont = tkfont.Font(family = 'Verdana', size = 10,
                                        slant='roman')
        self.container = tk.Frame()
        self.container.grid(row=0, column=0, sticky='nesw')
        
        #default values
        self.default_text = "enter string"
        self.default_int = 0.0

        #store selected modules and parameters
        self.selected_modules_and_param = {}

        #startpage
        self.mandatory_module = tk.StringVar()
        self.mandatory_module.set("Mandatory Modules")
        self.optional_modules = tk.StringVar()
        self.optional_modules.set("Optional Modules")

        #page 1
        self.mandatory_input_params = tk.StringVar()
        self.mandatory_input_params.set("Mandatory Params")
        self.optional_input_params = tk.StringVar()
        self.optional_input_params.set("Optional Params")        

        self.page_listing = {} #stores information of the pages
            
        self.module_selection_page = ModuleSelectionPage(parent=self.container, controller = self)
        self.module_selection_page.grid(row=0, column = 0, sticky="nesw")
        self.module_selection_page.tkraise()

        #self.up_frame('ModuleSelectionPage')

    def to_parameter_selection_page(self):
        all_mandatory_selected = True
        for mand_mod in self.sc._current_mandatory_modules:
            if self.sc._currently_selected_modules.get(mand_mod) is None:
                all_mandatory_selected = False
        if all_mandatory_selected:
            page = ParameterSelectionPage(self.container, controller=self)
            page.grid(row=0, column = 0, sticky="nesw")
            page.tkraise()
        else:
            from tkinter import messagebox
            messagebox.showerror("Error", "Not all mandatory modules selected!")
        
    def to_module_selection_page(self):
        self.module_selection_page.tkraise()

    def store_input_args(self, page_name, modules):
        """Store input arguments in a dictionary"""  
        self.to_parameter_selection_page()     

    def save_and_exit(self, page_name, param_to_var ):
        for param, var in param_to_var.items():
            val = var.get()
            if val != "":
                self.sc.select_param(param, val)
        all_mandatory_params_selected = True
        for mand_param in self.sc._current_mandatory_params:
            if self.sc._currently_selected_parameters.get(mand_param) is None:
                all_mandatory_params_selected = False
        if all_mandatory_params_selected:
            f_p = self.sc.create_filled_scenario_df()
            from tkinter import messagebox
            messagebox.showinfo("Config created!", f"The config-file is saved at {f_p}")
            exit()
        else:
            from tkinter import messagebox
            messagebox.showerror("Error", "Not all mandatory parameters selected!")


if __name__ == '__main__':

    app = ScenarioCreatorMainFrame()
    app.mainloop()

    #default values to the params
    #set params type
