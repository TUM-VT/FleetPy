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
        container = tk.Frame()
        container.grid(row=0, column=0, sticky='nesw')
        
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

        for p in (ModuleSelectionPage, ParameterSelectionPage):
            page_name = p.__name__
            frame = p(parent=container, controller = self)
            frame.grid(row=0, column = 0, sticky="nesw")
            self.page_listing[page_name] = frame

        self.up_frame('ModuleSelectionPage')

    def up_frame(self, page_name):
        """Raise the given page"""
        page = self.page_listing[page_name]
        page.tkraise()

    def store_input_args(self, page_name, modules):
        """Store input arguments in a dictionary"""  

        for module in modules:
            self.selected_modules_and_param[module[0]] = module[1].get()                       
        self.up_frame(page_name)

    def save_to_csv(self,page_name, modules):
        """Saves the selected modules/param to a csv"""
        self.store_input_args(page_name, modules)
        csv_file = "scenario_selected.csv"
        csv_columns = ['Parameter','Value']
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for key in self.selected_modules_and_param.keys():
                    csvfile.write("%s, %s\n" % (key, self.selected_modules_and_param[key]))
            # csvfile.close()
        except IOError:
            print("I/O error")        

    def save_and_exit(self, page_name, modules ):

        self.sc.create_filled_scenario_df()
        self.save_to_csv(page_name, modules)
        exit()


if __name__ == '__main__':

    app = ScenarioCreatorMainFrame()
    app.mainloop()

    #default values to the params
    #set params type
