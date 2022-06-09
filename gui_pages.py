import tkinter as tk
from tkinter import font as tkfont
from scenario_creator import ScenarioCreator
from src.scenario_gui.startpage import StartPage
from src.scenario_gui.pageone import PageOne
# build pages all together and raise the page we need over other pages

class MainFrame(tk.Tk):
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

        for p in (StartPage, PageOne):
            page_name = p.__name__
            frame = p(parent=container, controller = self)
            frame.grid(row=0, column = 0, sticky="nesw")
            self.page_listing[page_name] = frame

        self.up_frame('StartPage')

    def up_frame(self, page_name):
        page = self.page_listing[page_name]
        page.tkraise()

if __name__ == '__main__':

    sc = ScenarioCreator()
    app = MainFrame()
    app.mainloop()

    #default values to the params
    #set params type
