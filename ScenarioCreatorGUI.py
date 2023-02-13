import sys
import os

from PyQt6.QtWidgets import QApplication, QMainWindow,QLabel, QPushButton, QStackedWidget, QWidget,QMessageBox,QGridLayout
from PyQt6.QtGui import QIcon,QPixmap,QFont
from PyQt6.QtCore import Qt
import pandas as pd
import numpy as np

from src.scenario_gui.ModuleSelectionPage import ModuleSelectionPage
from src.scenario_gui.ParameterSelectionPage import ParametersSelectionPage
from scenario_creator import ScenarioCreator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #add the icon to the main window
        #create grid layout with spacing between items
        grid = QGridLayout()
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.pixmap = QPixmap(QPixmap(os.path.join(os.path.dirname(__file__),'src','scenario_gui', 'icon.png')))
        self.pixmap = self.pixmap.scaled(64, 64)
        self.icon = QLabel(self)
        self.icon.setPixmap(self.pixmap)
        #change size of icon
        self.icon.resize(64, 64)
        #fix the icon to the topleft corner
        self.icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        grid.addWidget(self.icon, 0, 0)
        #add a text label to the main window
        self.label = QLabel("Scenario Creator for Fleetpy", self)
        #align the text label to the top left corner
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)     
        #change the font size of the text label
        self.label.setFont(QFont("Verdana", 20, QFont.Weight.Bold))
        # add the text label to the main window next to the icon
        grid.addWidget(self.label, 0, 1)
        #create a help button
        self.help_button = QPushButton("Help", self)
        self.help_button.setToolTip("Click to see the help page")
        self.help_button.clicked.connect(self.help)
        grid.addWidget(self.help_button, 0, 2)
        

        self.setWindowTitle("FleetPy")
        self.setGeometry(100, 100, 600, 600)
        self.sc = ScenarioCreator()
        self.window1 = ModuleSelectionPage(self.sc)
        self.window2 = ParametersSelectionPage(self.sc)
        
        
        #create stacked widget to hold two pages
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.window1)
        self.stacked_widget.addWidget(self.window2)
        grid.addWidget(self.stacked_widget, 1, 1, 1, 2)
        
        self.button = QPushButton("Next", self)
        self.button.setToolTip("Click to go to the next page")
        self.button.clicked.connect(self.next_page)
        grid.addWidget(self.button, 2, 1)
        self.save_button = QPushButton("Save", self)
        self.save_button.setToolTip("Click to save the parameters")
        self.save_button.clicked.connect(self.save)
        grid.addWidget(self.save_button, 3, 1)

        self.main_widget.setLayout(grid)
        self.show()


    def next_page(self):
        #create a message box to show the user that they have not selected all the mandatory modules
        if self.stacked_widget.currentIndex() == 0 and not self.window1.flag:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("You have not selected all the mandatory modules")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()


        if self.stacked_widget.currentIndex() == 0 and self.window1.flag:
            sc = self.window1.get_module_dict()
            self.window2.update(sc)
            self.stacked_widget.setCurrentIndex(1)
            self.button.setText("Go back")
            self.modules,self.labels = self.window1.get_modules()

        else:
            #self.window2 = None
            self.stacked_widget.setCurrentIndex(0)
            self.button.setText("Next")
    
    def save(self):
        flag = self.window2.get_flag_status()
        if not flag or not self.window1.flag:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("You have not selected all the mandatory modules")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        else:
            self.labels_for_modules, self.modules = self.window1.get_modules()
            self.labels_for_params, self.params = self.window2.get_params()
            study_name_idx = self.labels_for_params.index('study_name')
            scenario_name_idx = self.labels_for_params.index('scenario_name')
            study_name = self.params[study_name_idx]
            scenario_name = self.params[scenario_name_idx]
            #check if a folder is laready created with the name study_name
            if os.path.exists(os.path.join(os.path.dirname(__file__),'studies',study_name)):
                path = os.path.join(os.path.dirname(__file__),'studies',study_name)
            if not os.path.exists(os.path.join(os.path.dirname(__file__),'studies',study_name)):
                path = os.path.join(os.path.dirname(__file__),'studies',study_name)
                os.mkdir(path)
            file_path = os.path.join(path, scenario_name)
            df = pd.DataFrame()
            labels = self.labels_for_modules + self.labels_for_params
            entries = self.modules + self.params
            df['labels'] = labels
            df['entries'] = entries 
            #drop the rows that contains the string 'None'
            df['entries'].replace('None', np.nan, inplace=True)
            df['entries'].replace('', np.nan, inplace=True)
            #drop the rows that contains the string 'NO SELECTION'
            df['entries'].replace('NO SELECTION', np.nan, inplace=True)
            df.dropna()
            df.to_csv(file_path + '.csv',header=False, index=False)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("The scenario file has been saved to the selected study folder")
            msg.setWindowTitle("Success")
            msg.exec()
    def help(self):
        #create a pop up window to show the help page
        window = QMessageBox()
        window.setIcon(QMessageBox.Icon.Information)
        prompt = "<br>1. Please select the modules that you want to use in your scenario.<br>"
        prompt += "<br>2. Please make sure that all the mandatory modules are selected.<br>"
        prompt += "<br>3. Then, click on the 'Next' button to go to the next page.<br>"
        prompt += "<br>4. Select the parameters that you want to use in your scenario.<br>"
        prompt += "<br>5. Click on the 'Save' button to save the scenario file.<br>"
        prompt += "<br>6. Click on the 'Go back' button to go back to the previous page.<br>"
        prompt += "<br>7. The scenario file will be saved in the selected study folder.<br>"
        prompt += "<br>For more information, please visit the FleetPy <a href='https://github.com/TUM-VT/FleetPy'> Github page </a>"

        #add a hyperlink to a string
        window.setTextFormat(Qt.TextFormat.RichText)
        window.setText(prompt)
        window.exec()

            
            

        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    #window = ModuleSelectionPage()
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__),'src','scenario_gui', 'icon.png')))
    sys.exit(app.exec())
