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
            #self.main_modules =
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
            


            """
            #create a dataframe and fill it with modules names and labels 
            df = pd.DataFrame()
            df['Modules'] = self.modules
            df['Labels'] = self.labels_for_modules
            #create a dataframe and fill it with parameters and labels
            df2 = pd.DataFrame()
            df2['Parameters'] = self.params
            df2['Labels'] = self.labels_for_params
            #concatinate the two dataframes
            df = pd.concat([df, df2], axis=1)
            #drop the rows with empty or None values
            df = df.dropna()
            #drop the rows that contains the string 'None'
            df = df[~df['Parameters'].str.contains('None')]
            #save the dataframe to a csv file
            df.to_csv('config.csv')
            
            #concat the two lists
            #self.labels_for_modules.extend(self.labels_for_params)
            #self.modules.extend(self.params)

            df = pd.DataFrame()
            df['labels'] = self.labels_for_modules 
            df['modules'] = self.modules
            df.dropna()
            #drop the rows that contains the string 'None'
            df = df[~df['modules'].str.contains('None')]
            df.to_csv('config.csv')
            """
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
            df.to_csv('config.csv',header=False, index=False)

            
            

        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    #window = ModuleSelectionPage()
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__),'src','scenario_gui', 'icon.png')))
    sys.exit(app.exec())
