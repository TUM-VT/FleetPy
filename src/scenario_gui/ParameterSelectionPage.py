
import os
from PyQt6.QtWidgets import QMessageBox,QLabel, QWidget,QVBoxLayout,QGridLayout,QScrollArea,QComboBox,QLineEdit
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from os.path import dirname as up

class ParametersSelectionPage(QWidget):
    def __init__(self,sc):
        super().__init__()
        self.setGeometry(100, 100, 600, 600)
        self.sc = sc
        self.man_counter = 0
        self.demand_flag = self.network_flag = False
        self.flag = False
        self.abnormal_params_list = ['network_name','demand_name','rq_file','nr_mod_operators']
        self.vbox = QVBoxLayout()
        self.setWindowTitle("Parameters Selection Page")
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scrollcontent = QWidget(self.scroll)
        self.scrollLayout = QGridLayout(self.scrollcontent)
        self.scrollcontent.setLayout(self.scrollLayout)
        self.man_param_dict, self.op_param_dict = self.sc.get_current_mandatory_and_optional_parameters()
        self.man_params_label = QLabel("Mandatory parameters:", self)
        self.man_params_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.man_params_label.setFont(QFont("Verdana", 15, QFont.Weight.Bold))
        self.scrollLayout.addWidget(self.man_params_label,0,0,1,2)
        self.rows = 1
        self.man_param_boxes = []
        self.current_man_param_labels = [] # to keep track of the labels that are currently displayed
        for param, parameter in self.man_param_dict.items():
            self.current_man_param_labels.append(param)
            self.label = QLabel(param, self)
            self.label.setToolTip(f"{parameter.doc_string} \n Expected dtype: {parameter.type}")
            if param not in self.abnormal_params_list:
                self.textbox = QLineEdit(self)
                self.textbox.resize(280,30)
                self.textbox.textChanged.connect(self.set_flag)
                if parameter.default_value is not None:
                    self.textbox.setText(f"{parameter.default_value}")
            else:
                self.textbox = self.abnormal_params(param,parameter)

            self.man_param_boxes.append(self.textbox)
            self.scrollLayout.addWidget(self.man_param_boxes[-1],self.rows,1,1,2)
            self.scrollLayout.addWidget(self.label,self.rows,0,1,1)
            self.rows += 1
        self.man_rows = self.rows
        self.op_rows = self.man_rows + 2000 # to account for the optional parameters label
        self.op_params_label = QLabel("Optional parameters:", self)
        self.op_params_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.op_params_label.setFont(QFont("Verdana", 15, QFont.Weight.Bold))
        self.scrollLayout.addWidget(self.op_params_label,self.op_rows,0,1,2)
        self.op_rows += 1  
        self.op_param_boxes = []
        self.current_op_param_labels = [] # to keep track of the labels that are currently displayed
        for param, parameter in self.op_param_dict.items():
            self.current_op_param_labels.append(param)
            self.label = QLabel(param, self)
            self.label.setToolTip(f"{parameter.doc_string} \n Expected dtype: {parameter.type}")
            self.scrollLayout.addWidget(self.label,self.op_rows,0,1,1) 
            if param not in self.abnormal_params_list:
                self.textbox = QLineEdit(self)
                self.textbox.resize(280,30)
                self.textbox.textChanged.connect(self.set_flag)
                if parameter.default_value is not None:
                    self.textbox.setText(f"{parameter.default_value}")
            else:
                self.textbox = self.abnormal_params(param,parameter)
            self.op_param_boxes.append(self.textbox)
            self.scrollLayout.addWidget(self.op_param_boxes[-1],self.op_rows,1,1,2)
            self.op_rows += 1    
        self.scroll.setWidget(self.scrollcontent)
        self.vbox.addWidget(self.scroll)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)


    def update(self,sc):

        self.sc = sc
        self.man_module_dict, self.op_module_dict = self.sc.get_current_mandatory_and_optional_modules()
        self.man_param_dict, self.op_param_dict = self.sc.get_current_mandatory_and_optional_parameters()
        for param, parameter in self.man_param_dict.items():
            if param not in self.current_man_param_labels:
                self.current_man_param_labels.append(param)
                self.label = QLabel(param, self)
                self.label.setToolTip(f"{parameter.doc_string} \n Expected dtype: {parameter.type}")
                if param not in self.abnormal_params_list:
                    self.textbox = QLineEdit(self)
                    self.textbox.resize(280,30)
                    self.textbox.textChanged.connect(self.set_flag)
                    if parameter.default_value is not None:
                        self.textbox.setText(f"{parameter.default_value}")
                else:
                    self.textbox = self.abnormal_params(param,parameter)

                self.man_param_boxes.append(self.textbox)
                self.scrollLayout.addWidget(self.man_param_boxes[-1],self.man_rows,1,1,2)
                self.scrollLayout.addWidget(self.label,self.man_rows,0,1,1)
                self.man_rows += 1
        for param, parameter in self.op_param_dict.items():
            if param not in self.current_op_param_labels:
                self.current_op_param_labels.append(param)
                self.label = QLabel(param, self)
                self.label.setToolTip(f"{parameter.doc_string} \n Expected dtype: {parameter.type}")
                if param not in self.abnormal_params_list:
                    self.textbox = QLineEdit(self)
                    self.textbox.resize(280,30)
                    self.textbox.textChanged.connect(self.set_flag)
                    if parameter.default_value is not None:
                        self.textbox.setText(f"{parameter.default_value}")
                else:
                    self.textbox = self.abnormal_params(param,parameter)
                self.op_param_boxes.append(self.textbox)
                self.scrollLayout.addWidget(self.op_param_boxes[-1],self.op_rows,1,1,2)
                self.scrollLayout.addWidget(self.label,self.op_rows,0,1,1)
                self.op_rows += 1

    
    def abnormal_params(self,param,parameter):
        if param == "network_name":
            textbox = QComboBox(self)
            path = os.path.join(os.path.dirname(up(up(__file__))),"data","networks")
            options = [" "] + os.listdir(path)
            textbox.addItems(options)
            textbox.currentIndexChanged.connect(lambda: self.update_flag(textbox.currentText(),"network"))

            return textbox
        elif param == "demand_name":
            textbox = QComboBox(self)
            path = os.path.join(os.path.dirname(up(up(__file__))),"data","demand")
            options = [" "] + os.listdir(path)
            textbox.addItems(options)
            textbox.currentIndexChanged.connect(lambda: self.update_flag(textbox.currentText(),"demand"))
            return textbox
        elif param == "rq_file":
            self.textbox = QComboBox(self)
            self.textbox.setEnabled(False)
            return self.textbox
        elif param == "nr_mod_operators":
            textbox = QLineEdit(self)
            textbox.textChanged.connect(self.nr_mod_operators_changed)
            return textbox
            """
        elif param == "study_name":
            self.textbox = QComboBox(self)
            path = os.path.join(os.path.dirname(up(up(__file__))),"data","demand")
            """

    def nr_mod_operators_changed(self,text):
        if text == "1" or text == "": 
            return
        if int(text) >= 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            prompt = "You have selected more than one operator.\n"
            prompt += "Please make sure to select different parameters for each operator."
            prompt += "Otherwise, the parameters will be the same for all operators."
            prompt += " You can do this by seperating the different parameters by a comma ',' for all parameters starting with op_" 
            msg.setText(prompt)
            msg.setWindowTitle("Warning")
            msg.exec()
    def update_rq(self):
        idx_rq = self.current_man_param_labels.index("rq_file")
        if self.demand_flag and self.network_flag:
            idx_rq = self.current_man_param_labels.index("rq_file")
            idx_demand = self.current_man_param_labels.index("demand_name")
            idx_network = self.current_man_param_labels.index("network_name")
            current_network = self.man_param_boxes[idx_network].currentText()
            current_demand = self.man_param_boxes[idx_demand].currentText()
            if current_demand == " " or current_network == " ":
                self.man_param_boxes[idx_rq].setEnabled(False)
                return
            self.man_param_boxes[idx_rq].setEnabled(True)
            path = os.path.join(os.path.dirname(up(up(__file__))),"data","demand",current_demand,"matched",current_network)
            options = [" "] + os.listdir(path)
            self.man_param_boxes[idx_rq].addItems(options)
        else:
            self.man_param_boxes[idx_rq].setEnabled(False)
    def update_flag(self,box,text):
        if box != " ":
            if text == "demand":
                self.demand_flag = True
            if text == "network":
                self.network_flag = True
        self.update_rq()
        self.set_flag(box)

    def get_params(self):
        labels = []
        params = []
        for i in range(len(self.man_param_boxes)):
            labels.append(self.current_man_param_labels[i])
            try:
                params.append(self.man_param_boxes[i].text())
            except:
                params.append(self.man_param_boxes[i].currentText())
        for i in range(len(self.op_param_boxes)): 
            try:
                if (self.op_param_boxes[i].text() == " " or 
                self.op_param_boxes[i].text() == "" or 
                self.op_param_boxes[i].text() == None 
                or self.op_param_boxes[i].text() == "None"):
                    continue
                params.append(self.op_param_boxes[i].text())
                labels.append(self.current_op_param_labels[i])
            except:
                if (self.op_param_boxes[i].currentText() == " " or 
                self.op_param_boxes[i].currentText() == "" or
                self.op_param_boxes[i].currentText() == None
                or self.op_param_boxes[i].currentText() == "None"):
                    continue
                params.append(self.op_param_boxes[i].currentText())
                labels.append(self.current_op_param_labels[i])
        return labels, params
    #function to set a flag to indicate that the user has selected all the mandatory parameters
    def set_flag(self,text):
        indicator = 0
        for i in range(len(self.man_param_boxes)):
            try:
                label = self.man_param_boxes[i].text()
            except:
                label = self.man_param_boxes[i].currentText()

            if label == " " or label == "" or label == None:
                indicator += 1
                self.flag = False
                return
        if indicator == 0:
            self.flag = True
       # if text != " " and text != "" and text != None:
       #     self.man_counter += 1 
       # if self.man_counter == len(self.man_param_boxes):
       #     self.flag = True
    def get_flag_status(self):
        return self.flag
    def get_sc(self):
        return self.sc
