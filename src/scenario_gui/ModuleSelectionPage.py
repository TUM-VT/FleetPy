
from PyQt6.QtWidgets import QLabel, QWidget, QGridLayout, QComboBox
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt



#create a second window called ModuleSelectionPage
class ModuleSelectionPage(QWidget):
    def __init__(self,sc):
        super().__init__()
        self.setWindowTitle("ModuleSelectionPage")
        self.grid = QGridLayout()
        self.mandatory_modules = QLabel("Mandatory modules:", self)
        self.mandatory_modules.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.mandatory_modules.setFont(QFont("Verdana", 15, QFont.Weight.Bold))
        self.grid.addWidget(self.mandatory_modules,0,0,1,2)
        self.setGeometry(100, 100, 400, 400)
        self.sc = sc
        self.flag = False
        self.opt_modules_loaded = False
        self.man_module_dict, self.op_module_dict = self.sc.get_current_mandatory_and_optional_modules()
        # create a text label to show the mandatory module
        self.row = 0
        self.man_labels = []
        self.man_combos = []     
        for module_name in self.sc.possible_modules:
            options = []
            if self.man_module_dict.get(module_name):
                
                options = ["NO SELECTION"] + self.man_module_dict[module_name].options
                
                self.label = QLabel(module_name, self)
                self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
                self.man_labels.append(self.label)
                self.grid.addWidget(self.man_labels[self.row],self.row+1,0,1,1)
                #create options menu 
                self.combo = QComboBox(self)
                self.combo.addItems(options)
                self.combo.currentIndexChanged.connect(self.onActivated)
                self.man_combos.append(self.combo)
                self.grid.addWidget(self.man_combos[self.row],self.row+1,1,1,3)
                self.row += 1
        self.optional_modules = QLabel("Optional modules:", self)
        self.optional_modules.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.optional_modules.setFont(QFont("Verdana", 15, QFont.Weight.Bold))
        self.optional_modules.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.grid.addWidget(self.optional_modules,self.row+1,0,1,2)
        self.setLayout(self.grid)
    #create a function to load optional modules
    def load_optional_modules(self):
        self.row += 1
        
        self.op_labels = []
        self.op_combos = []
        self.row2 = 0
        for module_name in self.sc.possible_modules:
            options = []
            

            if self.man_module_dict.get(module_name) == None and self.op_module_dict.get(module_name):
                self.opt_modules_loaded = True
                options = ["NO SELECTION"] + self.op_module_dict[module_name].options
                self.label = QLabel(module_name, self)
                self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
                self.op_labels.append(self.label)
                self.grid.addWidget(self.op_labels[self.row2],self.row+1,0,1,1)
                #create options menu 
                self.combo_op = QComboBox(self)
                self.combo_op.addItems(options)
                self.combo_op.currentIndexChanged.connect(self.onActivatedOp)
                #disable the optional modules
                #self.combo_op.setEnabled(False)
                self.op_combos.append(self.combo_op)
                self.grid.addWidget(self.op_combos[self.row2],self.row+1,1,1,3)
                self.row += 1
                self.row2 += 1
        self.setLayout(self.grid)

    def onActivated(self, text):
        try:
            self.man_module_dict, self.op_module_dict = self.sc.get_current_mandatory_and_optional_modules()
        except:
            pass

        for i in range(len(self.man_combos)):
            if self.man_combos[i].currentText() == "NO SELECTION":
                self.flag = False
                return
            else:
                self.flag = True
        for idx, label in enumerate(self.man_labels):
            input = self.man_combos[idx].currentText()
            if input != "NO SELECTION":
                try:
                    self.sc.select_module(self.man_labels[idx].text(), self.man_combos[idx].currentText())
                except:
                    pass
        try:
            self.man_module_dict, self.op_module_dict = self.sc.get_current_mandatory_and_optional_modules()
        except:
            pass      
        if self.opt_modules_loaded == False:
            self.load_optional_modules()


    def onActivatedOp(self, text):
        for idx, label in enumerate(self.op_labels):
            input = self.op_combos[idx].currentText()
            if input != "NO SELECTION":
                try: 
                    self.sc.select_module(self.op_labels[idx].text(), self.op_combos[idx].currentText())
                except:
                    pass
        self.man_module_dict, self.op_module_dict = self.sc.get_current_mandatory_and_optional_modules()


    
    def update_selections(self):
        if len(self.combos_op) == 0:
            return
        for i in range(len(self.combos_op)):

            if self.combos_op[i].currentText() != "NO SELECTION":
                self.sc.select_module(self.labels[i].text(), self.combos_op[i].currentText())
            self.combos_op[i].options = ["NO SELECTION"] + self.op_module_dict[self.labels[i].text()].options
        self.sc.update_module_selections()


    def get_modules(self):
        man_modules = []
        for i in range(len(self.man_combos)):
            man_modules.append(self.man_combos[i].currentText())
        op_modules = []
        for i in range(len(self.op_combos)):
            if self.op_combos[i].currentText() != "NO SELECTION":
                op_modules.append(self.op_combos[i].currentText())
        labels = []
        for i in range(len(self.man_labels)):
            labels.append(self.man_labels[i].text())
        for i in range(len(self.op_labels)):
            if self.op_combos[i].currentText() != "NO SELECTION":
                labels.append(self.op_labels[i].text())
        modules = man_modules + op_modules
        return labels, modules

    def get_module_dict(self):
        
        return self.sc

    


        

        
