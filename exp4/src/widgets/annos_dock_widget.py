from PyQt5 import QtWidgets, QtCore, QtGui
from ui.category_dock import Ui_Form
from typing import List, Dict, Tuple


class AnnosDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(AnnosDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        