from PyQt5 import QtWidgets
from widgets.mainwindow import MainWindow
import sys

import torch
torch.cuda.is_available()

from qt_material import apply_stylesheet
# apply_stylesheet(app, theme='light_lightgreen.xml')

import qdarkstyle 

def main():
    print('start')
    app = QtWidgets.QApplication([''])
    mainwindow = MainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

