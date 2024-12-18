from PyQt5 import QtWidgets, QtCore, QtGui
from ui.category_dock import Ui_Form
from fuzzywuzzy import process

from typing import List, Dict, Tuple


class CategoriesDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super().__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        print("connect")
        self.listWidget.itemClicked.connect(self.item_choice)

    def update_categories(self, labels: List[Dict]):
        self.listWidget.clear()
        btngroup = QtWidgets.QButtonGroup(self)

        for label in labels:
            name = label.get("name", "UNKNOW")
            color = label.get("color", "#000000")
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))
            widget = QtWidgets.QWidget()

            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(9, 1, 9, 1)

            label_color = QtWidgets.QLabel()
            label_color.setFixedWidth(10)
            label_color.setStyleSheet("background-color: {};".format(color))
            label_color.setObjectName("label_color")

            label_radio = QtWidgets.QRadioButton("{}".format(name))
            label_radio.setObjectName("label_radio")
            label_radio.toggled.connect(self.radio_choice)

            btngroup.addButton(label_radio)

            if name == "__background__":
                label_radio.setChecked(True)

            layout.addWidget(label_color)
            layout.addWidget(label_radio)
            widget.setLayout(layout)

            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, widget)

    def update_widget(self):
        self.listWidget.clear()
        btngroup = QtWidgets.QButtonGroup(self)
        labels = self.mainwindow.cfg.get("label", [])
        search_text = self.lineEdit_search_category.text()

        name_label_dict = {label.get("name", "UNKNOW"): label for label in labels}

        label_names = [label.get("name", "UNKNOW") for label in labels]

        if search_text == "":
            show_label_names = label_names
        elif search_text.strip(" ") == "":
            show_label_names = label_names
        else:
            matches = process.extract(search_text, label_names, limit=5)
            show_label_names = [name for name, score in matches if score > 0]

        for index in range(len(show_label_names)):
            label = name_label_dict[show_label_names[index]]
            name = label.get("name", "UNKNOW")
            color = label.get("color", "#000000")
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))
            widget = QtWidgets.QWidget()

            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(9, 1, 9, 1)

            label_color = QtWidgets.QLabel()
            label_color.setFixedWidth(10)
            label_color.setStyleSheet("background-color: {};".format(color))
            label_color.setObjectName("label_color")

            label_radio = QtWidgets.QRadioButton("{}".format(name))
            label_radio.setObjectName("label_radio")
            label_radio.toggled.connect(self.radio_choice)

            btngroup.addButton(label_radio)
            if name == "__background__":
                label_radio.setChecked(True)

            layout.addWidget(label_color)
            layout.addWidget(label_radio)
            widget.setLayout(layout)

            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, widget)

    def radio_choice(self):
        sender = self.sender()
        if isinstance(sender, QtWidgets.QRadioButton):
            # sender: QtWidgets.QRadioButton
            if sender.isChecked():
                print("Current category: {}".format(sender.text()))
                self.mainwindow.current_category = sender.text()

    def item_choice(self, item_now):
        print(f"item_choice: {item_now.text()}")
        for index in range(self.listWidget.count()):
            item = self.listWidget.item(index)
            widget = self.listWidget.itemWidget(item)
            label_radio = widget.findChild(QtWidgets.QRadioButton, "label_radio")
            label_radio.setChecked(item == item_now)
