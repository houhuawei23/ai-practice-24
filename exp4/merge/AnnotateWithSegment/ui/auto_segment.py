# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/mnt/disk2/PycharmProjects/ISAT_with_segment_anything/ISAT/ui/auto_segment.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1000, 680)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        Dialog.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_image_dir = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_image_dir.setObjectName("lineEdit_image_dir")
        self.horizontalLayout.addWidget(self.lineEdit_image_dir)
        self.pushButton_image_dir = QtWidgets.QPushButton(self.widget)
        self.pushButton_image_dir.setObjectName("pushButton_image_dir")
        self.horizontalLayout.addWidget(self.pushButton_image_dir)
        self.verticalLayout.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(Dialog)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_label_dir = QtWidgets.QLineEdit(self.widget_2)
        self.lineEdit_label_dir.setObjectName("lineEdit_label_dir")
        self.horizontalLayout_2.addWidget(self.lineEdit_label_dir)
        self.pushButton_label_dir = QtWidgets.QPushButton(self.widget_2)
        self.pushButton_label_dir.setObjectName("pushButton_label_dir")
        self.horizontalLayout_2.addWidget(self.pushButton_label_dir)
        self.verticalLayout.addWidget(self.widget_2)
        self.widget_3 = QtWidgets.QWidget(Dialog)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_save_dir = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_save_dir.setObjectName("lineEdit_save_dir")
        self.horizontalLayout_3.addWidget(self.lineEdit_save_dir)
        self.pushButton_save_dir = QtWidgets.QPushButton(self.widget_3)
        self.pushButton_save_dir.setObjectName("pushButton_save_dir")
        self.horizontalLayout_3.addWidget(self.pushButton_save_dir)
        self.verticalLayout.addWidget(self.widget_3)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("宋体")
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.widget_4 = QtWidgets.QWidget(Dialog)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(461, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.pushButton_cancel = QtWidgets.QPushButton(self.widget_4)
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        self.horizontalLayout_4.addWidget(self.pushButton_cancel)
        self.pushButton_start = QtWidgets.QPushButton(self.widget_4)
        self.pushButton_start.setObjectName("pushButton_start")
        self.horizontalLayout_4.addWidget(self.pushButton_start)
        self.verticalLayout.addWidget(self.widget_4)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Auto segment"))
        self.lineEdit_image_dir.setPlaceholderText(_translate("Dialog", "Image dir"))
        self.pushButton_image_dir.setText(_translate("Dialog", "image root"))
        self.lineEdit_label_dir.setPlaceholderText(_translate("Dialog", "VOC xmls dir (for object detection)"))
        self.pushButton_label_dir.setText(_translate("Dialog", "xml root"))
        self.lineEdit_save_dir.setPlaceholderText(_translate("Dialog", "ISAT jsons save dir"))
        self.pushButton_save_dir.setText(_translate("Dialog", "save root"))
        self.label.setText(_translate("Dialog", "Auto segment with bounding box.(\'contour_mode\' and \'use_polydp\' also effect the results.)"))
        self.pushButton_cancel.setText(_translate("Dialog", "Cancel"))
        self.pushButton_start.setText(_translate("Dialog", "Start"))
