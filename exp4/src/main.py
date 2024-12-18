from PyQt5 import QtWidgets
import sys
from widgets.main_window import MainWindow


def main():
    app = QtWidgets.QApplication([""])
    # window = QtWidgets.QWidget()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
