from PyQt5 import QtWidgets
from ui.main_window import UI_MainWindow
from widgets.canvas import AnnotationScene, AnnotationView
from widgets.categories_dock_widget import CategoriesDockWidget

from configs import load_config, save_config, CONFIG_FILE, SOFTWARE_CONFIG_FILE


class MainWindow(QtWidgets.QMainWindow, UI_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setup_ui(self)

        self.current_image_path: str = None
        self.current_image = None

        self.current_label = "__background__"
        self.current_group = 1

        self.config_file = CONFIG_FILE
        self.software_config_file = SOFTWARE_CONFIG_FILE

        self.init_ui()
        self.load_config()
        self.init_connections()

    def show_image(self, image_path: str):
        self.scene.load_image(image_path)
        self.view.zoom_fit()

    def init_ui(self):
        self.scene = AnnotationScene(mainwindow=self)
        self.view = AnnotationView(parent=self)
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)

        self.categories_dock_widget = CategoriesDockWidget(mainwindow=self)
        self.categories_dock.setWidget(self.categories_dock_widget)

    def init_connections(self):
        self.action_open_image.triggered.connect(self.open_image)

    def open_image(self):
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", filter="Image Files (*.jpg *.png *.jpeg)"
        )

        if not image_path:
            return

        self.current_image_path = image_path
        self.show_image(image_path)
        # self.cur_image = QtWidgets.QImage(image_path)

    def load_config(self):
        self.config = load_config(self.software_config_file)

        softwate_config = self.config.get("software", {})

        # self.config.update
        self.categories_dock_widget.update_categories(self.config.get("label"))
