from PyQt5 import QtCore, QtGui, QtWidgets


class UI_MainWindow(object):
    def __init__(self):
        self.info_dock: QtWidgets.QDockWidget = None
        self.files_dock: QtWidgets.QDockWidget = None
        self.annos_dock: QtWidgets.QDockWidget = None
        self.categories_dock: QtWidgets.QDockWidget = None

        self.action_open_image: QtWidgets.QAction = None

    def setup_ui(self, main_window: QtWidgets.QMainWindow):
        self.main_window = main_window
        main_window.setObjectName("MainWindow")
        main_window.resize(1280, 764)
        main_window.setMinimumSize(QtCore.QSize(800, 600))
        # set font
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        main_window.setFont(font)
        # set window icon
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("./icons/icon_picture.svg"),  # from pwd(main.py)
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )
        main_window.setWindowIcon(icon)

        self.setup_central_widget()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_tool_bar()
        self.setup_dock_widgets()
        self.init_actions()
        self.setup_actions()


        
    def setup_central_widget(self):
        # set central widget
        self.central_widget = QtWidgets.QWidget(self.main_window)
        self.central_widget.setObjectName("central_widget")
        self.vertical_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setSpacing(0)
        self.vertical_layout.setObjectName("vertical_layout")
        self.main_window.setCentralWidget(self.central_widget)

    def setup_menu_bar(self):
        # set menu bar
        self.menu_bar = QtWidgets.QMenuBar(self.main_window)
        self.menu_bar.setObjectName("menu_bar")
        self.menu_bar.setEnabled(True)
        self.menu_bar.setGeometry(QtCore.QRect(0, 0, 1280, 29))
        self.menu_bar.setAutoFillBackground(False)
        self.menu_bar.setDefaultUp(False)
        self.menu_bar.setNativeMenuBar(False)
        # add menus
        # add "File" menu
        self.menu_file = QtWidgets.QMenu("File", self.menu_bar)
        self.menu_file.setObjectName("menu_file")

        self.menu_bar.addAction(self.menu_file.menuAction())

        # add "Edit" menu
        self.menu_edit = QtWidgets.QMenu("Edit", self.menu_bar)
        self.menu_edit.setObjectName("menu_edit")

        self.menu_bar.addAction(self.menu_edit.menuAction())

        # add "View" menu
        self.menu_view = QtWidgets.QMenu("View", self.menu_bar)
        self.menu_view.setObjectName("menu_view")
        self.menu_bar.addAction(self.menu_view.menuAction())

        # add "SAM" menu
        self.menu_sam = QtWidgets.QMenu("SAM", self.menu_bar)
        self.menu_sam.setObjectName("menu_sam")
        self.menu_bar.addAction(self.menu_sam.menuAction())

        # add "Help" menu
        self.menu_help = QtWidgets.QMenu("Help", self.menu_bar)
        self.menu_help.setObjectName("menu_help")
        self.menu_bar.addAction(self.menu_help.menuAction())

        self.main_window.setMenuBar(self.menu_bar)

    def setup_status_bar(self):
        # set status bar
        self.status_bar = QtWidgets.QStatusBar(self.main_window)
        self.status_bar.setObjectName("status_bar")
        self.status_bar.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.main_window.setStatusBar(self.status_bar)

    def setup_tool_bar(self):
        # set tool bar
        self.tool_bar = QtWidgets.QToolBar(self.main_window)
        self.tool_bar.setObjectName("tool_bar")

        self.tool_bar.setIconSize(QtCore.QSize(24, 24))
        self.tool_bar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.tool_bar.setFloatable(False)

        self.main_window.addToolBar(QtCore.Qt.TopToolBarArea, self.tool_bar)

    def setup_dock_widgets(self):
        # set dock widget
        # info dock
        self.info_dock = QtWidgets.QDockWidget("info_dock", self.main_window)
        self.info_dock.setObjectName("info_dock")
        self.info_dock.setMinimumSize(QtCore.QSize(85, 43))
        self.info_dock.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)  # ?
        self.dock_widget_contents_1 = QtWidgets.QWidget()
        self.dock_widget_contents_1.setObjectName("dock_widget_contents_1")
        self.info_dock.setWidget(self.dock_widget_contents_1)
        self.main_window.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.info_dock)
        # self.main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.info_dock)

        # annos dock
        self.annos_dock = QtWidgets.QDockWidget("annos_dock", self.main_window)
        self.annos_dock.setObjectName("annos_dock")
        self.annos_dock.setMinimumSize(QtCore.QSize(85, 43))
        self.annos_dock.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)  # ?
        self.dock_widget_contents_2 = QtWidgets.QWidget()
        self.dock_widget_contents_2.setObjectName("dock_widget_contents_2")
        self.annos_dock.setWidget(self.dock_widget_contents_2)
        self.main_window.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.annos_dock)

        # files dock
        self.files_dock = QtWidgets.QDockWidget("files_dock", self.main_window)
        self.files_dock.setObjectName("files_dock")
        self.files_dock.setMinimumSize(QtCore.QSize(85, 43))
        self.files_dock.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)  # ?
        self.dock_widget_contents_3 = QtWidgets.QWidget()
        self.dock_widget_contents_3.setObjectName("dock_widget_contents_3")
        self.files_dock.setWidget(self.dock_widget_contents_3)
        self.main_window.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.files_dock)

        # categories dock
        self.categories_dock = QtWidgets.QDockWidget(
            "categories_dock", self.main_window
        )
        self.categories_dock.setObjectName("categories_dock")
        self.categories_dock.setMinimumSize(QtCore.QSize(85, 43))
        self.categories_dock.setFeatures(
            QtWidgets.QDockWidget.AllDockWidgetFeatures
        )  # ?
        self.dock_widget_contents_4 = QtWidgets.QWidget()
        self.dock_widget_contents_4.setObjectName("dock_widget_contents_4")
        self.categories_dock.setWidget(self.dock_widget_contents_4)
        self.main_window.addDockWidget(
            QtCore.Qt.DockWidgetArea(2), self.categories_dock
        )

    def init_actions(self):
        # action_open_image
        self.action_open_image = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_picture.svg"), "Open Image", self.main_window
        )
        self.action_open_image.setShortcut("Ctrl+O")
        self.action_open_image.setObjectName("action_open_image")

        # action_open_label
        self.action_open_label = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_tag.svg"), "Open Label", self.main_window
        )
        self.action_open_label.setObjectName("action_open_label")

        # action_settings
        self.action_settings = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_settings.svg"), "Settings", self.main_window
        )
        self.action_settings.setShortcut("Ctrl+S")
        self.action_settings.setObjectName("action_settings")

        # action_exit
        self.action_exit = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_power.svg"), "Exit", self.main_window
        )
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.setObjectName("action_exit")

        # action_zoom_in
        self.action_zoom_in = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_zoom_in.svg"), "Zoom In", self.main_window
        )
        self.action_zoom_in.setShortcut("Ctrl++")
        self.action_zoom_in.setObjectName("action_zoom_in")

        # action_zoom_out
        self.action_zoom_out = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_zoom_out.svg"), "Zoom Out", self.main_window
        )
        self.action_zoom_out.setShortcut("Ctrl+-")
        self.action_zoom_out.setObjectName("action_zoom_out")

        # action_zoom_fit
        self.action_zoom_fit = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_zoom_fit.svg"), "Zoom Fit", self.main_window
        )
        self.action_zoom_fit.setShortcut("Ctrl+F")
        self.action_zoom_fit.setObjectName("action_zoom_fit")

        # action_segment_anything
        self.action_segment_anything = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_meta.svg"), "Segment Anything", self.main_window
        )
        self.action_segment_anything.setObjectName("action_segment_anything")

        # action_polygon_annotation
        self.action_polygon_annotation = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_anchor.svg"),
            "Polygon Annotation",
            self.main_window,
        )
        self.action_polygon_annotation.setObjectName("action_polygon_annotation")

        # action_backspace
        self.action_backspace = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_backspace.svg"), "Backspace", self.main_window
        )
        self.action_backspace.setShortcut("Backspace")
        self.action_backspace.setObjectName("action_backspace")

        # action_cancel
        self.action_cancel = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_cancel.svg"), "Cancel", self.main_window
        )
        self.action_cancel.setShortcut("Escape")
        self.action_cancel.setObjectName("action_cancel")

        # action_finish
        self.action_finish = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_right.svg"), "Finish", self.main_window
        )
        self.action_finish.setObjectName("action_finish")

        # action_prev_image
        self.action_prev_image = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_prev.svg"), "Previous Image", self.main_window
        )
        # self.action_prev_image.setShortcut("Ctrl+Left")
        self.action_prev_image.setObjectName("action_prev_image")

        # action_next_image
        self.action_next_image = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_next.svg"), "Next Image", self.main_window
        )
        # self.action_next_image.setShortcut("Ctrl+Right")
        self.action_next_image.setObjectName("action_next_image")

        # action_union
        self.action_union = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_union.svg"), "Union", self.main_window
        )
        self.action_union.setObjectName("action_union")

        # action_intersect
        self.action_intersect = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_intersect.svg"), "Intersection", self.main_window
        )
        self.action_intersect.setObjectName("action_intersect")

        # action_subtract
        self.action_subtract = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_subtract.svg"), "Subtract", self.main_window
        )
        self.action_subtract.setObjectName("action_subtract")

        # action_exclude
        self.action_exclude = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_exclude.svg"), "Exclude", self.main_window
        )
        self.action_exclude.setObjectName("action_exclude")

        # action_model_manage
        self.action_model_manage = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_list.svg"), "Model Manage", self.main_window
        )
        self.action_model_manage.setObjectName("action_model_manage")

        # action_visible
        self.action_visible = QtWidgets.QAction(
            QtGui.QIcon("./icons/icon_eye.svg"), "Visible", self.main_window
        )
        self.action_visible.setObjectName("action_visible")

    def setup_actions(self):
        # set actions
        # file menu actions
        self.menu_file.addAction(self.action_open_image)
        self.menu_file.addAction(self.action_open_label)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_settings)
        self.menu_file.addAction(self.action_exit)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_prev_image)
        self.menu_file.addAction(self.action_next_image)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_settings)
        self.menu_file.addAction(self.action_exit)

        # view menu actions
        self.menu_view.addAction(self.action_zoom_in)
        self.menu_view.addAction(self.action_zoom_out)
        self.menu_view.addAction(self.action_zoom_fit)
        self.menu_view.addSeparator()
        self.menu_view.addAction(self.action_visible)

        # SAM menu actions
        self.menu_sam.addAction(self.action_model_manage)

        # edit menu actions
        self.menu_edit.addAction(self.action_segment_anything)
        self.menu_edit.addAction(self.action_polygon_annotation)
        self.menu_edit.addSeparator()
        self.menu_edit.addAction(self.action_backspace)
        self.menu_edit.addAction(self.action_cancel)
        self.menu_edit.addAction(self.action_finish)
        self.menu_edit.addSeparator()
        self.menu_edit.addAction(self.action_union)
        self.menu_edit.addAction(self.action_intersect)
        self.menu_edit.addAction(self.action_subtract)
        self.menu_edit.addAction(self.action_exclude)

        # tool bar
        self.tool_bar.addAction(self.action_prev_image)
        self.tool_bar.addAction(self.action_next_image)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.action_segment_anything)
        self.tool_bar.addAction(self.action_polygon_annotation)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.action_backspace)
        self.tool_bar.addAction(self.action_cancel)
        self.tool_bar.addAction(self.action_finish)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.action_union)
        self.tool_bar.addAction(self.action_intersect)
        self.tool_bar.addAction(self.action_subtract)
        self.tool_bar.addAction(self.action_exclude)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.action_zoom_in)
        self.tool_bar.addAction(self.action_zoom_out)
        self.tool_bar.addAction(self.action_zoom_fit)
        self.tool_bar.addAction(self.action_visible)
