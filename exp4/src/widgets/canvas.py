from PyQt5 import QtWidgets, QtGui, QtCore
import cv2

class AnnotationScene(QtWidgets.QGraphicsScene):
    def __init__(self, mainwindow):
        super(AnnotationScene, self).__init__()
        self.mainwindow = mainwindow


    def load_image(self, image_path: str):
        self.clear()
        image_data = cv2.imread(image_path)
        self.image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.addItem(self.image_item)

        self.mask_item = QtWidgets.QGraphicsPixmapItem()
        self.mask_item.setZValue(1)
        self.addItem(self.mask_item)

        self.image_item.setPixmap(QtGui.QPixmap(image_path))
        self.setSceneRect(self.image_item.boundingRect())
        # self.change_mode_to_view()
 

class AnnotationView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(AnnotationView, self).__init__(parent)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # DragMod: This enum describes the default action for the view when pressing and dragging the mouse over the viewport.

        # NoDrag: Ignores the mouse events.

        # ScrollHandDrag: The cursor changes into a pointing hand, and dragging the mouse around will scroll the scrolbars.
        # This mode works both in interactive and non-interactive mode.

        # RubberBandDrag: The view will zoom in or out based on the dragged rectangle.
        # A rubber band will appear. Dragging the mouse will set the rubber band geometry,
        # and all items covered by the rubber band are selected. This mode is disabled for non-interactive views.
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.factor = 1.2

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # return super().wheelEvent(event)
        angel = event.angleDelta()
        angelX, angelY = angel.x(), angel.y()
        point = event.pos()  # 当前鼠标位置
        if angelY > 0:
            self.zoom(self.factor, point)
        else:
            self.zoom(1 / self.factor, point)

    def zoom_in(self):
        self.zoom(self.factor)

    def zoom_out(self):
        self.zoom(1 / self.factor)

    def zoom_fit(self):
        self.fitInView(
            0,
            0,
            self.scene().width(),
            self.scene().height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )

    def zoom(self, factor, point=None):
        mouse_old = self.mapToScene(point) if point is not None else None
        # 缩放比例

        pix_widget = (
            self.transform()
            .scale(factor, factor)
            .mapRect(QtCore.QRectF(0, 0, 1, 1))
            .width()
        )
        if pix_widget > 30 and factor > 1:
            return
        if pix_widget < 0.01 and factor < 1:
            return

        self.scale(factor, factor)
        if point is not None:
            mouse_now = self.mapToScene(point)
            center_now = self.mapToScene(
                self.viewport().width() // 2, self.viewport().height() // 2
            )
            center_new = mouse_old - mouse_now + center_now
            self.centerOn(center_new)
