import sys
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QInputDialog,
    QLabel,
    QSplitter,
    QToolBar,
    QAction,
    QFileDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal

from vedo import Plotter, Mesh, Sphere, Text3D, settings
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

settings.default_backend = "qt"


class Viewer3D(QWidget):
    pointPicked = pyqtSignal(object, float)

    def __init__(self, mesh_path):
        super().__init__()

        self.interactor = QVTKRenderWindowInteractor(self)
        self.plotter = Plotter(qt_widget=self.interactor, bg="white", axes=1)

        layout = QVBoxLayout()
        layout.addWidget(self.interactor)
        self.setLayout(layout)

        self.mesh_path = mesh_path
        self.mesh = Mesh(mesh_path).normalize()
        self.plotter.show(self.mesh, resetcam=True)

        self.selected_coords = []
        self.measured_values = []

        self.plotter.add_callback("mouse click", self._on_click)

    def _on_click(self, evt):
        if not evt.actor:
            return
        picked = evt.picked3d
        val, ok = QInputDialog.getDouble(
            self, "Measured Value", "Enter value:", decimals=3
        )
        if not ok:
            return

        self.selected_coords.append(picked)
        self.measured_values.append(val)

        self.plotter.add(Sphere(picked, r=0.01, c="red"))
        self.plotter.add(Text3D(f"{val:.2f}", pos=picked + [0.02, 0.015, 0], s=0.01))
        self.plotter.render()

        self.pointPicked.emit(picked, val)
        self.update_interpolation()

        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

        style = vtkInteractorStyleTrackballCamera()
        self.plotter.interactor.SetInteractorStyle(style)

    def update_values(self, values):
        self.measured_values = values
        self.update_interpolation()

    def update_interpolation(self):
        if len(self.selected_coords) < 4:
            return
        arr = np.array(self.selected_coords)
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        values = np.array(self.measured_values)
        rbf = Rbf(x, y, z, values)
        xi, yi, zi = np.split(self.mesh.vertices, 3, axis=1)
        interpolated = rbf(xi, yi, zi)
        self.mesh.cmap("turbo", interpolated).add_scalarbar(title="Interpolated Value")
        self.plotter.render()


class PointTable(QWidget):
    valuesChanged = pyqtSignal(list)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Index", "X", "Y", "Z", "Measured Value"])
        layout.addWidget(QLabel("Measured Points:"))
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.table.cellChanged.connect(self._on_cell_changed)
        self._block_signal = False
        self.data = []

    def add_point(self, coord, value):
        self._block_signal = True
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.table.setItem(row, 1, QTableWidgetItem(f"{coord[0]:.3f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{coord[1]:.3f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{coord[2]:.3f}"))
        val_item = QTableWidgetItem(f"{value:.3f}")
        val_item.setFlags(val_item.flags() | Qt.ItemIsEditable)
        self.table.setItem(row, 4, val_item)
        self.data.append((coord, value))
        self._block_signal = False

    def _on_cell_changed(self, row, column):
        if self._block_signal or column != 4:
            return
        try:
            new_val = float(self.table.item(row, column).text())
            coord = self.data[row][0]
            self.data[row] = (coord, new_val)
            updated_values = [val for _, val in self.data]
            self.valuesChanged.emit(updated_values)
        except ValueError:
            pass


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Viewer + Point Table")
        self.resize(1200, 700)

        self.viewer = Viewer3D(
            "../3DModels/3D_printed_ICE_iphone/Scaniverse_2025_04_08_095715.obj"
        )
        self.table = PointTable()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.viewer)
        splitter.addWidget(self.table)
        splitter.setSizes([800, 400])

        container = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.viewer.pointPicked.connect(self.table.add_point)
        self.table.valuesChanged.connect(self.viewer.update_values)

        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        save_action = QAction("\U0001f4be Save", self)
        save_action.triggered.connect(self.save_points)
        toolbar.addAction(save_action)

        load_action = QAction("\U0001f4c2 Load", self)
        load_action.triggered.connect(self.load_points)
        toolbar.addAction(load_action)

    def save_points(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "NPZ files (*.npz)"
        )
        if not path:
            return
        coords = np.array(self.viewer.selected_coords)
        values = np.array(self.viewer.measured_values)
        model_path = self.viewer.mesh_path
        np.savez(path, coords=coords, values=values, model_path=model_path)

    def load_points(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "NPZ files (*.npz)"
        )
        if not path:
            return
        data = np.load(path, allow_pickle=True)
        coords = data["coords"]
        values = data["values"]
        model_path = str(data["model_path"])

        self.viewer.selected_coords = list(coords)
        self.viewer.measured_values = list(values)
        self.viewer.mesh = Mesh(model_path).normalize()
        self.viewer.mesh_path = model_path

        self.viewer.plotter.clear()
        self.viewer.plotter.show(self.viewer.mesh, resetcam=True)

        self.table.table.setRowCount(0)
        self.table.data = []

        for p, v in zip(coords, values):
            self.table.add_point(p, v)
            self.viewer.plotter.add(Sphere(p, r=0.01, c="red"))
            self.viewer.plotter.add(
                Text3D(f"{v:.2f}", pos=p + [0.02, 0.015, 0], s=0.01)
            )

        self.viewer.update_interpolation()
        self.viewer.plotter.render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
