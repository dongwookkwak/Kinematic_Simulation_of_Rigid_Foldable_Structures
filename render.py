import sys
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QApplication, QFileDialog, QLabel,
    QVBoxLayout, QWidget, QColorDialog, QPushButton, QTextEdit,
    QSlider, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont, QFontDatabase

# == JSON

import json
import os

# === Utility classes ===

class FileLoader:
    @staticmethod
    def read_ply(path):
        return pv.read(path)

    @staticmethod
    def read_deformation(path):
        return np.loadtxt(path)


class DeformationController:
    def __init__(self):
        self.data = None
        self.T = 0
        self.N = 0

    def load(self, array):
        self.data = array
        self.T, D = array.shape
        self.N = D // 3

    def get_frame(self, index):
        if self.data is not None and 0 <= index < self.T:
            return self.data[index].reshape(self.N, 3)
        return None

    def total_frames(self):
        return self.T


class CameraController:
    def __init__(self, plotter):
        self.plotter = plotter

    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.update()


class StatusManager:
    def __init__(self, label):
        self.label = label

    def update(self, message):
        self.label.setText(f"Status: {message}")


# === Scene and Event ===

class SceneManager:
    def __init__(self, plotter):
        self.plotter = plotter
        self.plotter.show_axes()
        self.mesh = None
        self.actor = None

    def set_mesh(self, mesh):
        self.plotter.clear()
        self.mesh = mesh
        self.actor = self.plotter.add_mesh(mesh, show_edges=True, scalars="rgba", rgba=True,  preference="cell", use_transparency=False)
        return self.mesh.points, self.mesh.faces

    def update_vertices(self, new_vertices):
        if self.mesh is not None:
            self.mesh.points = new_vertices
            self.mesh.Modified()
            self.plotter.update()

    def update_color(self, rgb):
        if self.actor:
            self.actor.prop.color = rgb

    def parse_faces(self):
        face_list = []
        flat_faces = self.mesh.faces
        i = 0
        while i < len(flat_faces):
            n = flat_faces[i]
            face = flat_faces[i + 1:i + 1 + n]
            face_list.append(face)
            i += n + 1
        return face_list


class EventHandler:
    def __init__(self, plotter, vertices, update_status_func, display_info_func):
        self.plotter = plotter
        self.vertices = vertices
        self.update_status = update_status_func
        self.display_info = display_info_func

    def register(self):
        self.plotter.add_key_event('s', self.on_key_s)
        self.plotter.track_click_position(self.on_click, side='left')

    def on_key_s(self):
        msg = f"[S] key pressed. Vertex count: {len(self.vertices)}"
        print(msg)
        self.update_status(msg)

    def on_click(self, coords):
        msg = f"Mouse clicked at: {coords}"
        print(msg)
        self.update_status(msg)
        distances = np.linalg.norm(self.vertices - coords, axis=1)
        idx = np.argmin(distances)
        nearest_vertex = self.vertices[idx]
        self.display_info(f"Nearest Vertex [{idx}]: {nearest_vertex}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_path = "settings.json"
        self.config = {
            "last_ply": "",
            "last_deform": "",
            "mesh_color": [0.95, 0.95, 0.8],
            "play_speed": 100
        }

        self.setWindowTitle("VTK Viewer (Single File Version)")
        self.setWindowIcon(QIcon("Icon.ico"))
        self.resize(1000, 800)

        # Layout & Render
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: white")
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.plotter = BackgroundPlotter(show=False, auto_update=True, notebook=False)
        self._override_quit_key()
        layout.addWidget(self.plotter.interactor)

        # Status bar and info display
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        self.status_manager = StatusManager(self.status_label)

        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(100)
        layout.addWidget(self.info_display)

        # Time slider and label
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.slider)

        self.time_label = QLabel("Frame: 0")
        self.time_label.setStyleSheet("font-weight: bold; font-size: 14px; color: darkblue;")
        slider_layout.addWidget(self.time_label)
        layout.addLayout(slider_layout)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.play_direction = 1

        # Controllers
        self.scene = SceneManager(self.plotter)
        self.deformation = DeformationController()
        self.camera = CameraController(self.plotter)
        self.file_loader = FileLoader()
        self.vertices = None

        # GUI Setup
        self.init_menu()
        self.init_toolbar()
        self.register_arrow_keys()

    def _override_quit_key(self):
        def override_key_event(obj, event):
            key = obj.GetKeySym()
            if key.lower() == 'q':
                print("[INFO] Quit key disabled.")
            else:
                obj.OnKeyPress()

        self.plotter.iren.remove_observers("KeyPressEvent")
        self.plotter.iren.add_observer("KeyPressEvent", override_key_event)

    def init_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open PLY", self)
        open_action.triggered.connect(self.load_mesh)
        file_menu.addAction(open_action)

        edit_menu = menubar.addMenu("Edit")
        color_action = QAction("Change Color", self)
        color_action.triggered.connect(self.change_color)
        edit_menu.addAction(color_action)

        view_menu = menubar.addMenu("View")
        reset_action = QAction("Reset Camera", self)
        reset_action.triggered.connect(self.reset_camera)
        view_menu.addAction(reset_action)

        save_cfg = QAction("Save Settings", self)
        save_cfg.triggered.connect(self.save_settings)
        file_menu.addAction(save_cfg)

        load_cfg = QAction("Load Settings", self)
        load_cfg.triggered.connect(self.load_settings)
        file_menu.addAction(load_cfg)

    def init_toolbar(self):
        button_layout = QHBoxLayout()

        btn_save = QPushButton("Save Mesh")
        btn_save.clicked.connect(self.save_mesh)
        button_layout.addWidget(btn_save)

        btn_deform = QPushButton("Load Deformation")
        btn_deform.clicked.connect(self.load_deformation)
        button_layout.addWidget(btn_deform)

        btn_play = QPushButton("Play")
        btn_play.clicked.connect(self.play_forward)
        button_layout.addWidget(btn_play)

        btn_rev = QPushButton("Reverse")
        btn_rev.clicked.connect(self.play_reverse)
        button_layout.addWidget(btn_rev)

        btn_pause = QPushButton("Pause")
        btn_pause.clicked.connect(self.pause)
        button_layout.addWidget(btn_pause)

        self.centralWidget().layout().addLayout(button_layout)

    def register_arrow_keys(self):
        self.plotter.add_key_event('Left', self.previous_frame)
        self.plotter.add_key_event('Right', self.next_frame_manual)

    def load_mesh(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open VTP File", "", "VTP files (*.vtp)")
        if path:
            mesh = self.file_loader.read_ply(path)
            self.vertices, _ = self.scene.set_mesh(mesh)
            EventHandler(self.plotter, self.vertices, self.status_manager.update, self.display_info).register()
            self.status_manager.update(f"Loaded: {path}")
            self.config["last_ply"] = path

    def load_deformation(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Deformation", "", "NPY files (*.npy)")
        if path:
            data = self.file_loader.read_deformation(path)
            self.deformation.load(data)
            self.slider.setRange(0, self.deformation.total_frames() - 1)
            self.slider.setEnabled(True)
            self.status_manager.update(f"Loaded deformation: {self.deformation.total_frames()} frames")
            self.config["last_deform"] = path

    def on_slider_changed(self, value):
        frame = self.deformation.get_frame(value)
        if frame is not None:
            self.scene.update_vertices(frame)
            self.time_label.setText(f"Frame: {value}")
            self.time_label.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
            self.status_manager.update(f"Frame updated to {value}")

    def next_frame(self):
        value = self.slider.value() + self.play_direction
        if 0 <= value <= self.slider.maximum():
            self.slider.setValue(value)
        else:
            self.pause()

    def next_frame_manual(self):
        value = min(self.slider.value() + 1, self.slider.maximum())
        self.slider.setValue(value)
        self.status_manager.update(f"Moved to frame {value}")

    def previous_frame(self):
        value = max(self.slider.value() - 1, self.slider.minimum())
        self.slider.setValue(value)
        self.status_manager.update(f"Moved to frame {value}")

    def play_forward(self):
        self.play_direction = 1
        self.timer.start(100)
        self.status_manager.update("Playing forward")

    def play_reverse(self):
        self.play_direction = -1
        self.timer.start(100)
        self.status_manager.update("Playing reverse")

    def pause(self):
        self.timer.stop()
        self.time_label.setStyleSheet("font-weight: bold; font-size: 14px; color: darkblue;")
        self.status_manager.update("Paused")

    def reset_camera(self):
        self.camera.reset_view()
        self.status_manager.update("Camera reset")

    def change_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = tuple(c / 255 for c in color.getRgb()[:3])
            self.scene.update_color(rgb)
            self.status_manager.update(f"Color changed to {rgb}")
            self.config["mesh_color"] = rgb

    def save_mesh(self):
        if self.scene.mesh is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Mesh As", "", "VTP files (*.VTP)")
            if path:
                self.scene.mesh.save(path)
                self.status_manager.update(f"Saved to: {path}")

    def display_info(self, text):
        self.info_display.append(text)

    def save_settings(self):
        self.config["play_speed"] = self.timer.interval()
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            self.status_manager.update("Settings saved.")
        except Exception as e:
            self.status_manager.update(f"Failed to save settings: {e}")
    
    def load_settings(self):
        if not os.path.exists(self.config_path):
            self.status_manager.update("No settings file found.")
            return
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            # Apply settings
            rgb = self.config.get("mesh_color", [0.95, 0.95, 0.8])
            self.scene.update_color(rgb)
            self.timer.setInterval(self.config.get("play_speed", 100))
            self.status_manager.update("Settings loaded.")
    
            # Optional auto-reload of mesh
            ply = self.config.get("last_ply")
            if ply and os.path.exists(ply):
                mesh = self.file_loader.read_ply(ply)
                self.vertices, _ = self.scene.set_mesh(mesh)
                EventHandler(self.plotter, self.vertices, self.status_manager.update, self.display_info).register()
    
            deform = self.config.get("last_deform")
            if deform and os.path.exists(deform):
                data = self.file_loader.read_deformation(deform)
                self.deformation.load(data)
                self.slider.setRange(0, self.deformation.total_frames() - 1)
                self.slider.setEnabled(True)
    
            self.status_manager.update("Settings loaded and applied.")
    
        except Exception as e:
            self.status_manager.update(f"Failed to load settings: {e}")


font = QFont("Times", 10, QFont.Bold)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
