# ImportDialog.py

import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class ImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import COCO Dataset")
        self.setWindowIcon(QIcon("res/PolyVisionLogo.png"))
        self.setMinimumWidth(500)
        self.setModal(True) # Blocks the RetrainUI while this is open

        self.image_dir_path = ""
        self.json_path = ""

        # --- Layouts ---
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # --- Widgets ---
        # Image Directory Selection
        self.image_dir_label = QLabel("<i>No folder selected...</i>")
        image_dir_button = QPushButton("Select Image Folder...")
        
        # Annotation File Selection
        self.json_path_label = QLabel("<i>No file selected...</i>")
        json_path_button = QPushButton("Select COCO .json File...")

        # Add to form layout
        form_layout.addRow("Image Directory:", self.image_dir_label)
        form_layout.addRow("", image_dir_button)
        form_layout.addRow("Annotation File:", self.json_path_label)
        form_layout.addRow("", json_path_button)

        # --- Button Box ---
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setText("Import")
        self.ok_button.setEnabled(False) # Disabled until both paths are selected

        # --- Connections ---
        image_dir_button.clicked.connect(self.select_image_dir)
        json_path_button.clicked.connect(self.select_json_file)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # --- Finalize Layout ---
        layout.addLayout(form_layout)
        layout.addWidget(button_box)

    def select_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if path:
            self.image_dir_path = path
            self.image_dir_label.setText(f"<b>{os.path.basename(path)}</b>")
            self.check_if_ready()

    def select_json_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select COCO Annotation File", "", "JSON Files (*.json)")
        if path:
            self.json_path = path
            self.json_path_label.setText(f"<b>{os.path.basename(path)}</b>")
            self.check_if_ready()

    def check_if_ready(self):
        """Enables the 'Import' button only if both paths are valid."""
        if self.image_dir_path and self.json_path:
            self.ok_button.setEnabled(True)

    def get_paths(self):
        """Public method to get the selected paths after the dialog is accepted."""
        return self.image_dir_path, self.json_path