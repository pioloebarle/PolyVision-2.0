import os
import sys 
import cv2
import time
import psutil
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QInputDialog, QLineEdit, QPushButton
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
from bbox import BoundingBox
from localization import Detector, loadModel, initialize_models, is_models_ready

class DetectUI(QDialog):
    close_signal = pyqtSignal()

    def __init__(self, port, parent=None):
        super().__init__(parent)
        self.port = port
        self.setWindowTitle("Images")
        self.setWindowIcon(QIcon("res/PolyVisionLogo.png"))
        self.setFixedSize(1400, 800)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Models should already be pre-loaded by MainWindow - no need to initialize again
        print("Detect.py using pre-loaded models from main application")
        
        main_layout = QHBoxLayout(self)

        # Create the left-side widgets
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        left_layout.addStretch()

        self.change_dir_button = QPushButton("Choose Image")
        self.change_dir_button.setFixedSize(175,30)
        self.change_dir_button.setStyleSheet("QPushButton {\n""    background-color: #00853f;\n""    color: #FFFFFF;\n""    font: bold 15px;\n""    border-radius: 5px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        
        self.detect_button = QPushButton("Detect")
        self.detect_button.setFixedSize(175,30)
        self.detect_button.setStyleSheet("QPushButton {\n""    background-color: #00853f;\n""    color: #FFFFFF;\n""    font: bold 15px;\n""    border-radius: 5px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        
        self.close_button = QPushButton("Close")
        self.close_button.setFixedSize(175,30)
        self.close_button.setStyleSheet("QPushButton {\n""    background-color: #00853f;\n""    color: #FFFFFF;\n""    font: bold 15px;\n""    border-radius: 5px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        
        self.image_pathway = None
        self.file_path = None
        
        # Connect buttons
        self.detect_button.clicked.connect(self.detect_mp)
        self.change_dir_button.clicked.connect(self.change_path)
        self.close_button.clicked.connect(self.closeUI)
        
        left_layout.addWidget(self.change_dir_button)
        left_layout.addWidget(self.detect_button)
        left_layout.addWidget(self.close_button)
        self.left_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.left_widget.setMaximumWidth(int(self.width() * 0.2))  

        self.scroll_widget = QLabel()
        self.scroll_widget.setMinimumSize(1175,770)
        self.scroll_widget.setStyleSheet("background-color: white;")
        self.scroll_widget_layout = QVBoxLayout()
        self.scroll_widget.setLayout(self.scroll_widget_layout)
        self.scroll_widget_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        main_layout.addWidget(self.scroll_widget)
        main_layout.addWidget(self.left_widget, stretch=0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        brush = QBrush(QColor(224, 224, 212, 255))
        painter.setBrush(brush)
        rect = self.rect()
        painter.drawRoundedRect(rect, 10, 10)

    def closeUI(self):
        self.close_signal.emit()
        self.close()

    def detect_mp(self):
        # Check if models are ready using the global state
        if not is_models_ready():
            QMessageBox.warning(self, "Models Not Ready", "Models are still loading. Please wait a moment and try again.")
            return
        
        # Check if image is selected
        if self.file_path is None:
            QMessageBox.warning(self, "No Image Selected", "Please choose an image first before running detection.")
            return
        
        # Check if file exists
        if not os.path.exists(self.file_path):
            QMessageBox.critical(self, "File Not Found", f"The selected image file does not exist:\n{self.file_path}")
            return
        
        
        original_text = self.detect_button.text()
        self.detect_button.setText("Detecting...")
        self.detect_button.setEnabled(False)
        
        # Force UI update
        QApplication.processEvents()
        
        original_priority = None
        process = psutil.Process()
        original_priority = process.nice()
        
        # Set high priority for detection
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        
        start_time = time.perf_counter()
        detections = loadModel(self.file_path)
        end_time = time.perf_counter()
        
        # Restore original priority
        if original_priority is not None:
            process.nice(original_priority)
            
                
        
        inference_time = end_time - start_time
        print(f"Total inference time: {inference_time:.3f} seconds")
        
        if detections is None:
            
            QMessageBox.critical(self, "Detection Error", 
                                "Detection failed. Please check the image and try again.")
            # Restore button state
            self.detect_button.setText(original_text)
            self.detect_button.setEnabled(True)
            return
        
        if detections:
            # SUCCESS: Draw bounding boxes
            new_image = BoundingBox(self.file_path, detections, self.port)
            rgb_image = new_image.get_image()
            
            # Convert ndarray image to QImage
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            qpixmap = QtGui.QPixmap.fromImage(qimage)
            self.scroll_widget.setPixmap(qpixmap)
            self.scroll_widget.setScaledContents(True)
            
            # Detections found
            num_objects = len(detections)
            QMessageBox.information(self, "Detection Complete", 
                                #   f"Found {num_objects} object{'s' if num_objects != 1 else ''}!\nInference time: {inference_time:.3f} seconds")
                                  f"Found {num_objects} object{'s' if num_objects != 1 else ''}!")
        else:
            # No detections found
            self.scroll_widget.setPixmap(QtGui.QPixmap(self.file_path))
            self.scroll_widget.setScaledContents(True)
            
            QMessageBox.information(self, "Detection Complete", 
                                #   f"No objects found in the image.\nInference time: {inference_time:.3f} seconds")
                                  f"No objects found in the image.")
                                  
        # UI FEEDBACK: Restore button state
        self.detect_button.setText(original_text)
        self.detect_button.setEnabled(True)

    def change_path(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            new_db, _ = QFileDialog.getOpenFileName(self, "Select Photo", self.file_path, 
                                                  "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
            if new_db:
                if not os.path.exists(new_db):
                    QMessageBox.warning(self, "File Error", "Selected file does not exist.")
                    return
                
                try:
                    test_image = cv2.imread(new_db)
                    if test_image is None:
                        QMessageBox.warning(self, "Invalid Image", 
                                          "Selected file is not a valid image format.")
                        return
                    
                    # Displaying image in Detect.Py
                    self.file_path = new_db
                    print(f"Selected image: {self.file_path}")
                    self.scroll_widget.setPixmap(QtGui.QPixmap(self.file_path))
                    self.scroll_widget.setScaledContents(True)
                    
                except Exception as img_error:
                    QMessageBox.warning(self, "Image Error", 
                                      f"Cannot load image: {str(img_error)}")
                    
        except Exception as e:
            print(f"Error in change_path: {e}")
            QMessageBox.critical(self, "File Selection Error", 
                               f"Error selecting file: {str(e)}")

def main():
    app = QApplication(sys.argv)
    stat_ui = DetectUI(0)  # Binary detection port
    stat_ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()