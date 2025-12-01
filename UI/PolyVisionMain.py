import os
import sys 
from PIL import Image
from PyQt5 import QtGui, QtWidgets, QtMultimedia
from PyQt5.QtGui import *
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QInputDialog, QLineEdit, QPushButton, QMessageBox
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.uic import loadUi
import cv2
import gc 
import numpy as np
import math
import serial
import serial.tools.list_ports
import time
from Statistics import StatisticsUI
from Images import ImagesUI
from NewFile import NewFileUI
from Retrain import RetrainUI
from Database import *
from Capture import *
from Settings import SettingsUI
import json
from PIL import ImageEnhance
from OkayMessageBox import *
from Detect import DetectUI
from GRBL import GrblUI
from LiveDetect import *
from CalibrationUI import CalibrateUI
from CoordinateUI import CoordinateUI
import threading
from VerificationMessageBox import VerificationBox
from AnnotationReviewDialog import AnnotationReviewDialog
from PIL.ImageQt import ImageQt
from collections import deque
import winsound
from localization import LocalDetectMP, loadModel, initialize_models, is_models_ready, is_model_loading

#Memory Usage
# import os
# import psutil
# from PyQt5.QtCore import QTimer  


# def start_memory_monitor(window, interval_ms: int = 2000, show_in_status_bar: bool = True):
#     """Start a repeating memory sampler for the running UI."""
#     process = psutil.Process(os.getpid())
#     timer = QTimer(window)
#     timer.setInterval(interval_ms)

#     def report():
#         try:
#             info = process.memory_full_info()
#             rss_mb = info.rss / (1024 ** 2)
#             vms_mb = info.vms / (1024 ** 2)
#             message = f"Memory RSS: {rss_mb:.2f} MB | VMS: {vms_mb:.2f} MB"
#             print(message)
#             if show_in_status_bar:
#                 status_bar = window.statusBar()
#                 if status_bar is not None:
#                     status_bar.showMessage(message, interval_ms)
#         except (psutil.Error, AttributeError):
#             timer.stop()

#     timer.timeout.connect(report)
#     timer.start()
#     return timer
#======END=======

class Ui_MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.autoScanning = None
        self.autoFocusing = None
        self.image_queue = deque()
        self.VideoCapture = None
        self.available_ports = list(serial.tools.list_ports.comports())
        self.comports = [f"{port.device} - {port.description.split(' (')[0].strip()}" for port in self.available_ports]
        self.ser = None
        self.blurThreshold = 100 
        self.x = 0
        self.y = 0
        self.measuring = False
        self.paused = False
        self.points = []
        self.distance = 0
        self.capturing = False
        self.original_autoscan_frame = None  # Store original frame during AutoScan for measuring
        self.calibrating = False
        self.settings = None
        self.z = 0
        self.xLimit = 30
        self.yLimit = 30
        self.totalScan = 0
        self.currentScan = 0
        self.currentMP = 0
        self.widthClicked = None
        self.lengthClicked = None
        with open("user_settings.json", "r") as f:
                self.settings_data = json.load(f)
        self.image_settings = self.settings_data.get("image_settings", {})
        self.grbl_settings = self.settings_data.get("grbl_settings", {})
        self.general_settings = self.settings_data.get("general_features", {})
        self.model_port = 0  # Separate variable for model type (0=Binary, 1=Multiclass)
        self.current_model_type = self.general_settings.get("model", "Binary")
        if self.current_model_type == "Binary":
            self.model_port = 0
        else:
            self.model_port = 1
        self.captureDone = None
        create_main_database(os.getcwd())
        create_retraining_database(os.getcwd())

        print("PolyVision initialized - models will be pre-loaded after UI setup")
        # self._memory_timer = start_memory_monitor(self) # MEMORY USAGE
        
    #updating live feed in different Thread
    def ImageUpdateSlot(self, Image):
        if not self.paused:
            # Define the crop parameters (adjust as needed)
            left = 100
            top = 100
            right = Image.width() - 240
            bottom = Image.height() - 395

            # Create a copy of a region from the original image
            cropped_image = Image.copy(left, top, right - left, bottom - top)
            
            # Live overlays
            if getattr(self, "show_grid", False):
                # Grid + hover during tension calibration only
                self.applyGridOverlay(cropped_image)
                
            # Assuming graphicsView is a QGraphicsView
            self.graphicsView.setPixmap(QPixmap.fromImage(cropped_image))
            if self.autoFocusing is not None:
                self.image_queue.append(self.captureCurrentFrame())

        else:
            pass

    def calculate_blur_and_color(self, pic):

        np_image = self.qimage_to_numpy(pic)
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        blur_value = np.mean(gradient_magnitude)
        average_pixel_value = np.mean(np_image)

        return blur_value, average_pixel_value

    def qimage_to_numpy(self, qimage):
        width = qimage.width()
        height = qimage.height()
        byte_count = qimage.byteCount()
        format = qimage.format()

        # Convert the QImage to a NumPy array
        ptr = qimage.bits()
        ptr.setsize(byte_count)
        arr = np.frombuffer(ptr, dtype=np.uint8)

        if format == QImage.Format_RGB888:
            arr = arr.reshape(height, width, 3)
        elif format == QImage.Format_RGB32:
            arr = arr.reshape(height, width, 4)
            # Convert RGBA to RGB by removing alpha channel to fix tensor size mismatch
            # arr = arr[:, :, :3]

        return arr

    #for canceling live feed (To add pa)
    def CancelFeed(self):
        self.ImageCapture.stop()

    #for capturing images
    def captureButtonClicked(self):
        try:
            self.frame = self.captureCurrentFrame()
            self.paused = True
            self.capturing = True
            self.calibrating = False
            self.capture = CaptureUI()
            self.capture.length_clicked.connect(self.onLengthClicked)
            self.capture.width_clicked.connect(self.onWidthClicked)
            self.capture.save_clicked.connect(self.saving)
            self.capture.on_rejected.connect(self.captureClose)
            toReturn = self.capture.exec()
            return toReturn
        except:
            pass
    
    def captureClose(self):
        if self.VideoCapture is not None:
            self.VideoCapture.start()
        self.capturing = False
        self.paused = False
        self.scanButton.setEnabled(True)
        if (self.autoScanning is not None and self.scanBTN.text() == "Continue"):
            self.grblUP.setEnabled(True)
            self.grblDOWN.setEnabled(True )
            self.grblLEFT.setEnabled(True )
            self.grblRIGHT.setEnabled(True )
      
    def captureCurrentFrame(self):
        pixmap = self.graphicsView.pixmap()
        return pixmap.toImage()

    def saving(self):
        try:
            image = Image.fromqimage(self.frame)
            sized = image.resize((1600, 900), Image.LANCZOS)
            sharpened = self.adjust_sharpness(sized, (self.imageSharpnes() / 100))
            saturated = self.adjust_saturation(sharpened, (self.imageSaturation() / 100))
            self.saveFrame(saturated)
        except Exception as e:
            print("Error occurred during image processing and saving:", e)

    def imageQuality(self):
        quality = self.image_settings.get("image_quality")
        if quality == "Low":
            return 10   
        elif quality  == "Medium":
            return 42 
        else:
            return 95

    def adjust_sharpness(self,image, factor):
        sharpener = ImageEnhance.Sharpness(image)
        return sharpener.enhance(1 + factor)

    def adjust_saturation(self,image, factor):
        saturater = ImageEnhance.Color(image)
        return saturater.enhance(1 + factor)

    def imageSharpnes(self):
        return self.image_settings.get("image_sharpness")

    def imageSaturation(self):
        return self.image_settings.get("image_saturation")

    def saveFrame(self, frame):
        try:
            particle_name = self.capture.particle_name_edit.text() or None
            if particle_name is None:
                # Display an error message
                QMessageBox.critical(self, "Error", "Particle name cannot be empty.", QMessageBox.Ok)
                return  # Do not proceed further if particle_name is None

            length = self.capture.length_edit.text() or ""
            width = self.capture.width_edit.text() or ""
            color = self.capture.color_edit.text() or ""
            shape = self.capture.shape_edit.currentText() or ""
            magnification = self.capture.magnification_edit.text() or ""
            note = self.capture.note_edit.text() or ""
            file_type = self.capture.photo_options_combo.currentText()
            
            frame_filename = f"{particle_name}.{file_type.upper()}" 
            save_folder = self.file_name  # path to folder
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, frame_filename)
            frame.save(save_path, quality=self.imageQuality())

            # Adjust the insert_data function to handle NULL values appropriately
            insert_data(self.file_name, save_path, particle_name, length, width, color, shape, magnification, note)

            self.capture.close()
            self.capturing = False
            self.paused = False
            self.currentMP += 1
            self.focusValue.setText(str(self.currentMP))
        except Exception as e:
            print("Exception occurred saving image:", e)

        
    def retrainSave(self, frame, annotations, is_microplastic=None):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        save_folder = "retrainingImages"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(current_directory, save_folder)
        # Get the current row number
        current_row_number = count_rows_in_retraining_database(current_directory)
        # Create a unique image name based on the row number
        image_name = f"image_{current_row_number}.png"
        # Save the image using Pillow (PIL)
        frame.save(os.path.join(save_path, image_name))
        bounding_box_str = json.dumps(annotations)
        if is_microplastic is None:
            is_microplastic = bool(annotations)
        is_microplastic = 1 if is_microplastic else 0
        # Now call retrain_data with the image name
        retrain_data(current_directory, image_name, is_microplastic, bounding_box_str)
        
    def getAnnotationClass(self, detections=None):

        choices = []
        model_type = self.current_model_type
        if model_type == "Multiclass":
            choices = [
                (1, "Filament"),
                (2, "Film"),
                (3, "Fragment"),
            ]
        else:
            # Binary model: background vs. microplastic
            choices = [
                (1, "Microplastic"),
                (0, "Background"),
            ]

        if detections:
            known_ids = {value for value, _ in choices}
            for det in detections:
                det_id = det.get("class_id")
                if det_id is not None and det_id not in known_ids:
                    choices.append((det_id, f"Class {det_id}"))
                    known_ids.add(det_id)
        return choices

    def AnnotationReview(self, preview_image, detections):
        detections = detections or []
        class_choices = self.getAnnotationClass(detections)
        dialog = AnnotationReviewDialog(preview_image, detections, class_choices, self)
        if dialog.exec_() == QDialog.Accepted:
            accepted = dialog.accepted_annotations()
            if accepted:
                return accepted, True
            if detections:
                rejected = []
                for det in detections:
                    if isinstance(det, dict):
                        item = dict(det)
                        item["review_status"] = item.get("review_status", "rejected")
                        rejected.append(item)
                return rejected, False
            return [], False
        return None

    def SaveNegativeSample(self, frame, detections=None):
        dialog = VerificationBox("Save this frame as a negative sample for retraining?")
        if dialog.exec_() == QDialog.Accepted:
            annotations = detections if detections is not None else []
            self.retrainSave(frame, annotations, is_microplastic=False)
            return True
        return False
        
    def onLengthClicked(self):
        self.capture.hide()
        self.lengthClicked = 1
        self.measureLength()

    def onWidthClicked(self):
        self.capture.hide()
        self.widthClicked = 1
        self.measureLength()

    def measureClicked(self):
        self.calibrating = False
        self.measureLength()

    #start-measuring
    def measureLength(self):
        try:
           self.paused = True
           if not self.measuring:
                if not self.capturing:
                    self.frame = self.captureCurrentFrame()
                self.measuring = True
                self.measureButton.setEnabled(True)
                self.points = []
                self.measureButton.setText("Finish")
                QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
                self.graphicsView.mouseMoveEvent = self.mouseMoveEvent
                self.calibration = self.image_settings.get("calibration")
           else:
                self.stopMeasureLength()
                self.graphicsView.setPixmap(QPixmap.fromImage(self.frame))
        except Exception as e:
            print("Error occurred during measuring:", e)

    def stopMeasureLength(self):
        self.measuring = False
        self.measureButton.setText("Measure")
        QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))
        if self.capturing:
            self.paused = True
        else:
            self.paused = False
        self.lengthLabel.setVisible(False)
        if self.capturing and self.measuring != True:
            if self.lengthClicked == 1:
                self.capture.length_edit.setText(str(self.distance * self.calibration))
                self.lengthClicked = 0
            elif self.widthClicked == 1:
                self.capture.width_edit.setText(str(self.distance * self.calibration))
                self.widthClicked = 0
            self.capture.show()
        self.distance = 0 
        self.calibrating = False

    def mousePressEvent(self, event):
        if self.measuring and (event.button() == Qt.LeftButton):
            posGlobal = event.globalPos()
            pos = self.graphicsView.mapFromGlobal(posGlobal)
            self.points.append((pos.x(), pos.y()))
            self.currentPos = (pos.x(), pos.y())
            pixmap = self.graphicsView.pixmap()
            qimage = pixmap.toImage()
            painter = QPainter(qimage)
            pen = QPen(Qt.red)
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawPoint(pos.x(),pos.y())
            painter.end()
            self.graphicsView.setPixmap(QPixmap.fromImage(qimage))
            self.paintEvent()
        else:
            super().mousePressEvent(event)

    def paintEvent(self):
        if not self.paused:
            return
        pixmap = self.graphicsView.pixmap()
        if pixmap is None:
            return
        distance = []
        distanceLoc = 0
        qimage = pixmap.toImage()
        painter = QPainter(qimage)
        pen = QPen(QColor(255,0,0), 1)
        pen.setDashPattern([2, 2, 2, 2])
        painter.setPen(pen)
        # Draw lines based on stored points
        for i in range(len(self.points) - 1):
            pt1 = self.points[i]
            pt2 = self.points[i + 1]
            painter.drawLine(pt1[0], pt1[1],pt2[0],pt2[1])
            self.lengthLabel.setGeometry(QtCore.QRect((pt2[0]+255), (pt2[1]+80), 50, 22))
            self.lengthLabel.setAlignment(Qt.AlignCenter)
            self.lengthLabel.setStyleSheet("background-color: rgba(255, 255, 255, 128);")
            self.lengthLabel.setVisible(True)
            distanceLoc = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
            formatted_distance = "{:.2f}".format(distanceLoc)
        distance.append(distanceLoc)
        for i in range(len(distance)):
            self.distance = self.distance + distance[i]

        painter.end()
        pixmap = QPixmap.fromImage(qimage)
        self.graphicsView.setPixmap(pixmap)
        if (self.calibrating and (len(self.points) == 2)):
            self.calculatePixelDistanceRation()

        distanceShow = (self.distance*self.calibration) + (0.002*self.z)
        formatted_distance = "{:.2f}".format(distanceShow)
        self.lengthLabel.setText(str(formatted_distance))

    def calibrate(self):
        try:
            if self.settings is not None:
                self.settings.hide()

            if self.calibrateBTN.text() == "Finish":
                self.cancelCalibration()
                self.moveHome()
                return
            
            chooser = CalibrationChoiceDialog(self)
            if chooser.exec_() == QtWidgets.QDialog.Accepted:
                if chooser.selection == "distance":
                    self.calibrating = True
                    self.measuring = False
                    self.measureLength()  
                elif chooser.selection == "tension":
                    self.calibrateTension()
            else:
                if self.settings is not None:
                    self.settings.show()
        except Exception as e:
            print("Error occurred during calibration selection:", e)

    def calculatePixelDistanceRation(self):
        calibrateUI = CalibrateUI()
        if calibrateUI.exec_() == QtWidgets.QDialog.Accepted:
            actual = calibrateUI.distance_edit.text()
            try:
                integer_value = int(actual)
                self.calibration = integer_value/self.distance  
                #add saving of calibration and close show setting UI again
                self.image_settings["calibration"] = self.calibration
                if self.settings is not None:
                    self.settings.show()
                # Update settings_data dictionary
                self.settings_data["image_settings"] = self.image_settings
                file_path = "user_settings.json"  
                with open(file_path, "w") as f:
                    json.dump(self.settings_data, f, indent=4) 
                self.calibrating = False
                self.stopMeasureLength()
            except ValueError:
                self.stopMeasureLength()

            file_path = "user_settings.json"  
            settings_data = {} 
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    settings_data = json.load(f)
        else:
            self.stopMeasureLength()
            
    def mouseReleaseEvent(self, event):
        pass
    
    def gridHoverMove(self, event):
        try:
            # Use local position for reliability
            pos = event.pos()
            pixmap = self.graphicsView.pixmap()
            if pixmap is None:
                return
            qimage = pixmap.toImage()
            width = qimage.width()
            height = qimage.height()

            # Account for QLabel centering (pixmap may be centered inside label)
            label_w = self.graphicsView.width()
            label_h = self.graphicsView.height()
            off_x = max(0, (label_w - width) // 2)
            off_y = max(0, (label_h - height) // 2)

            # Translate widget coords to image coords
            img_x = pos.x() - off_x
            img_y = pos.y() - off_y

            # Clamp within image bounds
            x = max(0, min(width - 1, img_x))
            y = max(0, min(height - 1, img_y))
            self._hover_pos = (x, y)
        except:
            self._hover_pos = None
        
    def updateHoverFromGlobal(self):
        try:
            # Poll global cursor pos and map into image coords to keep hover active during modal dialogs
            gpos = QCursor.pos()
            local = self.graphicsView.mapFromGlobal(gpos)
            pixmap = self.graphicsView.pixmap()
            if pixmap is None:
                return
            qimage = pixmap.toImage()
            width = qimage.width()
            height = qimage.height()
            label_w = self.graphicsView.width()
            label_h = self.graphicsView.height()
            off_x = max(0, (label_w - width) // 2)
            off_y = max(0, (label_h - height) // 2)
            img_x = local.x() - off_x
            img_y = local.y() - off_y
            if img_x < 0 or img_y < 0 or img_x >= width or img_y >= height:
                return
            self._hover_pos = (img_x, img_y)
        except:
            pass

    def goToDetect(self):
        
        video_was_active = False
        # Check if video capture is active
        if self.VideoCapture is not None and hasattr(self.VideoCapture, 'ThreadActive') and self.VideoCapture.ThreadActive:
            video_was_active = True
            if hasattr(self.VideoCapture, 'set_detection_priority'):
                self.VideoCapture.set_detection_priority(True)
       
        self.paused = True
        self.detect = DetectUI(self.model_port)
        self.detect.close_signal.connect(self.setPausedFalse)
        
        result = self.detect.exec_()
        
        if video_was_active and self.VideoCapture is not None:
            if hasattr(self.VideoCapture, 'set_detection_priority'):
                self.VideoCapture.set_detection_priority(False)

        # Resume UI updates
        self.paused = False

    def goToSettings(self):
        self.settings = SettingsUI()
        self.settings.calibration_clicked.connect(self.calibrate)
        self.settings.apply_clicked.connect(self.refreshSettings)
        continued = self.settings.exec_()

    # to be edited.
    def goToRetrain(self):
        self.paused = True
        self.retrain = RetrainUI()
        self.retrain.close_signal.connect(self.setPausedFalse)
        self.retrain.exec_()

    def refreshSettings(self):
        previous_model_type = getattr(self, "current_model_type", self.current_model_type)
        with open("user_settings.json", "r") as f:
                self.settings_data = json.load(f)
                self.image_settings = self.settings_data.get("image_settings", {})
                self.grbl_settings = self.settings_data.get("grbl_settings", {})
                self.general_settings = self.settings_data.get("general_features", {})
                new_model_type = self.general_settings.get("model", "Binary")
                if new_model_type == "Binary":
                    self.model_port = 0
                else:
                    self.model_port = 1
        self.current_model_type = new_model_type
        if new_model_type != previous_model_type:
            print(f"Model selection changed from {previous_model_type} to {new_model_type}.")
            if hasattr(self, "_host_mainwindow") and self._host_mainwindow is not None:
                self._host_mainwindow.handle_model_selection_change(new_model_type)
   
    def goToCurrentImages(self):
        self.paused = True
        self.images = ImagesUI(self.file_name)
        continued = self.images.exec_()
        if continued == QDialog.Rejected:
            self.setPausedFalse()

    def goToStatistics(self):
        self.paused = True
        self.statistics = StatisticsUI(self.file_name)
        self.statistics.close_signal.connect(self.setPausedFalse)
        self.statistics.exec_()


    def setPausedFalse(self):
        self.paused = False

    #accessing COM ports
    def serialConnect(self):
        if not self.ser:
            self.port = self.dropDown.currentText().split(" - ")[0] 
            try:
                self.ser = serial.Serial(self.port, baudrate=115200)
                time.sleep(2)
                self.connectGRBL.setText("Disconnect")
                self.grblzUP.setEnabled(True)
                self.grblzDOWN.setEnabled(True)
                self.grblUP.setEnabled(True)
                self.grblDOWN.setEnabled(True)
                self.grblLEFT.setEnabled(True)
                self.grblRIGHT.setEnabled(True)
                self.grblHOME.setEnabled(True)
                self.emergencyGRBL.setEnabled(True)
                self.focusBTN.setEnabled(True)
                self.grbl = GrblUI()
                #====== ADD AUTOMATION HERE=======#
                self.grbl.auto_clicked.connect(self.autoScan)
                self.grbl.manual_clicked.connect(self.manualScan)
                self.grbl.exec_()
                    
            except Exception as e:
                print(e)
                prompt = OkayMessageBox("Could not connect to Serial port")
                result = prompt.exec_()
                self.ser = None
                
        else:
            try:
                self.ser.close() 
                self.connectGRBL.setText("Connect")  
                self.ser = None
            except Exception as e:
                print("Error closing serial port:", e)

    def focusUp(self):
        if self.ser:
            if self.z != 120:
                string = "G21 G91 G1 Z" 
                string += str(self.grbl_settings["steps_per_mm"]) + " F" + str(self.grbl_settings["max_feedrate"]) + "\r\n"
                toSend = string.encode('utf-8')
                self.ser.write(toSend)
                self.z += self.grbl_settings["steps_per_mm"]
                self.zValue.setText(str(self.z))

    def focusDown(self):
        if self.ser:
            if self.z != 0:
                string = "G21 G91 G1 Z-" 
                string += str(self.grbl_settings["steps_per_mm"]) + " F" + str(self.grbl_settings["max_feedrate"]) + "\r\n"
                toSend = string.encode('utf-8')
                self.ser.write(toSend)
                self.z -= self.grbl_settings["steps_per_mm"]
                self.zValue.setText(str(self.z))

    def moveUp(self):
        if self.ser:
            if self.grbl_settings["area_scan"] == False:
                if self.y != self.yLimit:
                    string = "G21 G91 G1 Y" + str(self.grbl_settings["steps_per_mm"])
                    string += " F" + str(self.grbl_settings["max_feedrate"]) + " \r\n"
                    toSend = string.encode('utf-8')
                    self.ser.write(toSend)
                    self.y += self.grbl_settings["steps_per_mm"]
                    display_y = -self.y
                    if abs(float(display_y)) < 1e-6:
                        display_y = 0.0
                    self.yValue.setText(str(display_y))
            else:
                if self.y != 0: 
                    string = "G21 G91 G1 Y-" + str(self.grbl_settings["steps_per_mm"])
                    string += " F" + str(self.grbl_settings["max_feedrate"]) + " \r\n"
                    toSend = string.encode('utf-8')
                    self.ser.write(toSend)
                    self.y -= self.grbl_settings["steps_per_mm"]
                    display_y = -self.y
                    if abs(float(display_y)) < 1e-6:
                        display_y = 0.0
                    self.yValue.setText(str(display_y))
    
    def moveDown(self):
        if self.ser:
            if self.grbl_settings["area_scan"] == False:
                if self.y != 0: 
                    string = "G21 G91 G1 Y-" + str(self.grbl_settings["steps_per_mm"])
                    string += " F" + str(self.grbl_settings["max_feedrate"]) + " \r\n"
                    toSend = string.encode('utf-8')
                    self.ser.write(toSend)
                    self.y -= self.grbl_settings["steps_per_mm"]
                    display_y = -self.y
                    if abs(float(display_y)) < 1e-6:
                        display_y = 0.0
                    self.yValue.setText(str(display_y))
            else:
                if self.y != self.yLimit:
                    string = "G21 G91 G1 Y" + str(self.grbl_settings["steps_per_mm"])
                    string += " F" + str(self.grbl_settings["max_feedrate"]) + " \r\n"
                    toSend = string.encode('utf-8')
                    self.ser.write(toSend)
                    self.y += self.grbl_settings["steps_per_mm"]
                    display_y = -self.y
                    if abs(float(display_y)) < 1e-6:
                        display_y = 0.0
                    self.yValue.setText(str(display_y))
                    
    def moveLeft(self):
        if self.ser:
            if self.grbl_settings["area_scan"] == False:
                if self.x != self.xLimit:
                    string = "G21 G91 G1 X-" + str(self.grbl_settings["steps_per_mm"])
                    string += " F" + str(self.grbl_settings["max_feedrate"]) + " \r\n"
                    toSend = string.encode('utf-8')
                    self.ser.write(toSend)
                    self.x += self.grbl_settings["steps_per_mm"]
                    self.xValue.setText(str(self.x))
            else:
                 if self.x != 0:
                    string = "G21 G91 G1 X" + str(self.grbl_settings["steps_per_mm"])
                    string += " F" + str(self.grbl_settings["max_feedrate"]) + " \r\n"
                    toSend = string.encode('utf-8')
                    self.ser.write(toSend)
                    self.x -= self.grbl_settings["steps_per_mm"]
                    self.xValue.setText(str(self.x))

    def moveRight(self): 
            if self.ser:
                if self.grbl_settings["area_scan"] == False:
                    if self.x != 0:
                        string = "G21 G91 G1 X" + str(self.grbl_settings["steps_per_mm"])
                        string += " F" + str(self.grbl_settings["max_feedrate"]) + " \r\n"
                        toSend = string.encode('utf-8')
                        self.ser.write(toSend)
                        self.x -= self.grbl_settings["steps_per_mm"]
                        self.xValue.setText(str(self.x))
                else:
                    if self.x != self.xLimit:
                        string = "G21 G91 G1 X-" + str(self.grbl_settings["steps_per_mm"])
                        string += " F" + str(self.grbl_settings["max_feedrate"]) + " \r\n"
                        toSend = string.encode('utf-8')
                        self.ser.write(toSend)
                        self.x += self.grbl_settings["steps_per_mm"]
                        self.xValue.setText(str(self.x))

    def moveHome(self):
        if self.ser:
            self.gcode_command = b"$H\r\n"
            self.ser.write(self.gcode_command)
            self.x = 0
            self.xValue.setText(str(self.x))
            self.y = 0
            self.yValue.setText(str(self.y))
        else:
            pass

    def emergencyStop(self):
        
        if self.autoFocusing is not None:
            self.autoFocusing.updateThread(False)

        if self.autoScanning is not None:
            self.autoScanning.stop()

        if self.ser:
            self.gcode_command = b"!\r\n"
            self.ser.write(self.gcode_command)
            self.ser.close()
            time.sleep(1)
            self.ser = serial.Serial(self.port, baudrate=115200)
            time.sleep(3)
            self.gcode_command = b"$X\r\n"
            self.ser.write(self.gcode_command)
            self.moveHome()
            self.scanning_active = False
            self.paused = False
            self.currentScan = 0
            self.totalScan = 0
            self.x = 0
            self.y = 0
            self.xValue.setText(str(self.x))
            self.yValue.setText(str(self.y))
            self.connectGRBL.setEnabled(True)
            self.grblUP.setEnabled(True)
            self.grblDOWN.setEnabled(True )
            self.grblLEFT.setEnabled(True )
            self.grblRIGHT.setEnabled(True )
            self.grblzUP.setEnabled(True )
            self.grblzDOWN.setEnabled(True )
            self.focusBTN.setEnabled(True )
            self.grblHOME.setEnabled(True)
            self.scanButton.setVisible(False)
            self.statusValue.setText("Idle")
            self.detectionValue.setText("OFF")
            self.progressBar.setProperty("value", 0)
            self.scanBTN.setText("Scan")
        else:
            self.moveHome()
            pass
        
        dialog = OkayMessageBox("Emergency Stop, resets all progress.")
        result = dialog.exec_()
        
        
    def autoScan(self):
        self.connectGRBL.setEnabled(False)
        self.grblUP.setEnabled(False)
        self.grblDOWN.setEnabled(False)
        self.grblLEFT.setEnabled(False)
        self.grblRIGHT.setEnabled(False)
        self.grblzUP.setEnabled(False)
        self.grblzDOWN.setEnabled(False)
        self.focusBTN.setEnabled(False)
        self.grblHOME.setEnabled(False)
        self.scanButton.setEnabled(False)
        self.detectionValue.setText("ON")
        self.statusValue.setText("Scanning")
        self.grbl.hide()
        QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))
        self.high = 0
        self.med = 0
        self.low = 0
        self.scanButton.setVisible(True)
        self.scanBTN.setText("Continue")
        self.statusValue.setText("Scanning..")
        self.detectionValue.setText("ON")
        #============== Home XY Table ================#
        self.paused = False #pause while moving      
        self.coordinates = CoordinateUI()
        self.coordinates.exec_()
        start_x = self.coordinates.largest_x
        start_y = self.coordinates.largest_y
        rows = self.coordinates.distance_y
        cols = self.coordinates.distance_x
        if start_x is not None and start_y is not None:
            self.autoScanning = AutoScan(self.ser, start_x, start_y, rows, cols)
            self.totalScan = rows * cols
            self.autoScanning.start()
            self.autoScanning.ImageScan.connect(self.scanForMP)
            self.autoScanning.Homing.connect(self.homingPrompt)
            self.autoScanning.Finished.connect(self.finishedScan)

    def startFocusing(self):   
        self.autoFocusing = AutoFocus(self.ser,self.image_queue,self.zValue)
        self.autoFocusing.focused.connect(self.doneFocus)
        self.image_queue.append(self.captureCurrentFrame())
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.stopFocusing)
        self.timer.start(7000)
        self.autoFocusing.start()

    def stopFocusing(self):
        if self.autoFocusing is not None:
            self.autoFocusing.ThreadActive = False 
            text = self.zValue.text()
            self.z = float(text)
            string = "G21 G91 G1 Z-" 
            string +=  str(self.z) + " F1000\r\n"
            print(string)
            toSend = string.encode('utf-8')
            self.ser.write(toSend)
            self.z = 0
            self.zValue.setText(str(self.z))
            # prompt = OkayMessageBox("Manual Focusing is Needed")
            # prompt.exec_()
            self.timer.stop()
            self.autoFocusing = None

    def doneFocus(self):
        self.timer.stop()
        self.autoFocusing = None
        text = self.zValue.text()
        self.z = float(text)

    def calculate_blur(self, Pic):
        # Convert the QImage to a numpy array
        npImage = self.qimage_to_numpy(Pic)
        grayImage = cv2.cvtColor(npImage, cv2.COLOR_RGB2GRAY)
        # Use Tenengrad (Sobel) operator for blur detection
        sobelx = cv2.Sobel(grayImage, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(grayImage, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        blur_value = np.mean(gradient_magnitude)

        return blur_value

    def qimage_to_numpy(self, qimage):
        width = qimage.width()
        height = qimage.height()
        byte_count = qimage.byteCount()
        format = qimage.format()

        # Convert the QImage to a NumPy array
        ptr = qimage.bits()
        ptr.setsize(byte_count)
        arr = np.frombuffer(ptr, dtype=np.uint8)

        if format == QImage.Format_RGB888:
            arr = arr.reshape(height, width, 3)
        elif format == QImage.Format_RGB32:
            arr = arr.reshape(height, width, 4)
            # Convert RGBA to RGB by removing alpha channel to fix tensor size mismatch
            # arr = arr[:, :, :3]

        return arr
    
    def manualScanBTN(self):   
        if self.scanBTN.text() == "Continue":
            if self.autoScanning is not None:
                target_x = getattr(self.autoScanning, "x", self.x)
                target_y = getattr(self.autoScanning, "y", self.y)
                delta_x = target_x - self.x
                delta_y = target_y - self.y
                if self.ser is not None and (abs(delta_x) >= 1e-6 or abs(delta_y) >= 1e-6):
                    command_parts = ["G21", "G91", "G1"]
                    if abs(delta_x) >= 1e-6:
                        command_parts.append(f"X{-delta_x:.3f}")
                    if abs(delta_y) >= 1e-6:
                        command_parts.append(f"Y{delta_y:.3f}")
                    feedrate = self.grbl_settings.get("max_feedrate", 1000)
                    command_parts.append(f"F{feedrate}")
                    command = " ".join(command_parts) + "\r\n"
                    try:
                        self.ser.write(command.encode("utf-8"))
                    except Exception as e:
                        print(f"Failed to realign AutoScan position: {e}")
                self.x = target_x
                self.y = target_y
                self.xValue.setText(f"{self.x:.1f}")
                self.yValue.setText(f"{-self.y:.1f}")
            self.detectionValue.setText("ON")
            if self.autoScanning is not None:
                self.autoScanning.event.set()
        else:
            self.scanButton.setVisible(False)
            self.manualScanMP()

    def manualScanMP(self):
        self.statusValue.setText("Scanning")   
        
        # Start overall timing
        overall_start_time = time.perf_counter()
        
        # Measure frame capture time
        frame_start = time.perf_counter()
        currentFrame = self.captureCurrentFrame()
        frame_end = time.perf_counter()
        
        # Measure image processing time
        processing_start = time.perf_counter()
        image = Image.fromqimage(currentFrame)
        sized = image.resize((640, 640), Image.LANCZOS)
        processing_end = time.perf_counter()
        
        # Measure video setup time
        video_start = time.perf_counter()
        if self.VideoCapture is not None and hasattr(self.VideoCapture, 'ThreadActive') and self.VideoCapture.ThreadActive:
            video_was_active = True
            if hasattr(self.VideoCapture, 'set_detection_priority'):
                self.VideoCapture.set_detection_priority(True)

        self.paused = True 
        # Removed manual gc.collect() for better performance
        video_end = time.perf_counter()

        start_detect = time.perf_counter()
        detections = loadModel(sized)   
        end_detect = time.perf_counter()
        
        # Detailed timing breakdown
        frame_time = frame_end - frame_start
        processing_time = processing_end - processing_start
        video_setup_time = video_end - video_start
        print(f"Frame capture: {frame_time:.3f}s, Image processing: {processing_time:.3f}s, Video setup: {video_setup_time:.3f}s")
       
        class FastDetector:
            def __init__(self, results):
                self.result = results if results else []
            def get_json(self):
                return self.result
        
        detector = FastDetector(detections)
        
        overall_end_time = time.perf_counter()
        total_scan_time = overall_end_time - overall_start_time
        print(f"Total scanning time: {total_scan_time:.3f} seconds")
        
        if len(detector.get_json()) != 0:
            if self.general_settings["sound"] == True:
                winsound.Beep(800, 500)
                winsound.Beep(800, 500)
                winsound.Beep(800, 750)

            low = 0
            med = 0
            high = 0
            for items in detector.get_json():
                if items["score"] > 0.70 and items["score"] < 0.80:
                    low += 1
                elif items["score"] > 0.80 and items["score"] < 0.90:
                    med += 1
                elif items["score"] >= 0.90  and items["score"] < 1.0:
                    high += 1
            
            self.highValue.setText(str(high)) 
            self.moderateValue.setText(str(med))  
            self.lowValue.setText(str(low)) 
            
            new_image = BoundingBox(sized, detector.get_json(), self.model_port)
            rgb_image = new_image.get_image()
            
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            qpixmap = QPixmap.fromImage(qimage)
            scaled = qpixmap.scaled(1280,720)
            self.graphicsView.setPixmap(scaled)
            dialog = VerificationBox("Microplastic Found. Capture Image?") 
            continued = dialog.exec_()    
            if continued == QDialog.Accepted:
                verify = VerificationBox("Is this a Microplastic?")
                go = verify.exec_()
                if go == QDialog.Accepted:
                    self.paused = True
                    qpixmap = QPixmap.fromImage(currentFrame)
                    image = Image.fromqimage(qpixmap)
                    self.graphicsView.setPixmap(qpixmap)
                    review_result = self.AnnotationReview(ImageQt(sized).copy(), detector.get_json())
                    if review_result is not None:
                        annotations, is_positive = review_result
                        self.retrainSave(currentFrame, annotations, is_microplastic=is_positive)
                        self.captureButtonClicked()
                    else:
                        self.paused = False
                        self.statusValue.setText("Idle")
                        if 'video_was_active' in locals() and video_was_active and self.VideoCapture is not None:
                            if hasattr(self.VideoCapture, 'set_detection_priority'):
                                self.VideoCapture.set_detection_priority(False)
                        return
                else:
                    self.SaveNegativeSample(currentFrame, detector.get_json())
                    self.paused = False
            else:
                self.SaveNegativeSample(currentFrame, detector.get_json())
                self.paused = False

            if video_was_active and self.VideoCapture is not None:
                if hasattr(self.VideoCapture, 'set_detection_priority'):
                    self.VideoCapture.set_detection_priority(False)
                    
        else:
            overall_end_time = time.perf_counter()
            total_scan_time = overall_end_time - overall_start_time
            print(f"No detections found")
            print(f"Total scanning time: {total_scan_time:.3f} seconds")
            
            self.SaveNegativeSample(currentFrame, detector.get_json())
            
            if video_was_active and self.VideoCapture is not None:
                if hasattr(self.VideoCapture, 'set_detection_priority'):
                    self.VideoCapture.set_detection_priority(False)
            self.paused = False
            
        self.statusValue.setText("Idle")

    def autoFocus(self):
        self.autoFocusing.updateThread(False)
         

    def scanForMP(self):

        overall_start_time = time.perf_counter()
        
        if self.totalScan <= 0:
            self.progressBar.setProperty("value", 0)
            return
        
        self.currentScan += 1
        self.progressBar.setProperty("value", (self.currentScan/self.totalScan) * 100)
        self.x = self.autoScanning.x
        self.y = self.autoScanning.y
        self.xValue.setText(f"{self.x:.1f}")
        self.yValue.setText(f"{-self.y:.1f}")

        currentFrame = self.captureCurrentFrame()
        image = Image.fromqimage(currentFrame)
        sized = image.resize((640, 640), Image.LANCZOS)
        
        if self.VideoCapture is not None and hasattr(self.VideoCapture, 'ThreadActive') and self.VideoCapture.ThreadActive:
            video_was_active = True
            if hasattr(self.VideoCapture, 'set_detection_priority'):
                self.VideoCapture.set_detection_priority(True)
            
        self.paused = True #pause 

        # Uses the resized PIL Image
        detections = loadModel(sized)    
        
        # DEBUG: Confirm model selection
        model_name = "Binary" if self.model_port == 0 else "Multiclass"
        print(f"AutoScan using {model_name} model (model_port = {self.model_port})")
        
        # Create compatible detector object
        class FastDetector:
            def __init__(self, results):
                self.result = results if results else []
            def get_json(self):
                return self.result
        
        detector = FastDetector(detections)
            
        if len(detector.get_json()) != 0:
            if self.general_settings["sound"] == True:
                winsound.Beep(800, 500)
                winsound.Beep(800, 500)
                winsound.Beep(800, 750)
            low = 0
            med = 0
            high = 0
            for items in detector.get_json():
                if items["score"] > 0.70 and items["score"] < 0.80:
                    low += 1
                elif items["score"] > 0.80 and items["score"] < 0.90:
                    med += 1
                elif items["score"] >= 0.90  and items["score"] < 1.0:
                    high += 1

            self.highValue.setText(str(high)) 
            self.moderateValue.setText(str(med))  
            self.lowValue.setText(str(low)) 
            print("With MP")
            new_image = BoundingBox(sized, detector.get_json(), self.model_port)
            rgb_image = new_image.get_image()
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            qpixmap = QPixmap.fromImage(qimage)
            scaled = qpixmap.scaled(1280,720)
            self.graphicsView.setPixmap(scaled)
            dialog = VerificationBox("Microplastic Found. Capture Image?") 
            continued = dialog.exec_()    
            if continued == QDialog.Accepted:     
                verify = VerificationBox("Is this a Microplastic?")
                yes = verify.exec_()
                if yes == QDialog.Accepted:
                    self.paused = True
                    qpixmap = QPixmap.fromImage(currentFrame)
                    image = Image.fromqimage(qpixmap)
                    self.graphicsView.setPixmap(qpixmap)
                    review_result = self.AnnotationReview(ImageQt(sized).copy(), detector.get_json())
                    if review_result is not None:
                        annotations, is_positive = review_result
                        self.retrainSave(currentFrame, annotations, is_microplastic=is_positive)
                        self.captureDone = self.captureButtonClicked()
                    else:
                        self.paused = False
                        self.statusValue.setText("Scanning")
                        if self.autoScanning is not None and hasattr(self.autoScanning, 'event'):
                            self.autoScanning.event.set()
                        if 'video_was_active' in locals() and video_was_active and self.VideoCapture is not None:
                            if hasattr(self.VideoCapture, 'set_detection_priority'):
                                self.VideoCapture.set_detection_priority(False)
                        return
                else:
                    self.SaveNegativeSample(currentFrame, detector.get_json())
                    self.autoScanning.event.set()
                    self.paused = False 
            else:
                self.SaveNegativeSample(currentFrame, detector.get_json())
                self.autoScanning.event.set()
                self.paused = False

            if video_was_active and self.VideoCapture is not None:
                if hasattr(self.VideoCapture, 'set_detection_priority'):
                    self.VideoCapture.set_detection_priority(False)          
        else:
            # No detections found - just show overall time
            overall_end_time = time.perf_counter()
            total_scan_time = overall_end_time - overall_start_time
            print(f"No detections found")
            print(f"Total scanning time: {total_scan_time:.3f} seconds")
            
            self.SaveNegativeSample(currentFrame, detector.get_json())
            # Restart video capture after detection
            if video_was_active and self.VideoCapture is not None:
                if hasattr(self.VideoCapture, 'set_detection_priority'):
                    self.VideoCapture.set_detection_priority(False)
            self.paused = False
            self.autoScanning.event.set()

                
    def homingPrompt(self):
        dialog = OkayMessageBox("Homing... Click Okay when finished.")
        result = dialog.exec_()
        if result == QDialog.Accepted:
            self.autoScanning.event.set()

    def finishedScan(self):
        self.progressBar.setProperty("value", 0)
        self.currentScan = 0
        self.totalScan = 0
        self.paused = False #unpause
        self.connectGRBL.setEnabled(True)
        self.grblUP.setEnabled(True)
        self.grblDOWN.setEnabled(True )
        self.grblLEFT.setEnabled(True )
        self.grblRIGHT.setEnabled(True )
        self.grblzUP.setEnabled(True )
        self.grblzDOWN.setEnabled(True )
        self.focusBTN.setEnabled(True )
        self.grblHOME.setEnabled(True)
        self.scanButton.setVisible(False)
        self.statusValue.setText("Idle")
        self.detectionValue.setText("OFF")
        verify = VerificationBox("Do you want to rescan?")
        self.x = 0
        self.y = 0 
        self.xValue.setText(str(self.x))
        self.yValue.setText(str(self.y))
        continued = verify.exec_()
        if continued == QDialog.Accepted:
            self.autoScan()
            self.scanButton.setVisible(True)
        else:
            self.scanBTN.setText("Scan")

    def manualScan(self):
        self.moveHome()
        # pass

    def closeEvent(self, event):
        
        if hasattr(self, 'VideoCapture') and self.VideoCapture is not None:
            self.VideoCapture.ThreadActive = False
            self.VideoCapture.stop()
            self.VideoCapture.wait()  
            self.VideoCapture.deleteLater()  
        event.accept()

    def count_images_in_folder(self, folder_path):
        # List all files in the folder
        all_files = os.listdir(folder_path)

        # Filter for image files (you can add more extensions if needed)
        image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        # Count the number of image files
        num_images = len(image_files)

        return num_images

    def on_new_action(self):
        
        new_file_widget = NewFileUI()
        if new_file_widget.exec_() == QtWidgets.QDialog.Accepted and new_file_widget.file_name_edit.text() != '' and new_file_widget.location_edit.text() != '':
            self.file_name = new_file_widget.file_name_edit.text()
            location = new_file_widget.location_edit.text()
            sampling_date = new_file_widget.sampling_date_edit.text()
            self.placeValue.setText(self.file_name)        
            self.locationValue.setText(location)
            self.dateValue.setText(sampling_date)
            self.selected_camera_index = new_file_widget.camera_combo_box.currentIndex()
            
            #so many changes here
            self.cameraStatusLabel.setText("Camera starting...")
            self.cameraStatusLabel.setVisible(True)
            self.cameraStatusLabel.setAlignment(Qt.AlignCenter)
            
            if self.VideoCapture is not None:
                self.VideoCapture.ThreadActive = False
                self.VideoCapture.stop()
                self.VideoCapture.wait()  
                self.VideoCapture.deleteLater()  
                self.VideoCapture = None
            
            self.VideoCapture = VideoCapture(self.selected_camera_index)
            self.VideoCapture.start()
            self.VideoCapture.ImageUpdate.connect(self.ImageUpdateSlot)
            self.VideoCapture.camera_ready.connect(self.on_camera_ready)  
            
            add_database_entry(self.file_name, f"{location}/microplastic.db", sampling_date)
            create_microplastics_database(self.file_name)
            self.paused = False
            self.statusValue.setText("Idle....")
            self.detectionValue.setText("OFF")
            self.highValue.setText("0") 
            self.moderateValue.setText("0")  
            self.lowValue.setText("0") 

            self.currentMP = self.count_images_in_folder(self.file_name)
            self.focusValue.setText(str(self.currentMP))
            self.captureButton.setEnabled(True)
            self.connectGRBL.setEnabled(True)
            self.imagesButton.setEnabled(True)
            self.detectButton.setEnabled(True)
            self.retrainButton.setEnabled(True)
            self.statisticsButton.setEnabled(True)
            self.measureButton.setEnabled(True)
            self.calibrateBTN.setEnabled(True)
            self.scanBTN.setEnabled(True)
            self.available_ports = list(serial.tools.list_ports.comports())
            self.comports = [f"{port.device} - {port.description.split(' (')[0].strip()}" for port in self.available_ports]
            self.dropDown.addItems(self.comports)

    def on_camera_ready(self):
        """Hide the camera starting indicator when camera is ready"""
        self.cameraStatusLabel.setVisible(False)

    #defining pyQt5 widgets
    def setupUi(self, MainWindow):
        self._host_mainwindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.showMaximized()
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        MainWindow.setWindowIcon(QIcon("res/PolyVisionLogo.png"))


        #instantiation of items
        self.centralwidget      =   QtWidgets.QWidget(MainWindow)
        self.progressBar        =   QtWidgets.QProgressBar(self.centralwidget)
        self.graphicsView       =   QtWidgets.QLabel(self.centralwidget)
        self.placeValue         =   QtWidgets.QLabel(self.centralwidget)
        self.locationLabel      =   QtWidgets.QLabel(self.centralwidget)
        self.locationValue      =   QtWidgets.QLabel(self.centralwidget)
        self.dataLabel          =   QtWidgets.QLabel(self.centralwidget)
        self.lengthLabel        =   QtWidgets.QLabel(self.centralwidget)
        self.dateValue          =   QtWidgets.QLabel(self.centralwidget)
        self.focusLabel         =   QtWidgets.QLabel(self.centralwidget)
        self.focusValue         =   QtWidgets.QLabel(self.centralwidget)
        self.statusLabel        =   QtWidgets.QLabel(self.centralwidget)
        self.statusValue        =   QtWidgets.QLabel(self.centralwidget)
        self.detectionLabel     =   QtWidgets.QLabel(self.centralwidget)
        self.detectionValue     =   QtWidgets.QLabel(self.centralwidget)
        self.progressLabel      =   QtWidgets.QLabel(self.centralwidget)
        self.grblTitle          =   QtWidgets.QLabel(self.centralwidget)
        self.highLabel          =   QtWidgets.QLabel(self.centralwidget)
        self.realTimeLabel      =   QtWidgets.QLabel(self.centralwidget)
        self.highLabel          =   QtWidgets.QLabel(self.centralwidget)
        self.moderateLabel      =   QtWidgets.QLabel(self.centralwidget)
        self.lowLabel           =   QtWidgets.QLabel(self.centralwidget)
        self.highValue          =   QtWidgets.QLabel(self.centralwidget)
        self.moderateValue      =   QtWidgets.QLabel(self.centralwidget)
        self.lowValue           =   QtWidgets.QLabel(self.centralwidget)
        self.currentStatus      =   QtWidgets.QLabel(self.centralwidget)
        self.boxWidget          =   QtWidgets.QLabel(self.centralwidget)
        self.xWidget            =   QtWidgets.QLabel(self.centralwidget)
        self.zWidget            =   QtWidgets.QLabel(self.centralwidget)
        self.xLabel             =   QtWidgets.QLabel(self.centralwidget)
        self.xValue             =   QtWidgets.QLabel(self.centralwidget)
        self.zLabel             =   QtWidgets.QLabel(self.centralwidget)
        self.zValue             =   QtWidgets.QLabel(self.centralwidget)
        self.yLabel             =   QtWidgets.QLabel(self.centralwidget)
        self.yValue             =   QtWidgets.QLabel(self.centralwidget)
        self.appTitle           =   QtWidgets.QLabel(self.centralwidget)
        self.dropDown           =   QtWidgets.QComboBox(self.centralwidget)
        self.appLogo            =   QtWidgets.QLabel(self.centralwidget)  
        self.filamentValue      =   QtWidgets.QLabel(self.centralwidget)
        self.cameraStatusLabel  =   QtWidgets.QLabel(self.centralwidget)     

        self.dropDown.setFocusPolicy(Qt.NoFocus)
        self.centralwidget.setObjectName("centralwidget")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.appLogo.setPixmap(QtGui.QPixmap("res/PolyVisionLogo.png"))
        self.appLogo.setScaledContents(True)

        
        #setting font for labels
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.lowLabel.setFont(font)
        self.highLabel.setFont(font)
        self.moderateLabel.setFont(font)
        self.dataLabel.setFont(font)
        self.focusLabel.setFont(font)
        self.statusLabel.setFont(font)
        self.detectionLabel.setFont(font)
        self.progressLabel.setFont(font)
        self.locationLabel.setFont(font)
        
        #setting font for values
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.locationValue.setFont(font)
        self.lowValue.setFont(font)
        self.moderateValue.setFont(font)
        self.highValue.setFont(font)
        self.dateValue.setFont(font)
        self.focusValue.setFont(font)
        self.statusValue.setFont(font)
        self.detectionValue.setFont(font)
        self.lengthLabel.setFont(font)
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.xLabel.setFont(font)
        self.yLabel.setFont(font)
        self.zLabel.setFont(font)
        font.setPointSize(10)
        self.xValue.setFont(font)
        self.yValue.setFont(font)
        self.zValue.setFont(font)
        
        #seeting font for app Title
        font.setFamily("Gotham")
        font.setPointSize(14)
        font.setBold(True)
        self.appTitle.setFont(font)
        
        #setting font for section titles
        font.setPointSize(11)
        self.grblTitle.setFont(font)
        self.realTimeLabel.setFont(font)
        self.currentStatus.setFont(font)
        
        #setting font size for folder name
        font.setPointSize(22)
        self.placeValue.setFont(font)
        
        #===============Capture Button for Images=========================#
        self.captureButton      = QtWidgets.QPushButton(self.centralwidget) 
        self.measureButton      = QtWidgets.QPushButton(self.centralwidget)        
        self.detectButton       = QtWidgets.QPushButton(self.centralwidget)
        self.retrainButton      = QtWidgets.QPushButton(self.centralwidget)
        self.imagesButton       = QtWidgets.QPushButton(self.centralwidget)
        self.statisticsButton   = QtWidgets.QPushButton(self.centralwidget)
        self.settingsButton     = QtWidgets.QPushButton(self.centralwidget) 
        self.grblUP             = QtWidgets.QPushButton(self.centralwidget) 
        self.grblDOWN           = QtWidgets.QPushButton(self.centralwidget) 
        self.grblzUP            = QtWidgets.QPushButton(self.centralwidget) 
        self.grblzDOWN          = QtWidgets.QPushButton(self.centralwidget) 
        self.grblLEFT           = QtWidgets.QPushButton(self.centralwidget) 
        self.grblRIGHT          = QtWidgets.QPushButton(self.centralwidget) 
        self.grblHOME           = QtWidgets.QPushButton(self.centralwidget)
        self.connectGRBL        = QtWidgets.QPushButton(self.centralwidget)
        self.emergencyGRBL      = QtWidgets.QPushButton(self.centralwidget)
        self.calibrateBTN       = QtWidgets.QPushButton(self.centralwidget)
        self.scanBTN            = QtWidgets.QPushButton(self.centralwidget)
        self.focusBTN           = QtWidgets.QPushButton(self.centralwidget)
        self.scanButton         = QtWidgets.QPushButton(self.centralwidget) 


        #==================Stylesheets======================#
        self.graphicsView.setStyleSheet("background-color: rgb(0,0,0);") #Black
        self.cameraStatusLabel.setStyleSheet("color: #878787; background-color: rgba(0, 0, 0, 180); font: bold 16px; border-radius: 5px; padding: 10px;")
        self.lowLabel.setStyleSheet("color:#fbbf16;")
        self.moderateLabel.setStyleSheet("color:#fbbf16;")
        self.highLabel.setStyleSheet("color:#fbbf16;")
        self.progressLabel.setStyleSheet("color:#fbbf16;")
        self.detectionLabel.setStyleSheet("color:#fbbf16;")
        self.statusLabel.setStyleSheet("color:#fbbf16;")
        self.locationLabel.setStyleSheet("color:#fbbf16;")
        self.dataLabel.setStyleSheet("color:#fbbf16;")
        self.focusLabel.setStyleSheet("color:#fbbf16;")
        self.captureButton.setStyleSheet("QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 10px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        self.measureButton.setStyleSheet("QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 10px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        self.detectButton.setStyleSheet("QPushButton {\n""background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 133, 63, 255), stop:0.909091 rgba(255, 255, 255, 167));\n""    color: #000000;\n""    font: bold 16px;\n""    border-radius: 0px;\n""    border-color: #FFFFFF;\n""}\n""QPushButton:hover {\n""   \n""    color: #FFFFFF;\n""}\n""")
        self.retrainButton.setStyleSheet("QPushButton {\n""background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 133, 63, 255), stop:0.909091 rgba(255, 255, 255, 167));\n""    color: #000000;\n""    font: bold 16px;\n""    border-radius: 0px;\n""    border-color: #FFFFFF;\n""}\n""QPushButton:hover {\n""   \n""    color: #FFFFFF;\n""}\n""")
        self.imagesButton.setStyleSheet("QPushButton {\n""    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 133, 63, 255), stop:0.909091 rgba(255, 255, 255, 167));\n""    color: #000000;\n""    font: bold 16px;\n""    border-radius: 0px;\n""    border-color: #FFFFFF;\n""}\n""QPushButton:hover {\n""   \n""    color: #FFFFFF;\n""}\n""")
        self.statisticsButton.setStyleSheet("QPushButton {\n""    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 133, 63, 255), stop:0.909091 rgba(255, 255, 255, 167));\n""    color: #000000;\n""    font: bold 16px;\n""    border-radius: 0px;\n""    border-color: #FFFFFF;\n""}\n""QPushButton:hover {\n""   \n""    color: #FFFFFF;\n""}\n""")          
        self.settingsButton.setStyleSheet("QPushButton {\n""    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 133, 63, 255), stop:0.909091 rgba(255, 255, 255, 167));\n""    color: #000000;\n""    font: bold 16px;\n""    border-radius: 0px;\n""    border-color: #FFFFFF;\n""}\n""QPushButton:hover {\n""   \n""    color: #FFFFFF;\n""}\n""")
        self.dropDown.setStyleSheet(''' QComboBox {border: 1px solid #ccc;border-radius: 5px;padding: 1px;background-color: #ffffff;color: #000000;font-size: 12px;}QComboBox::drop-down {subcontrol-origin: padding;subcontrol-position: top right;width: 20px;}''')
        self.grblUP.setStyleSheet(u"QPushButton {\n""       background-color: qconicalgradient(cx:0.5, cy:0, angle:90.9, stop:0 rgba(255, 255, 255, 255), stop:0.37223 rgba(255, 255, 255, 255), stop:0.373991 rgba(251, 191, 22, 255), stop:0.62362 rgba(253, 202, 22, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 1px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""     background-color: qconicalgradient(cx:0.5, cy:0, angle:90.9, stop:0 rgba(255, 255, 255, 255), stop:0.37223 rgba(255, 255, 255, 255), stop:0.373991 rgba(158, 120, 14, 255), stop:0.62362 rgba(158, 120, 14, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""}")
        self.grblDOWN.setStyleSheet(u"QPushButton {\n""    background-color: qconicalgradient(cx:0.494318, cy:1, angle:270, stop:0 rgba(255, 255, 255, 255), stop:0.373989 rgba(255, 255, 255, 255), stop:0.373991 rgba(252, 191, 22, 255), stop:0.623986 rgba(252, 191, 22, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 1px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: qconicalgradient(cx:0.494318, cy:1, angle:270, stop:0 rgba(255, 255, 255, 255), stop:0.37223 rgba(255, 255, 255, 255), stop:0.373991 rgba(158, 120, 14, 255), stop:0.62362 rgba(158, 120, 14, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""}")
        self.grblzUP.setStyleSheet(u"QPushButton {\n""       background-color: qconicalgradient(cx:0.5, cy:0, angle:90.9, stop:0 rgba(255, 255, 255, 255), stop:0.37223 rgba(255, 255, 255, 255), stop:0.373991 rgba(251, 191, 22, 255), stop:0.62362 rgba(253, 202, 22, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 1px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""     background-color: qconicalgradient(cx:0.5, cy:0, angle:90.9, stop:0 rgba(255, 255, 255, 255), stop:0.37223 rgba(255, 255, 255, 255), stop:0.373991 rgba(158, 120, 14, 255), stop:0.62362 rgba(158, 120, 14, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""}")
        self.grblzDOWN.setStyleSheet(u"QPushButton {\n""    background-color: qconicalgradient(cx:0.494318, cy:1, angle:270, stop:0 rgba(255, 255, 255, 255), stop:0.373989 rgba(255, 255, 255, 255), stop:0.373991 rgba(252, 191, 22, 255), stop:0.623986 rgba(252, 191, 22, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 1px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: qconicalgradient(cx:0.494318, cy:1, angle:270, stop:0 rgba(255, 255, 255, 255), stop:0.37223 rgba(255, 255, 255, 255), stop:0.373991 rgba(158, 120, 14, 255), stop:0.62362 rgba(158, 120, 14, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""}")
        self.grblLEFT.setStyleSheet(u"QPushButton {\n""     background-color: qconicalgradient(cx:0, cy:0.499, angle:180.1, stop:0 rgba(255, 255, 255, 255), stop:0.375488 rgba(255, 255, 255, 255), stop:0.375911 rgba(252, 191, 22, 255), stop:0.622911 rgba(251, 191, 22, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 1px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: qconicalgradient(cx:0, cy:0.499, angle:180.1, stop:0 rgba(255, 255, 255, 255), stop:0.37548 rgba(255, 255, 255, 255), stop:0.375626 rgba(158, 120, 14, 255), stop:0.622911 rgba(158, 120, 14, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""}")
        self.grblRIGHT.setStyleSheet(u"QPushButton {\n""    background-color: qconicalgradient(cx:1, cy:0.499, angle:0.110478, stop:0 rgba(255, 255, 255, 255), stop:0.373989 rgba(255, 255, 255, 255), stop:0.373991 rgba(252, 191, 22, 255), stop:0.623986 rgba(252, 191, 22, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 1px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: qconicalgradient(cx:1, cy:0.499, angle:0.110478, stop:0 rgba(255, 255, 255, 255), stop:0.373989 rgba(255, 255, 255, 255), stop:0.373991 rgba(252, 191, 22, 255), stop:0.374003 rgba(158, 120, 14, 255), stop:0.623986 rgba(158, 120, 14, 255), stop:0.624043 rgba(255, 255, 255, 255), stop:1 rgba(255, 255, 255, 255));\n""}")
        self.grblHOME.setStyleSheet(u"QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 10px;\n""    border-radius: 1px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        self.boxWidget.setStyleSheet(u"\n""border: 2px solid #d3d3d3;\n""border-radius: 10px;\n" "background-color: transparent;\n")
        self.xWidget.setStyleSheet(u"\n""border: 1px solid #d3d3d3;\n""border-radius: 10px;\n" "background-color: transparent;\n")
        self.zWidget.setStyleSheet(u"\n""border: 1px solid #d3d3d3;\n""border-radius: 10px;\n" "background-color: transparent;\n")
        self.connectGRBL.setStyleSheet("QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 10px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        self.emergencyGRBL.setStyleSheet("QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 10px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        self.calibrateBTN.setStyleSheet("QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 10px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        self.scanBTN.setStyleSheet("QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 10px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        self.focusBTN.setStyleSheet("QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 10px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
        self.scanButton.setStyleSheet("QPushButton {\n""    background-color: #fbbf16;\n""    color: #FFFFFF;\n""    font: bold 16px;\n""    border-radius: 10px;\n""    border-color: #fbbf16;\n""}\n""QPushButton:hover {\n""    background-color: #9e780e;\n""}")
       
        #======================BUTTON ACTIONS===========================#
        self.captureButton.clicked.connect(self.captureButtonClicked)
        self.detectButton.clicked.connect(self.goToDetect)
        self.retrainButton.clicked.connect(self.goToRetrain)
        self.grblUP.clicked.connect(self.moveUp)
        self.grblDOWN.clicked.connect(self.moveDown)
        self.grblzUP.clicked.connect(self.focusUp)
        self.grblzDOWN.clicked.connect(self.focusDown)
        self.grblLEFT.clicked.connect(self.moveLeft)
        self.grblRIGHT.clicked.connect(self.moveRight)
        self.grblHOME.clicked.connect(self.moveHome)
        self.statisticsButton.clicked.connect(self.goToStatistics)
        self.settingsButton.clicked.connect(self.goToSettings)
        self.imagesButton.clicked.connect(self.goToCurrentImages)
        self.measureButton.clicked.connect(self.measureClicked)
        self.graphicsView.mousePressEvent = self.mousePressEvent
        self.graphicsView.mouseReleaseEvent = self.mouseReleaseEvent
        self.connectGRBL.clicked.connect(self.serialConnect)
        self.emergencyGRBL.clicked.connect(self.emergencyStop)
        self.calibrateBTN.clicked.connect(self.calibrate)
        self.scanBTN.clicked.connect(self.manualScanBTN)
        self.focusBTN.clicked.connect(self.startFocusing)
        self.scanButton.clicked.connect(self.manualScanMP)
        
        #======================APP&LOGOS COORDNT========================#
        self.appTitle       .setGeometry(QtCore.QRect(100, 70, 140,40))
        self.appLogo         .setGeometry(QtCore.QRect(30, 60, 60, 50))
        #=====================MAIN VIEW COORDINATES=====================#
        self.progressLabel   .setGeometry(QtCore.QRect(290, 817, 80, 30))
        self.progressBar     .setGeometry(QtCore.QRect(380, 823, 565, 25))
        self.graphicsView    .setGeometry(QtCore.QRect(290, 90, 1280, 720))
        self.cameraStatusLabel.setGeometry(QtCore.QRect(830, 425, 200, 50))
        #======================LOC. COORDINATES=========================#
        self.placeValue      .setGeometry(QtCore.QRect(1595, 60, 320, 50))
        self.locationLabel   .setGeometry(QtCore.QRect(1595, 110, 130, 30))
        self.locationValue   .setGeometry(QtCore.QRect(1745, 110, 200, 30))
        self.dataLabel       .setGeometry(QtCore.QRect(1595, 140, 130, 30))
        self.dateValue       .setGeometry(QtCore.QRect(1745, 140, 200, 30))
        #====================REPORT COORDINATES=========================#
        self.realTimeLabel   .setGeometry(QtCore.QRect(1595, 180, 175, 30))
        self.highLabel       .setGeometry(QtCore.QRect(1595, 210, 45, 30))
        self.highValue       .setGeometry(QtCore.QRect(1745, 210, 50, 30))
        self.moderateLabel   .setGeometry(QtCore.QRect(1595, 240, 90, 30))
        self.moderateValue   .setGeometry(QtCore.QRect(1745, 240, 50, 30))
        self.lowLabel        .setGeometry(QtCore.QRect(1595, 270, 40, 30))
        self.lowValue        .setGeometry(QtCore.QRect(1745, 270, 50, 30))
        #====================STATUS COORDINATES==========================
        self.currentStatus      .setGeometry(QtCore.QRect(1595, 310, 175, 30))
        self.focusLabel         .setGeometry(QtCore.QRect(1595, 340, 105, 30))
        self.focusValue         .setGeometry(QtCore.QRect(1745, 340, 100, 30))
        self.statusLabel        .setGeometry(QtCore.QRect(1595, 370, 60, 30))
        self.statusValue        .setGeometry(QtCore.QRect(1745, 370, 100, 30))
        self.detectionLabel     .setGeometry(QtCore.QRect(1595, 400, 135, 30))
        self.detectionValue     .setGeometry(QtCore.QRect(1745, 400, 100, 30))
        #====================PICTURE COORDINATES=========================#
        self.captureButton   .setGeometry(QtCore.QRect(1600, 440, 280, 35))
        self.measureButton   .setGeometry(QtCore.QRect(1600, 480, 280, 35))
        self.calibrateBTN    .setGeometry(QtCore.QRect(1600, 520, 280, 35))
        self.scanBTN        .setGeometry(QtCore.QRect(1600, 560, 280, 35))
        self.scanButton        .setGeometry(QtCore.QRect(1600, 870, 300, 35))
        #======================GRBL COORDINATES=========================#
        self.boxWidget       .setGeometry(QtCore.QRect(1595, 610, 300, 250))
        self.xWidget         .setGeometry(QtCore.QRect(1765, 690, 115, 150))
        self.zWidget         .setGeometry(QtCore.QRect(1610, 690, 150, 150))
        self.grblTitle       .setGeometry(QtCore.QRect(1645, 620, 200, 35))
        self.grblUP          .setGeometry(QtCore.QRect(1665, 695, 43, 43))
        self.grblDOWN        .setGeometry(QtCore.QRect(1665, 790, 43, 43))
        self.grblzUP          .setGeometry(QtCore.QRect(1800, 695, 43, 43))
        self.grblzDOWN        .setGeometry(QtCore.QRect(1800, 743, 43, 43))
        self.grblLEFT        .setGeometry(QtCore.QRect(1617, 743, 43, 43))
        self.grblRIGHT       .setGeometry(QtCore.QRect(1713, 743, 43, 43))
        self.grblHOME        .setGeometry(QtCore.QRect(1665, 743, 43, 43))
        self.xLabel          .setGeometry(QtCore.QRect(1622, 655, 45, 20))
        self.xValue          .setGeometry(QtCore.QRect(1652, 657, 50, 20))
        self.yLabel          .setGeometry(QtCore.QRect(1708, 655, 45, 20))
        self.yValue          .setGeometry(QtCore.QRect(1738, 657, 50, 20))
        self.zLabel          .setGeometry(QtCore.QRect(1800, 655, 45, 20))
        self.zValue          .setGeometry(QtCore.QRect(1820, 657, 50, 20))
        self.dropDown        .setGeometry(QtCore.QRect(950, 820, 300,30))
        self.connectGRBL     .setGeometry(QtCore.QRect(1260, 820, 150,30))
        self.emergencyGRBL   .setGeometry(QtCore.QRect(1420, 820, 150,30))
        self.focusBTN        .setGeometry(QtCore.QRect(1780, 795, 85, 30))
        #====================SETTINGS COORDINATES=========================#
        self.detectButton    .setGeometry(QtCore.QRect(0, 170, 261, 41))
        self.imagesButton    .setGeometry(QtCore.QRect(0, 260, 261, 41))
        self.statisticsButton.setGeometry(QtCore.QRect(0, 350, 261, 41))
        self.settingsButton  .setGeometry(QtCore.QRect(0, 440, 261, 41))
        self.retrainButton   .setGeometry(QtCore.QRect(0, 530, 261, 41))
        #====================menuBar============================#

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1253, 40))
        self.menubar.setStyleSheet("background-color: rgb(245, 245, 245);")
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.actionYT = QtWidgets.QAction(MainWindow)
        self.actionYT.setObjectName("actionYT")

        self.menubar.addAction(self.actionNew)
        self.menubar.addAction(self.actionYT)
        self.menubar.addAction(self.actionHelp)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "C.Scope.AI"))
        #==============Setting Label Texts=======================#
        self.locationLabel.setText(_translate("MainWindow", "Location:"))
        self.dataLabel.setText(_translate("MainWindow", "Date Sampled:"))
        self.focusLabel.setText(_translate("MainWindow", "Current MP:"))
        self.statusLabel.setText(_translate("MainWindow", "Status:"))
        self.detectionLabel.setText(_translate("MainWindow", "Auto Detection:"))
        self.progressLabel.setText(_translate("MainWindow", "Progress:"))
        self.realTimeLabel.setText(_translate("MainWindow", "Real-Time Report"))
        self.currentStatus.setText(_translate("MainWindow", "Current Status"))
        self.grblTitle.setText(_translate("MainWindow", "Platform Coordinates"))
        self.highLabel.setText(_translate("MainWindow", "High:"))
        self.moderateLabel.setText(_translate("MainWindow", "Moderate:"))
        self.lowLabel.setText(_translate("MainWindow", "Low:"))
        self.appTitle.setText(_translate("MainWindow", "C.Scope.AI"))
        self.captureButton.setText(_translate("MainWindow", "Capture"))
        self.measureButton.setText(_translate("MainWindow", "Measure"))
        self.grblzUP.setText(_translate("MainWindow", "↑"))
        self.grblUP.setText(_translate("MainWindow", "↑"))
        self.grblUP.setShortcut("Up")
        self.grblzDOWN.setText(_translate("MainWindow", "↓"))
        self.grblDOWN.setText(_translate("MainWindow", "↓"))
        self.grblDOWN.setShortcut("Down")
        self.grblLEFT.setText(_translate("MainWindow", "←"))
        self.grblLEFT.setShortcut("Left")
        self.grblRIGHT.setText(_translate("MainWindow", "→"))
        self.grblRIGHT.setShortcut("Right")
        self.grblHOME.setText(_translate("MainWindow", "HOME"))
        self.detectButton.setText(_translate("MainWindow", "Detect"))
        self.retrainButton.setText(_translate("MainWindow", "Retrain"))
        self.imagesButton.setText(_translate("MainWindow", "Images"))
        self.statisticsButton.setText(_translate("MainWindow", "Statistics"))
        self.settingsButton.setText(_translate("MainWindow", "Settings"))
        self.xLabel.setText(_translate("MainWindow","X  :"))
        self.xValue.setText(_translate("MainWindow","00.0"))
        self.yLabel.setText(_translate("MainWindow","Y  :"))
        self.yValue.setText(_translate("MainWindow","00.0"))
        self.zLabel.setText(_translate("MainWindow","Z  :"))
        self.zValue.setText(_translate("MainWindow","00.0"))
        self.connectGRBL.setText(_translate("MainWindow","Connect"))
        self.emergencyGRBL.setText(_translate("MainWindow","Emergency Stop"))
        self.calibrateBTN.setText(_translate("MainWindow","Calibrate"))
        self.scanBTN.setText(_translate("MainWindow","Scan"))
        self.focusBTN.setText(_translate("MainWindow","Focus"))
        self.scanButton.setText(_translate("MainWindow","Manual Scan"))

        #Menu Bar
        self.actionHelp.setText(_translate("MainWindow", "Help?"))
        self.actionHelp.setShortcut(_translate("MainWindow", "Ctrl+H"))
        self.actionYT.setText(_translate("MainWindow", "Guide"))
        self.actionYT.setShortcut(_translate("MainWindow", "Ctrl+Y"))
        self.actionNew.setText(_translate("MainWindow", "New File"))
        self.actionNew.setShortcut(_translate("MainWindow", "Ctrl+N"))

        #===========Menu Bar Actions=============#
        self.actionNew.triggered.connect(self.on_new_action)
        self.actionHelp.triggered.connect(SettingsUI.redirectDocumentation)
        self.actionYT.triggered.connect(SettingsUI.redirectYoutube)

        self.captureButton.setEnabled(False)
        self.measureButton.setEnabled(False)
        self.grblzUP.setEnabled(False)
        self.grblzDOWN.setEnabled(False)
        self.grblUP.setEnabled(False)
        self.grblDOWN.setEnabled(False)
        self.grblLEFT.setEnabled(False)
        self.grblRIGHT.setEnabled(False)
        self.grblHOME.setEnabled(False)
        self.connectGRBL.setEnabled(False)
        self.emergencyGRBL.setEnabled(False)
        self.imagesButton.setEnabled(False)
        self.detectButton.setEnabled(False)
        self.retrainButton.setEnabled(False)
        self.statisticsButton.setEnabled(False)
        self.calibrateBTN.setEnabled(False)
        self.scanBTN.setEnabled(False)
        self.focusBTN.setEnabled(False)
        self.scanButton.setVisible(False)
        # Initially hide camera status label
        self.cameraStatusLabel.setVisible(False)

        self.captureButton.setShortcut("C")
        self.measureButton.setShortcut("M")
        self.grblzUP.setShortcut("Ctrl+Up")
        self.grblzDOWN.setShortcut("Ctrl+Down")
        self.connectGRBL.setShortcut("G")
        self.emergencyGRBL.setShortcut("E")
        self.calibrateBTN.setShortcut("P")
        self.scanBTN.setShortcut("S")
        self.settingsButton.setShortcut("Ctrl+S")
        self.imagesButton.setShortcut("Ctrl+I")
        self.detectButton.setShortcut("Ctrl+D")
        self.statisticsButton.setShortcut("Ctrl+A")
        self.retrainButton.setShortcut("Ctrl+R")

    def showLargeResultsDialog(self, title, text):
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(title)
            dialog.resize(900, 700)
            layout = QVBoxLayout(dialog)
            scroll = QScrollArea(dialog)
            scroll.setWidgetResizable(True)
            container = QWidget()
            v = QVBoxLayout(container)
            label = QLabel(text, container)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setWordWrap(True)
            v.addWidget(label)
            scroll.setWidget(container)
            layout.addWidget(scroll)
            btn = QPushButton("Close", dialog)
            btn.clicked.connect(dialog.accept)
            layout.addWidget(btn)
            dialog.exec_()
        except Exception as e:
            print("showLargeResultsDialog error:", e)

    def TensionSummary(self, title, remarks=None, details=None):
        try:
            if details is None:
                details = remarks or ""
                remarks = None

            dialog = QDialog(self)
            dialog.setWindowTitle(title)
            dialog.setFixedSize(400, 300)  
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(15, 15, 15, 15)
            layout.setSpacing(12)
            
            # Title section
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_label)
            
            # Scrollable content area
            scroll = QScrollArea(dialog)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(12, 12, 12, 12)

            if remarks:
                remarks_label = QLabel(remarks)
                remarks_label.setWordWrap(True)
                v.addWidget(remarks_label)
            
            # Content label with better formatting
            content_label = QLabel(details)
            content_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            content_label.setWordWrap(True)
            content_label.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    color: #495057;
                    font-family: 'Consolas', 'Monaco', monospace;
                    line-height: 1.4;
                    padding: 8px;
                    font-size: 14px
                }
            """)
            v.addWidget(content_label)
            v.addStretch()
            
            scroll.setWidget(container)
            layout.addWidget(scroll)
            
            # Button section
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            close_btn = QPushButton("Close")
            close_btn.setFixedSize(100, 35)
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            
            # Center the dialog on the parent
            if self.parent():
                dialog.move(
                    self.parent().geometry().center() - dialog.rect().center()
                )
            
            dialog.exec_()
        except Exception as e:
            print("TensionSummary error:", e)

    # Custom positioned dialogs for tension calibration
    def TensionDialogPos(self, dialog_class, *args, **kwargs):
        try:
            dialog = dialog_class(self, *args, **kwargs)
            
            # Get graphics view position and size
            graphics_rect = self.graphicsView.geometry()
            graphics_global_pos = self.graphicsView.mapToGlobal(self.graphicsView.rect().topLeft())
            
            # Position in lower left of graphics view
            dialog_x = graphics_global_pos.x() + 20  # 20px from left edge
            dialog_y = graphics_global_pos.y() + graphics_rect.height() - dialog.height() - 20  # 20px from bottom
            
            dialog.move(dialog_x, dialog_y)
            return dialog
        except Exception as e:
            print(f"Error positioning tension dialog: {e}")
            return dialog_class(self, *args, **kwargs)

    def verificationBox(self, message):
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Tension Calibration")
            dialog.setFixedSize(250, 120)

            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 15, 20, 15)
            layout.setSpacing(15)

            message_label = QLabel(message)
            message_label.setWordWrap(True)
            layout.addWidget(message_label)

            button_layout = QHBoxLayout()

            yes_btn = QPushButton("Yes")
            no_btn  = QPushButton("No")
            no_btn.setObjectName("cancel")

            button_layout.addWidget(yes_btn)
            button_layout.addWidget(no_btn)
            layout.addLayout(button_layout)

            # Return distinct codes
            yes_btn.clicked.connect(lambda: dialog.done(1))  # YES
            no_btn.clicked.connect(lambda: dialog.done(2))   # NO

            graphics_rect = self.graphicsView.geometry()
            graphics_global_pos = self.graphicsView.mapToGlobal(self.graphicsView.rect().topLeft())
            dialog_x = graphics_global_pos.x() + 20
            dialog_y = graphics_global_pos.y() + graphics_rect.height() - dialog.height() - 20
            dialog.move(dialog_x, dialog_y)

            return dialog.exec_()  
        except Exception as e:
            print(f"Error in tension verification dialog: {e}")
            return VerificationBox(message).exec_()

        except Exception as e:
            print(f"Error in tension verification dialog: {e}")
            verify = VerificationBox(message)
            return verify.exec_()
    
    def TensionOkayBox(self, message):
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Tension Calibration")
            dialog.setFixedSize(300, 140)

            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 15, 20, 15)
            layout.setSpacing(15)

            # Message
            message_label = QLabel(message)
            message_label.setWordWrap(True)
            layout.addWidget(message_label)

            # Button
            button_layout = QHBoxLayout()

            ok_btn = QPushButton("OK")

            # Add OK button to the layout
            button_layout.addWidget(ok_btn)
            layout.addLayout(button_layout)

            # Connect button
            ok_btn.clicked.connect(dialog.accept)

            graphics_rect = self.graphicsView.geometry()
            graphics_global_pos = self.graphicsView.mapToGlobal(self.graphicsView.rect().topLeft())

            dialog_x = graphics_global_pos.x() + 20  
            dialog_y = graphics_global_pos.y() + graphics_rect.height() - dialog.height() - 20  
            dialog.move(dialog_x, dialog_y)

            # Show the dialog
            return dialog.exec_()

        except Exception as e:
            print(f"Error in tension OK dialog: {e}")
            return None

    def cancelCalibration(self):
        if self.calibrateBTN.text() == "Finish":
            self.calibrateBTN.setText("Calibrate")
            self.show_grid = False  
            self._hover_pos = None  
            self.graphicsView.setMouseTracking(False)  
            try:
                if hasattr(self, '_hover_timer') and self._hover_timer is not None:
                    self._hover_timer.stop()  
                    self._hover_timer = None
            except:
                pass
            self.paused = False  
        else:
            pass
           
    # Here are the changes
    def calibrateTension(self):
        try:
            self.calibrateBTN.setText("Finish")
            completed = False

            if not self.ser:
                prompt = OkayMessageBox("Connect to GRBL first")
                prompt.exec_()
                return

            # Keep live feed; enable grid overlay & hover tracking
            self.show_grid = True
            self._hover_pos = None
            self.graphicsView.setMouseTracking(True)
            try:
                self._hover_timer = QTimer(self)
                self._hover_timer.timeout.connect(self.updateHoverFromGlobal)
                self._hover_timer.start(33)
            except:
                pass

            # Move to center and confirm user is ready to start asking alignment questions
            self.moveToGridCenter()
            if self.graphicsView.pixmap() is None:
                OkayMessageBox("Camera is not ready. Start a file/camera first.").exec_()
                return

            go = self.TensionOkayBox("We will record alignment at each position.\nClick OK when you're ready to start.")
            if go != QDialog.Accepted:
                return

            results = []  # (label, bool)

            def ask_alignment(label):
                res = self.verificationBox("Is the crosshair aligned to the center after moving?")
                if res == 0:          
                    return False
                aligned = (res == 1)  
                results.append((label, aligned))
                time.sleep(0.05)
                return True

            travel_steps = 5  

            self.moveToGridCenter()

            self.moveLeftSteps(travel_steps)
            time.sleep(1.5)
            self.moveRightSteps(travel_steps)
            if not ask_alignment("Center After Left Movement"): return

            self.moveUpSteps(travel_steps)
            time.sleep(1.5)
            self.moveDownSteps(travel_steps)
            if not ask_alignment("Center After Up Movement"): return

            no_count = sum(1 for _, ok in results if not ok)
            status = "FAILED" if no_count >= 2 else "PASSED"

            lines = []
            for label, ok in results:
                mark = "✓ YES" if ok else "✗ NO"
                lines.append(f"{label:18} : {mark}")
            lines.append("")
            lines.append(f"NO count: {no_count}  →  {status}")

            # Show summary
            if status == "PASSED":
                remarks = "Alignment looks acceptable at most checkpoints."
            else:
                remarks = "Adjust tension and try again."
            self.TensionSummary(f"Tension Calibration {status}", remarks, "\n".join(lines))

            completed = True

        except Exception as e:
            print("Calibrate Tension error:", e)
        finally:
            if completed:
                # Turn off grid/hover and reset UI
                self.show_grid = False
                self._hover_pos = None
                self.graphicsView.setMouseTracking(False)
                self.calibrateBTN.setText("Calibrate")
                self.moveHome()
                try:
                    if hasattr(self, '_hover_timer') and self._hover_timer is not None:
                        self._hover_timer.stop()
                        self._hover_timer = None
                except:
                    pass

    def moveToGridCenter(self):
        try:
            if not self.ser:
                return
            feed = self.grbl_settings.get("max_feedrate", 1000)
            dx = 15 - self.x
            dy = 15 - self.y
            if dx != 0:
                if self.grbl_settings.get("area_scan", False):
                    cmd = f"G21 G91 G1 X{'-' if dx>0 else ''}{abs(dx)} F{feed}\r\n"
                else:
                    cmd = f"G21 G91 G1 X{'' if dx>0 else '-'}{abs(dx)} F{feed}\r\n"
                self.ser.write(cmd.encode('utf-8'))
                time.sleep(0.1 + abs(dx) * 0.05)
                self.x = 15
                self.xValue.setText(str(self.x))
            if dy != 0:
                if self.grbl_settings.get("area_scan", False):
                    cmd = f"G21 G91 G1 Y{'' if dy>0 else '-'}{abs(dy)} F{feed}\r\n"
                else:
                    cmd = f"G21 G91 G1 Y{'' if dy>0 else '-'}{abs(dy)} F{feed}\r\n"
                self.ser.write(cmd.encode('utf-8'))
                time.sleep(0.1 + abs(dy) * 0.05)
                self.y = 15
                self.yValue.setText(str(self.y))
        except:
            pass

    def moveLeftSteps(self, n):
        print(f"Moving left {n} steps...")
        for i in range(n):
            self.moveLeft()
            time.sleep(0.1)  
        print(f"Left movement completed. Current position: ({self.x}, {self.y})")

    def moveRightSteps(self, n):
        print(f"Moving right {n} steps...")
        for i in range(n):
            self.moveRight()
            time.sleep(0.1)  
        print(f"Right movement completed. Current position: ({self.x}, {self.y})")

    def moveUpSteps(self, n):
        print(f"Moving up {n} steps...")
        for i in range(n):
            self.moveUp()
            time.sleep(0.1)  
        print(f"Up movement completed. Current position: ({self.x}, {self.y})")

    def moveDownSteps(self, n):
        print(f"Moving down {n} steps...")
        for i in range(n):
            self.moveDown()
            time.sleep(0.1)  # Increased delay for better stability
        print(f"Down movement completed. Current position: ({self.x}, {self.y})")

    def drawGridOverlay(self):
        try:
            pixmap = self.graphicsView.pixmap()
            if pixmap is None:
                return
            qimage = pixmap.toImage()
            painter = QPainter(qimage)
            width = qimage.width()
            height = qimage.height()

            # --- ONLY the green center crosshair ---
            pen_center = QPen(QColor(0, 255, 0), 3)
            painter.setPen(pen_center)
            cx = int(15 * (width / 30.0))
            cy = int(15 * (height / 30.0))
            painter.drawLine(cx, 0, cx, height)
            painter.drawLine(0, cy, width, cy)

            painter.end()
            self.graphicsView.setPixmap(QPixmap.fromImage(qimage))
        except:
            pass


    def applyGridOverlay(self, qimage):
        try:
            painter = QPainter(qimage)
            width = qimage.width()
            height = qimage.height()

            # --- ONLY the green center crosshair (no yellow grid, no labels) ---
            pen_center = QPen(QColor(0, 255, 0), 3)
            painter.setPen(pen_center)
            cx = int(15 * (width / 30.0))
            cy = int(15 * (height / 30.0))
            painter.drawLine(cx, 0, cx, height)
            painter.drawLine(0, cy, width, cy)

            # (Optional) keep the small hover tooltip; remove this block if you want absolutely nothing else
            if self._hover_pos is not None:
                hx, hy = self._hover_pos
                gx = (hx / width) * 30.0
                gy = (hy / height) * 30.0
                txt = f"({gx:.1f}, {gy:.1f})"
                bg = QColor(0, 0, 0, 160)
                fg = QColor(255, 255, 255)
                metrics = painter.fontMetrics()
                tw = metrics.horizontalAdvance(txt)
                th = metrics.height()
                bx = min(max(0, hx + 10), width - tw - 6)
                by = min(max(th, hy + 10), height)
                painter.fillRect(bx, by - th, tw + 6, th, bg)
                painter.setPen(fg)
                painter.drawText(bx + 3, by - metrics.descent(), txt)

            painter.end()
        except:
            pass

    def applyHoverOverlay(self, qimage):
        try:
            if self._hover_pos is None:
                return
            painter = QPainter(qimage)
            width = qimage.width()
            height = qimage.height()
            hx, hy = self._hover_pos
            gx = (hx / width) * 30.0
            gy = (hy / height) * 30.0
            txt = f"({gx:.1f}, {gy:.1f})"
            # small target cross at cursor
            painter.setPen(QPen(QColor(0, 255, 0), 1))
            painter.drawLine(hx - 6, hy, hx + 6, hy)
            painter.drawLine(hx, hy - 6, hx, hy + 6)
            # tooltip box
            bg = QColor(0, 0, 0, 160)
            fg = QColor(255, 255, 255)
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(txt)
            th = metrics.height()
            bx = min(max(0, hx + 10), width - tw - 6)
            by = min(max(th, hy + 10), height)
            painter.fillRect(bx, by - th, tw + 6, th, bg)
            painter.setPen(fg)
            painter.drawText(bx + 3, by - metrics.descent(), txt)
            painter.end()
        except:
            pass


class VideoCapture(QThread):
    ImageUpdate = pyqtSignal(QImage)
    camera_ready = pyqtSignal()

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.detection_priority = False  # flag for AI Inference priority
        self.semaphore = QSemaphore(1)
        self.camera_initialized = False  
        
    def run(self):
        self.ThreadActive = True
        self.capture = cv2.VideoCapture(self.index)

        # Wait for camera to initialize and emit ready signal
        frame_count = 0
        while self.ThreadActive:
           
            if self.detection_priority:
                if not self.semaphore.tryAcquire(1, 1):
                    continue
                  
                self.semaphore.release(1)
                continue  
                
            ret, frame = self.capture.read()
            frame_count += 1
            
            if ret:
                # Emit camera ready signal on first successful frame
                if not self.camera_initialized:
                    self.camera_initialized = True
                    self.camera_ready.emit()
                
                FlippedImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], FlippedImage.strides[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1620,1215, Qt.KeepAspectRatio) 
                self.ImageUpdate.emit(Pic)
        self.capture.release()

    def set_detection_priority(self, priority):
        self.detection_priority = priority
        if priority:
            self.semaphore.acquire(1)
        else:
            if self.semaphore.available() == 0:
                self.semaphore.release(1)

    def stop(self):
        self.ThreadActive = False
        self.quit()

class AutoScan(QThread):
    ImageScan = pyqtSignal()
    Homing = pyqtSignal()
    Finished = pyqtSignal()
    def __init__(self,serial, start_x, start_y, rows, cols):
        super().__init__()
        self.ser = serial
        self.start_x = start_x
        self.start_y = start_y * (-1)
        self.rows = rows
        self.cols = cols 
        self.x = 0
        self.y = 0

    def run(self):
        self.ThreadActive = True
        self.event = threading.Event()
        while self.ThreadActive:
            try:
                self.ser.write(b"$H\r\n")
                time.sleep(5)
                self.Homing.emit()
                self.event.wait()  
                if self.event.is_set():
                    self.event.clear()
                #================= START AUTOMATED SCAN ================#
                string = "G21 G91 G1 X" 
                string += str(-(abs(self.start_x)+3.0)) + " F1000\r\n"
                toSend = string.encode('utf-8')
                self.ser.write(toSend)
                self.x -= self.start_x
                time.sleep((self.start_x/7*-1)+1)
                string = "G21 G91 G1 Y" 
                string += str(self.start_y) + " F1000\r\n"
                toSend = string.encode('utf-8')
                self.ser.write(toSend)
                self.y += self.start_y
                time.sleep((self.start_y/6.5)+1)

                #loop while petridish is unscanned
                for rows in range(int(self.rows)):
                    self.ImageScan.emit()
                    self.event.wait()  
                    if self.event.is_set():
                        self.event.clear()
                    for cols in range(int(self.cols)): 
                        if cols < self.cols - 1:
                            if rows%2 == 0:
                                self.ser.write(b"G21 G91 G1 X-5 F1000\r\n")
                                self.x += 5
                                time.sleep(3)
                            else:
                                self.ser.write(b"G21 G91 G1 X5 F1000\r\n")
                                self.x -= 5
                                time.sleep(3)
                            self.ImageScan.emit()
                            self.event.wait()  
                            if self.event.is_set():
                                self.event.clear()

                    if rows < self.rows - 1:  # Only move down if it's not the last row
                        self.ser.write(b"G21 G91 G1 Y3 F1000\r\n")
                        self.y += 3
                        time.sleep(3)                    

                self.ser.write(b"$H\r\n")

                self.Homing.emit()
                self.event.wait()  
                if self.event.is_set():
                    self.event.clear()

                self.Finished.emit()
                self.event.wait()  
                if self.event.is_set():
                    self.event.clear()
                self.ThreadActive = False
            except:
                pass


    def stop(self):
        self.ThreadActive = False
        self.quit()


class AutoFocus(QThread):
    focused = pyqtSignal()
    def __init__(self, serial, image_queue, zValue):
        super().__init__()
        self.ser = serial
        self.image_queue = image_queue
        text = zValue.text()
        self.z = float(text)
        self.zValue = zValue
        self.gcode_command = b"$X\r\n"
        self.ser.write(self.gcode_command)
        self.up = 1
        self.event = threading.Event()
        self.ThreadActive = True
        self.prev_blur_value = float('inf')
        self.blurThreshold = 600
        self.zinc = 5
        self.increment = 1


    def run(self):
        image = self.image_queue.pop()
        height = image.height()
        width = image.width()

        rows = 3
        cols = 3
        region_height = height // rows
        region_width = width // cols

        result_image =image.copy()

        y1, y2 = 1 * region_height, (1 + 1) * region_height
        x1, x2 = 1 * region_width, (1 + 1) * region_width

        region_rect = QRect(x1, y1, region_width, region_height)

        # Extract the region from the image
        region = image.copy(region_rect)

 
        self.blurValue, average_pixel_value = self.calculate_blur_and_color(region)
        while self.ThreadActive:  # Run indefinitely

            try:
                # Check if the image is near white or near black
                if average_pixel_value < 120:
                    self.blurThreshold = 500
                elif average_pixel_value > 150:
                    self.blurThreshold = 250
                else:
                   self.blurThreshold = 150

                if self.ThreadActive:

                    if self.blurValue < self.blurThreshold:
                        # Focus adjustment code here
                        if self.up:
                            string = "G21 G91 G1 Z" 
                            string +=  str(self.zinc) + " F1000\r\n"
                            toSend = string.encode('utf-8')
                            self.ser.write(toSend)
                            self.z += self.zinc
                        else:
                            if self.z > 0:
                                string = "G21 G91 G1 Z-" 
                                string +=  str(self.zinc) + " F1000\r\n"
                                toSend = string.encode('utf-8')
                                self.ser.write(toSend)
                                self.z -= self.zinc

                        self.zValue.setText(str(self.z))
                        time.sleep(1)

                        
                        difference = abs(self.blurValue - self.prev_blur_value)
                        if difference < 10:
                            self.zinc = 8
                        elif difference > 100:
                            self.zinc = 1
                        else:
                            self.zinc = 5


                        image = self.image_queue.pop()
                        self.prev_blur_value = self.blurValue
                        self.blurValue, average_pixel_value = self.calculate_blur_and_color(image)

                        if self.up:
                            if round(self.blurValue) < round(self.prev_blur_value):
                                self.up = not self.up
                        else:
                            if round(self.prev_blur_value) < round(self.blurValue):
                                self.up = not self.up

                            
                    else:
                        self.ThreadActive = False
                        self.focused.emit()
                        self.image_queue.clear()
                        self.quit()
                else:
                    pass
            except:
                pass

    def updateThread(self, value):
        self.ThreadActive = value

    def stop(self):
        self.ThreadActive = False
        self.quit()

    def calculate_blur_and_color(self, pic):

        np_image = self.qimage_to_numpy(pic)
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        blur_value = np.mean(gradient_magnitude)
        average_pixel_value = np.mean(np_image)

        return blur_value, average_pixel_value

    def qimage_to_numpy(self, qimage):
        width = qimage.width()
        height = qimage.height()
        byte_count = qimage.byteCount()
        format = qimage.format()

        # Convert the QImage to a NumPy array
        ptr = qimage.bits()
        ptr.setsize(byte_count)
        arr = np.frombuffer(ptr, dtype=np.uint8)

        if format == QImage.Format_RGB888:
            arr = arr.reshape(height, width, 3)
        elif format == QImage.Format_RGB32:
            arr = arr.reshape(height, width, 4)
            # Convert RGBA to RGB by removing alpha channel to fix tensor size mismatch
            # arr = arr[:, :, :3]

        return arr

class CalibrationChoiceDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection = None
        self.setWindowTitle("Calibration")
        self.setFixedSize(340, 100)
        self.setWindowIcon(QIcon("res/PolyVisionLogo.png"))

        layout = QtWidgets.QVBoxLayout(self)
        prompt = QtWidgets.QLabel("What do you want to calibrate?", self)
        prompt.setAlignment(Qt.AlignCenter)
        layout.addWidget(prompt)

        row = QtWidgets.QHBoxLayout()
        self.distance_btn = QtWidgets.QPushButton("Distance", self)
        self.tension_btn  = QtWidgets.QPushButton("Tension", self)
        row.addWidget(self.distance_btn)
        row.addWidget(self.tension_btn)
        layout.addLayout(row)

        self.distance_btn.clicked.connect(self.pick_distance)
        self.tension_btn.clicked.connect(self.pick_tension)

    def pick_distance(self):
        self.selection = "distance"
        self.accept()

    def pick_tension(self):
        self.selection = "tension"
        self.accept()

class ModelLoaderThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool)

    def __init__(self, model_types, warmup_callback=None, parent=None):
        super().__init__(parent)
        if isinstance(model_types, str):
            self.model_types = [model_types]
        else:
            self.model_types = list(model_types)
        self._warmup_callback = warmup_callback
        # Track the primary model requested so callers can detect duplicates.
        self.primary_model = self.model_types[0] if self.model_types else None

    def run(self):
        success = initialize_models(self.model_types)
        if success and self._warmup_callback:
            try:
                self._warmup_callback()
            except Exception as warmup_error:
                print(f"Warning: Model warmup failed: {warmup_error}")
        self.finished.emit(success)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)
        self.model_loader_thread = None
        self.model_preload_complete = False
        self._pending_model_type = None
        self.current_model_type = self.ui.general_settings.get("model", "Binary")

        self._start_model_preload(self.current_model_type)

    def _start_model_preload(self, model_type=None):
        target_model = model_type or self.ui.general_settings.get("model", "Binary")
        print(f"Pre-loading {target_model} model in background for optimal performance...")
        self.model_preload_complete = False

        if is_models_ready(target_model):
            print(f"{target_model} model already loaded; skipping background preload.")
            self.model_preload_complete = True
            return

        if is_model_loading(target_model):
            print(f"{target_model} model is already being loaded by another worker.")
            return

        if self.model_loader_thread and self.model_loader_thread.isRunning():
            running_model = getattr(self.model_loader_thread, "primary_model", None)
            if running_model == target_model:
                print(f"{target_model} model is currently loading via existing worker.")
                return
            print(f"Background loader busy; queuing request to load {target_model}.")
            self._pending_model_type = target_model
            return

        self.model_loader_thread = ModelLoaderThread(
            model_types=target_model,
            warmup_callback=self.warmup_models
        )
        self.model_loader_thread.finished.connect(self._on_model_preload_finished)
        self.model_loader_thread.start()

    def handle_model_selection_change(self, new_model_type):
        if new_model_type == self.current_model_type:
            return
        print(f"Main window updating model selection: {self.current_model_type} -> {new_model_type}")
        self.current_model_type = new_model_type
        self._start_model_preload(new_model_type)

    def _on_model_preload_finished(self, success):
        self.model_preload_complete = success
        if success:
            print("Model pre-loading completed in the background.")
        else:
            print("Model pre-loading failed; models will be loaded on demand.")
        self.model_loader_thread = None
        if self._pending_model_type:
            pending = self._pending_model_type
            self._pending_model_type = None
            if not is_models_ready(pending) and not is_model_loading(pending):
                self._start_model_preload(pending)

    def warmup_models(self):
        print("Warming up models with dummy inference...")
        try:
            dummy_image = Image.new("RGB", (224, 224), color=0)
            start_time = time.time()
            _ = loadModel(dummy_image)
            warmup_time = time.time() - start_time
            
            print(f"Model warmup completed in {warmup_time:.3f} seconds")
            print("Models are now ready for optimal performance!")
            
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")
            print("Models will still work but first detection may be slower")

    def closeEvent(self, event):
        if self.model_loader_thread and self.model_loader_thread.isRunning():
            self.model_loader_thread.wait(2000)

        if hasattr(self.ui, 'VideoCapture') and self.ui.VideoCapture is not None:
            self.ui.VideoCapture.ThreadActive = False
            self.ui.VideoCapture.stop()
            self.ui.VideoCapture.wait()  
            self.ui.VideoCapture.deleteLater()  
        
        if self.ui.ser:
            if self.ui.z != 0:
                string = "G21 G91 G1 Z-" 
                string += str(self.ui.z) + " F1000\r\n"
                toSend = string.encode('utf-8')
                self.ui.ser.write(toSend)
            string = "$H\r\n"
            toSend = string.encode('utf-8')
            self.ui.ser.write(toSend)

        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    #app.setQuitOnLastWindowClosed(False)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
