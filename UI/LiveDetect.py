# Not used as of the moment since the model is loaded in PolyVisionMain.py
# You can found in method manualScanMP and scanForMP

import sys
import os
import json
import io
import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PIL.ImageQt import ImageQt  # Import ImageQt from PIL to convert PIL Image to QImage
from PIL import Image
from localization import LocalDetectMP  

class_mapping = {
    1: "filament",
    2: "film",
    3: "fragment"
    }

class LiveDetectMP():
    def __init__(self,image, port, parent=None):
        
        self.result = []
        
        image_bytes = io.BytesIO()
        image.save(image_bytes, "PPM")
        image_bytes.seek(0)
        image = Image.open(image_bytes)

        print("Starting live detection...")
        start_time = time.perf_counter()
        detector = LocalDetectMP(image, port)
        self.result = detector.get_json()
        
        print(f"Live detection found {len(self.result)} objects")
        if self.result:
            print(json.dumps(self.result, indent=4))
            
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Live detection: {elapsed_time:.2f} seconds")
        
    def get_json(self):
        return self.result

class BoundingBox():
    def __init__(self, imageLoad, plots, port, parent=None):
        # Bounding box data from the JSON object
        bounding_boxes = plots

        image_np = np.array(imageLoad)

        if not port:
            # Draw bounding boxes on the image
            for bbox_data in bounding_boxes:
                bbox = bbox_data["bbox"]
                class_id = bbox_data["class_id"]
                score = bbox_data["score"]

                x_min, y_min, x_max, y_max = map(int, bbox)
                #add random numbers
                color = (50, 150, 50)
                thickness = 2

                # Draw the bounding box
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, 1)

                # Add label with class ID and score
                label = f"MP, Score: {score * 100:.2f}%"
                label_position = (x_min, y_min-5)
                cv2.putText(image_np, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            for bbox_data in bounding_boxes:
                bbox = bbox_data["bbox"]
                class_id = bbox_data["class_id"]
                score = bbox_data["score"]

                x_min, y_min, x_max, y_max = map(int, bbox)
                color = (50, 150, 50)

                thickness = 2

                # Draw the bounding box
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, thickness)

                # Add label with class name and score
                class_name = class_mapping.get(class_id, "Unknown")
                label = f"{class_name}, Score: {score * 100:.2f}%"
                label_position = (x_min, y_min - 10)
                cv2.putText(image_np, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Convert the modified NumPy array back to a QImage
        self.image = image_np


    def get_image(self):
        return self.image