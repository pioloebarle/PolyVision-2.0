import os
import sys
import cv2
import numpy as np
import torch
import glob
import gc
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

class Binary():
    def __init__(self):
        self.predictor = None
        self.model_loaded = False
        self.setup_detectron2() 
    
    def find_model_path(self, base_path, architecture):
        try:
            path = os.path.join(base_path, architecture, "*", "model_final.pth")
            model_files = glob.glob(path)

            if not model_files:
                print(f"No model files found matching pattern: {path}")
                return None
            
            # OLD METHOD: Using file modification time (commented out for testing)
            # latest_model = max(model_files, key=os.path.getmtime) 
            
            # NEW METHOD: Extract directory names (timestamps) and sort by them instead of file modification time
            def get_timestamp_from_path(file_path):
                # Extract the timestamp directory name (e.g., "2025-10-01-03-07-35")
                timestamp_dir = os.path.basename(os.path.dirname(file_path))
                return timestamp_dir
            
            # Sort by timestamp directory name (newest first)
            latest_model = max(model_files, key=get_timestamp_from_path)
            print(f"Found {len(model_files)} model(s), using latest: {latest_model}")
            return latest_model
        
        except Exception as e:
            print(f"Error finding model path: {e}")
            return None
        
    def setup_detectron2(self):
        """Setup detectron2 once - similar to Docker initialization"""
        DATA_SET_NAME = "SEAMaP-Binary-Full"
        ARCHITECTURE = "faster_rcnn_R_50_FPN_3x"
        try:
            base_path = f"../Models/{DATA_SET_NAME}"
            MODEL_PATH = self.find_model_path(base_path, ARCHITECTURE)
            
            if MODEL_PATH is None:
                print(f"Error: No model file found for architecture {ARCHITECTURE}")
                return False
            
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
                return False
            
            print(f"Loading Binary model from: {MODEL_PATH}")
            
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{ARCHITECTURE}.yaml"))
            cfg.MODEL.WEIGHTS = MODEL_PATH
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Binary: background + microplastic
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
            cfg.MODEL.DEVICE = "cpu"  
            
            # OPTIMIZATION: Increase CPU threads for faster inference
            torch.set_num_threads(6)  
            
            self.predictor = DefaultPredictor(cfg)
            self.model_loaded = True
            
            print("Binary model loaded and ready for inference")
            print(f"Model configured for {cfg.MODEL.ROI_HEADS.NUM_CLASSES} classes")
            return True
        
        except Exception as e:
            print(f"Error setting up Binary Detectron2: {e}")
            self.model_loaded = False
            return False
    
    def imagePrediction(self, image_path):
        """Fast inference without model reloading - mimics Docker /detect endpoint"""
        
        if not self.model_loaded or self.predictor is None:
            print("Binary predictor not initialized. Please run setup_detectron2() first.")
            return None
    
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image from {image_path}")
                return None
            
            height, width = image.shape[:2]
            original_size = max(height, width)
            
            if original_size > 1200:
                # Only resize significantly large images for speed
                scale_factor = 800 / original_size
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = cv2.resize(image, (new_width, new_height))
                print(f"Resized large image from {width}x{height} to {new_width}x{new_height} for faster processing")
            
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image_tensor, "height": image_tensor.shape[1], "width": image_tensor.shape[2]}]
            
            with torch.no_grad():
                gc.collect()
                outputs = self.predictor.model(inputs)[0]
            
            
            instances = outputs["instances"]
            filtered_instances = instances[instances.scores > 0.7]  # Confidence threshold
            
            # Convert to detection results format
            detection_results = []
            for i in range(len(filtered_instances)):
                instance = filtered_instances[i]

                bbox = instance.pred_boxes.tensor.cpu().detach().numpy()[0].tolist()
                class_id = instance.pred_classes.cpu().detach().numpy()[0].item()
                score = instance.scores.cpu().detach().numpy()[0].item()

                detection_results.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "score": score 
                })

            print(f"Binary model found {len(detection_results)} detections")
            return detection_results

        except Exception as e:
            print(f"Error during Binary image prediction: {e}")
            return None   
    
    def arrayPrediction(self, image_array):
        """Direct array prediction for PIL Image inputs - optimized for speed"""
        
        if not self.model_loaded or self.predictor is None:
            print("Binary predictor not initialized. Please run setup_detectron2() first.")
            return None
    
        try:
            # Image is already a numpy array from PIL conversion
            if image_array is None:
                print("Error: Empty image array")
                return None

            height, width = image_array.shape[:2]
            original_size = max(height, width)
            
            if original_size > 640:
                scale_factor = 640 / original_size
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image_array = cv2.resize(image_array, (new_width, new_height))

            image_tensor = torch.as_tensor(image_array.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image_tensor, "height": image_tensor.shape[1], "width": image_tensor.shape[2]}]
            
            with torch.no_grad():
                gc.collect()
                self.predictor.model.eval()
                outputs = self.predictor.model(inputs)[0]
            
            instances = outputs["instances"]
            filtered_instances = instances[instances.scores > 0.7]  # Confidence threshold
            
            # Convert to detection results format
            detection_results = []
            for i in range(len(filtered_instances)):
                instance = filtered_instances[i]

                bbox = instance.pred_boxes.tensor.cpu().detach().numpy()[0].tolist()
                class_id = instance.pred_classes.cpu().detach().numpy()[0].item()
                score = instance.scores.cpu().detach().numpy()[0].item()

                detection_results.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "score": score 
                })

            print(f"Binary model found {len(detection_results)} detections (direct array)")
            return detection_results

        except Exception as e:
            print(f"Error during Binary array prediction: {e}")
            return None   
        
    def customMetadata(self):
        """Setup metadata for visualization"""
        classes = ["background", "microplastic"]  # Class 0 = background, Class 1 = microplastic
        class_colors = [
            (128, 128, 128),  # background - gray
            (0, 255, 0)       # microplastic - green
        ]
        
        MetadataCatalog.get("microplastic_dataset").thing_classes = classes
        MetadataCatalog.get("microplastic_dataset").thing_colors = class_colors
        
        return "microplastic_dataset"
        
    def visualization_predictions(self, image_path, save_path=None):
        """Visualize predictions using preloaded model"""
        detections = self.imagePrediction(image_path)
        if detections is None:
            return None
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            dataset_name = self.customMetadata()
             
            v = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get(dataset_name),
                scale=1.2
            )
            
            # Convert detections back to Detectron2 format for visualization
            boxes = torch.tensor([det["bbox"] for det in detections])
            scores = torch.tensor([det["score"] for det in detections])
            classes = torch.tensor([det["class_id"] for det in detections])
            
            instances = Instances((image.shape[0], image.shape[1]))
            instances.pred_boxes = Boxes(boxes)
            instances.scores = scores
            instances.pred_classes = classes
            
            with torch.no_grad():
                out = v.draw_instance_predictions(instances)
                
            output_image = out.get_image()[:, :, ::-1]
            
            if save_path:
                cv2.imwrite(save_path, output_image)
                print(f"Saved Binary visualization to {save_path}")
            
            return output_image
        
        except Exception as e:
            print(f"Error during Binary visualization: {e}")
            return None
    
    def is_ready(self):
        """Check if model is loaded and ready"""
        return self.model_loaded and self.predictor is not None