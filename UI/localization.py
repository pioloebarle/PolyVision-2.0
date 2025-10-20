import os
import sys
import cv2
import numpy as np
import json
import torch
import time
from PIL import Image
from InferenceBinary import Binary
from InferenceMulticlass import Multiclass

# GLOBAL VARIABLES FOR MODEL MANAGEMENT
GLOBAL_BINARY_DETECTOR = None
GLOBAL_MULTICLASS_DETECTOR = None
MODELS_INITIALIZED = False

def initialize_models():
    global GLOBAL_BINARY_DETECTOR, GLOBAL_MULTICLASS_DETECTOR, MODELS_INITIALIZED
    
    if MODELS_INITIALIZED:
        print("Models already initialized, skipping...")
        return True
    
    print("Initializing models at startup...")
    start_time = time.perf_counter()
    
    try:
        # Preload Binary model
        print("Loading Binary model...")
        binary_start = time.perf_counter()
        GLOBAL_BINARY_DETECTOR = Binary()
        binary_end = time.perf_counter()
        print(f"Binary model loaded in {binary_end - binary_start:.3f} seconds")
        
        # Preload Multiclass model
        print("Loading Multiclass model...")
        multi_start = time.perf_counter()
        GLOBAL_MULTICLASS_DETECTOR = Multiclass()
        multi_end = time.perf_counter()
        print(f"Multiclass model loaded in {multi_end - multi_start:.3f} seconds")
        
        torch.set_num_threads(6)  
        
        MODELS_INITIALIZED = True
        total_time = time.perf_counter() - start_time
        print(f"All models initialized successfully in {total_time:.3f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        GLOBAL_BINARY_DETECTOR = None
        GLOBAL_MULTICLASS_DETECTOR = None
        MODELS_INITIALIZED = False
        return False

def get_current_detector():
    """
    Get the appropriate detector based on current settings
    No model loading - just returns the preloaded model
    """
    global GLOBAL_BINARY_DETECTOR, GLOBAL_MULTICLASS_DETECTOR, MODELS_INITIALIZED
    
    # Ensure models are initialized
    if not MODELS_INITIALIZED:
        print("Models not initialized, initializing now...")
        if not initialize_models():
            return None
    
    # Get model type from settings
    model_type = get_model_type_from_settings()
    
    if model_type == "Binary":
        if GLOBAL_BINARY_DETECTOR is None:
            print("Error: Binary detector is None")
            return None
        return GLOBAL_BINARY_DETECTOR
    else:  # Multiclass
        if GLOBAL_MULTICLASS_DETECTOR is None:
            print("Error: Multiclass detector is None")
            return None
        return GLOBAL_MULTICLASS_DETECTOR

def get_model_type_from_settings():
    try:
        if os.path.exists("user_settings.json"):
            with open("user_settings.json", 'r') as f:
                settings = json.load(f)
                return settings.get("general_features", {}).get("model", "Binary")
    except Exception as e:
        print(f"Could not read settings, using Binary model: {e}")
    return "Binary"

def Detector():
    return get_current_detector()

def loadModel(image_input):
    """
    Enhanced loadModel that accepts both file paths and PIL Images
    """
    # Ensure models are initialized
    if not MODELS_INITIALIZED:
        print("Models not initialized, initializing now...")
        if not initialize_models():
            return None
    
    # Get preloaded detector (no model loading here)
    detector = get_current_detector()
    if detector is None:
        print("Error: Could not get detector")
        return None
    
    try:
        detection_start = time.perf_counter()
        
        if isinstance(image_input, Image.Image):
            import numpy as np
            image_array = np.array(image_input)
            # Convert RGB to BGR for OpenCV compatibility
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            with torch.no_grad():
                detections = detector.arrayPrediction(image_array)
                
        elif isinstance(image_input, str):

            with torch.no_grad():
                detections = detector.imagePrediction(image_input)
        else:
            print(f"Unsupported image input type: {type(image_input)}")
            return None
        
        detection_end = time.perf_counter()
        detection_time = detection_end - detection_start
        
        model_type = get_model_type_from_settings()
        
        if detections is not None:
            print(f"Detection time: {detection_time:.3f} seconds using {model_type} model")
            return detections
        
        print(f"Detection completed in {detection_time:.3f} seconds with no objects")
        return []
        
    except Exception as e:
        print(f"Detection error: {e}")
        return None

class LocalDetectMP():
    
    def __init__(self, image_input, port=None, parent=None):
        self.result = []
        
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    print(f"Image file not found: {image_input}")
                    return
                
                detections = loadModel(image_input)
                
                if detections is not None:
                    self.result = detections
                    print(f"Local detection found {len(self.result)} objects")
                else:
                    print("Failed to get prediction results")
                    
            elif isinstance(image_input, Image.Image):
                temp_path = "temp_image.jpg"
                image_input.save(temp_path)
                
                detections = loadModel(temp_path)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if detections is not None:
                    self.result = detections
                    print(f"Local detection found {len(self.result)} objects")
                else:
                    print("Failed to get prediction results")
            else:
                print("Invalid image input type")
                return
            
        except Exception as e:
            print(f"Error in local detection: {e}")
    
    def get_json(self):
        """Return results in the expected format"""
        return self.result

def imageDetection(image_path, confidence=0.7, save_result=True, model_type=None):
    print(f"Processing: {image_path}")
    
    try:
        
        if not os.path.exists(image_path):
            print(f"Could not find image: {image_path}")
            return None
        
        
        detections = loadModel(image_path)
        
        if detections is None:
            print("Failed to get detections")
            return None
        
        print(f"Found {len(detections)} detections")
        
        
        if save_result and detections:
            try:
                detector = get_current_detector()
                if detector:
                    vis_image = detector.visualization_predictions(image_path)  
                    if vis_image is not None:
                        output_path = f"detected_{os.path.basename(image_path)}"
                        cv2.imwrite(output_path, vis_image)
                        print(f"Saved result to: {output_path}")
                        return output_path
            except Exception as vis_error:
                print(f"Visualization failed: {vis_error}")
        
        return {
            "boxes": [det["bbox"] for det in detections],
            "scores": [det["score"] for det in detections], 
            "classes": [det["class_id"] for det in detections],
            "detections": detections,
            "model_type": get_model_type_from_settings()
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def switch_model(new_model_type):
    if new_model_type not in ["Binary", "Multiclass"]:
        print(f"Invalid model type: {new_model_type}")
        return False
    
    # Update settings file
    try:
        settings = {}
        if os.path.exists("user_settings.json"):
            with open("user_settings.json", 'r') as f:
                settings = json.load(f)
        
        if "general_features" not in settings:
            settings["general_features"] = {}
        
        settings["general_features"]["model"] = new_model_type
        
        with open("user_settings.json", 'w') as f:
            json.dump(settings, f, indent=4)
            
        print(f"Switched to {new_model_type} model (no reloading needed)")
        return True
        
    except Exception as e:
        print(f"Error updating settings: {e}")
        return False

def get_current_model_type():
    return get_model_type_from_settings()

def is_models_ready():
    return MODELS_INITIALIZED and GLOBAL_BINARY_DETECTOR is not None and GLOBAL_MULTICLASS_DETECTOR is not None


if __name__ == "__main__":
    # If run directly, initialize models
    initialize_models()
