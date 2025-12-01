import os
import sys
import cv2
import numpy as np
import json
import torch
import time
import threading
from PIL import Image
from InferenceBinary import Binary
from InferenceMulticlass import Multiclass

# GLOBAL VARIABLES FOR MODEL MANAGEMENT
GLOBAL_BINARY_DETECTOR = None
GLOBAL_MULTICLASS_DETECTOR = None

_MODEL_STATUS_LOCK = threading.Lock()
_MODEL_STATUS = {
    "Binary": {"loaded": False, "loading": False, "error": None},
    "Multiclass": {"loaded": False, "loading": False, "error": None},
}

def initialize_models(model_types=None):
    """
    Lazily initialize one or more models. When model_types is None, both
    detectors are loaded (previous behaviour). Returns True when all requested
    models are ready.
    """
    global GLOBAL_BINARY_DETECTOR, GLOBAL_MULTICLASS_DETECTOR

    if model_types is None:
        target_models = ["Binary", "Multiclass"]
    elif isinstance(model_types, str):
        target_models = [model_types]
    else:
        target_models = list(model_types)

    success = True

    for model_type in target_models:
        if model_type not in _MODEL_STATUS:
            print(f"Unknown model type requested: {model_type}")
            success = False
            continue

        # Wait for an in-flight load to complete before deciding what to do
        while True:
            with _MODEL_STATUS_LOCK:
                status = _MODEL_STATUS[model_type]
                if status["loaded"]:
                    break
                if not status["loading"]:
                    status["loading"] = True
                    status["error"] = None
                    break
            # Another thread is loading this model – wait briefly
            time.sleep(0.05)

        with _MODEL_STATUS_LOCK:
            status = _MODEL_STATUS[model_type]
            if status["loaded"]:
                continue

        print(f"Loading {model_type} model...")
        model_start = time.perf_counter()

        try:
            if model_type == "Binary":
                detector = Binary()
                if not detector.is_ready():
                    raise RuntimeError("Binary detector failed to report ready")
                GLOBAL_BINARY_DETECTOR = detector
            elif model_type == "Multiclass":
                detector = Multiclass()
                if not detector.is_ready():
                    raise RuntimeError("Multiclass detector failed to report ready")
                GLOBAL_MULTICLASS_DETECTOR = detector

            model_end = time.perf_counter()
            print(f"{model_type} model loaded in {model_end - model_start:.3f} seconds")
            with _MODEL_STATUS_LOCK:
                status = _MODEL_STATUS[model_type]
                status["loaded"] = True
                status["error"] = None
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            with _MODEL_STATUS_LOCK:
                status = _MODEL_STATUS[model_type]
                status["loaded"] = False
                status["error"] = str(e)
            success = False
        finally:
            with _MODEL_STATUS_LOCK:
                _MODEL_STATUS[model_type]["loading"] = False

    if success:
        torch.set_num_threads(6)

    return success

def get_current_detector():
    """
    Get the appropriate detector based on current settings.
    If the detector is not ready, attempt to load it on demand.
    """
    model_type = get_model_type_from_settings()

    if not is_models_ready(model_type):
        print(f"{model_type} model not ready, initializing now...")
        if not initialize_models(model_type):
            return None

    if model_type == "Binary":
        if GLOBAL_BINARY_DETECTOR is None:
            print("Error: Binary detector is None")
            return None
        return GLOBAL_BINARY_DETECTOR

    if GLOBAL_MULTICLASS_DETECTOR is None:
        print("Error: Multiclass detector is None")
        return None
    return GLOBAL_MULTICLASS_DETECTOR

def get_model_type_from_settings():
    try:
        settings_file = "user_settings.json"
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
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
    model_type = get_model_type_from_settings()

    if not is_models_ready(model_type):
        print(f"{model_type} model not ready, initializing now...")
        if not initialize_models(model_type):
            return None

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
            
            # Cleanup after detection
            del image_array
                
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
                detections = loadModel(image_input)
                
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
        
        if model_type is not None and model_type in _MODEL_STATUS:
            if not is_models_ready(model_type):
                print(f"{model_type} model requested for image detection; initializing...")
                if not initialize_models(model_type):
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
        settings_file = "user_settings.json"
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        
        if "general_features" not in settings:
            settings["general_features"] = {}
        
        settings["general_features"]["model"] = new_model_type
        
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        
        print(f"Switched to {new_model_type} model (will load on demand if needed)")
        return True
        
    except Exception as e:
        print(f"Error updating settings: {e}")
        return False

def get_current_model_type():
    return get_model_type_from_settings()

def is_models_ready(model_type=None):
    if model_type is None:
        model_type = get_model_type_from_settings()
    with _MODEL_STATUS_LOCK:
        status = _MODEL_STATUS.get(model_type)
        if status is None:
            return False
        return status["loaded"]

def is_model_loading(model_type=None):
    if model_type is None:
        model_type = get_model_type_from_settings()
    with _MODEL_STATUS_LOCK:
        status = _MODEL_STATUS.get(model_type)
        if status is None:
            return False
        return status["loading"]


if __name__ == "__main__":
    # If run directly, initialize models
    initialize_models()
