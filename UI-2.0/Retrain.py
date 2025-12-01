# Retrain.py

import os
import sys
import json
import sqlite3
import subprocess # NEW: To run the external training script
import threading  # NEW: Although QThread handles it, it's good practice
import shutil
import random
import yaml
import glob
from benchmark import main as benchmark_main
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ImportDialog import ImportDialog
from Database import create_retraining_database
from datetime import datetime
from ComparisonDialog import ComparisonDialog
from pathlib import Path
import cv2
import time

# Ensure project root is available on sys.path so we can import Models package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Models.Retraining.data_curation import (
    DEFAULT_SELECTION_MIX,
    DIFFICULTY_ORDER,
    build_or_load_manifest,
    select_anchor_subset,
    get_anchor_loss_statistics,
    compute_filtering_threshold,
    filter_new_data_by_loss,
    calculate_dynamic_iterations,
    format_iteration_summary,
)

BASE_OUTPUT_DIRECTORY = "../Models/"



def get_retraining_data():
    db_path = 'retrain_images.db'
    if not os.path.exists(db_path): return []
    try:
        connection = sqlite3.connect(db_path)
        c = connection.cursor()
        c.execute("SELECT image_name, is_microplastic, bounding_box FROM database_list ORDER BY rowid DESC")
        data = c.fetchall()
        connection.close()
        return data
    except Exception as e:
        print(f"Error reading retraining database: {e}")
        return []

def import_coco_data(image_dir, json_path, progress_callback):
    """
    Processes a COCO dataset and imports it into the project's retraining database.
    Returns the number of images successfully imported.
    """
    target_img_dir = "retrainingImages"
    db_path = "retrain_images.db"
    os.makedirs(target_img_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_map = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    images_to_process = list(images_map.values())
    total_images = len(images_to_process)
    imported_count = 0

    for i, image_info in enumerate(images_to_process):
        progress_callback.emit(f"Processing image {i+1}/{total_images}: {image_info['file_name']}")
        
        original_image_path = os.path.join(image_dir, image_info['file_name'])
        if not os.path.exists(original_image_path):
            progress_callback.emit(f"  -> Warning: Image not found. Skipping.")
            continue

        # Get current row count to create a unique name
        cursor.execute("SELECT COUNT(*) FROM database_list")
        next_idx = cursor.fetchone()[0]
        
        new_image_name = f"image_{next_idx}.png"
        target_image_path = os.path.join(target_img_dir, new_image_name)
        
        shutil.copy(original_image_path, target_image_path)
        
        is_mp = False
        bounding_boxes_for_db = []
        
        img_id = image_info['id']
        if img_id in annotations_by_image:
            is_mp = True # If there are annotations, it's a positive sample
            for ann in annotations_by_image[img_id]:
                # Convert COCO [x, y, w, h] to Detectron2 [xmin, ymin, xmax, ymax]
                x, y, w, h = ann['bbox']
                db_bbox = {
                    "bbox": [x, y, x + w, y + h],
                    "class_id": ann['category_id'],
                    "score": 1.0 # Ground truth
                }
                bounding_boxes_for_db.append(db_bbox)

        bounding_box_str = json.dumps(bounding_boxes_for_db)
        cursor.execute("INSERT INTO database_list (image_name, is_microplastic, bounding_box) VALUES (?, ?, ?)",
                      (new_image_name, int(is_mp), bounding_box_str))
        
        imported_count += 1

    connection.commit()
    connection.close()
    return imported_count

class ImportThread(QThread):
    progress_update = pyqtSignal(str)
    import_finished = pyqtSignal(int) # Sends number of images imported

    def __init__(self, image_dir, json_path):
        super().__init__()
        self.image_dir = image_dir
        self.json_path = json_path

    def run(self):
        imported_count = import_coco_data(self.image_dir, self.json_path, self.progress_update)
        self.import_finished.emit(imported_count)

# NEW: The QThread class for handling the background process
class RetrainingThread(QThread):
    """
    Handles the full Train-Evaluate-Compare pipeline in the background.
    """
    # Signals for UI communication
    log_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    # NEW SIGNAL: Passes a dictionary with all results back to the UI.
    retraining_finished = pyqtSignal(dict)
    # NEW SIGNAL: Request user confirmation for filtering results
    filtering_checkpoint = pyqtSignal(dict)

    def __init__(self, model_type, filter_easy_data = False, filtering_strategy = "hybrid", filtering_strictness = 0.5,
                 auto_combine_data = False, use_dynamic_scaling = False, config_overrides={}):
        super().__init__()
        polyvision_root = Path(__file__).resolve().parents[1]
        self.model_type = model_type
        self.config_overrides = config_overrides # For HPO-optimized params
        self.process = None
        self.mining_process = None  # Track mining subprocess for cancellation
        self._is_running = True
        self.project_root = str(polyvision_root)   
        self.anchor_mix = {"hard": 0.20, "medium": 0.35, "easy": 0.45}

        self.filter_easy_data = filter_easy_data
        self.filtering_strategy = filtering_strategy
        self.filtering_strictness = filtering_strictness
        self.auto_combine_data = auto_combine_data
        self.use_dynamic_scaling = use_dynamic_scaling
        self._filtering_approved = None
        self.filtered_image_names = set()   
        
        # Option 2: Aggressive - 20% easy, 60% medium, 20% hard
        # self.anchor_mix = {"hard": 0.20, "medium": 0.60, "easy": 0.20}
        
        # Option 3: Original default - 40% easy, 35% medium, 25% hard
        # self.anchor_mix = DEFAULT_SELECTION_MIX.copy()
    
    def set_filtering_decision(self, approved: bool):       # for DEBUGGING PURPOSES, SAFELY DELETE FOR DEPLOYMENT
        """Called by UI to set user's decision on filtering results."""
        self._filtering_approved = approved
        
    #not used as of the moment. safely delete it to have clean code
    def _find_project_root_containing(self, marker_dir: str = "production_models") -> str:
        """Walk up from this file until we find a folder that contains `marker_dir`."""
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            if (parent / marker_dir).exists():
                return str(parent)
        # Fallback: current working directory
        return str(Path.cwd())

    def stop(self):
        """ Requests the thread and its subprocess to stop. """
        self.log_update.emit("Retraining cancellation requested...")
        self._is_running = False
        # Terminate mining subprocess if running
        if self.mining_process:
            try:
                self.mining_process.terminate()
                self.mining_process.wait(timeout=5)
                self.log_update.emit("Mining process terminated.")
            except Exception as e:
                self.log_update.emit(f"Force killing mining process: {e}")
                self.mining_process.kill()
            self.mining_process = None
        # Terminate training subprocess if running
        if self.process:
            self.process.terminate()

    def cancelTraining(self, challenger_output_dir):
        """
        Cleans up any files and directories created during cancelled training.
        """
        try:
            if challenger_output_dir and os.path.exists(challenger_output_dir):
                self.log_update.emit(f"Cleaning up cancelled training files from: {challenger_output_dir}")
                shutil.rmtree(challenger_output_dir)
                self.log_update.emit("Cleanup completed successfully.")
            
            # Also clean up any temporary annotation files
            temp_annotation_file = "retraining_data/annotations_merged.json"
            if os.path.exists(temp_annotation_file):
                os.remove(temp_annotation_file)
                self.log_update.emit("Cleaned up temporary annotation file.")
            
            # Reset progress bar to 0
            self.progress_update.emit(0, 100)
                
        except Exception as e:
            self.log_update.emit(f"Warning: Could not complete cleanup. Error: {e}")

    # --- NEW HELPER METHODS ---

    def _find_latest_model_in_paths(self, path_list):
        if isinstance(path_list, str):
            path_list = [path_list]

        all_models = []
        for base_path in path_list:
            if not os.path.exists(base_path):
                continue
            # Accept timestamped subfolders...
            all_models.extend(glob.glob(os.path.join(base_path, "*", "model_final.pth")))
            # ...and a flat file directly inside the folder
            flat_path = os.path.join(base_path, "model_final.pth")
            if os.path.exists(flat_path):
                all_models.append(flat_path)

        if not all_models:
            return None
        return max(all_models, key=os.path.getmtime)

    def _find_champion_model(self):
        """
        Finds the current champion model in the base model directories.
        Returns (model_path, is_base_model) tuple.
        """
        # Define base model directories and protected base model names
        if self.model_type == 'Binary':
            base_model_dir = os.path.join(self.project_root, "Models", "SEAMaP-Binary-Full", "faster_rcnn_R_50_FPN_3x")
            protected_base_model = "2025-10-01-03-07-35"  # Binary base model
        else:
            base_model_dir = os.path.join(self.project_root, "Models", "SEAMaP-Multi-class-100", "faster_rcnn_R_50_FPN_3x")
            protected_base_model = "2025-10-01-03-54-34"  # Multiclass base model
        
        # Look for any model_final.pth in the base directory (including timestamp subdirectories)
        champion_model = self._find_latest_model_in_paths([base_model_dir])
        
        if champion_model:
            # Extract the directory name containing the model
            model_dir = os.path.dirname(champion_model)
            model_dir_name = os.path.basename(model_dir)
            
            # Check if this is the protected base model
            if model_dir_name == protected_base_model:
                return champion_model, True  # This is the base model
            else:
                return champion_model, False  # This is a retrained model
                
        return None, False

    def _run_benchmark(self, model_path, run_name):
        
        self.log_update.emit(f"\n--- Benchmarking '{run_name}' Model ---")
        
        if not model_path or not os.path.exists(model_path):
            self.log_update.emit(f"Warning: Model file not found. Cannot benchmark. Path provided: {model_path}")
            return None

        num_classes = 4 if self.model_type == "Multiclass" else 2
        
        # Convert to absolute paths to avoid working directory issues
        abs_model_path = os.path.abspath(model_path)
        
        # Construct test set path relative to project root
        project_root = Path(__file__).resolve().parents[1]  # PolyVision-2.0
        
        #Final test set path
        if self.model_type == "Binary":
            test_set_json = os.path.join(project_root, "Models", "SEAMaP-Binary-Full-6", "test", "_annotations.coco.json")
        else: #Multiclass
            test_set_json = os.path.join(project_root, "Models", "SEAMaP-Multi-class-100-1", "test", "_annotations.coco.json")
        
        # test_set_json = os.path.join(project_root, "Models", "Retraining", "original_datasets", f"{self.model_type.lower()}_90_percent", "test", "_annotations.coco.json")
        
        if not os.path.exists(test_set_json):
            self.log_update.emit(f"FATAL: Test set not found at '{test_set_json}'. Cannot benchmark.")
            return None

        try:
            # Import and call benchmark function directly instead of subprocess
            sys.path.append(os.path.join(project_root, "UI"))
        
            # Create a mock args object
            class MockArgs:
                def __init__(self, model_path, annotations_path, num_classes, run_name):
                    self.model_path = model_path
                    self.annotations_path = annotations_path
                    self.num_classes = num_classes
                    self.run_name = run_name
            
            args = MockArgs(abs_model_path, test_set_json, num_classes, run_name)
            
            # Call benchmark function directly and get results in memory
            self.log_update.emit("Running benchmark evaluation...")
            results = benchmark_main(args)
            
            if results:
                ap_score = results.get('bbox', {}).get('AP', 'N/A')
                if isinstance(ap_score, (int, float)):
                    self.log_update.emit(f"--- Benchmark PASSED for {run_name}. AP: {ap_score:.2f} ---")
                else:
                    self.log_update.emit(f"--- Benchmark PASSED for {run_name}. AP: {ap_score} ---")
                return results
            else:
                self.log_update.emit(f"--- Benchmark returned no results for {run_name} ---")
                return None
                
        except Exception as e:
            self.log_update.emit(f"--- FAILED to run benchmark for {run_name}. Error: {e} ---")
            return None
    
    def _ensure_hard_examples_exist(self, dataset_root, champion_model_path):
        """
        Ensure hard_examples_ranked.json exists with detailed_losses field.
        If missing or incomplete, automatically run hard example mining.
        
        Args:
            dataset_root: Path to dataset
            champion_model_path: Path to champion model for mining
            
        Returns:
            True if hard examples file is ready, False if mining failed
        """
        ranked_list_path = dataset_root / "hard_examples_ranked.json"
        needs_mining = False
        
        # Check if file exists and has detailed_losses
        if not ranked_list_path.exists():
            self.log_update.emit(f"Hard examples file not found at {ranked_list_path}")
            needs_mining = True
        else:
            try:
                with open(ranked_list_path, 'r') as f:
                    mining_data = json.load(f)
                if "detailed_losses" not in mining_data:
                    self.log_update.emit("Hard examples file missing 'detailed_losses' field (outdated format)")
                    needs_mining = True
            except (json.JSONDecodeError, Exception) as e:
                self.log_update.emit(f"Hard examples file corrupted: {e}")
                needs_mining = True
        
        if not needs_mining:
            self.log_update.emit("Hard examples file is up-to-date.")
            return True
        
        # Run automatic mining
        self.log_update.emit("\n=== FIRST-TIME SETUP: Analyzing Dataset Difficulty ===")
        self.log_update.emit("This is a one-time operation that analyzes your anchor dataset.")
        self.log_update.emit("Future retraining sessions will skip this step.\n")
        
        if not champion_model_path or not os.path.exists(champion_model_path):
            self.log_update.emit(f"ERROR: Cannot mine without a valid champion model.")
            self.log_update.emit(f"Model path: {champion_model_path}")
            return False
        
        base_annotation_path = dataset_root / "train" / "_annotations.coco.json"
        base_image_root = dataset_root / "train"
        num_classes = 4 if self.model_type == "Multiclass" else 2
        
        # Path to mine_hard_examples.py in Models/Retraining directory
        mining_script_path = os.path.join(self.project_root, "Models", "Retraining", "mine_hard_examples.py")
        
        mining_command = [
            sys.executable, mining_script_path,
            "--model-path", champion_model_path,
            "--annotations-path", str(base_annotation_path),
            "--image-root", str(base_image_root),
            "--num-classes", str(num_classes),
            "--output-file", str(ranked_list_path),
            "--include-loss-details",  # Critical flag for new system
            "--batch-size", "4"  # Conservative batch size for stability
        ]
        
        self.log_update.emit(f"Running: mine_hard_examples.py on {dataset_root.name}...")
        self.log_update.emit(f"Script path: {mining_script_path}")
        self.log_update.emit("This may take several minutes depending on dataset size.\n")
        
        try:
            # Use Popen instead of run() for interruptibility
            self.mining_process = subprocess.Popen(
                mining_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Stream output in real-time while checking for cancellation
            stdout_lines = []
            while True:
                if not self._is_running:
                    self.log_update.emit("Cancellation detected - stopping mining process...")
                    # Check if process still exists (may have been terminated by stop())
                    if self.mining_process:
                        try:
                            self.mining_process.terminate()
                            self.mining_process.wait(timeout=5)
                        except Exception:
                            pass
                        self.mining_process = None
                    return False
                
                # Check if process still exists
                if not self.mining_process:
                    return False
                
                line = self.mining_process.stdout.readline()
                if line:
                    stdout_lines.append(line)
                    if line.strip():
                        self.log_update.emit(line.rstrip())
                elif self.mining_process.poll() is not None:
                    break  # Process finished
            
            # Safely get any remaining output
            if not self.mining_process:
                return False
                
            remaining_stdout, stderr = self.mining_process.communicate()
            if remaining_stdout:
                for line in remaining_stdout.split('\n'):
                    if line.strip():
                        self.log_update.emit(line)
            
            returncode = self.mining_process.returncode
            self.mining_process = None
            
            if returncode != 0:
                self.log_update.emit(f"\n--- MINING FAILED ---")
                if stderr:
                    self.log_update.emit(stderr)
                return False
            
            # Verify the output file was created correctly
            if not ranked_list_path.exists():
                self.log_update.emit("ERROR: Mining completed but output file not found.")
                return False
            
            with open(ranked_list_path, 'r') as f:
                mining_data = json.load(f)
            if "detailed_losses" not in mining_data:
                self.log_update.emit("ERROR: Mining completed but 'detailed_losses' field missing.")
                return False
            
            self.log_update.emit("\n=== Dataset Analysis Complete ===")
            self.log_update.emit(f"Difficulty rankings saved to: {ranked_list_path}")
            self.log_update.emit("Proceeding with retraining...\n")
            return True
            
        except Exception as e:
            self.log_update.emit(f"\n--- MINING FAILED: {e} ---")
            return False

    def _filter_new_data_by_difficulty(self, champion_model_path, new_data_from_db):
        """
        Run loss mining on new data and filter out low-difficulty images.
        
        Uses the anchor dataset's loss statistics to compute a threshold,
        then filters new images that fall below this threshold.
        
        Args:
            champion_model_path: Path to champion model for mining
            new_data_from_db: List of (image_name, is_microplastic, bounding_box) tuples
            
        Returns:
            Tuple of (filtered_data, filter_stats) or (None, None) if cancelled
        """
        import tempfile
        import numpy as np
        
        self.log_update.emit("\n=== FILTERING LOW-DIFFICULTY NEW DATA ===")
        self.log_update.emit("Analyzing new images to identify challenging examples...\n")
        
        # --- 1. Get anchor dataset statistics ---
        if self.model_type == 'Binary':
            dataset_root = Path(self.project_root) / "Models" / "SEAMaP-Binary-Full-6"
        else:
            dataset_root = Path(self.project_root) / "Models" / "SEAMaP-Multi-class-100-1"
        
        ranked_list_path = dataset_root / "hard_examples_ranked.json"
        
        try:
            anchor_stats = get_anchor_loss_statistics(str(ranked_list_path))
            self.log_update.emit(f"Anchor dataset statistics:")
            self.log_update.emit(f"  Mean loss: {anchor_stats['mean']:.4f}")
            self.log_update.emit(f"  Std loss:  {anchor_stats['std']:.4f}")
            self.log_update.emit(f"  Count:     {anchor_stats['count']} images")
        except Exception as e:
            self.log_update.emit(f"ERROR: Could not load anchor statistics: {e}")
            return None, None
        
        # --- 2. Create temporary COCO file for new data ---
        self.log_update.emit("\nPreparing new data for analysis...")
        
        temp_coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "other"},
                {"id": 1, "name": "microplastic"},
            ] if self.model_type == 'Binary' else [
                {"id": 0, "name": "other"},
                {"id": 1, "name": "fiber"},
                {"id": 2, "name": "film"},
                {"id": 3, "name": "fragment"},
            ]
        }
        
        image_name_to_id = {}
        ann_id = 0
        
        for img_idx, (image_name, is_mp, bbox_str) in enumerate(new_data_from_db):
            image_path = os.path.abspath(os.path.join("retrainingImages", image_name))
            if not os.path.exists(image_path):
                continue
            
            try:
                img_cv = cv2.imread(image_path)
                if img_cv is None:
                    continue
                height, width = img_cv.shape[:2]
            except Exception:
                continue
            
            image_id = img_idx + 1
            image_name_to_id[image_name] = image_id
            
            temp_coco["images"].append({
                "id": image_id,
                "file_name": image_path,
                "height": height,
                "width": width
            })
            
            # Add annotations if it's a positive sample
            if is_mp == 1 and bbox_str:
                try:
                    bboxes_list = json.loads(bbox_str)
                    for bbox_info in bboxes_list:
                        x1, y1, x2, y2 = bbox_info['bbox']
                        w, h = x2 - x1, y2 - y1
                        category_id = 1 if self.model_type == 'Binary' else bbox_info.get('class_id', 1)
                        temp_coco["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x1, y1, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        
        if not temp_coco["images"]:
            self.log_update.emit("ERROR: No valid images found in new data.")
            return None, None
        
        # --- 3. Run mining on new data ---
        self.log_update.emit(f"\nRunning loss analysis on {len(temp_coco['images'])} new images...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_annotations_path = os.path.join(temp_dir, "temp_annotations.json")
            temp_output_path = os.path.join(temp_dir, "new_data_losses.json")
            
            with open(temp_annotations_path, 'w') as f:
                json.dump(temp_coco, f)
            
            num_classes = 2 if self.model_type == 'Binary' else 4
            
            # Use absolute path for mining script
            mining_script_path = os.path.join(self.project_root, "Models", "Retraining", "mine_hard_examples.py")
            
            mining_command = [
                sys.executable, mining_script_path,
                "--model-path", champion_model_path,
                "--annotations-path", temp_annotations_path,
                "--image-root", ".",  # Absolute paths in annotations
                "--num-classes", str(num_classes),
                "--output-file", temp_output_path,
                "--include-loss-details",
                "--batch-size", "2"
            ]
            
            try:
                # Use Popen for interruptibility during filtering
                self.mining_process = subprocess.Popen(
                    mining_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # Monitor process with cancellation support
                while True:
                    if not self._is_running:
                        self.log_update.emit("Cancellation detected - stopping filtering analysis...")
                        # Check if process still exists (may have been terminated by stop())
                        if self.mining_process:
                            try:
                                self.mining_process.terminate()
                                self.mining_process.wait(timeout=5)
                            except Exception:
                                pass
                            self.mining_process = None
                        return None, None
                    
                    # Check if process still exists and finished
                    if not self.mining_process or self.mining_process.poll() is not None:
                        break
                    
                    # Small sleep to avoid busy-waiting
                    import time
                    time.sleep(0.1)
                
                # Safely get output if process still exists
                if self.mining_process:
                    stdout, stderr = self.mining_process.communicate()
                    returncode = self.mining_process.returncode
                    self.mining_process = None
                else:
                    # Process was terminated by stop()
                    return None, None
                
                if returncode != 0:
                    self.log_update.emit(f"ERROR: Mining failed for new data")
                    if stderr:
                        self.log_update.emit(stderr[:500])
                    return None, None
                
                # Load mining results
                with open(temp_output_path, 'r') as f:
                    mining_results = json.load(f)
                    
            except Exception as e:
                self.log_update.emit(f"ERROR: Exception during mining: {e}")
                self.mining_process = None
                return None, None
        
        # --- 4. Compute threshold and filter ---
        # Map image IDs back to names and get losses first
        id_to_name = {v: k for k, v in image_name_to_id.items()}
        image_losses = []
        
        for item in mining_results.get("detailed_losses", []):
            img_id = item["image_id"]
            loss = item["total_loss"]
            if img_id in id_to_name:
                image_losses.append((id_to_name[img_id], loss))
        
        # Compute new data statistics
        import numpy as np
        new_data_losses = np.array([loss for _, loss in image_losses])
        new_data_mean = float(np.mean(new_data_losses))
        new_data_std = float(np.std(new_data_losses))
        
        self.log_update.emit(f"\nNew data statistics:")
        self.log_update.emit(f"  Mean loss: {new_data_mean:.4f}")
        self.log_update.emit(f"  Std loss:  {new_data_std:.4f}")
        
        # --- Compute threshold based on selected strategy ---
        # Strictness: 0 = keep all, 1 = aggressive filtering
        # Higher strictness = higher threshold = fewer images kept
        strictness = self.filtering_strictness
        
        # For anchor-relative: interpolate between anchor_mean (s=0) and anchor_mean + 2*std (s=1)
        # This makes strictness=0.5 set threshold at anchor_mean + 1*std
        anchor_threshold = anchor_stats['mean'] + (strictness * 2 * anchor_stats['std'])
        
        # For self-relative: use percentile-based approach
        # strictness=0 → keep all (threshold at min)
        # strictness=0.5 → keep top 50% hardest (threshold at median)
        # strictness=1 → keep top ~16% hardest (threshold at mean + 1*std)
        self_threshold = np.percentile(new_data_losses, strictness * 100)
        
        self.log_update.emit(f"\n--- Filtering Strategy: {self.filtering_strategy.upper()} (strictness={strictness}) ---")
        self.log_update.emit(f"  Anchor threshold: {anchor_threshold:.4f} (anchor_mean + {strictness*2:.1f}*std)")
        self.log_update.emit(f"  Self threshold:   {self_threshold:.4f} (percentile {strictness*100:.0f}% of new data)")
        
        if self.filtering_strategy == "anchor":
            # Pure anchor-relative: use anchor statistics only
            threshold = anchor_threshold
            self.log_update.emit(f"  Using ANCHOR threshold: {threshold:.4f}")
            self.log_update.emit(f"  (Keeps images harder than anchor + {strictness*2:.1f}*std)")
        elif self.filtering_strategy == "self":
            # Pure self-relative: use new data statistics only
            threshold = self_threshold
            self.log_update.emit(f"  Using SELF threshold: {threshold:.4f}")
            self.log_update.emit(f"  (Keeps top {(1-strictness)*100:.0f}% hardest images)")
        else:  # hybrid (default)
            # Hybrid: use the MAXIMUM threshold (most strict applicable)
            # But cap at the self_threshold to avoid filtering "hard" images relative to new data
            threshold = max(anchor_threshold, self_threshold)
            # Safety cap: never filter more than what self_threshold suggests for 0.8 strictness
            max_safe_threshold = np.percentile(new_data_losses, 80)
            if threshold > max_safe_threshold and strictness < 0.8:
                threshold = max_safe_threshold
                self.log_update.emit(f"  Using HYBRID threshold: {threshold:.4f} (safety capped)")
            else:
                self.log_update.emit(f"  Using HYBRID threshold: {threshold:.4f} (max of both)")
            
            if anchor_threshold >= self_threshold:
                self.log_update.emit(f"  (Anchor-driven: filtering based on anchor difficulty)")
            else:
                self.log_update.emit(f"  (Self-driven: filtering based on new data distribution)")
        
        # Filter
        kept, filtered, stats = filter_new_data_by_loss(image_losses, threshold)
        
        # Store filtered image names for later use
        self.filtered_image_names = {name for name, _ in filtered}
        
        # --- 5. Log results ---
        self.log_update.emit(f"\n--- Filtering Results ---")
        self.log_update.emit(f"Total new images analyzed: {stats['total_images']}")
        self.log_update.emit(f"Images KEPT (loss >= {threshold:.4f}): {stats['kept_count']}")
        self.log_update.emit(f"Images FILTERED (loss < {threshold:.4f}): {stats['filtered_count']}")
        self.log_update.emit(f"Original mean loss: {stats['original_mean']:.4f}")
        self.log_update.emit(f"Kept images mean loss: {stats['kept_mean']:.4f}")
        if stats['filtered_count'] > 0:
            self.log_update.emit(f"Filtered images mean loss: {stats['filtered_mean']:.4f}")
        
        # Add anchor comparison
        self.log_update.emit(f"\nComparison with anchor dataset:")
        self.log_update.emit(f"  Anchor mean: {anchor_stats['mean']:.4f}")
        self.log_update.emit(f"  New data (kept) mean: {stats['kept_mean']:.4f}")
        
        return kept, stats

    def _combine_new_data_with_anchor(self, new_data_from_db):
        """
        Permanently combine filtered new data into the anchor dataset.
        
        This function:
        1. Copies images from retrainingImages/ to anchor train folder
        2. Updates the _annotations.coco.json with new entries
        3. Invalidates the anchor_manifest.json (must re-mine)
        4. Clears the retraining database
        
        Args:
            new_data_from_db: List of (image_name, is_mp, bbox_str) tuples from DB
                             (already filtered by self.filtered_image_names)
        
        Returns:
            dict with combine statistics
        """
        self.log_update.emit("\n" + "="*60)
        self.log_update.emit("COMBINING NEW DATA WITH ANCHOR DATASET")
        self.log_update.emit("="*60)
        
        # Resolve paths
        if self.model_type == 'Binary':
            dataset_root = Path(self.project_root) / "Models" / "SEAMaP-Binary-Full-6"
        else:
            dataset_root = Path(self.project_root) / "Models" / "SEAMaP-Multi-class-100-1"
        
        train_dir = dataset_root / "train"
        annotations_path = train_dir / "_annotations.coco.json"
        manifest_path = dataset_root / "anchor_manifest.json"
        mining_path = dataset_root / "hard_examples_ranked.json"
        
        if not annotations_path.exists():
            self.log_update.emit(f"ERROR: Anchor annotations not found at {annotations_path}")
            return {"success": False, "error": "Annotations file not found"}
        
        # Load existing annotations
        with open(annotations_path, 'r', encoding='utf-8') as f:
            anchor_coco = json.load(f)
        
        # Find max IDs
        max_image_id = max([img['id'] for img in anchor_coco['images']]) if anchor_coco['images'] else 0
        max_ann_id = max([ann['id'] for ann in anchor_coco['annotations']]) if anchor_coco['annotations'] else 0
        
        # Track statistics
        images_added = 0
        annotations_added = 0
        images_skipped = 0
        
        # Get existing filenames to avoid duplicates
        existing_filenames = {img['file_name'] for img in anchor_coco['images']}
        
        self.log_update.emit(f"Current anchor dataset: {len(anchor_coco['images'])} images, {len(anchor_coco['annotations'])} annotations")
        
        for image_name, is_mp, bbox_str in new_data_from_db:
            # Skip filtered images
            if image_name in self.filtered_image_names:
                continue
            
            # Source and destination paths
            src_path = Path("retrainingImages") / image_name
            
            # Generate a unique name if needed
            base_name = f"combined_{image_name}"
            if base_name in existing_filenames:
                # Add timestamp to make unique
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                name_parts = image_name.rsplit('.', 1)
                if len(name_parts) == 2:
                    base_name = f"combined_{name_parts[0]}_{timestamp}.{name_parts[1]}"
                else:
                    base_name = f"combined_{image_name}_{timestamp}"
            
            dst_path = train_dir / base_name
            
            if not src_path.exists():
                self.log_update.emit(f"  Warning: Source image not found: {src_path}")
                images_skipped += 1
                continue
            
            try:
                # Copy image
                shutil.copy2(str(src_path), str(dst_path))
                
                # Read image dimensions
                img_cv = cv2.imread(str(dst_path))
                if img_cv is None:
                    self.log_update.emit(f"  Warning: Could not read image: {dst_path}")
                    dst_path.unlink()  # Remove the copied file
                    images_skipped += 1
                    continue
                height, width = img_cv.shape[:2]
                
                # Create image entry
                max_image_id += 1
                new_image = {
                    "id": max_image_id,
                    "file_name": base_name,
                    "height": height,
                    "width": width
                }
                anchor_coco['images'].append(new_image)
                images_added += 1
                existing_filenames.add(base_name)
                
                # Add annotations if present
                if is_mp == 1 and bbox_str:
                    try:
                        bboxes_list = json.loads(bbox_str)
                        for bbox_info in bboxes_list:
                            x1, y1, x2, y2 = bbox_info['bbox']
                            w, h = x2 - x1, y2 - y1
                            category_id = bbox_info['class_id']
                            
                            # For binary model, all classes map to 1
                            if self.model_type == 'Binary':
                                category_id = 1
                            
                            max_ann_id += 1
                            new_ann = {
                                "id": max_ann_id,
                                "image_id": max_image_id,
                                "category_id": category_id,
                                "bbox": [x1, y1, w, h],
                                "area": w * h,
                                "iscrowd": 0
                            }
                            anchor_coco['annotations'].append(new_ann)
                            annotations_added += 1
                    except (json.JSONDecodeError, TypeError) as e:
                        self.log_update.emit(f"  Warning: Could not parse bbox for {image_name}: {e}")
                        
            except Exception as e:
                self.log_update.emit(f"  Error processing {image_name}: {e}")
                images_skipped += 1
                continue
        
        # Save updated annotations
        self.log_update.emit(f"\nSaving updated annotations to {annotations_path}...")
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(anchor_coco, f, indent=2)
        
        # Invalidate manifest (force re-mining next time)
        if manifest_path.exists():
            backup_path = manifest_path.with_suffix('.json.bak')
            shutil.move(str(manifest_path), str(backup_path))
            self.log_update.emit(f"Manifest backed up to {backup_path}")
            self.log_update.emit("⚠️ anchor_manifest.json invalidated - re-run hard example mining before next retrain!")
        
        # Also mark mining results as outdated
        if mining_path.exists():
            backup_mining = mining_path.with_suffix('.json.bak')
            shutil.move(str(mining_path), str(backup_mining))
            self.log_update.emit(f"Mining results backed up to {backup_mining}")
            self.log_update.emit("⚠️ hard_examples_ranked.json invalidated - re-run mining!")
        
        # Clear retraining database
        self.log_update.emit("\nClearing retraining database...")
        try:
            db_path = 'retrain_images.db'
            if os.path.exists(db_path):
                connection = sqlite3.connect(db_path)
                cursor = connection.cursor()
                cursor.execute("DELETE FROM database_list")
                connection.commit()
                connection.close()
                self.log_update.emit("✓ Retraining database cleared")
        except Exception as e:
            self.log_update.emit(f"Warning: Could not clear database: {e}")
        
        # Clear retrainingImages folder (optional, keep for safety)
        # We'll just log that user can manually clean it
        self.log_update.emit("Note: retrainingImages/ folder preserved for backup purposes")
        
        # Summary
        self.log_update.emit("\n" + "-"*60)
        self.log_update.emit("COMBINE COMPLETE:")
        self.log_update.emit(f"  Images added to anchor: {images_added}")
        self.log_update.emit(f"  Annotations added: {annotations_added}")
        self.log_update.emit(f"  Images skipped: {images_skipped}")
        self.log_update.emit(f"  New anchor dataset size: {len(anchor_coco['images'])} images")
        self.log_update.emit("-"*60)
        self.log_update.emit("\n⚠️ IMPORTANT: Run hard example mining before next retraining!")
        self.log_update.emit("="*60 + "\n")
        
        return {
            "success": True,
            "images_added": images_added,
            "annotations_added": annotations_added,
            "images_skipped": images_skipped,
            "new_anchor_size": len(anchor_coco['images'])
        }
        
    def prepare_coco_annotations(self):
        """
        Merges new data with a "bridge" dataset composed of both RANDOM and HARD
        examples from the original training set to improve fine-tuning robustness.
        """
        self.log_update.emit("--- Starting Data Preparation (Bridge Strategy) ---")
        
        # --- 1. Load New Data from DB ---
        self.log_update.emit("Loading new data from retrain_images.db...")
        new_data_from_db = get_retraining_data()
        num_new_samples = len(new_data_from_db)
        self.log_update.emit(f"Found {num_new_samples} new data points.")

        # --- 2. Load the Original Base Training Dataset ---
        
        if self.model_type == 'Binary':
            dataset_root = Path(self.project_root) / "Models" / "SEAMaP-Binary-Full-6"
        else: # Multiclass
            dataset_root = Path(self.project_root) / "Models" / "SEAMaP-Multi-class-100-1"
        
        base_annotation_path = dataset_root / "train" / "_annotations.coco.json"
        base_image_root = dataset_root / "train"
        manifest_path = dataset_root / "anchor_manifest.json"
        ranked_list_path = dataset_root / "hard_examples_ranked.json"
        
        # if self.model_type == 'Binary':
        #     # Final Binary Model Paths
        #     base_annotation_path = "../Models/SEAMaP-Binary-Full-6/train/_annotations.coco.json"
        #     base_image_root = "../Models/SEAMaP-Binary-Full-6/train/"
        #     manifest_path = "../Models/SEAMaP-Binary-Full-6/anchor_manifest.json"
        #     ranked_list_path = "../Models/SEAMaP-Binary-Full-6/hard_examples_ranked.json"
            
        #     # For testing onle
        #     # base_annotation_path = "../Models/Retraining/original_datasets/binary_90_percent/train/_annotations.coco.json"
        #     # base_image_root = "../Models/Retraining/original_datasets/binary_90_percent/train/"
        #     # ranked_list_path = "../Models/Retraining/original_datasets/binary_90_percent/hard_examples_ranked.json"
        # else: # Multiclass
        #     # Final Multiclass Model Paths
        #     base_annotation_path = "../Models/SEAMaP-Multi-class-100-1/train/_annotations.coco.json"
        #     base_image_root = "../Models/SEAMaP-Multi-class-100-1/train/"
        #     manifest_path = "../Models/SEAMaP-Multi-class-100-1/anchor_manifest.json"
        #     ranked_list_path = "../Models/SEAMaP-Multi-class-100-1/hard_examples_ranked.json"
            
            # For testing onle
            # base_annotation_path = "../Models/Retraining/original_datasets/multiclass_90_percent/train/_annotations.coco.json"
            # base_image_root = "../Models/Retraining/original_datasets/multiclass_90_percent/train/"
            # ranked_list_path = "../Models/Retraining/original_datasets/multiclass_90_percent/hard_examples_ranked.json"

        if not os.path.exists(base_annotation_path):
            self.log_update.emit(f"FATAL ERROR: Base annotation file not found at {base_annotation_path}")
            raise FileNotFoundError(f"Base annotation file not found: {base_annotation_path}")

        self.log_update.emit(f"Loading base dataset from: {base_annotation_path}")
        with open(base_annotation_path, 'r') as f:
            base_coco = json.load(f)
        
        # --- 3. Ensure manifest is available and deterministically select anchors ---
        if not ranked_list_path.exists():
            self.log_update.emit(f"FATAL ERROR: Hard example ranking file not found at {ranked_list_path}")
            self.log_update.emit("Please run hard example mining with --include-loss-details flag first.")
            raise FileNotFoundError(f"Hard example ranking file not found: {ranked_list_path}")

        manifest = build_or_load_manifest(
            str(manifest_path),
            base_coco,
            dataset_name=dataset_root.name,
            mining_json_path=str(ranked_list_path),
        )

        # Calculate how many new images will actually be used AFTER filtering
        # This ensures anchor ratio is based on filtered count, not raw count
        filtered_new_count = sum(
            1 for (image_name, _, _) in new_data_from_db 
            if image_name not in self.filtered_image_names
        )
        self.log_update.emit(f"New data after filtering: {filtered_new_count} images (filtered out {num_new_samples - filtered_new_count})")
        
        anchor_requested = int(filtered_new_count) 
        base_image_count = len(base_coco.get('images', []))
        anchor_requested = min(anchor_requested, base_image_count)

        selected_anchor_ids, bucket_counts, actual_anchor_total = select_anchor_subset(
            manifest,
            anchor_requested,
            mix=self.anchor_mix,
        )
        bucket_summary = ", ".join(
            f"{level}={bucket_counts.get(level, 0)}" for level in DIFFICULTY_ORDER
        )
        self.log_update.emit(
            f"Requested {anchor_requested} anchor images; using {actual_anchor_total} ({bucket_summary})."
        )

        sampled_image_ids = set(selected_anchor_ids)
        final_coco = {
            "images": [img for img in base_coco['images'] if img['id'] in sampled_image_ids],
            "annotations": [ann for ann in base_coco['annotations'] if ann['image_id'] in sampled_image_ids],
            "categories": base_coco['categories'],
        }

        base_image_root_abs = base_image_root.resolve()
        for img in final_coco['images']:
            img_path = (base_image_root_abs / img['file_name']).resolve()
            img['file_name'] = str(img_path)
        # # --- 3. Build the "Bridge" Anchor Dataset ---
        # # Use fixed 100% ratio for anchor data
        
        # # --- 3a. Calculate Quotas for Random and Hard Anchors ---
        # # Use all new samples as basis for anchor calculation
        # num_random_to_keep = num_new_samples
        # num_random_to_keep = min(num_random_to_keep, len(base_coco['images']))
        
        # # We add a fixed ratio of HARD anchors (e.g., 50% of the random anchor count)
        # HARD_ANCHOR_RATIO = 0.5 
        # num_hard_to_keep = int(num_random_to_keep * HARD_ANCHOR_RATIO)
        
        # # --- 3b. Select Random Anchor Images ---
        # self.log_update.emit(f"Selecting {num_random_to_keep} RANDOM old images...")
        # all_base_image_ids = [img['id'] for img in base_coco['images']]
        # if num_random_to_keep > 0:
        #     random_ids = set(random.sample(all_base_image_ids, num_random_to_keep))
        # else:
        #     random_ids = set()

        # # --- 3c. Select Hard Anchor Images ---
        # ranked_list_path = os.path.join(os.path.dirname(base_image_root), "../hard_examples_ranked.json")
        # hard_ids = set()
        # if num_hard_to_keep > 0 and os.path.exists(ranked_list_path):
        #     self.log_update.emit(f"Selecting top {num_hard_to_keep} HARDEST old images...")
        #     with open(ranked_list_path, 'r') as f:
        #         ranked_data = json.load(f)
        #     # Take the top N IDs from the ranked list
        #     hard_ids = set(ranked_data['ranked_image_ids'][:num_hard_to_keep])
        # elif num_hard_to_keep > 0:
        #     self.log_update.emit(f"Warning: hard_examples_ranked.json not found. Skipping hard anchor selection.")

        # # --- 3d. Combine and De-duplicate ---
        # sampled_image_ids = random_ids | hard_ids # Set union automatically handles duplicates
        # self.log_update.emit(f"Total unique anchor images selected: {len(sampled_image_ids)}")

        # # Create the starting point for our final dataset
        # final_coco = {
        #     "images": [img for img in base_coco['images'] if img['id'] in sampled_image_ids],
        #     "annotations": [ann for ann in base_coco['annotations'] if ann['image_id'] in sampled_image_ids],
        #     "categories": base_coco['categories']
        # }
        
        # # Correct image paths for the anchor images
        # for img in final_coco['images']:
        #     img['file_name'] = os.path.abspath(os.path.join(base_image_root, img['file_name']))
        
        # --- 4. Merge the New Data (This part is unchanged) ---
        if not new_data_from_db:
            self.log_update.emit("No new data to merge.")
        
        max_image_id = max([img['id'] for img in final_coco['images']]) if final_coco['images'] else -1
        max_ann_id = max([ann['id'] for ann in final_coco['annotations']]) if final_coco['annotations'] else -1
        image_id_offset = max_image_id + 1
        annotation_id_offset = max_ann_id + 1
        new_annotations_count = 0
        new_images_added = 0
        filtered_count = 0
        for i, (image_name, is_mp, bbox_str) in enumerate(new_data_from_db):
            if image_name in self.filtered_image_names:
                filtered_count += 1
                continue   
            
            image_path = os.path.abspath(os.path.join("retrainingImages", image_name))
            if not os.path.exists(image_path):
                self.log_update.emit(f"Warning: Image file not found, skipping: {image_path}")
                continue
            try:
                img_cv = cv2.imread(image_path)
                height, width, _ = img_cv.shape
            except Exception:
                self.log_update.emit(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            new_image_id = i + image_id_offset
            final_coco["images"].append({"id": new_image_id, "file_name": image_path, "height": height, "width": width})
            if is_mp == 1 and bbox_str:
                try:
                    bboxes_list = json.loads(bbox_str)
                    for ann_idx, bbox_info in enumerate(bboxes_list):
                        x1, y1, x2, y2 = bbox_info['bbox']
                        w, h = x2 - x1, y2 - y1
                        category_id = bbox_info['class_id']
                        final_coco["annotations"].append({
                            "id": new_annotations_count + annotation_id_offset, "image_id": new_image_id,
                            "category_id": category_id, "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0
                        })
                        new_annotations_count += 1
                except (json.JSONDecodeError, TypeError):
                    self.log_update.emit(f"Warning: Could not decode bbox for {image_name}.")
                    
        # Log filtering results if any
        if filtered_count > 0:
            self.log_update.emit(f"Note: {filtered_count} low-difficulty images were excluded from training.")
        self.log_update.emit(f"Added {new_images_added} new images to training data.")
        
        # Store counts for dynamic iteration calculation
        final_new_data_count = new_images_added
        final_anchor_count = actual_anchor_total
        
        if self.model_type == 'Binary':
            self.log_update.emit("Binary model selected. Downcasting all categories to a single 'microplastic' class.")
            
            # Remap all annotation category_ids to 1
            final_coco['categories'] = [
                {"id": 0, "name": "other", "supercategory": "root"},
                {"id": 1, "name": "microplastic", "supercategory": "root"},
            ]

            # (B) annotations: force ALL binary GT boxes to id = 1 (the foreground class your champion uses)
            for ann in final_coco["annotations"]:
                ann["category_id"] = 1

        # --- 5. Save the Final Merged Annotation File ---
        self.log_update.emit("Merge complete. Saving final annotation file.")
        output_dir = "retraining_data"
        os.makedirs(output_dir, exist_ok=True)
        final_annotations_path = os.path.join(output_dir, "annotations_merged.json")
        with open(final_annotations_path, 'w') as f:
            json.dump(final_coco, f, indent=4)
        
        # Return path and data counts for dynamic iteration calculation
        data_counts = {
            "anchor_count": final_anchor_count,
            "new_data_count": final_new_data_count,
            "total_count": len(final_coco['images']),
            "filtered_count": filtered_count
        }
        return final_annotations_path, data_counts

    def prepare_training_command(self, annotations_path, output_dir):
        """
        Prepares the command to execute the external training script,
        including the HPO overrides.
        """
        train_script = "../Models/Retraining/train.py"
        config_file = f"../Models/Retraining/{self.model_type.lower()}_config.yaml"

        if not os.path.exists(train_script) or not os.path.exists(config_file):
            return None

        command = [
            sys.executable, "-u", train_script,
            "--config-file", config_file,
            "--annotations-path", annotations_path,
            "--image-root", "." # Assumes absolute paths in your merged JSON
        ]

        # --- Add all overrides ---
        opts = []
        for key, value in self.config_overrides.items():
            opts.extend([key, str(value)])
        
        # Also override the output directory
        opts.extend(["OUTPUT_DIR", output_dir])
        
        search_paths = []
        if self.model_type == 'Binary':
            # Search in the base model directory (including timestamp subdirectories)
            search_paths.append(os.path.join(self.project_root, "Models", "SEAMaP-Binary-Full", "faster_rcnn_R_50_FPN_3x"))
        else: # Multiclass
            # TEMPORARY: Using compromised/testing base model for multiclass
            # search_paths.append(os.path.join(self.project_root, "Models", "SEAMaP-90%-Real-Multiclass", "faster_rcnn_R_50_FPN_3x"))
            # COMMENTED OUT: Original final base model
            search_paths.append(os.path.join(self.project_root, "Models", "SEAMaP-Multi-class-100", "faster_rcnn_R_50_FPN_3x"))
        
        # Use the helper function to search all relevant paths
        base_model_path = self._find_latest_model_in_paths(search_paths)

        if base_model_path:
            opts.extend(["MODEL.WEIGHTS", base_model_path])
        
        command.extend(["--opts"] + opts)

        return command

    def run(self):
        """
        The main logic for the full Train, Evaluate, and Compare pipeline.
        """
        # This dictionary will hold all our results to pass back to the UI.
        final_result = {'success': False, 'message': 'Pipeline started...'}
        challenger_output_dir = None

        try:
            # --- STAGE 1: IDENTIFY CHAMPION MODEL ---
            self.log_update.emit("\nStage 1/6: Identifying current Champion model...")

            # Use the existing _find_champion_model method which searches in the correct base model directories
            champion_model_path, is_base_model = self._find_champion_model()
            if champion_model_path:
                champion_type = "Base Model" if is_base_model else "Retrained Model"
                self.log_update.emit(f"Champion model found: {champion_type} at {champion_model_path}")
            else:
                self.log_update.emit("No champion model found in base model directories.")

            # --- STAGE 2: ENSURE HARD EXAMPLES EXIST (automatic mining if needed) ---
            self.log_update.emit("\nStage 2/6: Verifying dataset preparation...")
            
            if self.model_type == 'Binary':
                dataset_root = Path(self.project_root) / "Models" / "SEAMaP-Binary-Full-6"
            else:
                dataset_root = Path(self.project_root) / "Models" / "SEAMaP-Multi-class-100-1"
            
            mining_success = self._ensure_hard_examples_exist(dataset_root, champion_model_path)
            if not mining_success:
                raise RuntimeError(
                    "Failed to prepare hard examples file. Cannot proceed with intelligent data selection. "
                    "Please check the logs above for details."
                )
            
            if not self._is_running:
                raise InterruptedError("Process cancelled during dataset verification.")
            
            # --- STAGE 3 (OPTIONAL): FILTER LOW-DIFFICULTY NEW DATA ---
            if self.filter_easy_data:
                self.log_update.emit(f"\nStage 3/6: Filtering low-difficulty new data...")
                self.log_update.emit("=" * 60)
                self.log_update.emit("⚠️  IMPORTANT: A confirmation prompt will appear shortly.")
                self.log_update.emit("A confirmation dialog will appear after filtering completes.")
                self.log_update.emit("You will need to review and approve the filtered data")
                self.log_update.emit("before training can begin.")
                self.log_update.emit("=" * 60)
                
                if not champion_model_path:
                    self.log_update.emit("WARNING: Cannot filter without champion model. Skipping filter stage.")
                else:
                    # Get current new data
                    new_data_from_db = get_retraining_data()
                    
                    if len(new_data_from_db) > 0:
                        kept_data, filter_stats = self._filter_new_data_by_difficulty(
                            champion_model_path, new_data_from_db
                        )
                        
                        if kept_data is None:
                            self.log_update.emit("WARNING: Filtering failed. Proceeding with all data.")
                        else:
                            # Emit signal to request user confirmation
                            self._filtering_approved = None
                            self.filtering_checkpoint.emit(filter_stats)
                            
                            # Wait for user decision (UI will call set_filtering_decision)
                            import time
                            timeout_seconds = 3600  # 1 hour timeout (user may be away)
                            waited = 0
                            while self._filtering_approved is None and waited < timeout_seconds:
                                if not self._is_running:
                                    raise InterruptedError("Process cancelled during filtering checkpoint.")
                                time.sleep(0.1)
                                waited += 0.1
                            
                            if self._filtering_approved is None:
                                raise RuntimeError("Filtering checkpoint timed out (1 hour). Please restart retraining.")
                            
                            if not self._filtering_approved:
                                raise InterruptedError("User cancelled after reviewing filtering results.")
                            
                            self.log_update.emit("\nUser approved filtering. Proceeding with filtered data...")
                    else:
                        self.log_update.emit("No new data to filter.")
                
                if not self._is_running:
                    raise InterruptedError("Process cancelled during filtering stage.")
                
                # Adjust stage numbers for remaining stages
                data_prep_stage = 4
                train_stage = 5
                eval_stage = 6
            else:
                data_prep_stage = 3
                train_stage = 4
                eval_stage = 5
            
            # --- STAGE 4: DATA PREPARATION ---
            self.log_update.emit("Stage 4/6: Preparing training data...")
            annotations_path, data_counts = self.prepare_coco_annotations()
            if not self._is_running:
                raise InterruptedError("Process cancelled during data preparation.")
            
            # --- DYNAMIC ITERATION SCALING ---
            if self.use_dynamic_scaling:
                # Calculate optimal iterations based on dataset size using epoch-based formula
                iter_config = calculate_dynamic_iterations(
                    new_data_count=data_counts['new_data_count'],
                    anchor_count=data_counts['anchor_count']
                )
                
                # Apply dynamic iterations to config overrides
                self.config_overrides["SOLVER.MAX_ITER"] = iter_config["max_iter"]
                self.config_overrides["SOLVER.STEPS"] = f"({iter_config['step1']}, {iter_config['step2']})"
                self.config_overrides["SOLVER.WARMUP_ITERS"] = iter_config["warmup_iters"]
                
                # Log the dynamic configuration
                self.log_update.emit(f"\n📊 Dynamic Iteration Scaling Applied:")
                self.log_update.emit(f"   Dataset: {data_counts['new_data_count']} new + {data_counts['anchor_count']} anchor = {data_counts['total_count']} total images")
                self.log_update.emit(f"   Iterations: {iter_config['max_iter']} (≈{iter_config['estimated_epochs']:.1f} epochs)")
                self.log_update.emit(f"   LR Steps: [{iter_config['step1']}, {iter_config['step2']}]")
                self.log_update.emit(f"   Warmup: {iter_config['warmup_iters']} iters")
                
                # Estimate training time (based on ~0.265 sec/iter from HPO baseline)
                estimated_minutes = (iter_config['max_iter'] * 0.265) / 60
                self.log_update.emit(f"   Est. Training Time: {estimated_minutes:.1f} minutes")
            else:
                self.log_update.emit(f"\n📊 Using config file iterations (dynamic scaling disabled)")

            # --- STAGE 4: TRAIN CHALLENGER MODEL ---
            self.log_update.emit("\nStage 4/6: Training new Challenger model...")
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            if self.model_type == 'Binary':
                # Save retrained models to base model directory with timestamp
                challenger_base_path = os.path.join(self.project_root, "Models", "SEAMaP-Binary-Full", "faster_rcnn_R_50_FPN_3x")
            else: # Multiclass
                # TEMPORARY: Using compromised/testing base model for multiclass
                # challenger_base_path = os.path.join(self.project_root, "Models", "SEAMaP-90%-Real-Multiclass", "faster_rcnn_R_50_FPN_3x")
                # COMMENTED OUT: Original final base model
                challenger_base_path = os.path.join(self.project_root, "Models", "SEAMaP-Multi-class-100", "faster_rcnn_R_50_FPN_3x")
            
            # Create the final, unique directory for this specific run.
            challenger_output_dir = os.path.join(challenger_base_path, timestamp)
            
            # Prepare the training command with all overrides
            command = self.prepare_training_command(annotations_path, challenger_output_dir)

            if not command:
                raise RuntimeError("Could not prepare the training command.")
            if not self._is_running:
                self.cancelTraining(challenger_output_dir)
                raise InterruptedError("Process cancelled before training started.")

            # This is your existing training subprocess loop
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace'
            )
            
            # Read MAX_ITER from the config file for accurate progress bar
            config_file_path = f"../Models/Retraining/{self.model_type.lower()}_config.yaml"
            total_iterations = 2000  # Default fallback
            try:
                with open(config_file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    total_iterations = config_data.get('SOLVER', {}).get('MAX_ITER', 2000)
                # self.log_update.emit(f"Progress bar set for {total_iterations} iterations from config file.")
            except Exception as e:
                self.log_update.emit(f"Warning: Could not read MAX_ITER from config. Using default {total_iterations}. Error: {e}")
            
            # Override with any config_overrides if present
            total_iterations = int(self.config_overrides.get("SOLVER.MAX_ITER", total_iterations))
            for line in iter(self.process.stdout.readline, ''):
                if not self._is_running: 
                    self.process.terminate()
                    break
                self.log_update.emit(line.strip())
                if "iter:" in line:
                    try:
                        current_iter = int(line.split("iter:")[1].strip().split(" ")[0])
                        self.progress_update.emit(current_iter, total_iterations)
                    except: pass
            
            self.process.wait()
            if not self._is_running:
                self.cancelTraining(challenger_output_dir)
                raise InterruptedError("Training was cancelled.")
            if self.process.returncode != 0:
                raise RuntimeError("Training script failed. Check logs for details.")
            
            challenger_model_path = os.path.join(challenger_output_dir, "model_final.pth")
            self.log_update.emit(f"Training complete. Verifying model file exists at: {challenger_model_path}")
            
            max_retries = 5
            retry_delay_seconds = 1
            
            for i in range(max_retries):
                if os.path.exists(challenger_model_path):
                    self.log_update.emit("Model file found. Proceeding to evaluation.")
                    break # Exit the loop, file is found
                else:
                    self.log_update.emit(f"Model file not yet visible. Retrying in {retry_delay_seconds} second(s)... (Attempt {i+1}/{max_retries})")
                    time.sleep(retry_delay_seconds)
            else: # This 'else' belongs to the 'for' loop. It runs if the loop finishes without a 'break'.
                raise FileNotFoundError(f"Challenger model file was not found after {max_retries} retries.")
            final_result['challenger_path'] = challenger_model_path
            self.log_update.emit(f"Challenger model trained successfully at: {challenger_model_path}")
            
            # --- AUTO-COMBINE (after training, before evaluation) ---
            # Only combines if training completed successfully (not cancelled mid-training)
            if self.auto_combine_data:
                self.log_update.emit("\n" + "=" * 60)
                self.log_update.emit("AUTO-COMBINE: Integrating new data into anchor dataset...")
                self.log_update.emit("(Training completed successfully - combining data)")
                self.log_update.emit("=" * 60)
                
                new_data_from_db = get_retraining_data()
                if len(new_data_from_db) > 0:
                    combine_result = self._combine_new_data_with_anchor(new_data_from_db)
                    if combine_result.get('success'):
                        self.log_update.emit(f"Successfully combined {len(new_data_from_db)} images into anchor dataset.")
                    else:
                        self.log_update.emit(f"Warning: Combine failed - {combine_result.get('error', 'Unknown error')}")
                else:
                    self.log_update.emit("No new data to combine.")
            
            # --- STAGE 5: EVALUATION ---
            self.log_update.emit("\nStage 5/5: Evaluating Champion vs. Challenger...")
            
            # Benchmark the champion (if one exists)
            final_result['champion_scores'] = self._run_benchmark(champion_model_path, f"{self.model_type}_Champion")
            if not self._is_running:
                self.cancelTraining(challenger_output_dir)
                raise InterruptedError("Process cancelled during evaluation.")

            # Benchmark the new challenger
            final_result['challenger_scores'] = self._run_benchmark(challenger_model_path, f"{self.model_type}_Challenger_{timestamp}")
            
            final_result['success'] = True
            final_result['message'] = "Evaluation complete. Please review the results."

        except InterruptedError as e:
            self.log_update.emit("Training was interrupted by user.")
            if challenger_output_dir and os.path.exists(challenger_output_dir):
                self.cancelTraining(challenger_output_dir)
            final_result['success'] = False
            final_result['message'] = str(e)
        except Exception as e:
            self.log_update.emit(f"\n--- PIPELINE FAILED: An unexpected error occurred ---")
            self.log_update.emit(str(e))
            if challenger_output_dir and os.path.exists(challenger_output_dir):
                self.cancelTraining(challenger_output_dir)
            final_result['success'] = False
            final_result['message'] = f"An error occurred in the pipeline: {e}"
        finally:
            # Always emit the final result, whether success or failure.
            self.retraining_finished.emit(final_result)


class RetrainUI(QDialog):
    close_signal = pyqtSignal()
    settings_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_type = self._read_current_model_type()
        self.retraining_thread = None
        # Set project root for model path calculations
        self.project_root = str(Path(__file__).resolve().parents[1])
        self.init_ui()
        self.load_data_summary()

    def _read_current_model_type(self):
        try:
            with open("user_settings.json", "r") as f:
                settings_data = json.load(f)
                return settings_data.get("general_features", {}).get("model", "Binary")
        except (FileNotFoundError, json.JSONDecodeError): return "Binary"

    def init_ui(self):
        self.setWindowTitle(f"Retrain Model ({self.model_type})")
        self.setWindowIcon(QIcon("res/PolyVisionLogo.png"))
        self.setFixedSize(800, 600)
        self.setStyleSheet("""
            QDialog { background-color: #f0f0f0; }
            QGroupBox { background-color: transparent; border: 1px solid #c0c0c0; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        """)

        layout = QVBoxLayout(self)
        
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        model_label = QLabel("Select model to retrain:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Binary", "Multiclass"])
        current_index = self.model_combo.findText(self.model_type)
        if current_index != -1: self.model_combo.setCurrentIndex(current_index)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        
        # --- Summary Group ---
        summary_group = QGroupBox("Retraining Data Summary")
        summary_layout = QGridLayout()
        self.positive_label = QLabel("Positive Samples (is MP): 0")
        self.negative_label = QLabel("Negative Samples (not MP): 0")
        self.total_label = QLabel("Total Samples: 0")
        self.context_label = QLabel("") # It starts empty
        self.context_label.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(self.positive_label, 0, 0)
        summary_layout.addWidget(self.negative_label, 0, 1)
        summary_layout.addWidget(self.total_label, 0, 2)
        summary_layout.addWidget(self.context_label, 1, 0, 1, 3)
        #summary_layout.addWidget(self.multiclass_label, 1, 0, 1, 2) # Span across two columns
        #self.multiclass_label.setStyleSheet("color: green; font-weight: bold;")
        #summary_layout.addWidget(self.binary_label, 1, 2)
        #self.binary_label.setStyleSheet("color: #888;")
        summary_group.setLayout(summary_layout)

        # --- Progress Group ---
        progress_group = QGroupBox("Retraining Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Courier", 9))
        self.log_view.setPlaceholderText("Training logs will appear here...")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_view)
        progress_group.setLayout(progress_layout)
        
        # --- Button Group ---
        button_layout = QHBoxLayout()
        self.import_button = QPushButton("Import Dataset")
        self.reset_button = QPushButton("Reset Data")
        self.reset_button.setStyleSheet("color: red;")
        self.reset_button.hide()
        self.start_button = QPushButton("Start Retraining")
        self.cancel_button = QPushButton("Cancel")
        self.close_button = QPushButton("Close")
        self.cancel_button.setEnabled(False)
        # self.import_button.setEnabled(False)         #set to True if you use it
        button_layout.addWidget(self.import_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(self.reset_button, alignment=Qt.AlignLeft)
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.close_button)

        # --- Final Assembly ---
        layout.addWidget(model_group)
        layout.addWidget(summary_group)
        layout.addWidget(progress_group)
        layout.addLayout(button_layout)
        
        # --- Connections ---
        self.start_button.clicked.connect(self.start_retraining)
        self.cancel_button.clicked.connect(self.cancel_retraining)
        self.close_button.clicked.connect(self.close)
        self.model_combo.currentIndexChanged.connect(self._on_model_type_changed)
        self.reset_button.clicked.connect(self.reset_retraining_data)
        self.import_button.clicked.connect(self.run_import_dialog)
        self.model_combo.currentIndexChanged.connect(self.load_data_summary)
        
    def run_import_dialog(self):
        dialog = ImportDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            image_dir, json_path = dialog.get_paths()
            
            # Disable buttons during import
            self.import_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.log_view.clear()

            # Run the import in a background thread
            self.import_thread = ImportThread(image_dir, json_path)
            self.import_thread.progress_update.connect(self.update_log)
            self.import_thread.import_finished.connect(self.on_import_finished)
            self.import_thread.start()

    # In Retrain.py -> RetrainUI

# In Retrain.py -> RetrainUI
# REPLACE the entire function with this one.

    def load_data_summary(self):
        data = get_retraining_data()
        
        total_count = len(data)
        pos_count = 0
        neg_count = 0
        multiclass_ready_count = 0

        # --- Pass 1: Discover the nature of the data ---
        max_class_id = 0
        all_class_ids = []
        for row in data:
            bbox_str = row[2]
            try:
                bboxes = json.loads(bbox_str)
                if isinstance(bboxes, list):
                    for bbox_info in bboxes:
                        all_class_ids.append(bbox_info.get("class_id", 0))
            except (json.JSONDecodeError, TypeError):
                continue
        if all_class_ids:
            max_class_id = max(all_class_ids)
        is_db_multiclass = max_class_id > 1

        # --- Pass 2: Count the samples ---
        for row in data:
            is_mp = row[1]
            bbox_str = row[2]
            if is_mp == 1:
                pos_count += 1
                if is_db_multiclass:
                    try:
                        bboxes = json.loads(bbox_str)
                        if isinstance(bboxes, list) and len(bboxes) > 0:
                            multiclass_ready_count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
            else:
                neg_count += 1

        # --- Update the standard labels ---
        self.positive_label.setText(f"Positive Samples (is MP): {pos_count}")
        self.negative_label.setText(f"Negative Samples (not MP): {neg_count}")
        self.total_label.setText(f"Total Samples: {total_count}")
        
        # --- FINAL ADAPTIVE UI LOGIC ---
        current_model_type = self.model_combo.currentText()
        
        if current_model_type == "Multiclass":
            # In Multiclass mode, show the count of annotated positive samples.
            self.context_label.setText(f"Multiclass-Ready Samples: {multiclass_ready_count}")
            self.context_label.setStyleSheet("color: green; font-weight: bold;")
            self.context_label.setToolTip(
                "Positive samples with one or more bounding boxes, suitable for multiclass training."
            )

            # Disable button if no data for this specific task
            if multiclass_ready_count == 0:
                self.start_button.setEnabled(False)
                self.start_button.setText("No Multiclass Data")
            else:
                self.start_button.setEnabled(True)
                self.start_button.setText("Start Retraining")

        else: # "Binary" mode
            # In Binary mode, show the total count of all samples, as they are all useful.
            self.context_label.setText(f"Binary-Relevant Samples: {total_count}")
            self.context_label.setStyleSheet("color: blue; font-weight: bold;")
            self.context_label.setToolTip(
                "All samples (positive and negative) are used for binary training."
            )
            
            # Enable button as long as there is some data
            if total_count == 0:
                self.start_button.setEnabled(False)
                self.start_button.setText("No Data to Retrain")
            else:
                self.start_button.setEnabled(True)
                self.start_button.setText("Start Retraining")

        if total_count == 0:
            self.log_view.setText("No retraining data found. Please collect data for retraining.")
    def _on_model_type_changed(self):
        self.model_type = self.model_combo.currentText()
        self.setWindowTitle(f"Retrain Model ({self.model_type})")
        self.load_data_summary()

    # NEW: Slot to handle the start button click
    def start_retraining(self):

        RECOMMENDED_MINIMUM_SAMPLES = 50
        new_data = get_retraining_data()
        num_new_samples = len(new_data)

        if num_new_samples < RECOMMENDED_MINIMUM_SAMPLES:
            # The number of samples is below our recommendation, so we warn the user.
            warning_reply = QMessageBox.warning(
                self,
                'Low Data Warning',
                f"You have only collected {num_new_samples} new data samples.\n\n"
                f"It is recommended to have at least {RECOMMENDED_MINIMUM_SAMPLES} new samples "
                "to see a significant improvement in model performance.\n\n"
                "Do you want to continue with the retraining anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No # Default button is "No"
            )
            
            # If the user clicks "No", we simply stop the process here.
            if warning_reply == QMessageBox.No:
                return # Exit the function

        reply = QMessageBox.question(self, 'Confirm Retraining', 
                                     "This will start the model retraining process, which can take a long time and consume significant computer resources. We recommend charging your device before proceeding.\n\nAre you sure you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                with open("user_settings.json", "r+") as f:
                    settings_data = json.load(f)
                    settings_data["general_features"]["model"] = self.model_type
                    f.seek(0)
                    json.dump(settings_data, f, indent=4)
                    f.truncate()
                self.settings_updated.emit()
            except Exception as e:
                self.log_view.append(f"Warning: Could not save setting to user_settings.json. {e}")

            # 3. Read MAX_ITER from the correct config file
            max_iter = 300 # A safe default value
            #freeze_backbone = self.freeze_checkbox.isChecked()
            config_file_path = f"../Models/Retraining/{self.model_type.lower()}_config.yaml"
            try:
                with open(config_file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    # Navigate the YAML structure to find the value
                    max_iter = config_data.get('SOLVER', {}).get('MAX_ITER', 300)
                self.log_view.append(f"Loaded MAX_ITER = {max_iter} from config.")
            except Exception as e:
                self.log_view.append(f"Warning: Could not read MAX_ITER from {config_file_path}. Using default value of {max_iter}. Error: {e}")

            # 4. Lock the UI and prepare for launch
            self.start_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.model_combo.setEnabled(False)
            self.log_view.clear()
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(max_iter)  
            self.progress_bar.setFormat("%p%  (%v/%m iterations)")
            
            filter_easy_data = True # Set to false if you want to disable easy data filtering
            filtering_strictness = 0.2
            filtering_strategy = "hybrid"
            auto_combine = True    # Set to true to auto-combine new data into anchor dataset
            use_dynamic_scaling = True  # Set to true to enable dynamic iteration scaling based on dataset size
            
            
            # 5. Create and start the thread with all the correct parameters
            self.retraining_thread = RetrainingThread(
                self.model_type, 
                filter_easy_data=filter_easy_data,
                filtering_strategy=filtering_strategy,
                filtering_strictness=filtering_strictness,
                auto_combine_data=auto_combine,
                use_dynamic_scaling=use_dynamic_scaling
            )
            self.retraining_thread.log_update.connect(self.update_log)
            self.retraining_thread.progress_update.connect(self.update_progress)
            self.retraining_thread.retraining_finished.connect(self.on_retraining_finished)
            self.retraining_thread.filtering_checkpoint.connect(self.on_filtering_checkpoint)
            self.retraining_thread.start()

    # NEW: Slot to handle the cancel button click
    def cancel_retraining(self):
        if self.retraining_thread and self.retraining_thread.isRunning():
            self.retraining_thread.stop()
            self.cancel_button.setEnabled(False)
            self.reset_button.setEnabled(True)
            self.cancel_button.setText("Cancelling...")
            
    # NEW: Slots to receive signals from the background thread
    @pyqtSlot(int)
    def on_import_finished(self, imported_count):
        self.log_view.append(f"\n--- Import complete. Successfully added {imported_count} images. ---")
        QMessageBox.information(self, "Import Complete", f"Successfully imported {imported_count} images into the retraining dataset.")
        
        # Re-enable buttons and refresh the summary
        self.import_button.setEnabled(False)
        self.load_data_summary() # This will re-enable the start button if data exists

    @pyqtSlot(str)
    def update_log(self, message):
        self.log_view.append(message)

    @pyqtSlot(int, int)
    def update_progress(self, current_step, total_steps):
        if self.progress_bar.maximum() != total_steps:
            self.progress_bar.setMaximum(total_steps)
        self.progress_bar.setValue(current_step)

    @pyqtSlot(dict)
    def on_filtering_checkpoint(self, filter_stats):
        """
        Handle the filtering checkpoint - show results and ask user to proceed.
        """
        total = filter_stats['total_images']
        kept = filter_stats['kept_count']
        filtered = filter_stats['filtered_count']
        threshold = filter_stats['threshold_used']
        original_mean = filter_stats['original_mean']
        kept_mean = filter_stats['kept_mean']
        
        RECOMMENDED_MINIMUM = 50
        
        # Calculate percentage kept
        kept_percent = int((kept / total) * 100) if total > 0 else 0
        
        # Build the message - user-friendly wording
        msg_lines = [
            f"<b>🔍 Smart Data Selection Complete</b><br><br>",
            f"We analyzed your images to find the most valuable ones for improving the model. <br><br>"
            f"<b>Filtering Results:</b><br>",
            f"• Images analyzed: <b>{total}</b><br>",
            f"• Selected for training: <b>{kept}</b> ({kept_percent}% of total)<br>",
            f"• Images filtered out: <b>{filtered}</b><br>",
        ]
        
        # Add encouraging note if good amount kept
        if kept >= RECOMMENDED_MINIMUM:
            msg_lines.append(f"<br><span style='color: #4CAF50;'>✓ Great! You have enough challenging images for effective training.</span><br>")
        
        # Add warning if below recommended minimum
        if kept < RECOMMENDED_MINIMUM:
            msg_lines.append(f"<br><span style='color: #FF5722;'><b>⚠️ Note:</b> Only {kept} challenging images found. ")
            msg_lines.append(f"For best results, we recommend at least {RECOMMENDED_MINIMUM} images. ")
            msg_lines.append(f"You may want to collect more diverse samples.</span><br>")
        
        # Add auto-combine notice if enabled # DEBUG
        # if self.auto_combine_checkbox.isChecked():
        #     msg_lines.append(f"<br><span style='color: #2196F3;'><b>ℹ️ Note:</b> Auto-combine is enabled. ")
        #     msg_lines.append(f"The <b>{kept}</b> kept images will be permanently added to the anchor dataset after proceeding.</span><br>")
        
        msg_lines.append("<br>Do you want to proceed with the filtered data?")
        
        message = "".join(msg_lines)
        
        # Create custom message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Filtering Checkpoint")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Question)
        
        # Add custom buttons
        proceed_btn = msg_box.addButton("Proceed with Filtered Data", QMessageBox.AcceptRole)
        cancel_btn = msg_box.addButton("Cancel Retraining", QMessageBox.RejectRole)
        
        # Set default based on whether we're below minimum
        if kept < RECOMMENDED_MINIMUM:
            msg_box.setDefaultButton(cancel_btn)
        else:
            msg_box.setDefaultButton(proceed_btn)
        
        msg_box.exec_()
        
        # Set the decision for the thread (with safety check for testing)
        if self.retraining_thread is not None:
            if msg_box.clickedButton() == proceed_btn:
                self.retraining_thread.set_filtering_decision(True)
            else:
                self.retraining_thread.set_filtering_decision(False)
            
    @pyqtSlot(dict)
    def on_retraining_finished(self, result):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancel")
        self.model_combo.setEnabled(True)
        
        if not result.get('success'):
            QMessageBox.critical(self, "Retraining Failed", result.get('message', 'An unknown error occurred.'))
            self.finishedTraining()  # Reset UI to idle state
            return

        # Launch the comparison dialog
        dialog = ComparisonDialog(self, result.get('champion_scores'), result.get('challenger_scores'))
        
        if dialog.exec_() == QDialog.Accepted:
            # User clicked "Yes", so we deploy the new model
            self.log_view.append("\n--- User approved deployment. Deploying new model... ---")
            self.deploy_model(result.get('challenger_path'))
            self.finishedTraining()  # Reset UI to idle state after deployment
        else:
            # User clicked "No" - reject and clean up the challenger model
            self.log_view.append("\n--- User rejected deployment. Cleaning up new model... ---")
            self.reject_model(result.get('challenger_path'))
            QMessageBox.information(self, "Deployment Rejected", "The new model has been removed and the current model will be retained.")
            self.finishedTraining()  # Reset UI to idle state after rejection

    def finishedTraining(self):
        
        # Reset progress bar to idle state
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setFormat("Waiting to start...")
        
        # Add completion message to logs
        self.log_view.append("\n" + "="*50)
        self.log_view.append("Training session completed. Ready for next operation.")
        self.log_view.append("="*50)
        
        # Ensure buttons are in correct state (redundant but safe)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancel")
        self.model_combo.setEnabled(True)
        
        # Refresh data summary in case anything changed
        self.load_data_summary()



    def deploy_model(self, challenger_model_path):
        """
        Deploys the challenger model according to specific rules:
        - Always preserve base model (specific timestamp directories)
        - Remove any other retrained models (previous champions)
        - Keep challenger as new active champion
        """
        if not challenger_model_path or not os.path.exists(challenger_model_path):
            QMessageBox.critical(self, "Deployment Error", f"Challenger model not found at: {challenger_model_path}")
            return

        try:
            # Define base model directory and protected base model names
            if self.model_type == 'Binary':
                base_model_dir = os.path.join(self.project_root, "Models", "SEAMaP-Binary-Full", "faster_rcnn_R_50_FPN_3x")
                protected_base_model = "2025-10-01-03-07-35"  # Binary base model - NEVER DELETE
            else:
                base_model_dir = os.path.join(self.project_root, "Models", "SEAMaP-Multi-class-100", "faster_rcnn_R_50_FPN_3x")
                protected_base_model = "2025-10-01-03-54-34"  # Multiclass base model - NEVER DELETE
            
            # Get challenger directory info
            challenger_source_dir = os.path.dirname(challenger_model_path)
            challenger_dir_name = os.path.basename(challenger_source_dir)
            
            self.log_view.append(f"Deploying challenger: {challenger_dir_name}")
            self.log_view.append(f"Protected base model: {protected_base_model}")
            
            # STEP 1: Find and remove old retrained models (excluding challenger and base model)
            retrained_models_to_remove = []
            if os.path.exists(base_model_dir):
                self.log_view.append(f"Scanning directory: {base_model_dir}")
                for item in os.listdir(base_model_dir):
                    item_path = os.path.join(base_model_dir, item)
                    
                    # Skip if not a directory
                    if not os.path.isdir(item_path):
                        continue
                    
                    # Skip challenger directory
                    if item == challenger_dir_name:
                        self.log_view.append(f"Skipping challenger: {item}")
                        continue
                    
                    # Skip protected base model
                    if item == protected_base_model:
                        self.log_view.append(f"Skipping protected base model: {item}")
                        continue
                    
                    # Check if it has model_final.pth (valid model directory)
                    model_path = os.path.join(item_path, "model_final.pth")
                    if os.path.exists(model_path):
                        # This is a retrained model that should be removed
                        retrained_models_to_remove.append((item_path, item))
                        self.log_view.append(f"Marked for removal: {item}")
                    else:
                        self.log_view.append(f"Skipping {item} - no model_final.pth")
            
            # STEP 2: Remove old retrained models
            self.log_view.append(f"Found {len(retrained_models_to_remove)} retrained models to remove")
            for model_path, model_name in retrained_models_to_remove:
                try:
                    self.log_view.append(f"Removing old retrained model: {model_name}")
                    self._force_remove_directory(model_path)
                    self.log_view.append(f"Successfully removed: {model_name}")
                except Exception as e:
                    self.log_view.append(f"WARNING: Could not remove {model_name}: {e}")
                    # Continue with deployment even if removal fails
            
            # STEP 3: Verify challenger deployment
            if os.path.exists(challenger_model_path):
                self.log_view.append(f"Challenger model is now the active champion: {challenger_source_dir}")
                self.log_view.append(f"New active model verified at: {challenger_model_path}")
                QMessageBox.information(self, "Deployment Successful", 
                    f"New model has been successfully deployed!")
            else:
                raise FileNotFoundError(f"Challenger model not found at: {challenger_model_path}")
                
        except Exception as e:
            self.log_view.append(f"--- DEPLOYMENT FAILED: {e} ---")
            QMessageBox.critical(self, "Deployment Failed", f"Failed to deploy model: {str(e)}")

    def _force_remove_directory(self, dir_path):
        """
        Forcefully removes a directory with better error handling.
        """
        if not os.path.exists(dir_path):
            return
        
        import stat
        import time
        
        def handle_remove_readonly(func, path, exc):
            """Error handler for removing readonly files."""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        # Try normal removal first
        try:
            shutil.rmtree(dir_path)
            return
        except PermissionError:
            pass
        
        # Try with readonly handler
        try:
            shutil.rmtree(dir_path, onerror=handle_remove_readonly)
            return
        except Exception:
            pass
        
        # Last resort: try to remove files individually
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)
                except Exception:
                    pass
            for dir in dirs:
                try:
                    os.rmdir(os.path.join(root, dir))
                except Exception:
                    pass
        
        # Remove the main directory
        try:
            os.rmdir(dir_path)
        except Exception as e:
            raise Exception(f"Could not completely remove directory: {e}")

    def reject_model(self, challenger_model_path):
        """
        Rejects the challenger model by deleting it and its timestamp directory.
        This preserves the current champion and base model.
        """
        if not challenger_model_path or not os.path.exists(challenger_model_path):
            self.log_view.append("Challenger model path not found - nothing to clean up.")
            return

        try:
            # Remove the entire timestamp directory containing the rejected model
            challenger_dir = os.path.dirname(challenger_model_path)
            challenger_dir_name = os.path.basename(challenger_dir)
            
            # Safety check - only remove if it's a timestamp directory
            if '-' in challenger_dir_name and len(challenger_dir_name) == 19:  # Format: YYYY-MM-DD-HH-MM-SS
                self.log_view.append(f"Rejecting challenger model: {challenger_dir_name}")
                self._force_remove_directory(challenger_dir)
                self.log_view.append(f"Rejected model and directory removed: {challenger_dir}")
            else:
                # Fallback - just remove the model file if directory format is unexpected
                os.remove(challenger_model_path)
                self.log_view.append(f"Rejected model file removed: {challenger_model_path}")
                
        except Exception as e:
            self.log_view.append(f"--- ERROR during model rejection cleanup: {e} ---")

    def closeEvent(self, event):
        if self.retraining_thread and self.retraining_thread.isRunning():
            reply = QMessageBox.question(self, 'Retraining in Progress',
                                        "Retraining is currently in progress. Are you sure you want to close and cancel it?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.retraining_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            self.close_signal.emit()
            event.accept()
    def reset_retraining_data(self):
        """ Deletes the retraining database and associated images after confirmation. """
        
        reply = QMessageBox.question(self, 'Confirm Reset', 
                                     "This will permanently delete all collected retraining images and their labels. This action cannot be undone.\n\nAre you sure you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            db_path = 'retrain_images.db'
            img_dir = 'retrainingImages'
            
            # Delete the database file
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    self.log_view.append(f"Successfully deleted {db_path}.")
                except OSError as e:
                    self.log_view.append(f"Error deleting database: {e}")
                    QMessageBox.critical(self, "Error", f"Could not delete the database file.\nMake sure the application is not using it.\n\nError: {e}")
                    return


            if os.path.exists(img_dir):
                try:
                    for filename in os.listdir(img_dir):
                        file_path = os.path.join(img_dir, filename)
                        os.remove(file_path)
                    self.log_view.append(f"Successfully cleared all images in {img_dir}.")
                except OSError as e:
                    self.log_view.append(f"Error clearing images: {e}")
                    QMessageBox.critical(self, "Error", f"Could not delete images in the retraining folder.\n\nError: {e}")
                    return

            # Re-create the empty database
            create_retraining_database(os.getcwd())
            self.log_view.append("A new, empty retraining database has been created.")
            
            # Refresh the summary view
            self.load_data_summary()
            
            QMessageBox.information(self, "Success", "Retraining data has been reset.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    retrain_ui = RetrainUI() 
    retrain_ui.show()
    
    # --- high kept_count ---
    fake_filter_stats = {
        'total_images': 100,
        'kept_count': 80,
        'filtered_count': 20,
        'threshold_used': 0.1234,
        'original_mean': 0.0856,
        'kept_mean': 0.1123,
    }
    
    # --- low kept_count ---
    # fake_filter_stats = {
    # 'total_images': 60,
    # 'kept_count': 30,  # Below 50 threshold - will show warning
    # 'filtered_count': 30,
    # 'threshold_used': 0.2500,
    # 'original_mean': 0.0856,
    # 'kept_mean': 0.1500,
    # }   
    
    # retrain_ui.on_filtering_checkpoint(fake_filter_stats)
    # --- END TEMPORARY 
    
    sys.exit(app.exec_())