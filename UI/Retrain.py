# Retrain.py
# lacking anchor_manifest.json
import os
import sys
import json
import sqlite3
import subprocess 
import threading 
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

class RetrainingThread(QThread):
    log_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    retraining_finished = pyqtSignal(dict)

    def __init__(self, model_type, config_overrides={}):
        super().__init__()
        polyvision_root = Path(__file__).resolve().parents[1]
        self.model_type = model_type
        self.config_overrides = config_overrides 
        self.process = None
        self._is_running = True
        self.project_root = str(polyvision_root)   
        self.anchor_mix = DEFAULT_SELECTION_MIX.copy()
    
    #not used as of the moment. safely delete it to have clean code
    def _find_project_root_containing(self, marker_dir: str = "production_models") -> str:
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

    def prepare_coco_annotations(self):
        self.log_update.emit("--- Starting Data Preparation (Bridge Strategy) ---")
        
        # --- 1. Load New Data from DB ---
        self.log_update.emit("Loading new data from retrain_images.db...")
        new_data_from_db = get_retraining_data()
        num_new_samples = len(new_data_from_db)
        self.log_update.emit(f"Found {num_new_samples} new data points.")

        # --- 2. Load the Original Base Training Dataset ---
        
        if self.model_type == 'Binary':
            dataset_root = Path("../Models/SEAMaP-Binary-Full-6")
        else: # Multiclass
            dataset_root = Path("../Models/SEAMaP-Multi-class-100-1")
            
        base_annotation_path = dataset_root / "train" / "_annotations.coco.json"
        base_image_root = dataset_root / "train"
        manifest_path = dataset_root / "anchor_manifest.json"
        ranked_list_path = dataset_root / "hard_examples_ranked.json"
        
        # if self.model_type == 'Binary':
        #     # Final Binary Model Paths
        #     base_annotation_path = "../Models/SEAMaP-Binary-Full-6/train/_annotations.coco.json"
        #     base_image_root = "../Models/SEAMaP-Binary-Full-6/train/"
        #     ranked_list_path = "../Models/SEAMaP-Binary-Full-6/hard_examples_ranked.json"
            
        #     # For testing onle
        #     # base_annotation_path = "../Models/Retraining/original_datasets/binary_90_percent/train/_annotations.coco.json"
        #     # base_image_root = "../Models/Retraining/original_datasets/binary_90_percent/train/"
        #     # ranked_list_path = "../Models/Retraining/original_datasets/binary_90_percent/hard_examples_ranked.json"
        # else: # Multiclass
        #     # Final Multiclass Model Paths
        #     base_annotation_path = "../Models/SEAMaP-Multi-class-100-1/train/_annotations.coco.json"
        #     base_image_root = "../Models/SEAMaP-Multi-class-100-1/train/"
        #     ranked_list_path = "../Models/SEAMaP-Multi-class-100-1/hard_examples_ranked.json"
            
            # For testing onle
            # base_annotation_path = "../Models/Retraining/original_datasets/multiclass_90_percent/train/_annotations.coco.json"
            # base_image_root = "../Models/Retraining/original_datasets/multiclass_90_percent/train/"
            # ranked_list_path = "../Models/Retraining/original_datasets/multiclass_90_percent/hard_examples_ranked.json"

        if not os.path.exists(base_annotation_path):
            self.log_update.emit(f"FATAL ERROR: Base annotation file not found at {base_annotation_path}")
            raise FileNotFoundError(f"Base annotation file not found: {base_annotation_path}")

        self.log_update.emit(f"Loading base dataset from: {base_annotation_path}")
        with open(base_annotation_path, 'r', encoding='utf-8') as handle:
            base_coco = json.load(handle)
        
        # --- 3. Ensure manifest is available and deterministically select anchors ---
        ranked_ids = []
        if ranked_list_path.exists():
            try:
                with open(ranked_list_path, 'r', encoding='utf-8') as handle:
                    ranked_data = json.load(handle)
                ranked_ids = ranked_data.get('ranked_image_ids', [])
            except Exception as exc:
                self.log_update.emit(f"Warning: Failed to read {ranked_list_path}: {exc}")
        else:
            self.log_update.emit("Hard example ranking file not found; falling back to default ordering.")

        manifest = build_or_load_manifest(
            str(manifest_path),
            base_coco,
            ranked_ids,
            dataset_name=dataset_root.name,
        )

        # self.log_update.emit(f"Anchor Data Ratio Slider set to: {self.base_data_percentage}%")
        anchor_requested = num_new_samples
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
        
        # --- 3. Build the "Bridge" Anchor Dataset ---
        # Use fixed 100% ratio for anchor data
        
        # --- 3a. Calculate Quotas for Random and Hard Anchors ---
        # Use all new samples as basis for anchor calculation
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
        for i, (image_name, is_mp, bbox_str) in enumerate(new_data_from_db):
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
                    if not isinstance(bboxes_list, list):
                        bboxes_list = []
                    for ann_idx, bbox_info in enumerate(bboxes_list):
                        if not isinstance(bbox_info, dict):
                            continue
                        if bbox_info.get("review_status") == "rejected":
                            continue
                        bbox = bbox_info.get('bbox')
                        if not bbox or len(bbox) != 4:
                            continue
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        category_id = bbox_info.get('class_id', 1)
                        final_coco["annotations"].append({
                            "id": new_annotations_count + annotation_id_offset, "image_id": new_image_id,
                            "category_id": category_id, "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0
                        })
                        new_annotations_count += 1
                except (json.JSONDecodeError, TypeError):
                    self.log_update.emit(f"Warning: Could not decode bbox for {image_name}.")
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
            
        return final_annotations_path

    def prepare_training_command(self, annotations_path, output_dir):
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
            # --- STAGE 1: DATA PREPARATION ---
            self.log_update.emit("Stage 1/4: Preparing training data...")
            annotations_path = self.prepare_coco_annotations()
            if not self._is_running:
                raise InterruptedError("Process cancelled during data preparation.")

            # --- STAGE 2: IDENTIFY CHAMPION MODEL ---
            self.log_update.emit("\nStage 2/4: Identifying current Champion model...")

            # Use the existing _find_champion_model method which searches in the correct base model directories
            champion_model_path, is_base_model = self._find_champion_model()
            if champion_model_path:
                champion_type = "Base Model" if is_base_model else "Retrained Model"
                self.log_update.emit(f"Champion model found: {champion_type} at {champion_model_path}")
            else:
                self.log_update.emit("No champion model found in base model directories.")


            # --- STAGE 3: TRAIN CHALLENGER MODEL ---
            self.log_update.emit("\nStage 3/4: Training new Challenger model...")
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
            
            # --- STAGE 4: EVALUATION ---
            self.log_update.emit("\nStage 4/4: Evaluating Champion vs. Challenger...")
            
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
        self.start_button = QPushButton("Start Retraining")
        self.cancel_button = QPushButton("Cancel")
        self.close_button = QPushButton("Close")
        self.cancel_button.setEnabled(False)
        self.import_button.setEnabled(True)  #set to True if you use it
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
                        if isinstance(bbox_info, dict):
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
                        if isinstance(bboxes, list):
                            has_valid_bbox = any(
                                isinstance(bbox_info, dict) and bbox_info.get("bbox")
                                for bbox_info in bboxes
                            )
                            if has_valid_bbox:
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
            
            # 5. Create and start the thread with all the correct parameters
            self.retraining_thread = RetrainingThread(self.model_type)
            self.retraining_thread.log_update.connect(self.update_log)
            self.retraining_thread.progress_update.connect(self.update_progress)
            self.retraining_thread.retraining_finished.connect(self.on_retraining_finished)
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
    sys.exit(app.exec_())
