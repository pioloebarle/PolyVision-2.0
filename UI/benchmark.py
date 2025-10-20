# benchmark.py

import os
import time 
import argparse
import json
import torch
import detectron2
from detectron2 import model_zoo 
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

def main(args):
    print(f"--- Starting Benchmark for Model: {args.model_path} ---")

    # --- 1. Register the Test Dataset ---
    # The dataset must be in COCO format.
    dataset_name = "microplastic_benchmark_test"
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    register_coco_instances(
        dataset_name, 
        {}, 
        args.annotations_path, 
        os.path.dirname(args.annotations_path) # The image root is the same folder as the json file
    )
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    print(f"Registered '{dataset_name}' with {len(dataset_dicts)} images.")

    # Configure the Model for Inference ---
    cfg = get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"  
    
    cfg.MODEL.WEIGHTS = args.model_path
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

    # --- 3. Run COCO Evaluation ---
    print("\n--- Running COCO Evaluation (Calculating AP metrics)... ---")
    
    # Create an output directory for the results
    if args.run_name:
        folder_name = f"{args.run_name}_{args.num_classes}cls"
    else:
        model_filename = os.path.splitext(os.path.basename(args.model_path))[0]
        folder_name = f"{model_filename}_{args.num_classes}cls"
    
    output_dir = os.path.join("benchmarks", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the evaluator
    evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, dataset_name)
    
    # Run the evaluation
    predictor = DefaultPredictor(cfg)
    start_time = time.time()
    results = inference_on_dataset(predictor.model, test_loader, evaluator)

    # CALCULATE AND PRINT THE DURATION ---
    end_time = time.time()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60
    print(f"\n--- Evaluation Loop Finished ---")
    print(f"Total time taken: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")
    print("\n--- Benchmark Complete ---")
    
    # Return results in memory instead of saving to JSON
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script for PolyVision models.")
    parser.add_argument(
        "--model-path", 
        required=True, 
        help="Path to the trained model checkpoint file (.pth)."
    )
    parser.add_argument(
        "--annotations-path", 
        required=True, 
        help="Path to the COCO JSON annotation file for the test dataset."
    )
    parser.add_argument(
        "--num-classes", 
        required=True, 
        type=int,
        help="The number of classes the model was trained on (e.g., 1 for Binary, 3 for Multiclass)."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None, # It's optional
        help="A custom name for the benchmark run, used for the output folder (e.g., 'multiclass_50pct')."
    )
    
    args = parser.parse_args()
    results = main(args)
    
    # When run as script, save results to JSON for backward compatibility
    if args.run_name:
        folder_name = f"{args.run_name}_{args.num_classes}cls"
    else:
        model_filename = os.path.splitext(os.path.basename(args.model_path))[0]
        folder_name = f"{model_filename}_{args.num_classes}cls"
    
    output_dir = os.path.join("benchmarks", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")