import argparse
import pathlib
import time
import csv # For CSV output
from typing import Tuple, Dict, List, Optional

import cv2
import numpy as np
import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
HPATCHES_DIR_DEFAULT = ROOT_DIR / "data" / "hpatches-sequences-release"
NUM_POINTS_FOR_HOMOGRAPHY_EVAL = 1000
DETAILED_CSV_FILENAME = ROOT_DIR / "output/hpatches_detailed_results.csv"
SUMMARY_CSV_FILENAME = ROOT_DIR / "output/hpatches_summary_results.csv"

try:
    from roma_inference_adapted import get_roma_matches
    print("Successfully imported get_roma_matches from roma_inference_adapted.py")
except ImportError as e:
    print(f"ERROR: Could not import get_roma_matches from roma_inference_adapted.py: {e}")
    get_roma_matches = None

try:
    from homomatcher_lite_adapted import get_homomatcher_matches
    print("Successfully imported get_homomatcher_matches from homomatcher_lite_adapted.py")
except ImportError as e:
    print(f"ERROR: Could not import get_homomatcher_matches from homomatcher_lite_adapted.py: {e}")
    get_homomatcher_matches = None

def run_roma_matcher(img_path1: pathlib.Path, img_path2: pathlib.Path, device: torch.device) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if get_roma_matches is None:
        print("RoMA function (get_roma_matches) was not imported correctly. Skipping RoMA.")
        return None
    return get_roma_matches(str(img_path1), str(img_path2), device)

def run_homomatcher_lite(img_path1: pathlib.Path, img_path2: pathlib.Path, device: torch.device) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if get_homomatcher_matches is None:
        print("HomoMatcher function (get_homomatcher_matches) was not imported correctly. Skipping HomoMatcher.")
        return None
    return get_homomatcher_matches(str(img_path1), str(img_path2), device)

def load_hpatches_homography(h_path: pathlib.Path) -> Optional[np.ndarray]:
    try:
        H = np.loadtxt(str(h_path), dtype=np.float64)
        return H
    except Exception as e:
        print(f"Error loading homography {h_path}: {e}")
        return None

def compute_reprojection_error(H_gt: np.ndarray, H_est: np.ndarray, img_shape: Tuple[int, int], num_points: int = 1000) -> float:
    h, w = img_shape[:2]
    src_pts = np.random.rand(num_points, 1, 2) * np.array([w, h])
    src_pts = src_pts.astype(np.float32)
    dst_pts_gt = cv2.perspectiveTransform(src_pts, H_gt)
    dst_pts_est = cv2.perspectiveTransform(src_pts, H_est)
    if dst_pts_gt is None or dst_pts_est is None:
        return float('inf')
    errors = np.linalg.norm(dst_pts_gt - dst_pts_est, axis=2).squeeze()
    return np.mean(errors)

def estimate_homography_from_matches(kp1: np.ndarray, kp2: np.ndarray, reproj_thresh: float = 3.0) -> Optional[np.ndarray]:
    if kp1 is None or kp2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    H_est, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, reproj_thresh)
    return H_est

def evaluate_on_hpatches_sequence(sequence_dir: pathlib.Path,
                                  matcher_fn,
                                  matcher_name: str,
                                  device: torch.device,
                                  img_shape_for_eval: Tuple[int, int]) -> Dict:
    results = {
        "sequence": sequence_dir.name,
        "matcher": matcher_name,
        "pairs_processed": 0,
        "pairs_failed_matching": 0,
        "pairs_failed_homography_estimation":0,
        "total_reprojection_error": 0.0,
        "average_reprojection_error": float('inf'),
        "total_time_s": 0.0, # Renamed for clarity
    }
    img1_path = sequence_dir / "1.ppm"

    if not img1_path.exists():
        print(f"Warning: Reference image {img1_path} not found in {sequence_dir}. Skipping.")
        return results

    ref_img_for_shape = cv2.imread(str(img1_path))
    current_img_shape = img_shape_for_eval if ref_img_for_shape is None else ref_img_for_shape.shape[:2]

    for i in range(2, 7):
        img2_path = sequence_dir / f"{i}.ppm"
        h_path = sequence_dir / f"H_1_{i}"

        if not img2_path.exists() or not h_path.exists():
            continue

        H_gt = load_hpatches_homography(h_path)
        if H_gt is None:
            continue

        results["pairs_processed"] += 1
        start_time = time.time()
        matches = matcher_fn(img1_path, img2_path, device)
        match_time = time.time() - start_time
        results["total_time_s"] += match_time

        if matches is None:
            results["pairs_failed_matching"] += 1
            continue
        
        kp1, kp2 = matches
        H_est = estimate_homography_from_matches(kp1, kp2)

        if H_est is None:
            results["pairs_failed_homography_estimation"] +=1
            continue

        error = compute_reprojection_error(H_gt, H_est, current_img_shape, NUM_POINTS_FOR_HOMOGRAPHY_EVAL)
        results["total_reprojection_error"] += error

    valid_estimations = results["pairs_processed"] - results["pairs_failed_matching"] - results["pairs_failed_homography_estimation"]
    if valid_estimations > 0:
         results["average_reprojection_error"] = results["total_reprojection_error"] / valid_estimations
    else:
        results["average_reprojection_error"] = float('inf')
    return results

def write_detailed_to_csv(all_results: List[Dict], filename: str):
    """Writes the detailed results (per sequence, per matcher) to a CSV file."""
    if not all_results:
        print("No detailed results to write to CSV.")
        return
    
    fieldnames = [
        "matcher", "sequence", "pairs_processed", "pairs_failed_matching",
        "pairs_failed_homography_estimation", "average_reprojection_error", "total_time_s"
    ]
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_dict in all_results:
                filtered_row = {key: row_dict.get(key, 'N/A') for key in fieldnames}
                writer.writerow(filtered_row)
        print(f"Detailed results saved to {filename}")
    except IOError:
        print(f"Error: Could not write detailed results to CSV file {filename}.")

def write_summary_to_csv(summary_data: Dict, filename: str):
    """Writes the summary results (aggregated per matcher) to a CSV file."""
    if not summary_data:
        print("No summary data to write to CSV.")
        return

    fieldnames = [
        "matcher", "avg_reproj_err_overall", "total_unique_sequences", 
        "total_pairs_processed", "total_pairs_failed_matching", 
        "total_pairs_failed_homography", "total_processing_time_s", "avg_time_per_pair_s"
    ]
    
    rows_to_write = []
    for matcher_name, data in summary_data.items():
        avg_err = (data["cumulative_reprojection_error"] / data["num_valid_reprojections"]) if data["num_valid_reprojections"] > 0 else float('inf')
        avg_time_per_pair = (data['total_processing_time_s'] / data['total_pairs_processed']) if data['total_pairs_processed'] > 0 else 0
        
        row = {
            "matcher": matcher_name,
            "avg_reproj_err_overall": avg_err if avg_err != float('inf') else 'inf',
            "total_unique_sequences": data['total_sequences_processed_once'],
            "total_pairs_processed": data['total_pairs_processed'],
            "total_pairs_failed_matching": data['total_pairs_failed_matching'],
            "total_pairs_failed_homography": data['total_pairs_failed_homography'],
            "total_processing_time_s": data['total_processing_time_s'],
            "avg_time_per_pair_s": avg_time_per_pair
        }
        rows_to_write.append(row)

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_write)
        print(f"Summary results saved to {filename}")
    except IOError:
        print(f"Error: Could not write summary results to CSV file {filename}.")


def main():
    parser = argparse.ArgumentParser(description="HPatches Evaluation Pipeline for RoMA and HomoMatcher.")
    parser.add_argument("--hpatches_dir", type=pathlib.Path, default=HPATCHES_DIR_DEFAULT,
                        help="Path to the HPatches sequences directory.")
    parser.add_argument("--limit_sequences", type=int, default=None,
                        help="Limit the number of sequences to process (for quick testing).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run models on ('cuda' or 'cpu').")
    args = parser.parse_args()

    if not args.hpatches_dir.is_dir():
        print(f"Error: HPatches directory not found at {args.hpatches_dir}")
        return

    device = torch.device(args.device)
    print(f"Using device: {device}")

    img_shape_for_eval = (480, 640)
    try:
        first_seq = next((item for item in args.hpatches_dir.iterdir() if item.is_dir() and (item / "1.ppm").exists()), None)
        if first_seq:
            first_img = cv2.imread(str(first_seq / "1.ppm"))
            if first_img is not None:
                 img_shape_for_eval = first_img.shape[:2]
    except Exception as e:
        print(f"Could not determine image shape from HPatches, using default {img_shape_for_eval}. Error: {e}")

    all_results: List[Dict] = []
    sequence_dirs = sorted([d for d in args.hpatches_dir.iterdir() if d.is_dir()])
    if args.limit_sequences:
        sequence_dirs = sequence_dirs[:args.limit_sequences]

    print(f"Found {len(sequence_dirs)} sequences in {args.hpatches_dir}.")

    matchers_to_evaluate = {}
    if run_roma_matcher and get_roma_matches:
        matchers_to_evaluate["RoMA"] = run_roma_matcher
    if run_homomatcher_lite and get_homomatcher_matches:
        matchers_to_evaluate["HomoMatcherLite"] = run_homomatcher_lite
    
    if not matchers_to_evaluate:
        print("\nERROR: No matchers are available for evaluation.")
        return

    for i, seq_dir in enumerate(sequence_dirs):
        print(f"\nProcessing sequence {i+1}/{len(sequence_dirs)}: {seq_dir.name}")
        for matcher_name, matcher_fn in matchers_to_evaluate.items():
            print(f"  Running {matcher_name}...")
            seq_results = evaluate_on_hpatches_sequence(seq_dir, matcher_fn, matcher_name, device, img_shape_for_eval)
            all_results.append(seq_results)
            print(f"  Finished {matcher_name} for {seq_dir.name}: Avg Reproj Error: {seq_results['average_reprojection_error']:.4f}, "
                  f"Time: {seq_results['total_time_s']:.2f}s")

    print("\n\n--- Overall Results ---")
    if not all_results:
        print("No results to display.")
    else:
        write_detailed_to_csv(all_results, DETAILED_CSV_FILENAME) # Save detailed CSV

    summary = {}
    for res in all_results:
        matcher = res["matcher"]
        if matcher not in summary:
            summary[matcher] = {
                "total_sequences": 0, "total_pairs_processed": 0, "total_pairs_failed_matching": 0,
                "total_pairs_failed_homography": 0, "cumulative_reprojection_error": 0.0,
                "num_valid_reprojections": 0, "total_processing_time_s": 0.0 
            }
        summary[matcher]["total_sequences"] += 1
        summary[matcher]["total_pairs_processed"] += res["pairs_processed"]
        summary[matcher]["total_pairs_failed_matching"] += res["pairs_failed_matching"]
        summary[matcher]["total_pairs_failed_homography"] += res["pairs_failed_homography_estimation"]
        if res["average_reprojection_error"] != float('inf'):
             valid_estimations_for_seq = res["pairs_processed"] - res["pairs_failed_matching"] - res["pairs_failed_homography_estimation"]
             if valid_estimations_for_seq > 0:
                summary[matcher]["cumulative_reprojection_error"] += res["total_reprojection_error"]
                summary[matcher]["num_valid_reprojections"] += valid_estimations_for_seq
        summary[matcher]["total_processing_time_s"] += res["total_time_s"] 

    final_summary_for_print_and_csv = {}
    processed_matchers = set(res['matcher'] for res in all_results)

    for matcher_name in processed_matchers:
        data = summary[matcher_name]
        final_summary_for_print_and_csv[matcher_name] = data
        unique_sequences_for_matcher = len(set(r['sequence'] for r in all_results if r['matcher'] == matcher_name))
        final_summary_for_print_and_csv[matcher_name]["total_sequences_processed_once"] = unique_sequences_for_matcher

    for matcher_name, data in final_summary_for_print_and_csv.items():
        print(f"\nMatcher: {matcher_name}")
        avg_err = (data["cumulative_reprojection_error"] / data["num_valid_reprojections"]) if data["num_valid_reprojections"] > 0 else float('inf')
        print(f"  Average Reprojection Error (overall): {avg_err:.4f}")
        print(f"  Total Unique Sequences Processed: {data['total_sequences_processed_once']}")
        print(f"  Total Image Pairs Processed: {data['total_pairs_processed']}")
        print(f"  Total Pairs Failed Matching: {data['total_pairs_failed_matching']}")
        print(f"  Total Pairs Failed Homography Estimation: {data['total_pairs_failed_homography']}")
        print(f"  Total Processing Time: {data['total_processing_time_s']:.2f}s")
        avg_time_per_pair = (data['total_processing_time_s'] / data['total_pairs_processed']) if data['total_pairs_processed'] > 0 else 0
        print(f"  Average Time per Pair: {avg_time_per_pair:.3f}s")
    
    if final_summary_for_print_and_csv:
        write_summary_to_csv(final_summary_for_print_and_csv, SUMMARY_CSV_FILENAME) # Save summary CSV

    print("\nEvaluation finished.")

if __name__ == "__main__":
    main()
