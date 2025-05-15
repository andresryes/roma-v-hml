import os
import re
import csv
import time
import torch
import pathlib
import numpy as np
import cv2
from typing import List, Tuple, Optional

try:
    from romatch import roma_outdoor
except ImportError:
    print("ERROR: Could not import RoMA.")
    roma_outdoor = None

try:
    from homomatcher_lite_adapted import get_homomatcher_matches
except ImportError:
    print("WARNING: Could not import HomoMatcher.")
    get_homomatcher_matches = None

def load_intrinsics(calib_path: str) -> np.ndarray:
    with open(calib_path, 'r') as f:
        calib_data = f.read()
    cam0_match = re.search(r"cam0=\[([\d\. \-;]+)\]", calib_data)
    if not cam0_match:
        raise ValueError(f"No cam0 found in {calib_path}")
    vals = cam0_match.group(1).replace(';', ' ').split()
    fx, fy = float(vals[0]), float(vals[4])
    cx, cy = float(vals[2]), float(vals[5])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def triangulate_and_error(kpts1, kpts2, K, R, t):
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))
    pts1 = kpts1.T
    pts2 = kpts2.T
    pts_4d = cv2.triangulatePoints(P0, P1, pts1, pts2)
    pts_3d = pts_4d[:3] / pts_4d[3:]

    pts_4 = np.vstack((pts_3d, np.ones((1, pts_3d.shape[1]))))
    proj1 = P0 @ pts_4
    proj1 /= proj1[2]
    proj1 = proj1[:2]

    proj2 = P1 @ pts_4
    proj2 /= proj2[2]
    proj2 = proj2[:2]

    err1 = np.linalg.norm(proj1.T - kpts1, axis=1)
    err2 = np.linalg.norm(proj2.T - kpts2, axis=1)
    return len(pts_3d.T), float(np.mean(np.concatenate([err1, err2])))

def run_matching(img1, img2, K, matcher_name, matcher_fn, device):
    t0 = time.time()
    if matcher_name == "RoMA":
        warp, certainty = matcher_fn.match(img1, img2, device=device)
        matches, _ = matcher_fn.sample(warp, certainty)
        H0, W0 = cv2.imread(img1).shape[:2]
        H1, W1 = cv2.imread(img2).shape[:2]
        kpts1, kpts2 = matcher_fn.to_pixel_coordinates(matches, H0, W0, H1, W1)
        kpts1 = kpts1.cpu().numpy()
        kpts2 = kpts2.cpu().numpy()
    else:
        kpts1, kpts2 = matcher_fn(img1, img2, device=device)
    match_time = time.time() - t0

    if len(kpts1) < 8:
        return False, 0, float('nan'), match_time

    E, mask = cv2.findEssentialMat(kpts1, kpts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None or np.sum(mask) < 5:
        return False, 0, float('nan'), match_time

    _, R, t, mask_pose = cv2.recoverPose(E, kpts1, kpts2, cameraMatrix=K, mask=mask)
    if R is None or t is None:
        return False, 0, float('nan'), match_time

    inliers1 = kpts1[mask_pose.ravel() > 0]
    inliers2 = kpts2[mask_pose.ravel() > 0]
    if len(inliers1) == 0:
        return False, 0, float('nan'), match_time

    num_pts, err = triangulate_and_error(inliers1, inliers2, K, R, t)
    return True, num_pts, err, match_time

def main():
    file_root = pathlib.Path(__file__).resolve().parent.parent
    data_path = file_root / "data"
    dataset_root = pathlib.Path(data_path / "eth3d")
    output_csv = "output/eth3d_reconstruction_results.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roma_model = roma_outdoor(device=device) if roma_outdoor else None
    results = []

    for scene in sorted(os.listdir(dataset_root)):
        scene_dir = dataset_root / scene
        if not scene_dir.is_dir():
            continue
        img1 = str(scene_dir / "im0.png")
        img2 = str(scene_dir / "im1.png")
        calib = str(scene_dir / "calib.txt")
        if not all(os.path.exists(p) for p in [img1, img2, calib]):
            continue
        try:
            K = load_intrinsics(calib)
        except Exception as e:
            print(f"Skipping {scene} due to calibration error: {e}")
            continue

        if roma_model:
            success, npts, err, time_s = run_matching(img1, img2, K, "RoMA", roma_model, device)
            results.append(["RoMA", scene, success, npts, err, time_s])

        if get_homomatcher_matches:
            success, npts, err, time_s = run_matching(img1, img2, K, "HomoMatcher", get_homomatcher_matches, device)
            results.append(["HomoMatcher", scene, success, npts, err, time_s])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Scene", "Success", "TriangulatedPoints", "MeanReprojError", "MatchTime(s)"])
        for row in results:
            writer.writerow([row[0], row[1], row[2], row[3], f"{row[4]:.3f}", f"{row[5]:.3f}"])

    print(f"\nDone. Results saved to {output_csv}")

if __name__ == "__main__":
    main()
