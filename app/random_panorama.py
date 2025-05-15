import random
import pathlib
import cv2
import numpy as np
import torch
from typing import Tuple, Optional

def estimate_homography_from_matches(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    reproj_thresh: float = 3.0
) -> Optional[np.ndarray]:
    if pts_src is None or pts_dst is None or len(pts_src) < 4:
        return None

    H_est, mask = cv2.findHomography(
        pts_src,
        pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh
    )

    return H_est


def stitch_with_homography(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.array([[0,0], [0,h2], [w2,h2], [w2,0]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.vstack((warped_corners, np.array([[[0,0]], [[0,h1]], [[w1,h1]], [[w1,0]]], dtype=np.float32)))
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-x_min, -y_min]
    H_trans = np.array([[1, 0, translation[0]],
                        [0, 1, translation[1]],
                        [0, 0, 1]], dtype=np.float64)

    pano_size = (x_max - x_min, y_max - y_min)
    pano = cv2.warpPerspective(img2, H_trans.dot(H), pano_size)
    pano[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = img1
    return pano

def compare_random_sequence_panorama(
    hpatches_dir: pathlib.Path,
    device: torch.device = torch.device("cpu")
) -> Tuple[np.ndarray, np.ndarray]:
    
    seq_dirs = [d for d in hpatches_dir.iterdir()
                if d.is_dir() and (d / "1.ppm").exists() and (d / "2.ppm").exists()]
    if not seq_dirs:
        raise RuntimeError(f"No valid HPatches sequences found in {hpatches_dir}")
    
    seq = random.choice(seq_dirs)
    img1_path = seq / "1.ppm"
    img2_path = seq / "2.ppm"

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        raise RuntimeError(f"Could not load images from {seq.name}")


    cv2.imshow(f"Original Images - Sequence: {seq.name}", np.hstack((img1, img2)))

    matches_roma   = get_roma_matches(img1_path, img2_path, device)
    matches_homo   = get_homomatcher_matches(img1_path, img2_path, device)

    def stitch_from_matches(matches: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        if matches is None:
            return np.zeros((img1.shape[0], img1.shape[1]*2, 3), dtype=np.uint8)
        kp1, kp2 = matches
        H_est = estimate_homography_from_matches(kp1, kp2)
        if H_est is None:
            return np.zeros((img1.shape[0], img1.shape[1]*2, 3), dtype=np.uint8)
        return stitch_with_homography(img1, img2, H_est)

    pano_roma = stitch_from_matches(matches_roma)
    pano_homo = stitch_from_matches(matches_homo)

    h_max = max(pano_roma.shape[0], pano_homo.shape[0])
    w_sum = pano_roma.shape[1] + pano_homo.shape[1]
    canvas = np.zeros((h_max, w_sum, 3), dtype=np.uint8)
    canvas[:pano_roma.shape[0], :pano_roma.shape[1]] = pano_roma
    canvas[:pano_homo.shape[0], pano_roma.shape[1]:] = pano_homo

    cv2.imshow(f"RoMA (left) vs HomoMatcherLite (right) - Sequence: {seq.name}", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pano_roma, pano_homo


if __name__ == "__main__":
    from roma_inference_adapted import get_roma_matches
    from homomatcher_lite_adapted import get_homomatcher_matches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_root = pathlib.Path(__file__).resolve().parent.parent
    data_path = file_root / "data"
    hpatches_root = pathlib.Path(data_path / "hpatches-sequences-release")

    pano_r, pano_h = compare_random_sequence_panorama(hpatches_root, device)
    cv2.imwrite(file_root / "output/panorama_roma.png", pano_r)
    cv2.imwrite(file_root / "output/panorama_homo.png", pano_h)
