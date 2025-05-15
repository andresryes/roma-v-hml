from __future__ import annotations

import pathlib
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

class CoarseMatcher:
    def __init__(self, pretrained: str = "outdoor", device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.matcher = LoFTR(pretrained=pretrained).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def preprocess(img: np.ndarray) -> torch.Tensor:
            if img.ndim == 3: # HWC
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # HW
            img = torch.from_numpy(img).float() / 255.0 # HW
            return img.unsqueeze(0).unsqueeze(0).to(self.device) # LoFTR expects BCHW

        batch = {"image0": preprocess(img0), "image1": preprocess(img1)}
        
        # Inference
        output = self.matcher(batch)
        kpts0 = output["keypoints0"].cpu().numpy() # (N, 2)
        kpts1 = output["keypoints1"].cpu().numpy() # (N, 2)
        conf = output["confidence"].cpu().numpy()  # (N,)
        return kpts0, kpts1, conf


class HomographyRefiner:
    """Refine coarse matches by finding homography inliers (OpenCV RANSAC)."""
    def __init__(self, reproj_thresh: float = 3.0, min_inliers: int = 4):
        self.reproj_thresh = reproj_thresh
        self.min_inliers = min_inliers

    def __call__(self, kpts0: np.ndarray, kpts1: np.ndarray, conf: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        if len(kpts0) < self.min_inliers:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        H, inlier_mask = cv2.findHomography(kpts0, kpts1, cv2.RANSAC,
                                            ransacReprojThreshold=self.reproj_thresh,
                                            confidence=0.9999) # High confidence
        if H is None or inlier_mask is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        
        inliers = inlier_mask.ravel().astype(bool)
        rkpts0 = kpts0[inliers]
        rkpts1 = kpts1[inliers]
        return rkpts0, rkpts1

def get_homomatcher_matches(img0_path_str: str, 
                              img1_path_str: str, 
                              device: torch.device,
                              resize_to: Optional[Tuple[int, int]] = None, # (W, H)
                              ransac_reproj_thresh: float = 3.0
                              ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        # Load images
        img0 = cv2.imread(img0_path_str, cv2.IMREAD_COLOR) # Load as color first
        img1 = cv2.imread(img1_path_str, cv2.IMREAD_COLOR)

        if img0 is None or img1 is None:
            print(f"Error: Could not read images for HomoMatcher: {img0_path_str} or {img1_path_str}")
            return None

        if resize_to:
            w, h = resize_to
            img0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_AREA)
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        
        coarse_matcher = CoarseMatcher(device=device)
        
        k0_c, k1_c, _ = coarse_matcher(img0, img1) # Pass BGR or Grayscale numpy arrays

        if k0_c is None or len(k0_c) < 4:
            return None

        # Refine with homography RANSAC
        refiner = HomographyRefiner(reproj_thresh=ransac_reproj_thresh)
        rk0, rk1 = refiner(k0_c, k1_c) # Confidence from LoFTR not used by this simple refiner

        if rk0 is None or len(rk0) < 4: # Need at least 4 points for homography estimation
            return None
            
        return rk0, rk1

    except Exception as e:
        print(f"Error during HomoMatcher processing for {img0_path_str}, {img1_path_str}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Testing homomatcher_lite_adapted.py...")

    dummy_img_path0 = pathlib.Path("dummy_homo_img0.png")
    dummy_img_path1 = pathlib.Path("dummy_homo_img1.png")

    dummy_image_data0 = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    cv2.putText(dummy_image_data0, 'Img0', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5)
    cv2.imwrite(str(dummy_img_path0), dummy_image_data0)

    dummy_image_data1 = np.full((480, 640, 3), (180, 180, 180), dtype=np.uint8)
    cv2.putText(dummy_image_data1, 'Img1', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5)
    cv2.imwrite(str(dummy_img_path1), dummy_image_data1)

    if not dummy_img_path0.exists() or not dummy_img_path1.exists():
        print("Failed to create dummy images for testing.")
    else:
        print(f"Created dummy images: {dummy_img_path0}, {dummy_img_path1}")

        test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {test_device}")

        matches_output = get_homomatcher_matches(str(dummy_img_path0), str(dummy_img_path1), test_device)

        if matches_output:
            kpts0_test, kpts1_test = matches_output
            print(f"Successfully got {len(kpts0_test)} HomoMatcher matches.")
            print("Sample refined kpts0:", kpts0_test[:3])
            print("Sample refined kpts1:", kpts1_test[:3])

        else:
            print("HomoMatcher matching failed or returned no matches in the test.")
        print("Test finished.")
        print(f"Remember to delete dummy files: {dummy_img_path0}, {dummy_img_path1} if no longer needed.")
