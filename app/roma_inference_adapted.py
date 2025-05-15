import pathlib
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from romatch import roma_outdoor 

def draw_matches_visualization(img0: np.ndarray, img1: np.ndarray, kpts0: np.ndarray, kpts1: np.ndarray, out_path: pathlib.Path):
    if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:
        print("No matches to draw.")
        return

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1
    
    rng = np.random.default_rng(42)

    for i in range(len(kpts0)):
        x0, y0 = kpts0[i]
        x1, y1 = kpts1[i]
        
        colour = tuple(rng.integers(0, 255, 3).tolist())
        pt0 = int(round(x0)), int(round(y0))
        pt1 = int(round(x1 + w0)), int(round(y1)) # Offset x1 by width of img0
        
        cv2.circle(canvas, pt0, 3, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, colour, -1, cv2.LINE_AA)
        cv2.line(canvas, pt0, pt1, colour, 1, cv2.LINE_AA)
        
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved RoMA visualization to {out_path}")


def get_roma_matches(img0_path_str: str, 
                       img1_path_str: str, 
                       device: torch.device,
                       num_samples: int = 5000 # Number of matches to sample
                       ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        roma_model = roma_outdoor(device=device)

        img0_for_shape = cv2.imread(img0_path_str)
        img1_for_shape = cv2.imread(img1_path_str)

        if img0_for_shape is None or img1_for_shape is None:
            print(f"Error: Could not read images for RoMA: {img0_path_str} or {img1_path_str}")
            return None

        h0, w0 = img0_for_shape.shape[:2]
        h1, w1 = img1_for_shape.shape[:2]

        warp, certainty = roma_model.match(img0_path_str, img1_path_str, device=device)

        if warp is None:
            return None

        matches_sampled, certainty_sampled = roma_model.sample(warp, certainty, num=num_samples)

        if matches_sampled is None or matches_sampled.shape[0] < 4: # Need at least 4 points for homography
            return None

        kptsA_tensor, kptsB_tensor = roma_model.to_pixel_coordinates(matches_sampled, h0, w0, h1, w1)

        kptsA_np = kptsA_tensor.cpu().numpy()
        kptsB_np = kptsB_tensor.cpu().numpy()

        return kptsA_np, kptsB_np

    except Exception as e:
        print(f"Error during RoMA matching for {img0_path_str}, {img1_path_str}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Testing roma_inference_adapted.py...")

    dummy_img_path0 = pathlib.Path("dummy_roma_img0.png")
    dummy_img_path1 = pathlib.Path("dummy_roma_img1.png")
    
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

        matches_output = get_roma_matches(str(dummy_img_path0), str(dummy_img_path1), test_device)

        if matches_output:
            kpts0_test, kpts1_test = matches_output
            print(f"Successfully got {len(kpts0_test)} RoMA matches.")
            print("Sample kpts0:", kpts0_test[:3])
            print("Sample kpts1:", kpts1_test[:3])

            vis_out_path = pathlib.Path("roma_adapted_test_vis.png")
            draw_matches_visualization(dummy_image_data0, dummy_image_data1, kpts0_test, kpts1_test, vis_out_path)
        else:
            print("RoMA matching failed or returned no matches in the test.")
            
        print(f"Test finished. If successful, check for '{vis_out_path}' (if matches were found).")
        print(f"Remember to delete dummy files: {dummy_img_path0}, {dummy_img_path1} if no longer needed.")

