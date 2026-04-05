import numpy as np

def mask_top_fraction(img, frac=0.15):
    """
    Mask the top fraction of an image.

    Parameters
    ----------
    img : np.ndarray
        Image array of shape (H, W, C)
    frac : float
        Fraction of image height to mask (0.0 – 1.0)

    Returns
    -------
    np.ndarray
        Masked image (copy)
    """
    if frac <= 0:
        return img

    if not (0 <= frac <= 1):
        raise ValueError("frac must be between 0 and 1")

    h = img.shape[0]
    y = int(frac * h)

    out = img.copy()
    out[:y, :, :] = 0
    return out


import cv2
import numpy as np


def water_color_mask(
    img: np.ndarray,
    h_min: int = 80,
    h_max: int = 120,
    s_min: int = 30,
    v_min: int = 40,
) -> np.ndarray:
    """Return bool mask True where pixel is classified as water (whole image)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (
        (h >= h_min) & (h <= h_max) &
        (s >= s_min) &
        (v >= v_min)
    )


def bottom_exclude_mask_from_water(
    img: np.ndarray,
    bottom_frac: float = 0.30,
    h_min: int = 80,
    h_max: int = 120,
    s_min: int = 30,
    v_min: int = 40,
) -> np.ndarray:
    """
    Returns bool mask True where pixels should be EXCLUDED, but only in bottom region.
    Excluded = NOT water (in bottom ROI). Everywhere else False.
    """
    if not (0.0 < bottom_frac <= 1.0):
        raise ValueError("bottom_frac must be in (0, 1]")

    H, W = img.shape[:2]
    y0 = int((1 - bottom_frac) * H)

    water = water_color_mask(img, h_min=h_min, h_max=h_max, s_min=s_min, v_min=v_min)

    exclude = np.zeros((H, W), dtype=bool)
    exclude[y0:, :] = ~water[y0:, :]   # <-- ONLY bottom, and ONLY invert there
    return exclude


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color=(0, 255, 255),  # yellow in BGR
    alpha: float = 0.35,
) -> np.ndarray:
    """Transparent overlay of mask=True pixels."""
    vis = img.copy()
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)





import cv2
import numpy as np
from ultralytics import YOLO

def get_horizon_via_seg(frame, seg_model):
    """
    Returns the average horizon Y-coordinate, the binary mask, 
    and the raw YOLO results.
    """
    results = seg_model.predict(frame, conf=0.2, verbose=False)
    h, w = frame.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    horizon_y = None
    
    if results[0].masks is not None:
        # Generate the combined water/sea mask
        for mask in results[0].masks.data:
            m = mask.cpu().numpy()
            m = cv2.resize(m, (w, h))
            combined_mask = cv2.bitwise_or(combined_mask, (m * 255).astype(np.uint8))
        
        # Find the top-most edge (horizon)
        coords = np.column_stack(np.where(combined_mask > 0))
        if len(coords) > 0:
            # We use the 5th percentile of Y-coordinates to find the top edge
            horizon_y = int(np.percentile(coords[:, 0], 5))

    return horizon_y, combined_mask, results[0]

def mask_below_horizon(frame, horizon_y, frac=0.10):
    """
    Masks the sky and a small buffer area below the horizon.
    """
    if horizon_y is None:
        return frame # Return original if no horizon was found
        
    h = frame.shape[0]
    # Calculate the buffer (gap) to move the mask lower into the water
    gap = int(frac * h) 
    
    # Define the cutoff point
    mask_y = horizon_y + gap
    
    # Ensure mask_y doesn't exceed image height
    mask_y = min(mask_y, h)
    
    masked_frame = frame.copy()
    # Everything from the top (0) to mask_y becomes black
    masked_frame[0:mask_y, :] = 0  
    
    return masked_frame

import numpy as np
import cv2
import tempfile
import os
import utils.two_objectives_horizon_detection as tohd

def get_horizon_via_contrast(frame_bgr):
    """
    Returns: 
    - horizon_y: Average Y coordinate on original image
    - angle: The detected angle in degrees
    - horizon_points: ((x1, y1), (x2, y2)) for plotting on original image
    """
    h_orig, w_orig = frame_bgr.shape[:2]
    
    # 1. Handle the 15% Crop
    crop_limit = int(h_orig * 0.15)
    img_cropped = frame_bgr[crop_limit:, :]
    
    # Create temp file for the library
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, img_cropped)

    try:
        # 2. Global Search
        global_search = tohd.main(temp_path, img_reduction=0.1, angles=(-90, 91, 5), 
                                  distances=(5, 100, 5), buffer_size=3, local_objective=0)
        
        obj1 = np.max((global_search[:,:,0] - global_search[:,:,1]), 0) / (global_search[:,:,2])
        
        # 3. Local Search
        above_2s = (2 * np.nanstd(obj1)) + np.nanmean(obj1)
        l_idx = np.where(obj1 > above_2s)
        l_ang = global_search[l_idx[0], l_idx[1]][:,6]
        l_dist = global_search[l_idx[0], l_idx[1]][:,7]
        
        local_search = tohd.main(temp_path, img_reduction=0.25,
                                 angles=(int(np.min(l_ang))-2, int(np.max(l_ang))+3, 1), 
                                 distances=(int(np.min(l_dist))-2, int(np.max(l_dist))+3, 1), 
                                 buffer_size=5, local_objective=1)
        
        # 4. Extract Best Candidate
        obj2 = (local_search[:,:,4] - local_search[:,:,5])**2 / local_search[:,:,2]
        best_idx = np.unravel_index(obj2.argmax(), obj2.shape)
        best_cand = local_search[best_idx]
        
        angle = int(best_cand[6])
        dist = best_cand[7] / 100
        
        # 5. Coordinate Translation
        # Get coords relative to cropped image
        cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        line_coords = tohd.get_plane_indicator_coord(cropped_gray, angle, dist, 0)[2:4]
        
        # Shift Y-coordinates back to original scale
        p1 = (line_coords[0][0], line_coords[0][1] + crop_limit)
        p2 = (line_coords[1][0], line_coords[1][1] + crop_limit)
        
        # Calculate an average horizon_y for the masking function
        horizon_y = int((p1[1] + p2[1]) / 2)
        
        return horizon_y, angle, (p1, p2)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)