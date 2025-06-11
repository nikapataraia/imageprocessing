#!/usr/bin/env python3
# denoise_fixed_paths.py
import cv2 as cv
import numpy as np
import os

# --- edit these two lines ----------------------------------------------------
IN_PATH  = r"D:\WorkSpace\Personal\imageprocessing\noiseremover\candy-sweet-vol-2-main-game.png"   
OUT_PATH = r"D:\WorkSpace\Personal\imageprocessing\noiseremover\removed\candy-sweet-vol-2-main-game.png"  
# -----------------------------------------------------------------------------

BLUR_1 = np.array([
    [0.003, 0.013, 0.022, 0.013, 0.003],
    [0.013, 0.050, 0.078, 0.050, 0.013],
    [0.022, 0.078, 0.114, 0.078, 0.022],
    [0.013, 0.050, 0.078, 0.050, 0.013],
    [0.003, 0.013, 0.022, 0.013, 0.003]
], dtype=np.float32)

BLUR_2 = np.array([
    [0.003, 0.013, 0.022, 0.013, 0.003],
    [0.013, 0.060, 0.098, 0.060, 0.013],
    [0.022, 0.098, 0.162, 0.098, 0.022],
    [0.013, 0.060, 0.098, 0.060, 0.013],
    [0.003, 0.013, 0.022, 0.013, 0.003]
], dtype=np.float32)
LIGHT_BLUR = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16
SHARPEN = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)
def denoise(img: np.ndarray) -> np.ndarray:
    img = cv.filter2D(img, -1, LIGHT_BLUR)
    img = cv.medianBlur(img, 3)
    return cv.filter2D(img, -1, SHARPEN)

if __name__ == "__main__":
    img = cv.imread(IN_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read {IN_PATH}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv.imwrite(OUT_PATH, denoise(img))
    print(f"Saved denoised image → {OUT_PATH}")
