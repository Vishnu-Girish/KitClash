import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from skimage import color
from sklearn.cluster import KMeans

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def get_dominant_colors(image_path, k=3):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))

    # Center-crop (middle 60%) to suppress background borders
    h, w, _ = img.shape
    start_row, start_col = int(h * 0.2), int(w * 0.2)
    end_row, end_col = int(h * 0.8), int(w * 0.8)
    cropped = img[start_row:end_row, start_col:end_col]

    pixels = cropped.reshape((-1, 3))
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(pixels)
    centers = model.cluster_centers_.astype(int)

    # Compute proportion of each cluster
    counts = np.bincount(labels, minlength=k)
    proportions = counts / len(pixels)

    # Sort descending by proportion size
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_centers = centers[sorted_indices]
    sorted_proportions = proportions[sorted_indices]

    return sorted_centers, sorted_proportions


def calculate_similarity(color1_rgb, color2_rgb):
    c1_lab = color.rgb2lab(np.uint8([[color1_rgb]]) / 255.0)
    c2_lab = color.rgb2lab(np.uint8([[color2_rgb]]) / 255.0)
    return np.linalg.norm(c1_lab - c2_lab)


def evaluate_kit_clash(colors1, props1, colors2, props2, clash_threshold=22.0):
    min_distance = float("inf")
    weighted_clash_score = 0.0

    # Primary-to-primary direct check
    primary_dist = calculate_similarity(colors1[0], colors2[0])
    is_direct_primary_clash = primary_dist < clash_threshold

    # Pairwise comparison across all dominant clusters
    for i, c1 in enumerate(colors1):
        for j, c2 in enumerate(colors2):
            dist = calculate_similarity(c1, c2)
            if dist < min_distance:
                min_distance = dist

            pair_weight = props1[i] * props2[j]
            if dist < clash_threshold:
                # Accumulate penalty inversely scaled by perceptual distance
                weighted_clash_score += pair_weight * (1.0 - (dist / clash_threshold))

    # Trigger clash if direct primary conflict or heavy pattern overlap
    clash_detected = is_direct_primary_clash or (weighted_clash_score > 0.25)
    outcome = "CLASH" if clash_detected else "PASS"

    return outcome, min_distance, round(weighted_clash_score, 3)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/analyze")
async def analyze(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    temp_dir = BASE_DIR / "temp"
    os.makedirs(temp_dir, exist_ok=True)

    path1 = str(temp_dir / file1.filename)
    path2 = str(temp_dir / file2.filename)

    with open(path1, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(path2, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    colors1, props1 = get_dominant_colors(path1)
    colors2, props2 = get_dominant_colors(path2)

    result, min_dist, clash_score = evaluate_kit_clash(colors1, props1, colors2, props2)

    hex_colors1 = [rgb_to_hex(c) for c in colors1]
    hex_colors2 = [rgb_to_hex(c) for c in colors2]

    # Clean up temporary files
    if os.path.exists(path1):
        os.remove(path1)
    if os.path.exists(path2):
        os.remove(path2)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "distance": round(min_dist, 2),
            "clash_score": clash_score,
            "colors1": hex_colors1,
            "colors2": hex_colors2,
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)