import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import color
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_dominant_colors(image_path, k=3):
    img = cv2.imread(image_path)
    if img is None: 
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    
    # Focus on the center 60% of the image
    h, w, _ = img.shape
    start_row, start_col = int(h * 0.2), int(w * 0.2)
    end_row, end_col = int(h * 0.8), int(w * 0.8)
    cropped = img[start_row:end_row, start_col:end_col]
    
    pixels = cropped.reshape((-1, 3))
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(pixels)
    return model.cluster_centers_.astype(int)

def calculate_similarity(color1_rgb, color2_rgb):
    c1_lab = color.rgb2lab(np.uint8([[color1_rgb]]) / 255.0)
    c2_lab = color.rgb2lab(np.uint8([[color2_rgb]]) / 255.0)
    return np.linalg.norm(c1_lab - c2_lab)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    path1 = f"temp/{file1.filename}"
    path2 = f"temp/{file2.filename}"
    
    with open(path1, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(path2, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    colors1 = get_dominant_colors(path1)
    colors2 = get_dominant_colors(path2)
    
    distance = calculate_similarity(colors1[0], colors2[0])
    result = "CLASH" if distance < 25 else "PASS"

    hex_colors1 = [rgb_to_hex(c) for c in colors1]
    hex_colors2 = [rgb_to_hex(c) for c in colors2]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "distance": round(distance, 2),
        "colors1": hex_colors1,
        "colors2": hex_colors2
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)