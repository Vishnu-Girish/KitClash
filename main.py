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

# Setup web framework directories
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_dominant_colors(image_path, k=3):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    pixels = img.reshape((-1, 3))
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(pixels)
    return model.cluster_centers_.astype(int)

def calculate_similarity(color1_rgb, color2_rgb):
    c1_lab = color.rgb2lab(np.uint8([[color1_rgb]]) / 255.0)
    c2_lab = color.rgb2lab(np.uint8([[color2_rgb]]) / 255.0)
    return np.linalg.norm(c1_lab - c2_lab)

# --- Web App Routes ---

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Save uploaded files temporarily
    os.makedirs("temp", exist_ok=True)
    path1 = f"temp/{file1.filename}"
    path2 = f"temp/{file2.filename}"
    
    with open(path1, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(path2, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    # Execute math models
    colors1 = get_dominant_colors(path1)
    colors2 = get_dominant_colors(path2)
    
    # Compare primary colors
    distance = calculate_similarity(colors1[0], colors2[0])
    result = "CLASH" if distance < 20 else "PASS"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "distance": round(distance, 2)
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)