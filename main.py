import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_colors(image_path, k=3):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to speed up processing
    img = cv2.resize(img, (200, 200))
    
    # Reshape the image to be a list of pixels (flattening)
    pixels = img.reshape((-1, 3))
    
    # Use K-Means to find the most dominant colors
    # n_init=10 is the standard for stable results
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(pixels)
    
    # Get the RGB values of the cluster centers
    colors = model.cluster_centers_.astype(int)
    
    return colors

# This is a test block to ensure it works
if __name__ == "__main__":
    print("Logic initialized. Ready for testing.")