import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import color

def get_dominant_colors(image_path, k=3):
    """
    Extracts the top 'k' dominant colors from an image using K-Means clustering.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found"
    
    # 2. Convert from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize to speed up processing (200x200 is plenty for color analysis)
    img = cv2.resize(img, (200, 200))
    
    # 4. Reshape the image to be a list of pixels (flattening from 2D to 1D)
    pixels = img.reshape((-1, 3))
    
    # 5. Use K-Means to find the clusters
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(pixels)
    
    # 6. Get the RGB values of the cluster centers
    colors = model.cluster_centers_.astype(int)
    
    return colors

def calculate_similarity(color1_rgb, color2_rgb):
    """
    Calculates the perceptual distance (Delta E) between two RGB colors.
    """
    # Convert RGB to LAB (values must be normalized between 0 and 1)
    c1_lab = color.rgb2lab(np.uint8([[color1_rgb]]) / 255.0)
    c2_lab = color.rgb2lab(np.uint8([[color2_rgb]]) / 255.0)

    # Calculate Euclidean distance in the LAB color space
    delta_e = np.linalg.norm(c1_lab - c2_lab)
    
    return delta_e

# Initializer block
if __name__ == "__main__":
    print("KitClash Logic Loaded.")
    print("Ready to process colors and calculate distances.")