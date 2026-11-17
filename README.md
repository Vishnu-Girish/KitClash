# KitClash ⚽👕

**KitClash** is a real-time computer vision web application designed for football officials, referees, and kit managers to detect visual color clashes between opposing team jerseys before kickoff.

## Features
- **Dominant Color Extraction:** Uses **K-Means Clustering** in 3D pixel space to extract the top 3 color centroids and their proportional coverage on each jersey.
- **Background Suppression:** Implements center-focused spatial cropping (inner 60% ROI) to filter out pitch grass, stadium backgrounds, and studio borders.
- **Perceptual Color Science:** Converts RGB centroids into the **CIE LAB** color space to calculate **Delta E ($\Delta E$)** Euclidean distances, matching human visual perception rather than raw digital RGB differences.
- **Multi-Color Pattern Scoring:** Evaluates pairwise distances across all extracted clusters weighted by color proportion to catch clashes in striped, hooped, or patterned kits.

## Tech Stack
- **Backend:** Python, FastAPI, Uvicorn
- **Computer Vision & ML:** OpenCV, Scikit-Learn, Scikit-Image, NumPy
- **Frontend:** HTML5, CSS3, Jinja2 Templates

## Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Vishnu-Girish/KitClash.git](https://github.com/Vishnu-Girish/KitClash.git)
   cd KitClash

Note : Always wanted to make a football related project on GitHub so decided this was a good idea to work on 
