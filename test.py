import cv2
from services.reid import FeatureExtractor
# Load your pre-cropped image
img = cv2.imread("p1.jpg")

# Initialize and run
service = FeatureExtractor()
result = service.extract(img)

print(f"Extracted Features: {result}")