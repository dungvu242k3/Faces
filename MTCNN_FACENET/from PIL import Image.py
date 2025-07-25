import numpy as np
from PIL import Image

# Load image
img_path = "C:/Users/dungv/Pictures/Screenshots/Ảnh chụp màn hình 2025-06-25 181457.png"
img = Image.open(img_path).convert("RGBA")

# Convert to NumPy array
img_array = np.array(img)

# Create alpha mask to detect circle
alpha = img_array[:, :, 3]
non_empty_cols = np.where(alpha.max(axis=0) > 0)[0]
non_empty_rows = np.where(alpha.max(axis=1) > 0)[0]

# Crop bounds
crop_box = (min(non_empty_cols), min(non_empty_rows), max(non_empty_cols), max(non_empty_rows))
cropped_img = img.crop(crop_box)

# Save result
output_path = "C:/Users/dungv/Pictures/Screenshots/123.png"
cropped_img.save(output_path)
print("Saved cropped image to:", output_path)
