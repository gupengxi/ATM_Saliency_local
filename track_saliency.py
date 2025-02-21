from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the uploaded image
image_path = "flow.png"
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Step 1: Edge detection to isolate trajectories
# edges = cv2.Canny(image_gray, 100, 200)

# # Step 2: Compute saliency based on curvature (approximation using Sobel gradients)
# sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
# gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
# gradient_magnitude = np.uint8(255 * (gradient_magnitude / np.max(gradient_magnitude)))

# # Step 3: Normalize and create a saliency map
# saliency_map = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0)

# # Step 4: Overlay saliency map on the original image
# saliency_colormap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
# overlay = cv2.addWeighted(image, 0.6, saliency_colormap, 0.4, 0)

# # Display results
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title("Original Image")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.title("Saliency Map")
# plt.imshow(saliency_map, cmap='hot')
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.title("Overlay")
# plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# plt.tight_layout()
# plt.show()


# # Step 1: Use a smoothing approach for trajectory highlights
# # Compute gradients for the entire image, not just edges
# image_float = image_gray.astype(np.float32) / 255.0  # Normalize for smoother computation
# sobelx_full = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
# sobely_full = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)
# gradient_magnitude_full = np.sqrt(sobelx_full**2 + sobely_full**2)

# # Normalize the gradient magnitude for visualization
# saliency_full = (gradient_magnitude_full / np.max(gradient_magnitude_full)) * 255
# saliency_full = saliency_full.astype(np.uint8)

# # Step 2: Apply Gaussian smoothing for harmony
# saliency_smoothed = cv2.GaussianBlur(saliency_full, (15, 15), 5)

# # Step 3: Overlay saliency map on the original image harmoniously
# saliency_colormap_smooth = cv2.applyColorMap(saliency_smoothed, cv2.COLORMAP_JET)
# overlay_harmonious = cv2.addWeighted(image, 0.3, saliency_colormap_smooth, 0.7, 0)

# # Display the new results
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title("Original Image")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.title("Harmonized Saliency Map")
# plt.imshow(saliency_smoothed, cmap='hot')
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.title("Harmonized Overlay")
# plt.imshow(cv2.cvtColor(overlay_harmonious, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# Step 1: Use a smoothing approach for trajectory highlights
# Compute gradients for the entire image, not just edges
image_float = image_gray.astype(np.float32) / 255.0  # Normalize for smoother computation
sobelx_full = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
sobely_full = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude_full = np.sqrt(sobelx_full**2 + sobely_full**2)

# Normalize the gradient magnitude for visualization
saliency_full = (gradient_magnitude_full / np.max(gradient_magnitude_full)) * 255
saliency_full = saliency_full.astype(np.uint8)

# Step 2: Apply Gaussian smoothing for harmony
saliency_smoothed = cv2.GaussianBlur(saliency_full, (15, 15), 5)
# Step 1: Strong Gaussian blur to smooth relations between trajectories
strong_blur = cv2.GaussianBlur(saliency_smoothed, (31, 31), 15)

# Step 2: Enhance global trends by emphasizing large-scale gradients
# Compute large-scale gradients (directional flow)
large_scale_sobelx = cv2.Sobel(strong_blur, cv2.CV_64F, 1, 0, ksize=7)
large_scale_sobely = cv2.Sobel(strong_blur, cv2.CV_64F, 0, 1, ksize=7)
global_gradient = np.sqrt(large_scale_sobelx**2 + large_scale_sobely**2)
global_gradient_normalized = (global_gradient / np.max(global_gradient)) * 255
global_gradient_normalized = global_gradient_normalized.astype(np.uint8)

# Step 3: Merge global trend with smoothed saliency map
trend_emphasis = cv2.addWeighted(strong_blur, 0.5, global_gradient_normalized, 0.5, 0)

# Step 4: Overlay the enhanced saliency map onto the original image
saliency_colormap_trend = cv2.applyColorMap(trend_emphasis, cv2.COLORMAP_JET)
overlay_trend = cv2.addWeighted(image, 0, saliency_colormap_trend, 0.7, 0)

# Display the updated results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Global Trend Saliency Map")
plt.imshow(trend_emphasis, cmap='hot')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Trend Overlay")
plt.imshow(cv2.cvtColor(overlay_trend, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
