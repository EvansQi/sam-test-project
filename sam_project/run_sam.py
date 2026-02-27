import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from segment_anything import SamPredictor, sam_model_registry

# ===============================
# 0. Configuration
# ===============================
IMAGE_FILENAME = "dog_feeding.png"       # Your image file name
MODEL_CHECKPOINT = "sam_vit_h_4b8939.pth" # Model checkpoint file name
MODEL_TYPE = "vit_h"                     # Model type: vit_h, vit_l, vit_b

# Check if files exist
if not os.path.exists(IMAGE_FILENAME):
    print(f"❌ Error: Image file '{IMAGE_FILENAME}' not found.")
    print(f"   Please ensure the image is in the same folder: {os.getcwd()}")
    print(f"   Files in current folder: {os.listdir('.')}")
    exit()

if not os.path.exists(MODEL_CHECKPOINT):
    print(f"❌ Error: Model file '{MODEL_CHECKPOINT}' not found.")
    print(f"   Please download the model and place it in the same folder.")
    exit()

print(f"✅ Found image: {IMAGE_FILENAME}")
print(f"✅ Found model: {MODEL_CHECKPOINT}")

# ===============================
# 1. Load SAM Model
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading model to device: {device} ...")

sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)
print("✅ Model loaded successfully!")

# ===============================
# 2. Load and Prepare Image
# ===============================
image = cv2.imread(IMAGE_FILENAME)
if image is None:
    print("❌ Error: Could not read image. File may be corrupted.")
    exit()

# Convert BGR (OpenCV default) to RGB (SAM requirement)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image_rgb.shape
print(f"🖼️ Image dimensions: {w} x {h}")

# Set the image for the predictor (encodes the image)
predictor.set_image(image_rgb)

# ===============================
# 3. Define Bounding Boxes (Prompts)
# ===============================
# Format: [x_min, y_min, x_max, y_max]
# Coordinates are estimated based on your image resolution (1570x1054)
boxes_list = [
    [0, int(h*0.4), int(w*0.5), int(h*0.95)],       # Dog (Bottom-Left)
    [int(w*0.65), 0, int(w*0.95), int(h*0.8)],      # Person (Right legs/hands)
    [int(w*0.6), int(h*0.25), int(w*0.82), int(h*0.48)] # Bowl (Center-Right)
]

# English labels for display and saving
box_names = ["Dog", "Person", "Bowl"]

# ===============================
# 4. Run Prediction (One by One)
# ===============================
print("⏳ Generating masks one by one...")

masks_list = []
scores_list = []

for i, box in enumerate(boxes_list):
    # Convert single box to numpy array with shape (1, 4)
    # IMPORTANT: Must be numpy, not torch tensor, for SAM's internal transforms
    box_np = np.array(box).reshape(1, 4)
    
    # Predict mask for this specific box
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_np,          
        multimask_output=False  # Return only the best mask per box
    )
    
    # Store results
    masks_list.append(masks[0]) 
    scores_list.append(scores[0])
    
    print(f"   ✅ [{box_names[i]}] Score: {scores[0]:.4f}")

# ===============================
# 5. Visualize Results
# ===============================
plt.figure(figsize=(20, 5))

# Plot Original Image
plt.subplot(1, len(masks_list) + 1, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

# Define colors for masks (Red, Green, Blue)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# Plot each mask overlay
for i, mask in enumerate(masks_list):
    plt.subplot(1, len(masks_list) + 1, i + 2)
    
    # Create a colored mask layer
    colored_mask = np.zeros_like(image_rgb)
    color = colors[i % len(colors)]
    colored_mask[mask == 1] = color
    
    # Blend original image and mask (70% image, 30% mask)
    overlay = cv2.addWeighted(image_rgb, 0.7, colored_mask, 0.3, 0)
    
    plt.imshow(overlay)
    plt.title(f"{box_names[i]} (Score: {scores_list[i]:.2f})")
    plt.axis('off')

plt.tight_layout()
plt.show()

# ===============================
# 6. Save Results
# ===============================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for i, mask in enumerate(masks_list):
    # Convert boolean mask to uint8 (0 or 255)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Save with English name
    save_path = os.path.join(output_dir, f"mask_{i}_{box_names[i]}.png")
    cv2.imwrite(save_path, mask_uint8)
    print(f"💾 Saved: {save_path}")

print("\n🎉 All done! Check the popup window and the 'results' folder.")