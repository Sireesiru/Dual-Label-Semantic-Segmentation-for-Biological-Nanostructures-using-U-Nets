import os
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from skimage.measure import regionprops, label
from tqdm import tqdm

# ==========================================
#  Setup & Calibration
# ==========================================
MODEL_PATH = "model/best_unet_dice.pt"
DATA_DIR = "data"  
OUTPUT_CSV = "quantitative_morphometry_results.csv"

if not os.path.exists("outputs"):
    os.makedirs("outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NM_PER_PIX = 0.898  # replace with your calibration factor (nm/pixel)if using a custom image

# ==========================================
# 2. Load Model
# ==========================================
print(f"Loading model from {MODEL_PATH}...")
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

records = []

# ==========================================
# 3. Iterate Through "Data" Folder
# ==========================================
# Getting all images from your data folder
image_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))]

print(f"Found {len(image_files)} images. Starting prediction and metric extraction...")

with torch.no_grad():
    for filename in tqdm(image_files):
        # Load Image
        img_path = os.path.join(DATA_DIR, filename)
        raw_img = Image.open(img_path).convert('L')
        
        img_array = np.array(raw_img) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        # Get U-Net Predictions
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs)
        pred_masks = (probs > 0.5).cpu().numpy().astype(np.uint8)[0] # Shape: [2, 640, 640]

        # Extract OM (Channel 0) and IM (Channel 1)
        om_mask = pred_masks[0]
        im_mask = pred_masks[1]
        
        # ==========================================
        # 4. Extract Metrics for OM
        # ==========================================
        om_label = label(om_mask)
        om_props = regionprops(om_label)
        
        if om_props:
            # Select largest object (the main bacterium)
            om_region = max(om_props, key=lambda r: r.area)
            
            # Area and Perimeter calculations for OM
            area_nm2 = om_region.area * (NM_PER_PIX**2)
            perim_nm = om_region.perimeter * NM_PER_PIX
            
            # ==========================================
            # 5. Periplasmic Area Logic (OM Area - IM Area)
            # ==========================================
            im_label = label(im_mask)
            im_props = regionprops(im_label)
            periplasm_area_nm2 = 0
            
            if im_props:
                im_region = max(im_props, key=lambda r: r.area)
                im_area_nm2 = im_region.area * (NM_PER_PIX**2)
                # Logic: Periplasm is the space between OM and IM
                periplasm_area_nm2 = area_nm2 - im_area_nm2

            # Append to records
            records.append({
                "Filename": filename,
                "OM_Area_nm2": round(area_nm2, 2),
                "OM_Perimeter_nm": round(perim_nm, 2),
                "OM_Eccentricity": round(om_region.eccentricity, 4),
                "Periplasm_Area_Estimate_nm2": round(periplasm_area_nm2, 2),
                "OM_Orientation_deg": round(np.degrees(om_region.orientation), 2)
            })
            # ==========================================
            # 6. Professional Visualization (Contours)
            # ==========================================
            # Convert grayscale to BGR so we can draw in color
            viz_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_GRAY2BGR)
    
            # Draw Outer Membrane (Red)
            om_contours, _ = cv2.findContours(om_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz_img, om_contours, -1, (0, 0, 255), 2) # Red
    
            # Draw Inner Membrane (Blue)
            im_contours, _ = cv2.findContours(im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz_img, im_contours, -1, (255, 0, 0), 2) # Blue
    
            # Save the result
            viz_path = os.path.join("outputs", f"overlay_{filename}")
            cv2.imwrite(viz_path, viz_img)
# ==========================================
# 7. Save CSV for GitHub
# ==========================================
df = pd.DataFrame(records)
final_csv_path = os.path.join("outputs", OUTPUT_CSV)
df.to_csv(final_csv_path, index=False)