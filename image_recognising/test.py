from ultralytics import YOLO
import cv2
import os
import sys
from glob import glob

# === CONFIG ===
MODEL_PATH = 'runs/detect/train2/weights/best.pt'

# === INPUT: cartella di immagini ===
input_dir = "testing/test_images"

# === OUTPUT: cartella per salvare immagini annotate ===
output_dir = 'testing/output_pred'
os.makedirs(output_dir, exist_ok=True)

model = YOLO(MODEL_PATH)
image_paths = glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.png'))

for img_path in image_paths:
    img = cv2.imread(img_path)
    results = model(img_path, conf=0.5)[0]  # puoi alzare conf se vuoi meno box

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)  # viola

    filename = os.path.basename(img_path)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, img)
    print(f"✅ Salvata immagine pulita: {out_path}")