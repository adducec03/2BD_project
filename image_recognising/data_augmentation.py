import os
import cv2
import albumentations as A
import random

# Cartelle di input e output
IMG_INPUT_DIR = "images"
LBL_INPUT_DIR = "labels"
IMG_OUT_DIR = "augmented/images"
LBL_OUT_DIR = "augmented/labels"

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(LBL_OUT_DIR, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=30, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.RandomScale(scale_limit=0.3, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomShadow(p=0.3),
    A.RandomFog(p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# Leggi tutti i file
image_files = [f for f in os.listdir(IMG_INPUT_DIR) if f.endswith('.jpg')]

for img_name in image_files:
    image_path = os.path.join(IMG_INPUT_DIR, img_name)
    label_path = os.path.join(LBL_INPUT_DIR, img_name.replace('.jpg', '.txt'))

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Carica le bbox
    with open(label_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    class_labels = []

    for line in lines:
        items = line.strip().split()
        label = int(items[0])
        bbox = list(map(float, items[1:5]))
        bboxes.append(bbox)
        class_labels.append(label)

    # Quante varianti vuoi per ogni immagine?
    for i in range(20):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # Salva immagine e txt
        base_name = img_name.replace('.jpg', f'_aug{i}.jpg')
        cv2.imwrite(os.path.join(IMG_OUT_DIR, base_name), aug_img)

        label_file = os.path.join(LBL_OUT_DIR, base_name.replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            for bbox, label in zip(aug_bboxes, aug_labels):
                f.write(f"{label} {' '.join(str(round(x, 6)) for x in bbox)}\n")