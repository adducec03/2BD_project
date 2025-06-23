import os
import random
import shutil

# Directory di input
IMG_DIR = 'augmented/images'
LBL_DIR = 'augmented/labels'

# Directory di output
BASE_DIR = 'my_dataset'
IMG_OUT = os.path.join(BASE_DIR, 'images')
LBL_OUT = os.path.join(BASE_DIR, 'labels')

# Crea le cartelle train/val
for split in ['train', 'val']:
    os.makedirs(os.path.join(IMG_OUT, split), exist_ok=True)
    os.makedirs(os.path.join(LBL_OUT, split), exist_ok=True)

# Lista di immagini
images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
random.shuffle(images)

# 80/20 split
split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def process_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            class_id = int(float(parts[0]))  # Gestisce sia '15' che '15.0'
            parts[0] = '0' if class_id == 15 else str(class_id)
            new_lines.append(' '.join(parts))

    return '\n'.join(new_lines) + '\n'

def move_files(img_list, split):
    for img_name in img_list:
        img_src = os.path.join(IMG_DIR, img_name)
        lbl_src = os.path.join(LBL_DIR, img_name.replace('.jpg', '.txt'))

        img_dst = os.path.join(IMG_OUT, split, img_name)
        lbl_dst = os.path.join(LBL_OUT, split, img_name.replace('.jpg', '.txt'))

        shutil.copy(img_src, img_dst)

        # Correggi label prima di copiare
        new_label = process_labels(lbl_src)
        with open(lbl_dst, 'w') as f:
            f.write(new_label)

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

print(f"✔ Completato: {len(train_imgs)} train, {len(val_imgs)} val immagini.")