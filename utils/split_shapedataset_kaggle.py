import os
import random
import shutil
from pathlib import Path


def split_dataset(source_img_dir, source_label_dir, output_root, split_ratio=0.8):
    # Création de l'arborescence YOLO
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    # Lister toutes les images (en supposant du .jpg ou .png)
    images = [f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def move_files(files, subset):
        for img_name in files:
            # Chemin de l'image
            img_src = os.path.join(source_img_dir, img_name)
            img_dst = os.path.join(output_root, 'images', subset, img_name)

            # Chemin du label (on remplace l'extension par .txt)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_src = os.path.join(source_label_dir, label_name)
            label_dst = os.path.join(output_root, 'labels', subset, label_name)

            # Copie de l'image
            shutil.copy2(img_src, img_dst)

            # Copie du label s'il existe
            if os.path.exists(label_src):
                shutil.copy2(label_src, label_dst)
            else:
                # Créer un fichier vide si pas de panneau (Background image)
                open(label_dst, 'a').close()

    print(f"Traitement de {len(train_imgs)} images pour le train...")
    move_files(train_imgs, 'train')
    print(f"Traitement de {len(val_imgs)} images pour la val...")
    move_files(val_imgs, 'val')
    print("Terminé !")


# --- CONFIGURATION ---
split_dataset(
    source_img_dir='../datasets/archive/images',
    source_label_dir='../datasets/archive/labels',
    output_root='ShapeDetector_Kaggle'
)
