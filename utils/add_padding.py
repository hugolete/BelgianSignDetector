import cv2
import os
import numpy as np

# Dézoomer les images de test (en ajoutant un padding) pour tester différentes "distances" avec les mêmes images

def add_padding(image_path, output_path, padding_factor):
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None: return

    h, w = img.shape[:2]

    # Calculer la taille du nouveau canevas
    new_h, new_w = h * padding_factor, w * padding_factor

    # Créer une image noire de la nouvelle taille
    padded_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # Calculer les coordonnées pour centrer l'image originale
    off_h = (new_h - h) // 2
    off_w = (new_w - w) // 2

    # Coller l'image originale au centre
    padded_img[off_h:off_h + h, off_w:off_w + w] = img

    # Sauvegarder
    cv2.imwrite(output_path, padded_img)


if __name__ == '__main__':
    zooms = [2,3,4]
    input_folder = '../datasets/Dataset test/'

    for zoom in zooms:
        output_folder = f'../datasets/Dataset test_padded{zoom}'
        os.makedirs(output_folder, exist_ok=True)

        for img_name in os.listdir(input_folder):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                add_padding(os.path.join(input_folder, img_name),os.path.join(output_folder, img_name),zoom)  # l'image devient X fois plus petite dans un cadre noir

    print("Padding terminé ! Images prêtes.")
