import os


mapping = {
    # TRIANGLES (Classe 0)
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0,
    18: 0, 19: 0,
    # RONDS ROUGES (Classe 1)
    20: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 40: 1, 41: 1, 42: 1, 43: 1,
    # RONDS BLEUS (Classe 2)
    33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2,
    # RECTANGLES / CARRÉS (Classe 3)
    44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3, 51: 3, 52: 3, 53: 3, 54: 3, 55: 3, 56: 3, 57: 3, 58: 3, 59: 3,
    # LOSANGES (Classe 4)
    60: 4, 61: 4,
    # STOP / OCTOGONE (Classe 5)
    21: 5
}


def convert_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                data = line.split()
                old_id = int(data[0])

                if old_id in mapping:
                    new_id = mapping[old_id]
                    # garde les coordonnées telles quelles
                    data[0] = str(new_id)
                    new_lines.append(" ".join(data))

            with open(os.path.join(output_folder, filename), 'w') as f:
                f.write("\n".join(new_lines))


if __name__ == '__main__':
    convert_dataset('../datasets/BelgiumTSC_Training_YOLO/labels/', '../datasets/Shape_Dataset/labels/')
    convert_dataset('../datasets/BelgiumTSC_Testing_YOLO/labels/', '../datasets/Shape_Dataset_Testing/labels/')

    print("Conversion terminée avec succès !")
