import cv2


def get_best_camera():
    index = 0
    available_cameras = []
    w = 0
    h = 0

    print("Scan des caméras")
    while index < 10:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Index {index}] Caméra détectée : {w}x{h}")

            available_cameras.append({'index': index, 'width': w})
            cap.release()

        index += 1

    if not available_cameras:
        print("Aucune caméra trouvée.")
        return None, 0, 0

    available_cameras.sort(key=lambda x: x['width'], reverse=True)
    best_idx = available_cameras[0]['index']

    print(f"Sélection : Index {best_idx}")
    print(f"Flux initialisé en {w}x{h}")

    return best_idx
