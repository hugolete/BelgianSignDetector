import cv2


def list_active_cameras():
    index = 0
    available_cameras = []

    while index < 10:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            print(f"Caméra trouvée à l'index {index})")
            available_cameras.append(index)
            cap.release()
        else:
            print(f"Index {index} : pas de caméra.")

        index += 1

    return available_cameras


print("Recherche des caméras en cours...")
cameras = list_active_cameras()

if not cameras:
    print("Aucune caméra détectée. Vérifie que la Virtual Cam OBS est active !")
else:
    print(f"\nIndex suggéré pour ton code : {cameras[0]}")
