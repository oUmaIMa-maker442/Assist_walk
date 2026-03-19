# ═══════════════════════════════════════════════════════════
# ÉTAPE 1 : Acquisition de l'image
# Auteure : Oumaima
# Rôle    : Charger une image depuis n'importe quelle source
#           et la convertir en numpy array RGB standard
# ═══════════════════════════════════════════════════════════

import cv2
import numpy as np
from PIL import Image


def acquire_from_file(image_path: str) -> np.ndarray:
    """
    Charge une image depuis un fichier (JPG, PNG, BMP...).
    Retourne un numpy array en format RGB.
    """
    # OpenCV charge en BGR → on convertit en RGB
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f'Image introuvable : {image_path}')

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    print(f'[Étape 1] Image chargée : {image_path}')
    print(f'[Étape 1] Dimensions   : {rgb_image.shape}')
    return rgb_image


def acquire_from_video_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convertit une frame OpenCV (BGR) en RGB.
    Utilisé dans le Prototype 2 (vidéo) et Prototype 3 (webcam).
    """
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return rgb_frame


def acquire_from_pil(pil_image: Image.Image) -> np.ndarray:
    """
    Convertit une image PIL (Streamlit) en numpy array RGB.
    Utilisé dans l'interface Streamlit (Prototype 1).
    """
    return np.array(pil_image.convert('RGB'))
