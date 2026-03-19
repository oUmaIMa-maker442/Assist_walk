# ═══════════════════════════════════════════════════════════
# ÉTAPE 6 : Extraction des zones de texte (crops)
# Auteure : Oumaima
# Rôle    : Découper chaque zone détectée par CRAFT pour
#           la transmettre au module OCR de Fatiha
# ═══════════════════════════════════════════════════════════

import numpy as np

# Marge en pixels autour de chaque zone (améliore l'OCR)
CROP_MARGIN = 4


def extract_text_regions(image: np.ndarray, text_boxes: list) -> list:
    """
    Découpe l'image originale selon les coordonnées CRAFT.

    Paramètres:
        image      : image RGB originale (PAS le redimensionné)
        text_boxes : liste de (x1, y1, x2, y2)

    Retourne:
        text_regions : liste de numpy arrays (crops)
                       chaque crop = une zone de texte
    """
    h, w = image.shape[:2]
    text_regions = []

    for i, (x1, y1, x2, y2) in enumerate(text_boxes):
        # Ajouter une petite marge autour de la zone
        # (améliore la précision de l'OCR)
        x1_m = max(0, x1 - CROP_MARGIN)
        y1_m = max(0, y1 - CROP_MARGIN)
        x2_m = min(w, x2 + CROP_MARGIN)
        y2_m = min(h, y2 + CROP_MARGIN)

        # Découper la région
        crop = image[y1_m:y2_m, x1_m:x2_m]

        # Vérifier que le crop n'est pas vide
        if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
            text_regions.append(crop)
            print(f'[Étape 6] Crop {i+1} : {crop.shape[1]}×{crop.shape[0]} px')
        else:
            print(f'[Étape 6] Crop {i+1} ignoré (trop petit)')

    print(f'[Étape 6] {len(text_regions)} regions extraites → envoi à Fatiha')
    return text_regions
