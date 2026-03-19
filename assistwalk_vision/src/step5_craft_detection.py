# ═══════════════════════════════════════════════════════════
# ÉTAPE 5 : Détection des zones de texte avec CRAFT
# Auteure : Oumaima
# Rôle    : Localiser toutes les régions contenant du texte
#           dans l'image. PAS de lecture — juste la détection.
#           La lecture est faite par Fatiha (OCR).
# ═══════════════════════════════════════════════════════════

import easyocr
import numpy as np


class CRAFTDetector:

    def __init__(self, languages=['fr', 'en']):

        print('[Étape 5] Initialisation CRAFT (EasyOCR)...')
        print('          → Premier lancement : téléchargement des modèles')
        print('          → Peut prendre 1-2 minutes la première fois')

        # Reader pour textes latins
        self.reader_latin = easyocr.Reader(['fr', 'en'], gpu=False)

        # Reader pour arabe
        self.reader_arabic = easyocr.Reader(['ar'], gpu=False)

        print('[Étape 5] CRAFT prêt ✓')


    def detect_text_zones(self, image: np.ndarray) -> list:


        results = []

        # Détection latin
        latin = self.reader_latin.detect(image)

        # Détection arabe
        arabic = self.reader_arabic.detect(image)

        # sécuriser extraction
        if latin and latin[0] and latin[0][0]:
            results += latin[0][0]

        if arabic and arabic[0] and arabic[0][0]:
            results += arabic[0][0]

        text_boxes = []

        h, w = image.shape[:2]

        for bbox in results:

            try:
                # format classique
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]

            except Exception:

                try:
                    # format rectangle simple
                    x1, y1, x2, y2 = map(int, bbox)
                    xs = [x1, x2]
                    ys = [y1, y2]

                except Exception:
                    continue

            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(w, max(xs))
            y2 = min(h, max(ys))

            # filtrer zones trop petites
            if (x2 - x1) > 20 and (y2 - y1) > 10:
                text_boxes.append((x1, y1, x2, y2))

        merged_boxes = self.merge_boxes(text_boxes)

        print(f'[Étape 5] {len(text_boxes)} boxes → {len(merged_boxes)} lignes texte')

        return merged_boxes

    def merge_boxes(self, boxes, y_threshold=25, x_gap=40):

        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged = []

        current = list(boxes[0])

        for b in boxes[1:]:

            x1, y1, x2, y2 = b

            # même ligne
            if abs(y1 - current[1]) < y_threshold and x1 - current[2] < x_gap:
                current[2] = max(current[2], x2)
                current[3] = max(current[3], y2)

            else:
                merged.append(tuple(current))
                current = [x1, y1, x2, y2]

        merged.append(tuple(current))

        return merged