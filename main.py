# main.py — Point de liaison Oumaima + Fatiha

import sys
import os
import cv2
sys.path.insert(0, os.path.abspath('text_module'))
sys.path.insert(0, os.path.abspath('assistwalk_vision'))

from vision_module import VisionModule
from pipeline import process_frame

# ---- Initialisation ----
vision = VisionModule()


def adapt_objects(filtered):
    """
    Convertit le format Oumaima → format Fatiha
    Oumaima : [{'class': 'car', 'bbox': (x1,y1,x2,y2), 'confidence': 0.92}]
    Fatiha  : [{'name': 'car', 'bbox': [x1,y1,x2,y2]}]
    """
    adapted = []
    for obj in filtered:
        adapted.append({
            "name": obj["class"],
            "bbox": [int(b) for b in obj["bbox"]]
        })
    return adapted


def run_pipeline(source):
    """
    source : chemin image (str) ou image numpy
    """
    if isinstance(source, str):
        image = cv2.imread(source)
        if image is None:
            print(f"❌ Image introuvable : {source}")
            return
    else:
        image = source

    print(f"\n{'='*55}")
    print(f"[MAIN] Source : {source}")

    # ---- Étape 1 : Vision Oumaima ----
    results = vision.analyze(image)

    print(f"[MAIN] Objets     : {results['objects']}")
    print(f"[MAIN] Zones texte: {len(results['text_regions'])}")

    # ---- Étape 2 : Adapter format ----
    objects_adapted = adapt_objects(results['objects'])
    print(f"[MAIN] Adapté     : {objects_adapted}")

    # ---- Étape 3 : Pipeline Fatiha ----
    data = {
        "objects":      objects_adapted,
        "text_regions": results['text_regions'],
        "frame_shape":  image.shape
    }

    message, lang = process_frame(data, use_ai=True)

    print(f"\n✅ Message : '{message}'")
    print(f"✅ Langue  : {lang}")
    return message, lang


# ---- TEST ----
if __name__ == "__main__":
    run_pipeline("tests/test_images/test1.jpg")