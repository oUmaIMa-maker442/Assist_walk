# test_yolo_only.py

import cv2
import sys
import os
sys.path.insert(0, os.path.abspath('text_module'))

from ultralytics import YOLO
from speech import speak_if_new
from distance_estimator import distance_message, estimate_distance

yolo_model = YOLO("yolov8n.pt")
print("✅ YOLO chargé")

TRADUCTIONS = {
    "person":        "personne",
    "car":           "voiture",
    "bicycle":       "vélo",
    "bus":           "bus",
    "truck":         "camion",
    "dog":           "chien",
    "stop sign":     "panneau stop",
    "traffic light": "feu de circulation",
    "chair":         "chaise",
    "bottle":        "bouteille",
}

def test_image(image_path):
    print(f"\n{'='*50}")
    print(f"Image : {image_path}")
    print("-" * 50)

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image introuvable")
        return

    results = yolo_model(img, verbose=False)

    # ✅ REMPLACER l'ancienne boucle par celle-ci
    DISTANCES_IMPORTANTES = ["très proche", "proche"]
    objects = []
    seen = set()

    for det in results[0].boxes:
        name       = yolo_model.names[int(det.cls)]
        bbox       = [int(b) for b in det.xyxy[0].tolist()]
        confidence = float(det.conf)

        if confidence < 0.5:
            continue

        dist_text, level = estimate_distance(bbox, img.shape[1], img.shape[0])

        # Garder seulement proche et très proche
        if dist_text not in DISTANCES_IMPORTANTES:
            continue

        # Dédoublonner par nom
        if name in seen:
            continue

        seen.add(name)
        name_fr = TRADUCTIONS.get(name, name)
        dist    = distance_message(name_fr, bbox, img.shape, lang='fr')
        objects.append({"name": name_fr, "bbox": bbox, "distance": dist})
        print(f"[YOLO] {name_fr} | conf={confidence:.2f} | {dist}")

    if not objects:
        print("[YOLO] Aucun objet proche détecté")
        return

    message = ". ".join(o["distance"] for o in objects)
    print(f"\n🔊 Message : '{message}'")
    speak_if_new(message, lang='fr')


# ---- Lance les tests ----
# Mets tes images ici
images = [
    "tests/test_images/test1.jpg",
    "tests/test_images/test2.jpg",
    "tests/test_images/test3.jpg",
]

for img_path in images:
    test_image(img_path)

print("\n✅ Tous les tests terminés")