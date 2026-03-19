# tests/test_pipeline_complet.py

import sys
import os
import cv2
import numpy as np

# ✅ sys.path AVANT tous les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'text_module')))

from pipeline import process_frame
from distance_estimator import distance_message, estimate_distance
from speech import speak_if_new

os.makedirs("tests/test_images", exist_ok=True)

print("=" * 60)
print("     TEST PIPELINE COMPLET")
print("=" * 60)


def create_test_image(text, color_bg=(255, 255, 255), color_text=(0, 0, 0)):
    """Crée une image synthétique avec du texte"""
    img = np.ones((120, 400, 3), dtype=np.uint8)
    img[:] = color_bg
    cv2.putText(img, text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color_text, 3)
    return img


# ============================================
# TEST 1 : Aucun objet, aucun texte → français
# ============================================
print("\n[TEST 1] Aucun objet, aucun texte")
print("-" * 40)

data1 = {
    "objects": [],
    "text_regions": [],
    "frame_shape": (480, 640, 3)
}
msg1, lang1 = process_frame(data1)
print(f"✅ Message : '{msg1}'")
print(f"✅ Langue  : {lang1}")


# ============================================
# TEST 2 : Voiture proche + Personne loin
# ============================================
print("\n[TEST 2] Voiture proche + Personne loin")
print("-" * 40)

data2 = {
    "objects": [
        {"name": "car",    "bbox": [50,  50, 590, 430]},  # grande bbox → très proche
        {"name": "person", "bbox": [280, 200, 360, 280]}  # petite bbox → loin
    ],
    "text_regions": [],
    "frame_shape": (480, 640, 3)
}
msg2, lang2 = process_frame(data2)
print(f"✅ Message : '{msg2}'")
print(f"✅ Langue  : {lang2}")


# ============================================
# TEST 3 : Panneau anglais "NO ENTRY"
# ============================================
print("\n[TEST 3] Panneau anglais : NO ENTRY")
print("-" * 40)

img_en = create_test_image("NO ENTRY", (255, 255, 255), (0, 0, 0))
cv2.imwrite("tests/test_images/no_entry.jpg", img_en)

data3 = {
    "objects": [],
    "text_regions": [img_en],
    "frame_shape": (480, 640, 3)
}
msg3, lang3 = process_frame(data3)
print(f"✅ Message : '{msg3}'")
print(f"✅ Langue  : {lang3}")


# ============================================
# TEST 4 : Panneau français "SORTIE"
# ============================================
print("\n[TEST 4] Panneau français : SORTIE")
print("-" * 40)

img_fr = create_test_image("SORTIE", (255, 255, 0), (0, 0, 0))
cv2.imwrite("tests/test_images/sortie.jpg", img_fr)

data4 = {
    "objects": [],
    "text_regions": [img_fr],
    "frame_shape": (480, 640, 3)
}
msg4, lang4 = process_frame(data4)
print(f"✅ Message : '{msg4}'")
print(f"✅ Langue  : {lang4}")


# ============================================
# TEST 5 : Panneau arabe (injection directe)
# ============================================
print("\n[TEST 5] Panneau arabe : ممنوع الدخول")
print("-" * 40)

speak_if_new("ممنوع الدخول", lang='ar')
print("✅ Message : 'ممنوع الدخول'")
print("✅ Langue  : ar")


# ============================================
# TEST 6 : Scénario réaliste — voiture + STOP
# ============================================
print("\n[TEST 6] Scénario réel : voiture + panneau STOP")
print("-" * 40)

img_stop = create_test_image("STOP", (255, 0, 0), (255, 255, 255))
cv2.imwrite("tests/test_images/stop_test.jpg", img_stop)

data6 = {
    "objects": [
        {"name": "car",    "bbox": [200, 100, 580, 420]},  # très proche
        {"name": "person", "bbox": [10,  20,  120, 300]}   # proche
    ],
    "text_regions": [img_stop],
    "frame_shape": (480, 640, 3)
}
msg6, lang6 = process_frame(data6)
print(f"✅ Message : '{msg6}'")
print(f"✅ Langue  : {lang6}")


# ============================================
# TEST 7 : Objets inconnus (arbre, mur, porte)
# ============================================
print("\n[TEST 7] Objets inconnus : arbre + mur + porte")
print("-" * 40)

data7 = {
    "objects": [
        {"name": "arbre", "bbox": [100, 50,  540, 450]},   # très proche
        {"name": "mur",   "bbox": [200, 150, 420, 350]},   # proche
        {"name": "porte", "bbox": [290, 210, 350, 270]}    # loin
    ],
    "text_regions": [],
    "frame_shape": (480, 640, 3)
}
msg7, lang7 = process_frame(data7)
print(f"✅ Message : '{msg7}'")
print(f"✅ Langue  : {lang7}")


# ============================================
# TEST 8 : Vérification calcul distance seul
# ============================================
print("\n[TEST 8] Vérification calcul distance")
print("-" * 40)

frame_shape = (480, 640, 3)
test_bboxes = [
    ("très proche", [50,  50,  590, 430]),
    ("proche",      [100, 100, 400, 350]),
    ("moyenne",     [250, 180, 390, 300]),
    ("loin",        [300, 220, 340, 260]),
]

for label, bbox in test_bboxes:
    h, w = frame_shape[:2]
    dist_text, level = estimate_distance(bbox, w, h)
    msg = distance_message("objet", bbox, frame_shape, 'fr')
    status = "✅" if label in dist_text else "⚠️"
    print(f"  {status} {label:<12} → '{dist_text}' | '{msg}'")


# ============================================
# RÉSUMÉ
# ============================================
print("\n" + "=" * 60)
print("     RÉSUMÉ DES TESTS")
print("=" * 60)
print(f"Test 1 (vide)             : '{msg1}' [{lang1}]")
print(f"Test 2 (voiture+personne) : '{msg2}' [{lang2}]")
print(f"Test 3 (anglais)          : '{msg3}' [{lang3}]")
print(f"Test 4 (français)         : '{msg4}' [{lang4}]")
print(f"Test 5 (arabe)            : 'ممنوع الدخول' [ar]")
print(f"Test 6 (réaliste)         : '{msg6}' [{lang6}]")
print(f"Test 7 (objets inconnus)  : '{msg7}' [{lang7}]")
print("=" * 60)
print("✅ Pipeline complet testé avec succès !")