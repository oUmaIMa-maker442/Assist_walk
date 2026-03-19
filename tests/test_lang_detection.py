import sys
sys.path.append('../text_module')

from lang_detector import detect_language
from pipeline import process_frame
import numpy as np

print("=" * 50)
print("TEST 1 : Détection de langue")
print("=" * 50)

texts = [
    "interdit de stationner",
    "no entry private road",
    "ممنوع الدخول",
    "STOP",
]
for t in texts:
    lang = detect_language(t)
    print(f"'{t}' → {lang}")


print()
print("=" * 50)
print("TEST 2 : Pipeline sans texte → français par défaut")
print("=" * 50)

data_no_text = {
    "objects": ["car", "person"],
    "text_regions": []
}
message, lang = process_frame(data_no_text)
print(f"Résultat : '{message}' | Langue : {lang}")


print()
print("=" * 50)
print("TEST 3 : Pipeline avec texte anglais simulé")
print("=" * 50)

# Simuler une image avec texte "no entry" (image blanche avec texte)
import cv2
import numpy as np

img = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(img, "no entry", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

data_english = {
    "objects": [],
    "text_regions": [img]
}
message, lang = process_frame(data_english)
print(f"Résultat : '{message}' | Langue : {lang}")