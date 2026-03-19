# test_webcam.py — Navigation + Lecture instantanée

import cv2
import sys
import os
import time
import threading
import keyboard
import glob
sys.path.insert(0, os.path.abspath('text_module'))
sys.path.insert(0, os.path.abspath('assistwalk_vision'))

from src.step4_filtering import filter_objects
from src.step6_extraction import extract_text_regions
from vision_module import VisionModule
from pipeline import process_frame, process_lecture
from speech import speak_if_new, nettoyer_audio

# ── Nettoyage au démarrage ────────────────────────────────
nettoyer_audio()
for f in glob.glob("debug_*.jpg"):
    try: os.remove(f)
    except: pass

print("Chargement des modèles...")
vision = VisionModule()

# ✅ Précharger EasyOCR UNE SEULE FOIS — évite rechargement à chaque appel
print("Chargement EasyOCR...")
from ocr_engine import _get_reader
_get_reader('fr')   # reader latin
_get_reader('ar')   # reader arabe
print("✅ Système prêt")
print("─"*50)
print("  ESPACE = Mode Lecture (instantané)")
print("  Q      = Quitter")
print("─"*50)

cap = cv2.VideoCapture("http://10.1.151.79:8080/video")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Webcam non détectée !")
    exit()

# ── Variables partagées ───────────────────────────────────
last_texts          = []        # crops CRAFT pour navigation
craft_running       = False
lecture_active      = False
stop_programme      = False
lecture_pending     = False
lock                = threading.Lock()
frame_count         = 0
YOLO_EVERY          = 10
CRAFT_EVERY         = 50

# ── Cache lecture (résultat pré-calculé) ──────────────────
lecture_cache       = None      # dernier résultat OCR lecture
lecture_cache_frame = None      # frame correspondante
ocr_bg_running      = False     # OCR background en cours
OCR_BG_EVERY        = 60        # recalculer toutes les 60 frames


def adapt_objects(filtered):
    return [
        {"name": obj["class"], "bbox": [int(b) for b in obj["bbox"]]}
        for obj in filtered
    ]


# ── Thread clavier ────────────────────────────────────────
def ecoute_clavier():
    global stop_programme, lecture_pending

    def on_space(e):
        global lecture_pending
        if not lecture_active:
            lecture_pending = True

    def on_q(e):
        global stop_programme
        stop_programme = True

    keyboard.on_press_key("space", on_space)
    keyboard.on_press_key("q",     on_q)
    keyboard.wait()

threading.Thread(target=ecoute_clavier, daemon=True).start()
print("⌨️  Écoute clavier active...")


# ── OCR background — pré-calcule le résultat lecture ─────
def ocr_background(frame):
    """
    Lance l'OCR en arrière-plan sur la frame actuelle.
    Résultat stocké dans lecture_cache → ESPACE = instantané
    """
    global lecture_cache, lecture_cache_frame, ocr_bg_running
    try:
        result = process_lecture(frame, speak=False)  # speak=False → pas de voix
        with lock:
            lecture_cache       = result
            lecture_cache_frame = frame.copy()
        print(f"[OCR BG] ✅ Cache mis à jour : '{result[0][:50]}...'")
    except Exception as e:
        print(f"[OCR BG ERROR] {e}")
    finally:
        ocr_bg_running = False


# ── CRAFT background ──────────────────────────────────────
def run_craft_background(frame):
    global last_texts, craft_running
    try:
        text_boxes   = vision.craft.detect_text_zones(frame)
        text_regions = extract_text_regions(frame, text_boxes)
        with lock:
            last_texts    = text_regions
            craft_running = False
    except Exception as e:
        craft_running = False


# ── Mode Lecture — utilise le cache ──────────────────────
def mode_lecture():
    global lecture_active
    lecture_active = True

    print("\n📖 MODE LECTURE")

    with lock:
        cache = lecture_cache

    if cache is not None:
        # ✅ Résultat déjà prêt → instantané
        message, lang = cache
        print(f"[LECTURE] ⚡ Instantané : '{message}' [{lang}]")
        speak_if_new(message, lang=lang)
    else:
        # Pas encore de cache → analyser maintenant
        print("[LECTURE] Cache vide → analyse en cours...")
        speak_if_new("Analyse en cours, patientez", lang='fr')

        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite("debug_lecture.jpg", frame)   # ✅ sauvegarder pour debug
            print(f"[DEBUG] Frame : {frame.shape} — ouvrir debug_lecture.jpg pour vérifier")
            message, lang = process_lecture(frame)
            print(f"[LECTURE] ✅ '{message}' [{lang}]")

    print("📷 Retour navigation...")
    lecture_active = False


# ── Boucle principale ─────────────────────────────────────
print("📷 Démarrage... ESPACE=lire | Q=quitter\n")

try:
    while not stop_programme:

        # ── Déclencher lecture ────────────────────────────
        if lecture_pending and not lecture_active:
            lecture_pending = False
            threading.Thread(target=mode_lecture, daemon=True).start()
            continue

        # ── Mode Navigation ───────────────────────────────
        if not lecture_active:

            for _ in range(3):
                cap.read()
            ret, frame = cap.read()

            if not ret:
                print("⚠️  Reconnexion webcam...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture("http://10.1.151.79:8080/video")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_count += 1

            # YOLO rapide
            if frame_count % YOLO_EVERY == 0:
                try:
                    results_raw     = vision.yolo.detect(frame)
                    filtered        = filter_objects(results_raw)
                    objects_adapted = adapt_objects(filtered)
                    with lock:
                        texts = last_texts.copy()
                    data = {
                        "objects":      objects_adapted,
                        "text_regions": texts,
                        "frame_shape":  frame.shape
                    }
                    message, lang = process_frame(data, use_ai=False)
                    print(f"[NAV] '{message}' [{lang}]")
                except Exception as e:
                    print(f"[NAV ERROR] {e}")

            # CRAFT arrière-plan
            if frame_count % CRAFT_EVERY == 0 and not craft_running:
                craft_running = True
                threading.Thread(
                    target=run_craft_background,
                    args=(frame.copy(),),
                    daemon=True
                ).start()

            # ✅ OCR background — pré-calcule pour ESPACE instantané
            if frame_count % OCR_BG_EVERY == 0 and not ocr_bg_running:
                ocr_bg_running = True
                threading.Thread(
                    target=ocr_background,
                    args=(frame.copy(),),
                    daemon=True
                ).start()

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n✅ Arrêt par Ctrl+C")

cap.release()
nettoyer_audio()
for f in glob.glob("debug_*.jpg"):
    try: os.remove(f)
    except: pass
print("✅ Webcam fermée — fichiers temporaires supprimés")