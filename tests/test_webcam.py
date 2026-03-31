# tests/test_webcam.py — Lanceur (toute la logique dans stream_processor.py)

import sys
import os
import time
import threading
import keyboard
import glob
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('text_module'))
sys.path.insert(0, os.path.abspath('assistwalk_vision'))

from stream_processor import StreamProcessor
from speech import nettoyer_audio

# ── Nettoyage ─────────────────────────────────────────────
nettoyer_audio()
for f in glob.glob("debug_*.jpg"):
    try: os.remove(f)
    except: pass

# ── Démarrer ──────────────────────────────────────────────
sp = StreamProcessor("http://10.1.154.176:8080/video", rotate=False)
sp.start()

print("Initialisation OCR cache...")
time.sleep(5) 
print("✅ Prêt !")

print("─"*50)
print("  ESPACE = Mode Lecture")
print("  Q      = Quitter")
print("─"*50)

stop_programme = False


def ecoute_clavier():
    global stop_programme

    def on_space(e):
        if not sp._lecture_active:
            threading.Thread(target=sp.read_document, kwargs={"speak": True}, daemon=True).start()

    def on_q(e):
        global stop_programme
        stop_programme = True

    keyboard.on_press_key("space", on_space)
    keyboard.on_press_key("q",     on_q)
    keyboard.wait()

threading.Thread(target=ecoute_clavier, daemon=True).start()
print("⌨️  Écoute clavier active... ESPACE=lire | Q=quitter\n")

try:
    while not stop_programme:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

sp.stop()
nettoyer_audio()
for f in glob.glob("debug_*.jpg"):
    try: os.remove(f)
    except: pass
print("✅ Terminé")