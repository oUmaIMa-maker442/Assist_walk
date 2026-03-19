# text_module/speech.py

import time
import os
import glob
import threading
import queue
from gtts import gTTS
from playsound import playsound

MIN_INTERVAL  = 3
_last_message = ""
_last_time    = 0
_last_valid_t = 0

# ── File d'attente audio — thread dédié ────────────────────
_audio_queue  = queue.Queue()
_audio_thread = None


def _worker():
    """Thread dédié à la lecture audio — évite les blocages"""
    while True:
        item = _audio_queue.get()
        if item is None:
            break
        message, lang = item
        filename = f"temp_audio_{int(time.time())}.mp3"
        try:
            lang_code = {'fr': 'fr', 'en': 'en', 'ar': 'ar'}.get(lang, 'fr')
            gTTS(text=message, lang=lang_code).save(filename)
            playsound(filename)
        except Exception as e:
            print(f"[SPEECH ERROR] {e}")
        finally:
            try: os.remove(filename)
            except: pass
        _audio_queue.task_done()


def _start_worker():
    global _audio_thread
    if _audio_thread is None or not _audio_thread.is_alive():
        _audio_thread = threading.Thread(target=_worker, daemon=True)
        _audio_thread.start()


# Démarrer le worker au chargement du module
_start_worker()


def nettoyer_audio():
    """Supprime les fichiers audio temporaires résiduels"""
    fichiers = glob.glob("temp_audio_*.mp3")
    for f in fichiers:
        try: os.remove(f)
        except: pass
    if fichiers:
        print(f"[SPEECH] 🗑️ {len(fichiers)} fichiers audio supprimés")


def speak_if_new(message: str, lang: str = 'fr'):
    global _last_message, _last_time, _last_valid_t

    now = time.time()

    EMPTY_MESSAGES = [
        "aucun obstacle détecté",
        "no obstacles detected",
        "لا يوجد عوائق"
    ]

    if message in EMPTY_MESSAGES:
        if now - _last_valid_t < 3.0:
            return
    else:
        _last_valid_t = now

    if message == _last_message and (now - _last_time) < MIN_INTERVAL:
        return

    _last_message = message
    _last_time    = now

    # ✅ Vider la file si déjà en train de parler (navigation)
    # Garder le message lecture complet — ne pas vider pour mode lecture
    if _audio_queue.qsize() > 1:
        try:
            while not _audio_queue.empty():
                _audio_queue.get_nowait()
                _audio_queue.task_done()
        except: pass

    _start_worker()
    _audio_queue.put((message, lang))