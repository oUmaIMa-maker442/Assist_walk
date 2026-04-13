# text_module/speech.py
# Phase 1: default lang changed to 'en'; empty-message list updated.

import time
import os
import glob
import threading
import queue
from gtts      import gTTS
from playsound import playsound

MIN_INTERVAL  = 3          # seconds between identical messages
_last_message = ""
_last_time    = 0
_last_valid_t = 0

_audio_queue  = queue.Queue()
_audio_thread = None

# Set to True by StreamProcessor before OCR — blocks navigation audio
lecture_mode  = False

# Messages that mean "nothing happening" — rate-limited more aggressively
_EMPTY_MESSAGES = {
    "no obstacles detected",
    "aucun obstacle détecté",
    "لا يوجد عوائق",
}


def _worker():
    while True:
        item = _audio_queue.get()
        if item is None:
            break
        message, lang = item
        filename = f"temp_audio_{int(time.time())}.mp3"
        try:
            lang_code = {'fr': 'fr', 'en': 'en', 'ar': 'ar'}.get(lang, 'en')
            gTTS(text=message, lang=lang_code).save(filename)
            playsound(filename)
        except Exception as e:
            print(f"[SPEECH ERROR] {e}")
        finally:
            try:
                os.remove(filename)
            except Exception:
                pass
        _audio_queue.task_done()


def _start_worker():
    global _audio_thread
    if _audio_thread is None or not _audio_thread.is_alive():
        _audio_thread = threading.Thread(target=_worker, daemon=True)
        _audio_thread.start()

_start_worker()


def nettoyer_audio():
    files = glob.glob("temp_audio_*.mp3")
    for f in files:
        try:
            os.remove(f)
        except Exception:
            pass
    if files:
        print(f"[SPEECH] 🗑️ {len(files)} temp files removed")


def vider_file():
    """Drain the audio queue — call before entering reading mode."""
    try:
        while not _audio_queue.empty():
            _audio_queue.get_nowait()
            _audio_queue.task_done()
    except Exception:
        pass


def speak_if_new(message: str, lang: str = 'en'):
    """Queue a message only if it is new or enough time has passed."""
    global _last_message, _last_time, _last_valid_t

    if lecture_mode:
        return

    now = time.time()

    if message in _EMPTY_MESSAGES:
        # Repeat "clear" messages at most every 3 s
        if now - _last_valid_t < 3.0:
            return
    else:
        _last_valid_t = now

    if message == _last_message and (now - _last_time) < MIN_INTERVAL:
        return

    _last_message = message
    _last_time    = now

    _start_worker()
    _audio_queue.put((message, lang))


def speak_lecture(message: str, lang: str = 'en'):
    """Speak in reading mode — bypasses lecture_mode flag, blocks until done."""
    _start_worker()
    _audio_queue.put((message, lang))
    _audio_queue.join()