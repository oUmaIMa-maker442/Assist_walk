# stream_processor.py

import cv2
import sys
import os
import time
import threading
sys.path.insert(0, os.path.abspath('text_module'))
sys.path.insert(0, os.path.abspath('assistwalk_vision'))

from src.step4_filtering import filter_objects
from src.step6_extraction import extract_text_regions
from vision_module import VisionModule
from pipeline import process_frame, process_lecture
from speech import speak_if_new, speak_lecture
import speech as _speech_module
from ocr_engine import _get_reader


class StreamProcessor:

    def __init__(self, source, rotate=True):
        self.source  = source
        self.rotate  = rotate

        self._cap          = None
        self._latest_frame = None
        self._frame_lock   = threading.Lock()

        self._lock          = threading.Lock()
        self._last_message  = "Initialisation..."
        self._last_lang     = 'fr'
        self._last_objects  = []
        self._last_texts    = []
        self._frame_count   = 0

        self._running        = False
        self._craft_running  = False
        self._ocr_bg_running = False

        # ✅ Un seul flag clair — True = lecture en cours
        self._lecture_active = False

        self._lecture_cache       = None
        self._lecture_cache_frame = None

        self.YOLO_EVERY   = 10
        self.CRAFT_EVERY  = 50
        self.OCR_BG_EVERY = 60

        print("Chargement des modèles...")
        self.vision = VisionModule()
        print("Chargement EasyOCR...")
        _get_reader('fr')
        _get_reader('ar')
        print("✅ Système prêt")

    def _apply_rotation(self, frame):
        if self.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    # ══════════════════════════════════════════════════════
    def start(self):
        self._running = True
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()
        print(f"[STREAM] Démarré — {self.source}")

    def stop(self):
        self._running = False
        time.sleep(0.3)
        if self._cap:
            self._cap.release()
        print("[STREAM] Arrêté")

    def get_latest_frame(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_result(self):
        with self._lock:
            return {
                "message": self._last_message,
                "lang":    self._last_lang,
                "objects": self._last_objects,
                "texts":   self._last_texts,
            }

    # ══════════════════════════════════════════════════════
    # MODE LECTURE
    # ══════════════════════════════════════════════════════
    def read_document(self, speak=True):

        # ✅ Bloquer speak_if_new navigation IMMÉDIATEMENT
        self._lecture_active = True
        _speech_module.lecture_mode = True
        

        # ✅ Vider file audio
        from speech import vider_file
        vider_file()

        print("\n📖 MODE LECTURE — navigation en pause")

        message, lang = "Aucune frame disponible", 'fr'

        try:
            with self._lock:
                cache = self._lecture_cache

            if cache is not None:
                message, lang = cache
                print(f"[LECTURE] ⚡ Instantané : '{message[:60]}' [{lang}]")
                if speak:
                    speak_lecture(message, lang=lang)   # ← bloque jusqu'au dernier mot

            else:
                print("[LECTURE] Analyse en cours...")
                if speak:
                    speak_lecture("Analyse en cours, patientez", lang='fr')

                frame = None
                for _ in range(20):
                    frame = self.get_latest_frame()
                    if frame is not None:
                        break
                    time.sleep(0.1)

                if frame is not None:
                    frame = self._apply_rotation(frame)
                    h, w  = frame.shape[:2]
                    if w > 960:
                        frame = cv2.resize(frame, (960, int(h * 960 / w)))
                    cv2.imwrite("debug_lecture.jpg", frame)
                    print(f"[DEBUG] Frame OCR : {frame.shape}")

                    message, lang = process_lecture(frame, speak=False)
                    print(f"[LECTURE] ✅ [{lang}] {len(message)} chars lus")

                    if speak:
                        speak_lecture(message, lang=lang)   # ← bloque jusqu'au dernier mot

        except Exception as e:
            print(f"[LECTURE ERROR] {e}")

        finally:
            # ✅ Navigation reprend seulement ici — après dernier mot
            _speech_module.lecture_mode = False
            self._lecture_active = False
            print("📷 Retour navigation...")

        return message, lang

    # ══════════════════════════════════════════════════════
    # BOUCLE CAPTURE
    # ══════════════════════════════════════════════════════
    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                print("⚠️  Reconnexion webcam...")
                self._cap.release()
                time.sleep(1)
                self._cap = cv2.VideoCapture(self.source)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ══════════════════════════════════════════════════════
    # BOUCLE TRAITEMENT
    # ══════════════════════════════════════════════════════
    def _process_loop(self):
        for _ in range(50):
            with self._frame_lock:
                if self._latest_frame is not None:
                    break
            time.sleep(0.1)

        while self._running:

            # ✅ Skip complet si lecture active — pas de YOLO, pas de CRAFT
            if self._lecture_active:
                time.sleep(0.05)
                continue

            frame = self.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = self._apply_rotation(frame)
            self._frame_count += 1

            # ── YOLO ──────────────────────────────────────
            if self._frame_count % self.YOLO_EVERY == 0:
                if self._lecture_active:
                    continue
                try:
                    results_raw = self.vision.yolo.detect(frame)
                    
                    # ✅ Vérifier ENCORE après detect (peut prendre 150ms)
                    if self._lecture_active:
                        continue
                        
                    filtered        = filter_objects(results_raw)
                    objects_adapted = self._adapt_objects(filtered)

                    with self._lock:
                        texts = self._last_texts.copy()

                    data = {
                        "objects":      objects_adapted,
                        "text_regions": texts,
                        "frame_shape":  frame.shape
                    }
                    
                    # ✅ Vérifier ENCORE avant process_frame qui parle
                    if self._lecture_active:
                        continue
                        
                    message, lang = process_frame(data, use_ai=False)
                    print(f"[NAV] '{message}' [{lang}]")

                    with self._lock:
                        self._last_message = message
                        self._last_lang    = lang
                        self._last_objects = objects_adapted

                except Exception as e:
                    print(f"[NAV ERROR] {e}")

            # ── CRAFT background ──────────────────────────
            if self._frame_count % self.CRAFT_EVERY == 0 and not self._craft_running:
                if not self._lecture_active:
                    self._craft_running = True
                    threading.Thread(
                        target=self._craft_bg,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()

            # ── OCR background ────────────────────────────
            if self._frame_count % self.OCR_BG_EVERY == 0 and not self._ocr_bg_running:
                if not self._lecture_active:
                    self._ocr_bg_running = True
                    threading.Thread(
                        target=self._ocr_bg,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()

            time.sleep(0.01)

    def _craft_bg(self, frame):
        try:
            text_boxes   = self.vision.craft.detect_text_zones(frame)
            text_regions = extract_text_regions(frame, text_boxes)
            with self._lock:
                self._last_texts = text_regions
        except:
            pass
        finally:
            self._craft_running = False

    def _ocr_bg(self, frame):
        try:
            result = process_lecture(frame, speak=False)
            with self._lock:
                self._lecture_cache       = result
                self._lecture_cache_frame = frame.copy()
            print(f"[OCR BG] ✅ Cache : '{result[0][:50]}...'")
        except Exception as e:
            print(f"[OCR BG ERROR] {e}")
        finally:
            self._ocr_bg_running = False

    @staticmethod
    def _adapt_objects(filtered):
        return [
            {"name": obj["class"], "bbox": [int(b) for b in obj["bbox"]]}
            for obj in filtered
        ]