# ═══════════════════════════════════════════════════════════
# text_module/pipeline.py
# Phase 1: English-only pipeline — object names stay in
#          English throughout; voice output is English.
#          French / Arabic removed from navigation logic.
# ═══════════════════════════════════════════════════════════

from ocr_engine           import extract_text_easy, extract_text_lecture
from text_cleaning        import clean_text
from lang_detector        import detect_language
from distance_estimator   import distance_message, estimate_distance
from ai_message_generator import generate_smart_message as generate_ai_message
from speech               import speak_if_new, speak_lecture
from text_analysis        import interpret_text

# Only objects that are "close enough" get announced
IMPORTANT_DISTANCES = {'very_close', 'close'}

# "No obstacle" messages in supported languages
_CLEAR_MSGS = {
    'en': 'no obstacles detected',
    'fr': 'aucun obstacle détecté',
    'ar': 'لا يوجد عوائق',
}


# ══════════════════════════════════════════════════════════
# NAVIGATION MODE — real-time obstacle detection
# ══════════════════════════════════════════════════════════
def process_frame(data: dict, use_ai: bool = False,
                  lang: str = 'en', speak: bool = True):
    """
    Navigation pipeline — YOLO detections + CRAFT OCR crops.

    data = {
        "objects":      [{"name": str, "bbox": list}, ...],
        "text_regions": [numpy_image, ...],
        "frame_shape":  (h, w, c)
    }

    Returns: (final_message: str, final_lang: str)
    """
    objects      = data.get("objects", [])
    text_regions = data.get("text_regions", [])
    frame_shape  = data.get("frame_shape", (480, 640, 3))

    # ── OCR on CRAFT crops ───────────────────────────────
    texts = []
    for region in text_regions:
        raw = extract_text_easy(region, lang=lang)
        for item in raw:
            cleaned = clean_text(item["text"])
            if cleaned:
                texts.append(cleaned)

    # ── Language detection (from OCR text only) ──────────
    # Object names stay in English regardless.
    all_text   = " ".join(texts)
    final_lang = detect_language(all_text) if all_text.strip() else lang

    # ── Build distance messages for close objects ─────────
    objects_with_distances = []
    seen = set()

    for obj in objects:
        name_en = obj.get("name", "object")
        bbox    = obj.get("bbox", [0, 0, 50, 50])

        dist_key, _ = estimate_distance(bbox, frame_shape[1], frame_shape[0])
        if dist_key not in IMPORTANT_DISTANCES:
            continue
        if name_en in seen:
            continue
        seen.add(name_en)

        # Always generate the spoken message in English
        msg = distance_message(name_en, bbox, frame_shape, lang='en')
        objects_with_distances.append({
            "name":     name_en,
            "bbox":     bbox,
            "distance": msg,
        })

    # ── Build final message ───────────────────────────────
    if use_ai and (objects_with_distances or texts):
        ai_msg = generate_ai_message(objects_with_distances, texts, 'en')
        final_message = ai_msg if ai_msg else _classic_message(
            objects_with_distances, texts, final_lang)
    else:
        final_message = _classic_message(
            objects_with_distances, texts, final_lang)

    if speak:
        speak_if_new(final_message, lang='en')

    return final_message, final_lang


# ══════════════════════════════════════════════════════════
# READING MODE — read a full document
# ══════════════════════════════════════════════════════════
def process_lecture(image, speak: bool = True):
    """
    Reading pipeline — EasyOCR on the full image.
    speak=False → silent computation for background cache.
    speak=True  → reads result aloud.

    Returns: (final_message: str, detected_lang: str)
    """
    textes = extract_text_lecture(image)

    if not textes:
        msg = "No text detected. Please move closer and hold the document steady."
        if speak:
            speak_if_new(msg, lang='en')
        return msg, 'en'

    all_text   = " ".join(textes)
    final_lang = detect_language(all_text)
    message    = ". ".join(textes)

    if speak:
        speak_lecture(message, lang=final_lang)

    return message, final_lang


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def _classic_message(objects_with_distances: list,
                     texts: list,
                     lang: str = 'en') -> str:
    """Rule-based message — no AI."""
    parts = [obj["distance"] for obj in objects_with_distances]

    for text in texts:
        interpreted = interpret_text(text, lang)
        if interpreted:
            parts.append(interpreted)

    if not parts:
        return _CLEAR_MSGS.get(lang, _CLEAR_MSGS['en'])

    return ". ".join(parts[:3])