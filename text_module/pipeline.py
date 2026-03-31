# text_module/pipeline.py

from ocr_engine         import extract_text_easy, extract_text_lecture
from text_cleaning      import clean_text
from lang_detector      import detect_language
from distance_estimator import distance_message, estimate_distance
from ai_message_generator import generate_smart_message as generate_ai_message
from speech             import speak_if_new
from text_analysis      import interpret_text

DISTANCES_IMPORTANTES = ["très proche", "proche"]

TRADUCTIONS = {
    'fr': {
        "person": "personne", "car": "voiture", "truck": "camion",
        "bus": "bus", "motorcycle": "moto", "bicycle": "vélo",
        "traffic light": "feu de circulation", "stop sign": "panneau stop",
        "chair": "chaise", "bench": "banc", "laptop": "ordinateur portable",
        "backpack": "sac à dos", "dog": "chien", "cat": "chat",
        "fire hydrant": "bouche d'incendie", "suitcase": "valise",
        "couch": "canapé", "dining table": "table", "tv": "télévision",
        "bed": "lit", "book": "livre", "keyboard": "clavier",
    },
    'en': {
        "person": "person", "car": "car", "truck": "truck",
        "bus": "bus", "motorcycle": "motorcycle", "bicycle": "bicycle",
        "traffic light": "traffic light", "stop sign": "stop sign",
        "chair": "chair", "bench": "bench", "laptop": "laptop",
        "backpack": "backpack", "dog": "dog", "cat": "cat",
    },
    'ar': {
        "person": "شخص", "car": "سيارة", "truck": "شاحنة",
        "bus": "حافلة", "motorcycle": "دراجة نارية", "bicycle": "دراجة",
        "traffic light": "إشارة مرور", "stop sign": "إشارة قف",
        "chair": "كرسي", "bench": "مقعد", "laptop": "حاسوب محمول",
        "backpack": "حقيبة ظهر", "dog": "كلب", "cat": "قطة",
    }
}


def translate_object(name, lang='fr'):
    return TRADUCTIONS.get(lang, TRADUCTIONS['fr']).get(name, name)


# ══════════════════════════════════════════════════════════
# MODE NAVIGATION — obstacles en temps réel
# ══════════════════════════════════════════════════════════
def process_frame(data, use_ai=False, lang='fr', speak=True):
    """
    Pipeline navigation — YOLO + CRAFT crops
    data = {
        "objects":      [{"name": ..., "bbox": [...]}],
        "text_regions": [image_numpy, ...],
        "frame_shape":  (h, w, c)
    }
    Retourne : (message_final, langue_finale)
    """
    objects      = data.get("objects", [])
    text_regions = data.get("text_regions", [])
    frame_shape  = data.get("frame_shape", (480, 640, 3))

    # ── OCR sur crops CRAFT ──────────────────────────────
    texts = []
    for region in text_regions:
        raw = extract_text_easy(region, lang=lang)
        for item in raw:
            cleaned = clean_text(item["text"])
            if cleaned:
                texts.append(cleaned)

    # ── Langue ───────────────────────────────────────────
    all_text   = " ".join(texts)
    final_lang = detect_language(all_text) if all_text.strip() else lang

    # ── Distances objets ─────────────────────────────────
    objects_with_distances = []
    seen = set()

    for obj in objects:
        name_en   = obj.get("name", "objet")
        bbox      = obj.get("bbox", [0, 0, 50, 50])
        name_trad = translate_object(name_en, final_lang)

        dist_text, _ = estimate_distance(bbox, frame_shape[1], frame_shape[0])
        if dist_text not in DISTANCES_IMPORTANTES:
            continue
        if name_en in seen:
            continue
        seen.add(name_en)

        dist_msg = distance_message(name_trad, bbox, frame_shape, final_lang)
        objects_with_distances.append({
            "name":     name_trad,
            "bbox":     bbox,
            "distance": dist_msg
        })

    # ── Message ───────────────────────────────────────────
    if use_ai and (objects_with_distances or texts):
        ai_msg = generate_ai_message(objects_with_distances, texts, final_lang)
        final_message = ai_msg if ai_msg else _classic_message(objects_with_distances, texts, final_lang)
    else:
        final_message = _classic_message(objects_with_distances, texts, final_lang)

    speak_if_new(final_message, lang=final_lang)   # pipeline parle toujours
    return final_message, final_lang


# ══════════════════════════════════════════════════════════
# MODE LECTURE — lire un document complet
# ══════════════════════════════════════════════════════════
def process_lecture(image, speak=True):
    """
    Pipeline lecture — EasyOCR sur image complète
    speak=False → calcul silencieux pour cache background
    speak=True  → lit le résultat à voix haute
    Retourne : (message_final, langue_detectee)
    """
    textes = extract_text_lecture(image)

    if not textes:
        msg = "Aucun texte détecté. Approchez et stabilisez le document."
        if speak:
            speak_if_new(msg, lang='fr')
        return msg, 'fr'

    all_text   = " ".join(textes)
    final_lang = detect_language(all_text)
    message    = ". ".join(textes)  # ✅ tout le texte, pas seulement 6 lignes

    if speak:
        speak_if_new(message, lang=final_lang, is_lecture=True)

    return message, final_lang


def _classic_message(objects_with_distances, texts, lang='fr'):
    """Message classique sans IA"""
    parts = [obj["distance"] for obj in objects_with_distances]

    for text in texts:
        interpreted = interpret_text(text, lang)
        if interpreted:
            parts.append(interpreted)

    if not parts:
        msgs = {'fr': "aucun obstacle détecté", 'en': "no obstacles detected", 'ar': "لا يوجد عوائق"}
        return msgs.get(lang, msgs['fr'])

    return ". ".join(parts[:3])