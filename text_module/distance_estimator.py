DISTANCE_AR = {
    "très proche":        "قريب جداً",
    "proche":             "قريب",
    "à distance moyenne": "على بعد متوسط",
    "loin":               "بعيد",
}

DISTANCE_EN = {
    "très proche":        "very close",
    "proche":             "nearby",
    "à distance moyenne": "at medium distance",
    "loin":               "far away",
}

def estimate_distance(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    obj_width  = x2 - x1
    obj_height = y2 - y1

    ratio_w = obj_width  / frame_width
    ratio_h = obj_height / frame_height
    ratio   = max(ratio_w, ratio_h)

    if ratio > 0.5:
        return "très proche", "danger"
    elif ratio > 0.25:
        return "proche", "attention"
    elif ratio > 0.10:
        return "à distance moyenne", "info"
    else:
        return "loin", "info"


def distance_message(obj_name, bbox, frame_shape, lang='fr'):
    h, w = frame_shape[:2]
    distance_text, level = estimate_distance(bbox, w, h)

    distance_ar = DISTANCE_AR.get(distance_text, distance_text)
    distance_en = DISTANCE_EN.get(distance_text, distance_text)

    TEMPLATES = {
        'fr': {
            "danger":    f"danger ! {obj_name} {distance_text} devant vous",
            "attention": f"attention, {obj_name} {distance_text}",
            "info":      f"{obj_name} détecté, {distance_text}",
        },
        'en': {
            "danger":    f"danger! {obj_name} {distance_en} ahead",
            "attention": f"warning, {obj_name} {distance_en}",
            "info":      f"{obj_name} detected, {distance_en}",
        },
        'ar': {
            "danger":    f"خطر! {obj_name} {distance_ar} أمامك",
            "attention": f"انتبه، {obj_name} {distance_ar}",
            "info":      f"تم اكتشاف {obj_name}، {distance_ar}",
        }
    }

    template = TEMPLATES.get(lang, TEMPLATES['fr'])
    return template[level]