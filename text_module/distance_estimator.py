# ═══════════════════════════════════════════════════════════
# Distance estimator
# Phase 1: English-first output — French/Arabic kept for
#          future multilingual support but not used by default
# ═══════════════════════════════════════════════════════════

# Distance label → human-readable string per language
_DISTANCE_LABELS = {
    'en': {
        'very_close': 'very close',
        'close':      'nearby',
        'medium':     'at medium distance',
        'far':        'far away',
    },
    'fr': {
        'very_close': 'très proche',
        'close':      'proche',
        'medium':     'à distance moyenne',
        'far':        'loin',
    },
    'ar': {
        'very_close': 'قريب جداً',
        'close':      'قريب',
        'medium':     'على بعد متوسط',
        'far':        'بعيد',
    },
}

# Alert level → message template per language
_TEMPLATES = {
    'en': {
        'danger':    'danger! {name} {dist} ahead',
        'attention': 'warning, {name} {dist}',
        'info':      '{name} detected, {dist}',
    },
    'fr': {
        'danger':    'danger ! {name} {dist} devant vous',
        'attention': 'attention, {name} {dist}',
        'info':      '{name} détecté, {dist}',
    },
    'ar': {
        'danger':    'خطر! {name} {dist} أمامك',
        'attention': 'انتبه، {name} {dist}',
        'info':      'تم اكتشاف {name}، {dist}',
    },
}


def estimate_distance(bbox, frame_width, frame_height):
    """
    Estimate distance from bounding-box size ratio.

    Returns: (distance_key, alert_level)
        distance_key : 'very_close' | 'close' | 'medium' | 'far'
        alert_level  : 'danger'     | 'attention' | 'info'
    """
    x1, y1, x2, y2 = bbox
    ratio = max((x2 - x1) / frame_width,
                (y2 - y1) / frame_height)

    if ratio > 0.5:
        return 'very_close', 'danger'
    elif ratio > 0.25:
        return 'close', 'attention'
    elif ratio > 0.10:
        return 'medium', 'info'
    else:
        return 'far', 'info'


def distance_message(obj_name: str, bbox, frame_shape, lang: str = 'en') -> str:
    """
    Build a spoken distance message.

    Args:
        obj_name   : English class name (e.g. 'car', 'person')
        bbox       : (x1, y1, x2, y2)
        frame_shape: (h, w, c)  — numpy shape
        lang       : 'en' | 'fr' | 'ar'  (default: 'en')

    Returns:
        Natural-language string, e.g. "danger! car very close ahead"
    """
    h, w = frame_shape[:2]
    dist_key, level = estimate_distance(bbox, w, h)

    labels    = _DISTANCE_LABELS.get(lang, _DISTANCE_LABELS['en'])
    templates = _TEMPLATES.get(lang, _TEMPLATES['en'])

    dist_str = labels[dist_key]
    return templates[level].format(name=obj_name, dist=dist_str)