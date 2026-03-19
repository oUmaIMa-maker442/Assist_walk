from text_analysis import interpret_text

def decision_logic(objects, texts, lang='fr'):
    messages = []

    # Messages selon la langue choisie
    OBJECT_MESSAGES = {
        'fr': {
            "person":        (1, "attention, personne devant vous"),
            "car":           (1, "attention, voiture devant vous"),
            "bicycle":       (2, "vélo détecté"),
            "bus":           (1, "bus détecté, faites attention"),
            "truck":         (1, "camion détecté, danger"),
            "traffic light": (2, "feu de circulation détecté"),
        },
        'ar': {
            "person":        (1, "انتبه، يوجد شخص أمامك"),
            "car":           (1, "انتبه، توجد سيارة أمامك"),
            "bicycle":       (2, "تم اكتشاف دراجة"),
            "bus":           (1, "انتبه، يوجد حافلة أمامك"),
            "truck":         (1, "خطر، يوجد شاحنة أمامك"),
            "traffic light": (2, "إشارة مرور أمامك"),
        },
        'en': {
            "person":        (1, "warning, person ahead"),
            "car":           (1, "warning, car in front of you"),
            "bicycle":       (2, "bicycle detected"),
            "bus":           (1, "warning, bus ahead"),
            "truck":         (1, "danger, truck ahead"),
            "traffic light": (2, "traffic light detected"),
        }
    }

    lang_msgs = OBJECT_MESSAGES.get(lang, OBJECT_MESSAGES['fr'])

    for obj in objects:
        if obj in lang_msgs:
            messages.append(lang_msgs[obj])
        else:
            messages.append((3, obj))

    for text in texts:
        priority, message = interpret_text(text)
        messages.append((priority, message))

    messages.sort(key=lambda x: x[0])
    return messages


def generate_message(messages):
    if not messages:
        return "aucun obstacle détecté"
    top = messages[:3]
    return ". ".join(msg for _, msg in top)