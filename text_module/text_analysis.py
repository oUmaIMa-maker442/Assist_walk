# Dictionnaire de significations
# Tu peux l'agrandir au fur et à mesure

SIGN_MEANING = {

    # ---- Anglais ----
    "stop":             ("danger",        "panneau stop, arrêtez vous"),
    "no entry":         ("danger",        "sens interdit"),
    "no parking":       ("interdiction",  "stationnement interdit"),
    "no cycling":       ("interdiction",  "vélo interdit"),
    "exit":             ("information",   "sortie"),
    "entrance":         ("information",   "entrée"),
    "danger":           ("danger",        "zone de danger"),
    "warning":          ("danger",        "attention danger"),
    "school":           ("danger",        "zone scolaire, ralentissez"),
    "hospital":         ("information",   "hôpital à proximité"),
    "speed limit":      ("information",   "limitation de vitesse"),
    "one way":          ("information",   "sens unique"),
    "dead end":         ("information",   "impasse"),
    "pedestrian":       ("danger",        "passage piéton"),

    # ---- Français ----
    "arrêt":            ("danger",        "arrêtez vous"),
    "interdit":         ("interdiction",  "accès interdit"),
    "stationnement":    ("interdiction",  "stationnement interdit"),
    "sortie":           ("information",   "sortie"),
    "entrée":           ("information",   "entrée"),
    "sens unique":      ("information",   "sens unique"),
    "cédez":            ("danger",        "cédez le passage"),
    "école":            ("danger",        "zone scolaire, ralentissez"),
    "impasse":          ("information",   "impasse, sans issue"),
    "travaux":          ("danger",        "travaux en cours, attention"),
    "passage":          ("danger",        "passage piéton"),

    # ---- Arabe ----
    "قف":               ("danger",        "قف، توقف الآن"),
    "ممنوع":            ("interdiction",  "ممنوع المرور"),
    "ممنوع الدخول":     ("danger",        "ممنوع الدخول، اتجاه ممنوع"),
    "ممنوع الوقوف":     ("interdiction",  "ممنوع الوقوف هنا"),
    "خطر":              ("danger",        "خطر، كن حذراً"),
    "مدرسة":            ("danger",        "منطقة مدرسية، تمهل"),
    "مستشفى":           ("information",   "مستشفى قريب"),
    "مخرج":             ("information",   "مخرج"),
    "مدخل":             ("information",   "مدخل"),
    "انتبه":            ("danger",        "انتبه، خطر"),
    "أشغال":            ("danger",        "أشغال في الطريق، توخ الحذر"),
}

# Niveaux de priorité
PRIORITY = {
    "danger":       1,  # priorité haute
    "interdiction": 2,  # priorité moyenne
    "information":  3,  # priorité basse
}


def interpret_text(text):
    """
    Cherche si le texte correspond à un panneau connu
    Retourne (priorité, message) ou (3, texte original) si inconnu
    """

    text_lower = text.lower()

    for keyword, (category, message) in SIGN_MEANING.items():
        if keyword in text_lower:
            priority = PRIORITY[category]
            return priority, message

    # Texte inconnu mais valide → on le retourne tel quel
    return 3, f"texte détecté : {text}"