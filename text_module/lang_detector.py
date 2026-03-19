from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

# ---- Mots-clés connus par langue ----
# Si langdetect échoue, on vérifie les mots-clés
KEYWORDS = {
    'en': [
        "stop", "no entry", "no parking", "no cycling", "exit",
        "entrance", "danger", "warning", "caution", "private",
        "road", "street", "school", "hospital", "speed", "limit",
        "keep", "clear", "slow", "yield", "do not", "one way",
        "dead end", "no turn", "pedestrian", "crossing"
    ],
    'fr': [
        "arrêt", "interdit", "stationnement", "sens", "sortie",
        "entrée", "danger", "attention", "prudence", "privé",
        "rue", "route", "école", "hôpital", "vitesse", "limite",
        "cédez", "passage", "piéton", "voie", "impasse"
    ],
    'ar': [
        "ممنوع", "توقف", "خطر", "مدخل", "مخرج", "انتبه",
        "مدرسة", "مستشفى", "طريق", "شارع", "سرعة", "قف"
    ]
}


def detect_by_keywords(text):
    """
    Détecte la langue par mots-clés connus
    Retourne la langue si trouvée, None sinon
    """
    text_lower = text.lower()

    scores = {'en': 0, 'fr': 0, 'ar': 0}

    for lang, keywords in KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[lang] += 1

    best_lang = max(scores, key=scores.get)

    # Retourner seulement si au moins un mot-clé trouvé
    if scores[best_lang] > 0:
        return best_lang

    return None


def detect_language(text):
    """
    Détecte automatiquement la langue du texte.

    Stratégie :
    1. Vérifier mots-clés connus (plus fiable pour textes courts)
    2. Si rien trouvé → utiliser langdetect
    3. Si langdetect échoue → français par défaut
    """

    if not text or len(text.strip()) < 2:
        return 'fr'

    # Étape 1 : mots-clés (prioritaire pour panneaux courts)
    lang_kw = detect_by_keywords(text)
    if lang_kw:
        print(f"   [LANG via keywords] '{text}' → {lang_kw}")
        return lang_kw

    # Étape 2 : langdetect (pour textes plus longs)
    try:
        lang = detect(text)
        mapping = {'fr': 'fr', 'en': 'en', 'ar': 'ar'}
        result = mapping.get(lang, 'fr')
        print(f"   [LANG via langdetect] '{text}' → {result}")
        return result

    except LangDetectException:
        return 'fr'


# ---- TEST ----
if __name__ == "__main__":
    tests = [
        ("stop interdit de stationner", 'fr'),
        ("no entry private property",   'en'),
        ("no parking",                  'en'),
        ("ممنوع الدخول",                'ar'),
        ("STOP",                        'en'),   # mot-clé anglais
        ("sortie",                      'fr'),
        ("exit",                        'en'),
        ("caution school ahead",        'en'),
        ("attention école",             'fr'),
    ]

    print("=" * 55)
    print(f"{'Texte':<30} {'Détecté':<10} {'Attendu':<10} {'Statut'}")
    print("=" * 55)

    correct = 0
    for text, expected in tests:
        result = detect_language(text)
        status = "✅" if result == expected else "⚠️"
        if result == expected:
            correct += 1
        print(f"{status} '{text:<28}' → {result:<8} | attendu: {expected}")

    print("=" * 55)
    print(f"Score : {correct}/{len(tests)}")