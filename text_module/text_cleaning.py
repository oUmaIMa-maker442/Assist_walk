import re

def clean_text(text):
    """
    Nettoie le texte brut sorti de l'OCR
    """

    # Mettre en minuscules
    text = text.lower()

    # Remplacer les sauts de ligne par des espaces
    text = re.sub(r'\n+', ' ', text)

    # Corriger des erreurs OCR connues
    corrections = {
        '0': 'o',   # zéro → lettre o (selon contexte)
        '1': 'l',   # un → L
        '@': 'a',
        '$': 's',
        '3': 'e',
    }
    # Note : applique les corrections seulement si le texte n'est pas numérique
    if not text.strip().isdigit():
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)

    # Enlever les caractères spéciaux sauf lettres, chiffres, espaces
    text = re.sub(r'[^a-z0-9 ]', '', text)

    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)

    # Enlever espaces au début/fin
    text = text.strip()

    return text


def validate_text(text):
    """
    Vérifie si le texte est utilisable
    Retourne True si valide, False sinon
    """

    # Trop court
    if len(text) < 2:
        return False

    # Que des chiffres → probablement pas un panneau de texte
    if text.isdigit():
        return False

    # Trop peu de lettres (bruit OCR)
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 2:
        return False

    return True