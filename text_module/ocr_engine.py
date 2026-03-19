# text_module/ocr_engine.py

import cv2
import numpy as np
import easyocr
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── EasyOCR — chargé une seule fois au démarrage ───────────
reader_latin  = None
reader_arabic = None

def _get_reader(lang):
    global reader_latin, reader_arabic
    if lang == 'ar':
        if reader_arabic is None:
            reader_arabic = easyocr.Reader(['ar'], gpu=False)
        return reader_arabic
    else:
        if reader_latin is None:
            reader_latin = easyocr.Reader(['fr', 'en'], gpu=False)
        return reader_latin


def deskew(image):
    """
    Détecte et corrige l'inclinaison du texte
    Utile quand le document est tenu de côté
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ne corriger que si angle significatif (>5°)
    if abs(angle) < 5:
        return image

    print(f"[OCR] Rotation détectée : {angle:.1f}° → correction")
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_image(image):
    """Prétraitement optimal pour EasyOCR"""

    # ✅ Rogner les bandes noires sur les côtés
    gray_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray_check.mean(axis=0) > 15)[0]
    rows = np.where(gray_check.mean(axis=1) > 15)[0]
    if len(cols) > 0 and len(rows) > 0:
        image = image[rows[0]:rows[-1], cols[0]:cols[-1]]

    h, w = image.shape[:2]

    # ✅ Redimensionner à taille optimale
    target_w = 1200
    if w > target_w:
        ratio = target_w / w
        image = cv2.resize(image, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
    elif w < 800:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # ✅ Sharpen — améliore les lettres floues
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image  = cv2.filter2D(image, -1, kernel)

    # ✅ Contraste CLAHE
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Retourner en BGR pour EasyOCR
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _filtrer(lignes, seuil_alpha=0.4):
    """Rejette les lignes avec trop peu de vraies lettres"""
    result = []
    for l in lignes:
        l = l.strip()
        if len(l) < 2:
            continue
        ratio = sum(c.isalpha() for c in l) / len(l)
        if ratio >= seuil_alpha:
            result.append(l)
    return result


# ══════════════════════════════════════════════════════════
# MODE NAVIGATION — EasyOCR sur petits crops CRAFT
# ══════════════════════════════════════════════════════════
def extract_text_easy(image, lang='auto'):
    """
    Navigation — EasyOCR sur petits crops CRAFT
    Seuil confiance strict : 0.7
    """
    if image is None or image.size == 0:
        return []

    results = []

    for (_, text, conf) in _get_reader('fr').readtext(image, detail=1):
        text = text.strip()
        if conf >= 0.7 and len(text) >= 2:
            results.append({"text": text, "confidence": round(conf, 2)})

    for (_, text, conf) in _get_reader('ar').readtext(image, detail=1):
        text = text.strip()
        if conf >= 0.7 and len(text) >= 2:
            results.append({"text": text, "confidence": round(conf, 2)})

    return results


# ══════════════════════════════════════════════════════════
# MODE LECTURE — EasyOCR sur image complète prétraitée
# ══════════════════════════════════════════════════════════
def extract_text_lecture(image):
    """
    Lecture — EasyOCR sur image brute + version prétraitée (texte coloré)
    """
    if image is None or image.size == 0:
        return []

    # ── Rogner bandes noires ──────────────────────────────
    gray_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray_check.mean(axis=0) > 15)[0]
    rows = np.where(gray_check.mean(axis=1) > 15)[0]
    if len(cols) > 10 and len(rows) > 10:
        image = image[rows[0]:rows[-1], cols[0]:cols[-1]]

    textes = []

    # ── EasyOCR image brute ───────────────────────────────
    try:
        res = _get_reader('fr').readtext(image, detail=1)
        t   = [tx.strip() for (_, tx, cf) in res if cf >= 0.25 and len(tx.strip()) >= 2]
        print(f"[OCR EASY FR/EN] {t}")
        textes.extend(t)
    except Exception as e:
        print(f"[OCR ERROR] {e}")

    # ── Fallback Tesseract ────────────────────────────────
    if not textes:
        print("[OCR] EasyOCR vide → fallback Tesseract...")
        try:
            config = '--oem 3 --psm 6'
            text   = pytesseract.image_to_string(image, lang='fra+eng', config=config)
            lignes = [l.strip() for l in text.split('\n')
                      if len(l.strip()) > 2
                      and sum(c.isalpha() for c in l) > len(l) * 0.5]
            print(f"[OCR TESS] {lignes[:4]}")
            textes.extend(lignes)
        except Exception as e:
            print(f"[OCR TESS ERROR] {e}")

    return list(dict.fromkeys(textes))