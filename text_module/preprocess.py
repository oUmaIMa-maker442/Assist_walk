import cv2

def preprocess_for_ocr(image):
    """
    Prépare l'image pour l'OCR
    - Convertit en niveaux de gris
    - Améliore le contraste (Otsu threshold)
    - Réduit le bruit
    - Agrandit si trop petite
    """

    # Convertir en gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Agrandir si image trop petite (OCR fonctionne mieux sur grandes images)
    height, width = gray.shape
    if height < 100 or width < 100:
        scale = 2
        gray = cv2.resize(
            gray,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC
        )

    # Threshold automatique (Otsu)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Réduire le bruit
    clean = cv2.medianBlur(thresh, 3)

    return clean


# TEST RAPIDE
if __name__ == "__main__":
    img = cv2.imread("tests/test_images/stop.jpg")
    result = preprocess_for_ocr(img)
    cv2.imwrite("tests/test_images/stop_preprocessed.jpg", result)
    print("Prétraitement OK")