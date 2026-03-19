# ═══════════════════════════════════════════════════════════
# ÉTAPE 2 : Prétraitement de l'image
# Auteure : Oumaima
# Rôle    : Préparer l'image pour les modèles IA.
#           - Redimensionner à 640×640 (standard YOLO)
#           - Normaliser les pixels [0,255] → [0.0, 1.0]
#           - Garder l'original pour les crops CRAFT
# ═══════════════════════════════════════════════════════════

import cv2
import numpy as np

# Taille standard d'entrée pour YOLOv8
YOLO_INPUT_SIZE = 640


def preprocess(frame: np.ndarray) -> dict:
    """
    Prétraite une image pour le pipeline AssistWalk.

    Paramètres:
        frame  : numpy array RGB de taille quelconque

    Retourne un dict avec :
        processed   : image 640×640 normalisée (pour YOLO)
        original    : image RGB originale (pour crops CRAFT)
        orig_h, orig_w : dimensions originales (pour recalculer les bbox)
    """
    orig_h, orig_w = frame.shape[:2]

    # ── Étape 2a : Redimensionnement ──────────────────────
    # INTER_LINEAR = interpolation bilinéaire (meilleur équilibre
    # qualité/vitesse pour le redimensionnement d'images réelles)
    resized = cv2.resize(
        frame,
        (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # ── Étape 2b : Normalisation ──────────────────────────
    # Convertir uint8 [0,255] en float32 [0.0, 1.0]
    # Les réseaux de neurones apprennent mieux avec des valeurs normalisées
    normalized = resized.astype(np.float32) / 255.0

    print(f'[Étape 2] Image originale  : {orig_w}×{orig_h} px')
    print(f'[Étape 2] Après traitement : {YOLO_INPUT_SIZE}×{YOLO_INPUT_SIZE} px')

    return {
        'processed': normalized,    # → YOLO (étape 3)
        'original':  frame,         # → CRAFT crops (étape 6)
        'orig_h':    orig_h,
        'orig_w':    orig_w,
    }
