# ═══════════════════════════════════════════════════════════
# ÉTAPE 3 : Détection d'objets avec YOLO
# Auteure : Oumaima
# Rôle    : Utiliser YOLOv8n pour détecter tous les objets
#           présents dans l'image prétraitée
# ═══════════════════════════════════════════════════════════

from ultralytics import YOLO
import numpy as np
import os

# Dossier où stocker les modèles téléchargés
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)


class YOLODetector:
    def __init__(self, model_name='yolov8s.pt', confidence=0.3):
        """
        Charge le modèle YOLOv8.
        Le fichier .pt se télécharge automatiquement si absent.

        Paramètres:
            model_name : nom du modèle Ultralytics
            confidence : seuil de confiance minimum (0.0 à 1.0)
        """
        model_path = os.path.join(MODELS_DIR, model_name)
        print(f'[Étape 3] Chargement YOLO : {model_name}...')
        # Si le fichier n'existe pas, Ultralytics le télécharge
        self.model = YOLO(model_path)
        self.confidence = confidence
        print('[Étape 3] YOLO prêt ✓')

    def detect(self, frame: np.ndarray) -> list:
        """
        Détecte tous les objets dans l'image.

        Paramètres:
            frame : image numpy RGB (taille quelconque)

        Retourne:
            Liste de dicts : [{'class', 'bbox', 'confidence'}, ...]
        """
        # verbose=False : supprimer les logs de détection
        results = self.model(frame, verbose=False)[0]

        detected = []
        for box in results.boxes:
            class_name  = self.model.names[int(box.cls)]
            confidence  = float(box.conf)
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            if confidence >= self.confidence:
                detected.append({
                    'class':      class_name,
                    'bbox':       (x1, y1, x2, y2),
                    'confidence': round(confidence, 2)
                })

        print(f'[Étape 3] {len(detected)} objets détectés')
        return detected
