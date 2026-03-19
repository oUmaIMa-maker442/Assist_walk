# ═══════════════════════════════════════════════════════════
# ÉTAPE 4 : Filtrage des objets pertinents
# Auteure : Oumaima
# Rôle    : Ne garder que les objets qui représentent un
#           danger ou une information utile pour un malvoyant
# ═══════════════════════════════════════════════════════════

# ── Classes pertinentes pour l'assistance ─────────────────
# Organisées par niveau de priorité (HAUTE → BASSE)

# Priorité HAUTE : obstacles dangereux immédiats
HIGH_PRIORITY = {
    'person', 'car', 'truck', 'bus', 'motorcycle',
    'bicycle', 'traffic light', 'stop sign'
}

# Priorité MOYENNE : obstacles dans l'environnement
MEDIUM_PRIORITY = {
    'chair', 'bench', 'fire hydrant', 'parking meter',
    'trash can', 'suitcase', 'backpack',
    'book', 'keyboard', 'laptop',      # ← ajouter
    'couch', 'bed', 'dining table',    # ← ajouter
    'tv', 'monitor', 'sink',           # ← ajouter
}

# Priorité BASSE : animaux et autres éléments
LOW_PRIORITY = {
    'dog', 'cat', 'bird', 'horse',
    'potted plant', 'handbag'
}

# Ensemble de toutes les classes utiles
ALL_RELEVANT = HIGH_PRIORITY | MEDIUM_PRIORITY | LOW_PRIORITY


def filter_objects(objects: list) -> list:
    """
    Filtre et trie les objets détectés par YOLO.

    Paramètres:
        objects : liste des objets détectés
                  format attendu :
                  {
                      'class': str,
                      'confidence': float,
                      'bbox': [...]
                  }

    Retourne:
        Liste d'objets filtrés et triés par priorité
    """

    # Fonction interne pour déterminer la priorité
    def get_priority(obj):
        c = obj['class']
        if c in HIGH_PRIORITY:
            return 0
        elif c in MEDIUM_PRIORITY:
            return 1
        elif c in LOW_PRIORITY:
            return 2
        else:
            return 99

    # ── Filtrer les objets pertinents ──────────────────────
    filtered = [o for o in objects if o['class'] in ALL_RELEVANT]

    # ── Objets rejetés (pour debug) ────────────────────────
    rejected = [o for o in objects if o['class'] not in ALL_RELEVANT]

    for obj in rejected:
        print(f"          ✗ REJETÉ: {obj['class']} (confiance: {obj['confidence']})")

    # ── Trier par priorité puis par confiance ──────────────
    filtered.sort(key=lambda o: (get_priority(o), -o['confidence']))

    # ── Affichage debug ────────────────────────────────────
    print(f'[Étape 4] {len(objects)} détectés → {len(filtered)} pertinents')

    for obj in filtered:
        print(f"          ✓ {obj['class']} (confiance: {obj['confidence']})")

    return filtered