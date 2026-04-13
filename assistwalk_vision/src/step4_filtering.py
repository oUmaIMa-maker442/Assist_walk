# ═══════════════════════════════════════════════════════════
# STEP 4 : Object filtering
# Role   : Keep only objects relevant to blind/low-vision users
# Phase 1: English class names throughout — no translation here
# ═══════════════════════════════════════════════════════════

# ── HIGH priority : immediate danger ──────────────────────
HIGH_PRIORITY = {
    'person', 'car', 'truck', 'bus', 'motorcycle',
    'bicycle', 'traffic light', 'stop sign',
    # New classes (ready for future retraining)
    'door', 'stair', 'stairs', 'metal_barrier',
    'pole',              # lamppost, utility pole
    'hole',              # open hole, missing manhole cover
    'construction',      # construction zone, scaffolding
    'open_manhole',      # explicit open manhole
}

# ── MEDIUM priority : nearby obstacles ────────────────────
MEDIUM_PRIORITY = {
    'chair', 'bench', 'fire hydrant', 'parking meter',
    'trash can', 'suitcase', 'backpack',
    'book', 'keyboard', 'laptop',
    'couch', 'bed', 'dining table',
    'tv', 'monitor', 'sink',
    # New classes (ready for future retraining)
    'ramp', 'grate', 'curb', 'tree',
    'mailbox',           # post box
    'fence',             # fence (different from metal_barrier)
    'wall',              # wall
    'column',            # pillar
    'planter',           # flower pot, planter box
    'stroller',          # baby stroller
    'shopping_cart',     # shopping cart
    'sign',              # generic sign (not stop/traffic light)
}

# ── LOW priority : animals and minor elements ──────────────
LOW_PRIORITY = {
    'dog', 'cat', 'bird', 'horse',
    'potted plant', 'handbag',
}

ALL_RELEVANT = HIGH_PRIORITY | MEDIUM_PRIORITY | LOW_PRIORITY


def filter_objects(objects: list) -> list:
    """
    Filter and sort YOLO detections by priority.

    Input:  [{'class': str, 'confidence': float, 'bbox': tuple}, ...]
    Output: filtered list sorted by (priority, -confidence)
    NOTE:   class names are kept in English — no translation.
    """

    def get_priority(obj):
        c = obj['class']
        if c in HIGH_PRIORITY:   return 0
        elif c in MEDIUM_PRIORITY: return 1
        elif c in LOW_PRIORITY:  return 2
        else:                    return 99

    filtered = [o for o in objects if o['class'] in ALL_RELEVANT]
    rejected = [o for o in objects if o['class'] not in ALL_RELEVANT]

    for obj in rejected:
        print(f"          ✗ REJECTED: {obj['class']} (conf: {obj['confidence']})")

    filtered.sort(key=lambda o: (get_priority(o), -o['confidence']))

    print(f'[Step 4] {len(objects)} detected → {len(filtered)} relevant')
    for obj in filtered:
        print(f"          ✓ {obj['class']} (conf: {obj['confidence']})")

    return filtered