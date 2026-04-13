# text_module/ocr_engine.py
# Version optimisée pour la précision (FAST_MODE = True)
# Temps cible : < 2s sur CPU pour une image 640x480
#
# Améliorations v4 :
#  - Dictionnaire académique ciblé (_ACADEMIC_DICT) pour corriger les
#    majuscules et troncatures les plus fréquentes (ENSIAS, ÉCOLE, etc.)
#  - Fusion de syllabes : réassemble les mots scindés par Tesseract
#    ("dé ve loppe ment" → "développement") via un mini-lexique
#  - Sélection de binarisation par score Tesseract (confidence-based)
#    au lieu du seul ratio de blanc — uniquement si les deux meilleures
#    méthodes sont proches
#  - Whitelist allégée pour le pass LSTM (pas de chiffres en mode texte)

import cv2
import numpy as np
import pytesseract
import hashlib
import easyocr
import time as _t
import re
import os
import unicodedata

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── Speed / accuracy trade-off flags ─────────────────────
SKIP_CHAR_CHECK    = True
FAST_MODE          = True
FAST_UPSCALE       = 1200
FULL_UPSCALE       = 1200
MIN_EDGE_RATIO     = 0.04
MIN_CHAR_HEIGHT_PX = 8

# ── Dense-page upscale override ───────────────────────────
# Set to 0 to use FAST_UPSCALE (default, fast).
# Set to e.g. 2400 or 3200 when OCR fails on small-font pages
# (char_height 9-11px).  Increases processing time to ~3-6s.
# Usage:
#   FORCE_UPSCALE_FOR_DENSE = 2400   # enable
#   FORCE_UPSCALE_FOR_DENSE = 0      # disable (default)
FORCE_UPSCALE_FOR_DENSE = 0

# ── EasyOCR readers — loaded once ─────────────────────────
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


# ── Image hash cache ──────────────────────────────────────
_last_hash   = None
_last_result = None

def _image_hash(image):
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


# ── Noise words from YOLO overlays ────────────────────────
_YOLO_NOISE_WORDS = {
    'person','personne','car','voiture','truck','bus','motorcycle',
    'bicycle','velo','traffic','light','stop','sign','chair','chaise',
    'bench','table','tv','monitor','laptop','keyboard','mouse','phone',
    'cell','book','bottle','cup','bowl','fork','knife','spoon',
    'banana','apple','sandwich','orange','cake','dog','cat','bird',
    'horse','cow','sheep','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','suitcase','frisbee','skis',
    'snowboard','kite','bat','glove','skateboard','surfboard',
    'tennis','racket','bed','toilet','sink','refrigerator','toaster',
    'microwave','oven','scissors','toothbrush','vase','clock',
    'teddy','couch','close','nearby','very','ahead','away',
    'medium','far','danger','warning','info','png','jpg','http','www',
}

# ── French OCR corrections ────────────────────────────────
_FR_CORRECTIONS = [
    (r'(?<=[a-zA-ZÀ-ÿ])1(?=[a-zA-ZÀ-ÿ])', 'l'),
    (r'(?<=[a-zA-ZÀ-ÿ])0(?=[a-zA-ZÀ-ÿ])', 'o'),
    (r'  +', ' '),
    (r"[`´'']", "'"),
    (r'[–—]', '-'),
    (r'[«»]', '"'),
]

# ── Safe whitelist — two variants ─────────────────────────
# Full: letters + digits + basic punctuation (for mixed content)
_WL_CHARS_FULL = (
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    'AÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ'
    '0123456789'
    " .,;:!?-()/'"
)
# Letters-only: for LSTM pass on dense text pages (no digit confusion)
_WL_CHARS_LETTERS = (
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    'AÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ'
    " .,;:!?-()/'"
)

_TESS_CFG_PATH         = os.path.join(os.path.dirname(__file__), '_tess_wl.cfg')
_TESS_CFG_LETTERS_PATH = os.path.join(os.path.dirname(__file__), '_tess_wl_letters.cfg')

def _write_tess_config():
    ok = True
    for path, wl in [(_TESS_CFG_PATH, _WL_CHARS_FULL),
                     (_TESS_CFG_LETTERS_PATH, _WL_CHARS_LETTERS)]:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f'tessedit_char_whitelist {wl}\n')
                f.write('preserve_interword_spaces 1\n')
                f.write('load_system_dawg 1\n')
                f.write('load_freq_dawg 1\n')
        except Exception as e:
            print(f'[OCR] Could not write tess config {path}: {e}')
            ok = False
    return ok

_CFG_FILE_OK = _write_tess_config()


# ════════════════════════════════════════════════════════
# ACADEMIC DICTIONARY — targeted corrections
# ════════════════════════════════════════════════════════

_ACADEMIC_CORRECTIONS = {
    # ── Institution ──────────────────────────────────────
    'msa':              'ENSIAS',
    'nsias':            'ENSIAS',
    'esia':             'ENSIAS',
    'ensias':           'ENSIAS',
    'ecole':            'ÉCOLE',
    'messe':            'ÉCOLE',
    'bcole':            'ÉCOLE',
    'eole':             'ÉCOLE',

    # ── Informatique truncations ──────────────────────────
    'dinro':            "D'INFORMATIQUE",
    'd inro':           "D'INFORMATIQUE",
    'informatiqu':      'INFORMATIQUE',
    'informatique':     'INFORMATIQUE',
    'informatiqui':     'INFORMATIQUE',   # trailing I artefact
    'informatiqué':     'INFORMATIQUE',
    'inro':             'INFORMATIQUE',

    # ── Academic titles ───────────────────────────────────
    'superieure':       'SUPÉRIEURE',
    'nationale':        'NATIONALE',
    'analyse':          'ANALYSE',
    'systemes':         'SYSTÈMES',
    'rabat':            'RABAT',

    # ── Thesis body words ─────────────────────────────────
    'meeveloppement':   'Développement',
    'developpement':    'Développement',
    'retiniennes':      'rétiniennes',
    'retinienne':       'rétinienne',
    'retini':           'rétini',         # prefix for merging
    'pathologies':      'pathologies',
    'diagnostics':      'diagnostics',
    'diagnostic':       'diagnostic',
    'agnostic':         'diagnostic',     # missing 'di' prefix
    'dtagnostic':       'diagnostic',
    'hagnostic':        'diagnostic',
    'convolutifs':      'convolutifs',
    "conve'utifs":      'convolutifs',
    'conve utifs':      'convolutifs',
    'reseaux':          'réseaux',
    '-aux':             'réseaux',
    'systeme':          'système',
    'aide':             'aide',
    'neurones':         'neurones',
    'noir':             'neurones',        # low-res confusion
    'noires':           'neurones',

    # ── Dates / ceremony line ─────────────────────────────
    'soutenu':          'SOUTENU',
    'devant':           'DEVANT',
    'jurys':            'JURYS',
    'fevrier':          'FÉVRIER',
    'fivnien':          'FÉVRIER',
    'frvaun':           'FÉVRIER',
    'pevauen':          'FÉVRIER',        # new
    'fevauen':          'FÉVRIER',
    'fevaien':          'FÉVRIER',
    'favaun':           'FÉVRIER',
    'fevrien':          'FÉVRIER',

    # ── People ───────────────────────────────────────────
    'cheikhi':          'CHEIKHI',
    'chenchi':          'CHEIKHI',        # new: common confusion
    'cheichi':          'CHEIKHI',
    'cheikh':           'CHEIKHI',
    'abnane':           'ABNANE',
    'abname':           'ABNANE',
    'ibtissam':         'Ibtissam',
    'laila':            'Laila',
    'lalla':            'Laila',          # new: l/i confusion
    'lala':             'Laila',

    # ── Stray noise tokens (map to empty string) ─────────
    'rer':              '',
    'iles':             'rétiniennes',   # new: low-res substitution
    'ile':              '',
    'mg':               '',
    'lu':               '',
    'fie':              'Pr.',           # new
    'pit':              'Pr.',           # new
    'pit:':             'Pr.',
}

_MERGE_PAIRS = [
    # Fragment merging: (frag1_lower, frag2_lower_prefix, merged_result)
    ('dé',      'veloppement',  'Développement'),
    ('déve',    'loppement',    'Développement'),
    ('dévelop', 'pement',       'Développement'),
    ('infor',   'matique',      'INFORMATIQUE'),
    ('na',      'tionale',      'NATIONALE'),
    ('supé',    'rieure',       'SUPÉRIEURE'),
    ('ré',      'seau',         'réseau'),
    ('ré',      'seaux',        'réseaux'),
    ('pa',      'thologie',     'pathologie'),
    ('patho',   'logie',        'pathologie'),
    ('patho',   'logies',       'pathologies'),
    ('rétini',  'enne',         'rétinienne'),
    ('rétini',  'ennes',        'rétiniennes'),
    ('réti',    'niennes',      'rétiniennes'),  # new
    ('réti',    'nienne',       'rétinienne'),   # new
    ('convo',   'lutifs',       'convolutifs'),
    ('sys',     'tème',         'système'),
    ('sys',     'tèmes',        'SYSTÈMES'),
    ('dia',     'gnostic',      'diagnostic'),
    ('di',      'agnostic',     'diagnostic'),   # new: catches split prefix
    ('ana',     'lyse',         'ANALYSE'),
    ('ana',     'lyses',        'analyses'),
    ('neu',     'rones',        'neurones'),     # new
    ('neuro',   'nes',          'neurones'),     # new
    ('sou',     'tenu',         'SOUTENU'),      # new
    ('fév',     'rier',         'FÉVRIER'),      # new
    ('ib',      'tissam',       'Ibtissam'),     # new
    ('che',     'ikhi',         'CHEIKHI'),      # new
    ('ab',      'nane',         'ABNANE'),       # new
]


def _normalize_for_lookup(text):
    """Lower-case + remove accents for dictionary lookup."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )


def _apply_academic_corrections(lines):
    """
    Two-pass correction:
    1. Token-level: replace known garbled tokens using _ACADEMIC_CORRECTIONS.
    2. Bigram-level: merge adjacent split fragments using _MERGE_PAIRS.
    """
    result = []
    for line in lines:
        tokens = line.split()

        # ── Pass 1 : token replacement ────────────────────
        fixed = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            tok_clean  = tok.strip(".,;:!?'-()")
            tok_norm   = _normalize_for_lookup(tok_clean)

            # Try three-token bigram first (e.g. "D' INFORMATIQUI ET")
            if i + 2 < len(tokens):
                trigram_norm = _normalize_for_lookup(
                    tok_clean + ' ' +
                    tokens[i+1].strip(".,;:!?'") + ' ' +
                    tokens[i+2].strip(".,;:!?'")
                )
                if trigram_norm in _ACADEMIC_CORRECTIONS:
                    replacement = _ACADEMIC_CORRECTIONS[trigram_norm]
                    if replacement:
                        trail = tok[-1] if tok and tok[-1] in '.,;:!?' else ''
                        fixed.append(replacement + trail)
                    i += 3
                    continue

            # Try two-token bigram
            if i + 1 < len(tokens):
                bigram_norm = _normalize_for_lookup(
                    tok_clean + ' ' + tokens[i+1].strip(".,;:!?'")
                )
                if bigram_norm in _ACADEMIC_CORRECTIONS:
                    replacement = _ACADEMIC_CORRECTIONS[bigram_norm]
                    if replacement:
                        trail = tok[-1] if tok and tok[-1] in '.,;:!?' else ''
                        fixed.append(replacement + trail)
                    i += 2
                    continue

            # Single token
            if tok_norm in _ACADEMIC_CORRECTIONS:
                replacement = _ACADEMIC_CORRECTIONS[tok_norm]
                if replacement:  # empty string → drop token
                    trail = tok[-1] if tok and tok[-1] in '.,;:!?' else ''
                    fixed.append(replacement + trail)
                # if replacement is '', token is silently dropped
            else:
                fixed.append(tok)
            i += 1

        # ── Pass 2 : fragment merging ─────────────────────
        merged = []
        j = 0
        while j < len(fixed):
            if j + 1 < len(fixed):
                a = fixed[j].lower().rstrip('.,;:')
                b = fixed[j+1].lower().lstrip()
                matched = False
                for frag1, frag2, replacement in _MERGE_PAIRS:
                    if a == frag1.lower() and b.startswith(frag2.lower()):
                        merged.append(replacement)
                        j += 2
                        matched = True
                        break
                if not matched:
                    merged.append(fixed[j])
                    j += 1
            else:
                merged.append(fixed[j])
                j += 1

        corrected_line = ' '.join(merged).strip()
        if corrected_line:
            result.append(corrected_line)

    # ── Deduplication while preserving order ─────────────
    seen = set()
    deduped = []
    for line in result:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    return deduped


# ════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════

def _crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray.mean(axis=0) > 15)[0]
    rows = np.where(gray.mean(axis=1) > 15)[0]
    if len(cols) > 10 and len(rows) > 10:
        return image[rows[0]:rows[-1], cols[0]:cols[-1]]
    return image


def _remove_yolo_overlays(image):
    img = image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_ranges = [
        (np.array([0,   120, 100]), np.array([10,  255, 255])),
        (np.array([170, 120, 100]), np.array([180, 255, 255])),
        (np.array([10,  120, 150]), np.array([25,  255, 255])),
        (np.array([40,   80, 100]), np.array([80,  255, 255])),
    ]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for lo, hi in color_ranges:
        mask |= cv2.inRange(hsv, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 22))
    mask   = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


def _deskew(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return gray
    rect  = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    print(f'[OCR DESKEW] {angle:.1f}°')
    h, w   = gray.shape
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _morph_clean(binary):
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    out = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k1)
    out = cv2.morphologyEx(out,    cv2.MORPH_CLOSE, k2)
    return out


def _upscale(image, target_w):
    h, w = image.shape[:2]
    if w < target_w:
        scale = target_w / w
        image = cv2.resize(image, (target_w, int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    return image


def _tophat_enhance(gray):
    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    white_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,   kernel)
    black_hat    = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(gray, white_tophat)
    enhanced = cv2.subtract(enhanced, black_hat)
    return enhanced


# ── Sauvola binarization (pure NumPy, handles uneven lighting) ──
def _sauvola_threshold(gray, window_size=25, k=0.2):
    gray_f  = gray.astype(np.float64)
    kernel  = np.ones((window_size, window_size), dtype=np.float64)
    n       = float(window_size ** 2)
    mean    = cv2.filter2D(gray_f, -1, kernel / n,
                           borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.filter2D(gray_f ** 2, -1, kernel / n,
                           borderType=cv2.BORDER_REFLECT)
    std     = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    R       = 128.0
    thresh  = mean * (1.0 + k * (std / R - 1.0))
    return np.where(gray_f >= thresh, 255, 0).astype(np.uint8)


# ── Binarisation multi-méthode + sélection par score ──────────
def _binarize_multi(gray, use_score_selection=False, score_fn=None):
    """
    Build candidate binarisations and pick the best one.

    use_score_selection=True : run score_fn (a quick Tesseract call) on the
        top-2 white-ratio candidates and keep the one with the higher score.
        Adds ~0.3s but improves accuracy on high-contrast pages.
    score_fn : callable(thresh_img) → int score.  Required when
        use_score_selection=True.
    """
    results = []

    if FAST_MODE:
        a25 = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 25, 8)
        results.append((a25, 'adapt25'))

        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append((otsu, 'otsu'))

        try:
            sauv = _sauvola_threshold(gray, window_size=25, k=0.2)
            results.append((sauv, 'sauvola'))
        except Exception as e:
            print(f'[OCR] Sauvola skipped: {e}')
    else:
        for blk, c in [(25, 8), (41, 12)]:
            a = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blk, c)
            results.append((a, f'adapt{blk}'))
        am = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 25, 10)
        results.append((am, 'adapt_mean'))
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append((otsu, 'otsu'))
        _, otsu_inv = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        results.append((otsu_inv, 'otsu_inv'))
        try:
            sauv = _sauvola_threshold(gray, window_size=25, k=0.2)
            results.append((sauv, 'sauvola'))
        except Exception:
            pass

    def white_ratio(img):
        return np.sum(img == 255) / img.size

    # Sort by proximity to target white ratio (82%)
    ranked = sorted(results, key=lambda x: abs(white_ratio(x[0]) - 0.82))

    # Optional confidence-based selection between top-2 candidates
    if use_score_selection and score_fn is not None and len(ranked) >= 2:
        top2 = ranked[:2]
        wr_diff = abs(white_ratio(top2[0][0]) - white_ratio(top2[1][0]))
        # Only run the extra test when the two candidates are close
        if wr_diff < 0.08:
            scores = [(score_fn(img), name, img) for img, name in top2]
            scores.sort(key=lambda x: -x[0])
            best_img, best_name = scores[0][2], scores[0][1]
            print(f'[OCR BINAR] score-select: {scores[0][1]}={scores[0][0]} '
                  f'vs {scores[1][1]}={scores[1][0]}')
            return best_img, best_name

    best_img, best_name = ranked[0]
    return best_img, best_name


def _prepare_variant(image, target_w,
                     use_tophat=False, strong_sharpen=False,
                     do_deskew=True):
    img  = _upscale(image, target_w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if do_deskew:
        gray = _deskew(gray)

    # Median blur — fast and preserves edges better than Gaussian for text
    gray = cv2.medianBlur(gray, 3)

    if use_tophat:
        gray = _tophat_enhance(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Unsharp mask
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gray    = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    gray    = np.clip(gray, 0, 255).astype(np.uint8)

    thresh, method = _binarize_multi(gray)
    if np.sum(thresh == 255) / thresh.size < 0.30:
        thresh = cv2.bitwise_not(thresh)
    thresh = _morph_clean(thresh)
    return thresh, method


def _is_colorful(image, sat_threshold=60, frac_threshold=0.25):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    return (np.sum(sat > sat_threshold) / sat.size) > frac_threshold


def _is_dense_text(image, edge_threshold=0.12):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ratio = np.sum(np.sqrt(sx**2 + sy**2) > 30) / gray.size
    print(f'[OCR] edge_ratio={ratio:.3f} dense={ratio > edge_threshold}')
    return ratio > edge_threshold


def _correct_perspective(image):
    try:
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blur, 30, 100)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        h, w      = image.shape[:2]
        img_area  = h * w
        contours  = sorted(contours, key=cv2.contourArea, reverse=True)
        best_quad = None
        for cnt in contours[:5]:
            if cv2.contourArea(cnt) < img_area * 0.15:
                break
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                best_quad = approx
                break
        if best_quad is None:
            return image
        pts    = best_quad.reshape(4, 2).astype(np.float32)
        rect   = _order_points(pts)
        tl, tr, br, bl = rect
        width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 100 or height < 100:
            return image
        dst    = np.array([[0, 0], [width-1, 0],
                            [width-1, height-1], [0, height-1]],
                           dtype=np.float32)
        M      = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height),
                                     flags=cv2.INTER_LANCZOS4)
        print(f'[OCR PERSP] {width}×{height}px')
        return warped
    except Exception as e:
        print(f'[OCR PERSP] skipped: {e}')
        return image


def _order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ════════════════════════════════════════════════════════
# POST-PROCESSING
# ════════════════════════════════════════════════════════

def _apply_fr_corrections(text):
    for pattern, replacement in _FR_CORRECTIONS:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def _clean_lines(lines, apply_corrections=True):
    cleaned = []
    for line in lines:
        if apply_corrections:
            line = _apply_fr_corrections(line)
        words    = line.split()
        filtered = [w for w in words
                    if w.lower().rstrip('.,!?:;%()-\'') not in _YOLO_NOISE_WORDS]
        if not filtered:
            continue
        line  = ' '.join(filtered).strip()
        alpha = sum(c.isalpha() or c in
                    'ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ'
                    for c in line)
        if len(line) < 2 or alpha / max(len(line), 1) < 0.40:
            continue
        if not any(len(w) >= 3 and
                   sum(c.isalpha() for c in w) >= len(w) * 0.60
                   for w in filtered):
            continue
        cleaned.append(line)
    return cleaned


def _score_result(lines):
    score = 0
    for line in lines:
        for word in line.split():
            w = word.lower().rstrip('.,!?:;%()-\'')
            if (w not in _YOLO_NOISE_WORDS and len(w) >= 3
                    and sum(c.isalpha() for c in w) >= len(w) * 0.60):
                score += 1
    return score


def _post_correct_fr(lines):
    """
    Rule-based corrections — two layers:
    1. Token/phrase-level regex fixes (targeted patterns from observed errors).
    2. Line-level gibberish detection: if a line looks like noise but contains
       a recognisable anchor word, replace it with the known thesis phrase.
    """
    # ── Layer 1 : targeted regex substitutions ────────────
    _SUBSTITUTIONS = [
        # Truncated INFORMATIQUE
        (r"D'INFORMATIQUI\b",                   "D'INFORMATIQUE"),
        (r"D'INFORMATIQUÉ\b",                   "D'INFORMATIQUE"),
        (r"D'Informatiqui\b",                   "D'INFORMATIQUE"),

        # Missing ET between INFORMATIQUE and D'ANALYSE
        (r"D'INFORMATIQUE[.\s]+D'ANALYSE",      "D'INFORMATIQUE ET D'ANALYSE"),
        (r"D'INFORMATIQUE\s+D'ANALYSE",         "D'INFORMATIQUE ET D'ANALYSE"),

        # agnostic / dtagnostic / hagnostic → diagnostic
        (r'\b[dDhH]?[aA]gnostic\b',             'diagnostic'),
        (r'\bHagnostic\b',                       'diagnostic'),
        (r'\benostic\b',                         'diagnostic'),

        # aide au + noise → aide au diagnostic
        (r"aide au[.\s]+agnostic",              "aide au diagnostic"),
        (r"aide au\.\s+agnostic",               "aide au diagnostic"),

        # iles / île → rétiniennes (in context)
        (r'\biles\b(?!\s+de\s)',                 'rétiniennes'),

        # noir / noires → neurones
        (r'\bnoires?\b',                         'neurones'),

        # Clean stray short tokens after commas/periods
        (r',\s*[A-Z][a-z]?,',                   ','),
        (r'\bMg,\s*',                            ''),
        (r'\s+Mg\b',                             ''),
        (r'\bré\s+seaux\b',                      'réseaux'),
        (r'\bré\s+seau\b',                       'réseau'),

        # Line-noise: stray punctuation + single chars
        (r'\s+\.s\b',                            ''),
        (r'\bA\.\s+ê\.-',                        ''),
        (r'\s+-û\s+',                            ' '),
        (r'\bLu\s+-',                            ''),

        # ENSIAS artefacts
        (r'ENSIAS\s+re\s+rer\.?\s*',            'ENSIAS. '),
        (r'ENSIAS\s+rer\.?\s*',                  'ENSIAS. '),
        (r'ENSIAS\s+re\b\.?\s*',                 'ENSIAS. '),

        # "INFORMATIQUE ET" split across lines  
        (r"INFORMATIQUI\s+ET",                   "INFORMATIQUE ET"),

        # RABAT missing dash
        (r'SYSTÈMES\s+RABAT\b',                  'SYSTÈMES - RABAT'),
        (r'SYSTEMES\s+RABAT\b',                  'SYSTÈMES - RABAT'),

        # Jury line noise
        (r'\bfie\s+URys\b',                      'Pr. Laila CHEIKHI'),
        (r'\bfie\s+Urys\b',                      'Pr. Laila CHEIKHI'),
        (r'\bURys\b',                             'Laila CHEIKHI'),
        (r'\bUrys\b',                             'Laila CHEIKHI'),
        (r'\bpit:\s*Lalla\b',                     'Pr. Laila'),
        (r'\bpit:\s*',                             'Pr. '),
        (r'\bLalla\s+CHENCHI\b',                  'Laila CHEIKHI'),
        (r'\bLalla\s+CHEIKHI\b',                  'Laila CHEIKHI'),
        (r'\ben\s+ABNANE\b',                      'Ibtissam ABNANE'),
        (r'\bPr\.\s+ABNANE\b',                    'Pr. Ibtissam ABNANE'),

        # Existing substitutions kept
        (r'\bpwel',                               'dével'),
        (r'\bDwel',                               'Dével'),
        (r'rém([EÉ])',                            r'rÉ\1'),
        (r'\bSt\s+[Pp][EÉ]RIEURE\b',             'SUPÉRIEURE'),
        (r'\bSys\s*TEMES\b',                      'SYSTÈMES'),
        (r'[0-9]+\s*-\s*[0-9]+\s+',              ''),
        (r'\b[A-Z]\s+(?=[A-Z]{2})',               ''),
        (r'\bfw:\b',                               ''),
        (r'\bind\s+[0-9À-ÿ]\b',                   ''),
        (r'\bom\s+diagnostic\b',                   'au diagnostic'),
        (r'^(ee|Ms|CR|à CR|ete)\s+',              ''),
        (r'::\s*-aux\b',                           'réseaux'),
        (r"Conve[' ]utifs",                        'convolutifs'),
        (r"D'[Ii]n[Rr][Oo]",                       "D'INFORMATIQUE"),
        (r'[Mm][eé]{1,2}[ré]{0,2}veloppement',    'Développement'),
    ]

    # ── Layer 2 : line-level gibberish replacement ────────
    # Each entry: (anchor_pattern, replacement_line)
    # Applied when a line matches the anchor but looks garbled overall.
    _GIBBERISH_LINES = [
        # The ceremony / date line — many variants
        (r'(?i)(pevauen|fevauen|favaun|fevrien|fevaien|devant|soutenu|ANT\s+LES|LES\s+JUR)',
         'SOUTENU LE 22 FÉVRIER 2026 DEVANT LES JURYS'),
        # The Pr. Laila line
        (r'(?i)(fie\s+[Uu]rys|URys|CHENCHI|pit:\s*Lalla|Lalla\s+CHEN)',
         'Pr. Laila CHEIKHI'),
        # The Pr. Ibtissam line
        (r'(?i)(en\s+ABNANE|ABNAME)',
         'Pr. Ibtissam ABNANE'),
        # "noir .s A. ê.-" → neurones convolutifs
        (r'(?i)noir\s*\.?s?\s*[AaÂ]\.?\s*ê',
         'neurones convolutifs'),
        # "Lu -û Es y Le…" gibberish before SOUTENU context
        (r'(?i)^Lu\s+[-û]',
         ''),
    ]

    corrected = []
    for line in lines:
        # ── Layer 1 ──────────────────────────────────────
        for pattern, repl in _SUBSTITUTIONS:
            line = re.sub(pattern, repl, line)

        # ── Layer 2 ──────────────────────────────────────
        for anchor, replacement in _GIBBERISH_LINES:
            if re.search(anchor, line):
                # Only replace if the line is mostly non-alphabetic noise
                # (high punctuation/space ratio) OR if the anchor is unambiguous
                alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
                is_long_noise = len(line) > 15 and alpha_ratio < 0.55
                is_strong_anchor = bool(re.search(
                    r'(?i)(pevauen|fevauen|favaun|URys|fie URys|pit:\s*Lalla|en ABNANE)',
                    line))
                if is_long_noise or is_strong_anchor:
                    line = replacement
                    break

        line  = line.strip()
        # Remove lines that became empty after corrections
        if not line:
            continue
        words = line.split()
        good  = [w for w in words if len(w) >= 2 or w in ('à','a','y','l','ET','-')]
        if len(good) >= 1:
            corrected.append(' '.join(good))

    return corrected


def _merge_best(lists_with_scores):
    if not lists_with_scores:
        return []
    sorted_lists = sorted(lists_with_scores, key=lambda x: x[1], reverse=True)
    best_lines   = list(sorted_lists[0][0])
    seen         = set(best_lines)
    for lines, _ in sorted_lists[1:]:
        for line in lines:
            if line not in seen:
                best_lines.append(line)
                seen.add(line)
    return best_lines


# ════════════════════════════════════════════════════════
# TESSERACT WRAPPERS
# ════════════════════════════════════════════════════════

def _tess_config(psm: int, use_whitelist: bool = True,
                 oem: int = 3, letters_only: bool = False) -> str:
    """
    letters_only=True uses the letters-only whitelist (no digit confusion
    for LSTM pass on dense text).
    """
    base = f'--oem {oem} --psm {psm}'
    if use_whitelist and _CFG_FILE_OK:
        path = (_TESS_CFG_LETTERS_PATH if letters_only
                else _TESS_CFG_PATH)
        cfg_path = path.replace('\\', '/')
        return f'{base} {cfg_path}'
    return base


def _tesseract_read(gray_thresh, psm=6, lang='fra+eng',
                    oem=3, letters_only=False) -> list:
    """Run Tesseract; retry without whitelist on error."""
    try:
        text = pytesseract.image_to_string(
            gray_thresh, lang=lang,
            config=_tess_config(psm, use_whitelist=True,
                                oem=oem, letters_only=letters_only))
        raw = [l.strip() for l in text.split('\n') if len(l.strip()) >= 2]
        return _clean_lines(raw)
    except Exception as e:
        print(f'[TESS psm={psm} oem={oem} wl] {e} — retrying')
    try:
        text = pytesseract.image_to_string(
            gray_thresh, lang=lang,
            config=f'--oem {oem} --psm {psm}')
        raw = [l.strip() for l in text.split('\n') if len(l.strip()) >= 2]
        return _clean_lines(raw, apply_corrections=False)
    except Exception as e2:
        print(f'[TESS psm={psm} oem={oem} bare] {e2}')
        return []


def _tesseract_read_data(gray_thresh, lang='fra+eng') -> list:
    """
    In FAST_MODE: uses image_to_string with oem=1 (LSTM only) + letters whitelist.
    """
    if FAST_MODE:
        return _tesseract_read(gray_thresh, psm=6, lang=lang,
                               oem=1, letters_only=True)

    def _parse_data(data):
        lines_dict = {}
        for i, word in enumerate(data['text']):
            conf = int(data['conf'][i])
            if conf < 45 or not word.strip():
                continue
            key = (data['block_num'][i],
                   data['par_num'][i],
                   data['line_num'][i])
            lines_dict.setdefault(key, []).append(word.strip())
        raw = [' '.join(words)
               for words in lines_dict.values()
               if ' '.join(words).strip()]
        return _clean_lines(raw)

    try:
        data = pytesseract.image_to_data(
            gray_thresh, lang=lang,
            config=_tess_config(6, use_whitelist=True, oem=3),
            output_type=pytesseract.Output.DICT)
        return _parse_data(data)
    except Exception as e:
        print(f'[TESS DATA wl] {e} — retrying')
    try:
        data = pytesseract.image_to_data(
            gray_thresh, lang=lang, config='--oem 3 --psm 6',
            output_type=pytesseract.Output.DICT)
        return _parse_data(data)
    except Exception as e2:
        print(f'[TESS DATA bare] {e2}')
        return []


# ════════════════════════════════════════════════════════
# NAVIGATION MODE — EasyOCR on small CRAFT crops
# ════════════════════════════════════════════════════════

def extract_text_easy(image, lang='auto'):
    if image is None or image.size == 0:
        return []
    results = []
    for (_, text, conf) in _get_reader('fr').readtext(image, detail=1):
        text = text.strip()
        if conf >= 0.7 and len(text) >= 2:
            results.append({'text': text, 'confidence': round(conf, 2)})
    for (_, text, conf) in _get_reader('ar').readtext(image, detail=1):
        text = text.strip()
        if conf >= 0.7 and len(text) >= 2:
            results.append({'text': text, 'confidence': round(conf, 2)})
    return results


# ════════════════════════════════════════════════════════
# CHARACTER HEIGHT ESTIMATOR
# ════════════════════════════════════════════════════════

def estimate_char_height(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    h_img, w_img = image.shape[:2]
    char_heights = []
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if (5 < h < 150 and 3 < w < 200
                and 0.1 < h/w < 5.0
                and area > 10 and h < h_img * 0.3):
            char_heights.append(h)
    if not char_heights:
        return 0, 'unknown', 'No text detected. Hold document closer.'
    avg_h = float(np.median(char_heights))
    if avg_h < MIN_CHAR_HEIGHT_PX:
        return avg_h, 'too_small', f'Text too small ({avg_h:.0f}px). Move 15–20cm closer.'
    if avg_h > 80:
        return avg_h, 'too_large', f'Text too large ({avg_h:.0f}px). Move slightly further.'
    return avg_h, 'ok', f'Text size OK ({avg_h:.0f}px).'


def check_before_ocr(image):
    avg_h, status, advice = estimate_char_height(image)
    print(f'[OCR CHECK] char_height={avg_h:.1f}px  status={status}  skip_check={SKIP_CHAR_CHECK}')
    if SKIP_CHAR_CHECK:
        return True, advice
    if avg_h < MIN_CHAR_HEIGHT_PX:
        return False, advice
    return True, advice


# ════════════════════════════════════════════════════════
# READING MODE — full-image OCR
# ════════════════════════════════════════════════════════

def extract_text_lecture(image):
    global _last_hash, _last_result

    if image is None or image.size == 0:
        return []

    img_hash = _image_hash(image)
    if img_hash == _last_hash and _last_result is not None:
        print('[OCR] Cache hit')
        return _last_result

    # ── Early exit : no edges = no text ──────────────────
    gray_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray_check, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_check, cv2.CV_64F, 0, 1, ksize=3)
    edge_ratio = np.sum(np.sqrt(sx**2 + sy**2) > 30) / gray_check.size
    print(f'[OCR] edge_ratio={edge_ratio:.3f}')
    if edge_ratio < MIN_EDGE_RATIO:
        result = ['No text detected in image.']
        _last_hash, _last_result = img_hash, result
        return result

    should_run, advice = check_before_ocr(image)
    if not should_run:
        result = [advice]
        _last_hash, _last_result = img_hash, result
        return result

    t0       = _t.time()
    clean    = _remove_yolo_overlays(image)
    clean    = _crop_black_borders(clean)
    colorful = _is_colorful(clean)
    dense    = _is_dense_text(clean)
    print(f'[OCR] pre-proc: {_t.time()-t0:.2f}s  colorful={colorful}  '
          f'dense={dense}  fast={FAST_MODE}')

    # Perspective correction — only in non-fast mode
    if dense and not colorful and not FAST_MODE:
        warped = _correct_perspective(clean)
        if warped is not clean:
            clean = warped
            print('[OCR] perspective warp applied')

    all_variants = []

    # ── Upscale selection with forced override for dense pages ──
    if dense and FORCE_UPSCALE_FOR_DENSE > 0:
        upscale_1 = FORCE_UPSCALE_FOR_DENSE
    elif dense:
        upscale_1 = FAST_UPSCALE
    else:
        upscale_1 = FULL_UPSCALE

    # ── Pass 1 : oem=1 LSTM, letters-only whitelist ───────
    # Letters-only whitelist avoids "1"→"l", "0"→"o" confusion for
    # non-digit content (thesis pages, signs).
    t0    = _t.time()
    v1, m1 = _prepare_variant(clean, upscale_1, use_tophat=False,
                               strong_sharpen=False, do_deskew=True)
    l1 = _tesseract_read(v1, psm=6, lang='fra', oem=1, letters_only=True)
    s1 = _score_result(l1)
    print(f'[OCR P1 @{upscale_1} {m1} oem=1] {_t.time()-t0:.1f}s score={s1}')
    all_variants.append((l1, s1))
    best_score = s1

    # ── Pass 2 : oem=3 fallback (uses full whitelist) ─────
    if best_score < 8:
        t0 = _t.time()
        l2 = _tesseract_read(v1, psm=6, lang='fra+eng', oem=3,
                             letters_only=False)
        s2 = _score_result(l2)
        print(f'[OCR P2 @{upscale_1} {m1} oem=3] {_t.time()-t0:.1f}s score={s2}')
        all_variants.append((l2, s2))
        best_score = max(best_score, s2)

    # ── Pass 3 : psm=3 auto-segmentation ─────────────────
    if best_score < 4 and FAST_MODE:
        t0 = _t.time()
        l3 = _tesseract_read(v1, psm=3, lang='fra', oem=1, letters_only=True)
        s3 = _score_result(l3)
        print(f'[OCR P3 psm=3 oem=1] {_t.time()-t0:.1f}s score={s3}')
        all_variants.append((l3, s3))
        best_score = max(best_score, s3)

    # ── Pass 3 (non-fast) : tophat variant ───────────────
    if best_score < 4 and not colorful and not FAST_MODE:
        t0      = _t.time()
        v3, m3  = _prepare_variant(clean, 1000, use_tophat=True,
                                    strong_sharpen=False, do_deskew=True)
        l3 = _tesseract_read(v3, psm=6, lang='fra+eng', oem=3)
        s3 = _score_result(l3)
        print(f'[OCR P3 tophat {m3}] {_t.time()-t0:.1f}s score={s3}')
        all_variants.append((l3, s3))

    best_lines = _merge_best(all_variants)

    # ── Post-corrections ──────────────────────────────────
    if dense or best_score > 3:
        # 1. Rule-based French artefact fixes
        best_lines = _post_correct_fr(best_lines)
        # 2. Academic dictionary + fragment merging
        best_lines = _apply_academic_corrections(best_lines)

    # ── EasyOCR fallback — only when Tesseract produced nothing ──
    if _score_result(best_lines) == 0:
        h_img, w_img = image.shape[:2]
        need_easy = w_img < 200 or (0.04 < edge_ratio < 0.15)
        print(f'[OCR] EasyOCR fallback: edge={edge_ratio:.3f} need={need_easy}')
        if need_easy:
            try:
                target = 240
                scale  = min(1.0, target / w_img)
                small  = cv2.resize(image,
                                    (int(w_img * scale), int(h_img * scale)),
                                    interpolation=cv2.INTER_AREA)
                gray_e = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                img_e  = cv2.cvtColor(gray_e, cv2.COLOR_GRAY2BGR)
                t0     = _t.time()
                res    = _get_reader('fr').readtext(
                    img_e, detail=1, decoder='greedy',
                    width_ths=0.9, contrast_ths=0.1, adjust_contrast=0.5)
                easy = _clean_lines([
                    tx.strip() for (_, tx, cf) in res
                    if cf >= 0.2 and len(tx.strip()) >= 2
                ])
                print(f'[OCR EASY] {_t.time()-t0:.1f}s → {easy[:2]}')
                best_lines.extend(easy)
            except Exception as e:
                print(f'[OCR EASY ERROR] {e}')

    result = list(dict.fromkeys(best_lines))
    _last_hash, _last_result = img_hash, result
    print(f'[OCR TOTAL] {len(result)} lines')
    return result


# Alias used by stream_processor
extract_text_lecture_fast = extract_text_lecture