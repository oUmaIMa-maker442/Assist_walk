#!/usr/bin/env python3
"""
test_ocr_image.py — Evaluate OCR pipeline on a static image file.

Usage:
    python test_ocr_image.py path/to/image.jpg
    python test_ocr_image.py path/to/image.jpg --save          # saves result to .txt
    python test_ocr_image.py path/to/image.jpg --save --show   # also displays image
"""

import sys
import os
import time
import argparse

# ── Path setup ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'text_module'))
sys.path.insert(0, os.path.join(ROOT, 'assistwalk_vision'))

# Silence Groq so pipeline.py doesn't fail on import
os.environ.setdefault('GROQ_API_KEY', 'dummy')

import cv2
import numpy as np

# ── Imports from your pipeline ───────────────────────────
from pipeline import process_lecture


# ════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Could not load image: {path}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"[IMAGE] Loaded: {path}  ({w}×{h} px)")
    return img


def print_result(text: str, lang: str, elapsed: float):
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  OCR RESULT  ({elapsed:.2f}s)  |  Language: {lang}")
    print(bar)
    print(text if text.strip() else "(no text extracted)")
    print(bar)


def save_result(text: str, lang: str, image_path: str):
    base = os.path.splitext(image_path)[0]
    out_path = base + "_ocr_result.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Language: {lang}\n")
        f.write("─" * 60 + "\n")
        f.write(text + "\n")
    print(f"[SAVED] Result written to: {out_path}")


def show_image(img: np.ndarray, title: str = "OCR Test Image"):
    """Display the image; press any key to close."""
    # Downscale for display if too large
    h, w = img.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(title, img)
    print("[DISPLAY] Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test OCR pipeline on a static image file.")
    parser.add_argument("image", help="Path to the image file (jpg, png, etc.)")
    parser.add_argument("--save",  action="store_true",
                        help="Save OCR result to a .txt file next to the image")
    parser.add_argument("--show",  action="store_true",
                        help="Display the image in an OpenCV window")
    parser.add_argument("--max-width", type=int, default=0,
                        help="Optionally resize image to this width before OCR "
                             "(0 = no resize). Useful to simulate webcam resolution.")
    args = parser.parse_args()

    img = load_image(args.image)

    # Optional: simulate lower resolution (e.g. webcam at 640px)
    if args.max_width and args.max_width > 0:
        h, w = img.shape[:2]
        if w > args.max_width:
            scale = args.max_width / w
            img = cv2.resize(img, (args.max_width, int(h * scale)),
                             interpolation=cv2.INTER_AREA)
            print(f"[RESIZE] Downscaled to {img.shape[1]}×{img.shape[0]} px "
                  f"(--max-width {args.max_width})")

    if args.show:
        show_image(img.copy(), title=os.path.basename(args.image))

    print("\n[OCR] Starting pipeline…")
    t0 = time.time()
    text, lang = process_lecture(img, speak=False)
    elapsed = time.time() - t0

    print_result(text, lang, elapsed)

    if args.save:
        save_result(text, lang, args.image)


if __name__ == "__main__":
    main()