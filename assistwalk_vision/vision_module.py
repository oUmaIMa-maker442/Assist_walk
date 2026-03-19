import numpy as np
from src.step1_acquisition import acquire_from_file, acquire_from_pil
from src.step2_preprocessing import preprocess
from src.step3_yolo_detection import YOLODetector
from src.step4_filtering import filter_objects
from src.step5_craft_detection import CRAFTDetector
from src.step6_extraction import extract_text_regions


class VisionModule:
    def __init__(self):
        print('='*55)
        print('  Initialisation du Module Vision — AssistWalk')
        print('='*55)
        self.yolo  = YOLODetector(model_name='yolov8n.pt')
        self.craft = CRAFTDetector(languages=['fr', 'en'])
        print('[Module Vision] Prêt ✓')

    def analyze(self, image: np.ndarray) -> dict:
        print('─'*55)
        print('[Module Vision] Analyse démarrée...')

        pp           = preprocess(image)
        raw_objects  = self.yolo.detect(pp['original'])
        filtered     = filter_objects(raw_objects)
        text_boxes   = self.craft.detect_text_zones(pp['original'])
        text_regions = extract_text_regions(pp['original'], text_boxes)

        output = {
            'objects':      filtered,
            'text_regions': text_regions,
            'text_boxes':   text_boxes,
        }
        print("format",filtered[:2])
        print(f'[Module Vision] Terminé : {len(filtered)} objets, {len(text_regions)} zones texte')
        return output