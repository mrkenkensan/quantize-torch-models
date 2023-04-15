import os
from quantize import SAVE_DIR as _SAVE_DIR

SAVE_DIR = os.path.join(_SAVE_DIR, 'speech-to-text')
MODEL_DIR = os.path.join(SAVE_DIR, 'models')
INFERENCE_DIR = os.path.join(SAVE_DIR, 'inference')


def get_inference_files():
    return [os.path.join(INFERENCE_DIR, filename) for filename in sorted(os.listdir(INFERENCE_DIR))
            if filename[4:] == '.wav']
