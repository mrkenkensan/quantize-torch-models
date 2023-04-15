import logging
import time
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import torch.jit
import torch.nn as nn
import os
from transformers import pipeline
import sys

if __name__ == '__main__':
    _BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, _BASE_DIR)


from quantize import dynamic_quantize_model, save_model, load_model
from quantize.speech_to_text import MODEL_DIR as _MODEL_DIR
from quantize.speech_to_text import INFERENCE_DIR

_METHOD = 'openai-whisper'
MODEL_DIR = os.path.join(_MODEL_DIR, _METHOD)
ORIGINAL_MODEL_DIR = os.path.join(MODEL_DIR, 'original')
QUANTIZED_MODEL_DIR = os.path.join(MODEL_DIR, 'quantized')
TRAINED_MODEL_DIR = os.path.join(MODEL_DIR, 'quantized-and-trained')

_MODELS = [
    'tiny', 'base', 'small', 'medium', 'large', 'large-v2',
    'tiny.en', 'base.en', 'small.en', 'medium.en'
]


def _get_model_str(model_size: str, only_english: bool):
    model_str = model_size
    if only_english:
        model_str += '.en'
    assert model_str in _MODELS
    return model_str


def get_model(model_size: str, only_english: bool):
    model_str = _get_model_str(model_size=model_size, only_english=only_english)
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_str}").eval()
    return model


def get_parametric_layers(model: nn.Module):
    layers = list(model.children())

    out = []
    if len(layers) == 0:
        return model
    else:
        for child in layers:
            try:
                out.extend(get_parametric_layers(child))
            except TypeError:
                out.append(get_parametric_layers(child))
    return out


def run_inference(audio_file: os.path, pipe):
    start_time = time.time()
    print(pipe(audio_file)['text'])
    taken = time.time() - start_time
    print(f'time taken: {taken}s')


def main():
    model_size = 'base'
    only_english = True
    model_str = _get_model_str(model_size=model_size, only_english=only_english)
    original_file = os.path.join(ORIGINAL_MODEL_DIR, f'{model_str}.pt')
    if os.path.exists(original_file):
        model_fp32 = load_model(save_file=original_file)
    else:
        model_fp32 = get_model(model_size='base', only_english=True)
        save_model(model=model_fp32, save_file=original_file)

    quantize_file = os.path.join(QUANTIZED_MODEL_DIR, model_str + '.pt')
    if os.path.exists(quantize_file):
        model_int8 = load_model(save_file=quantize_file)
    else:
        model_int8 = dynamic_quantize_model(model_fp32)
        save_model(model_int8, save_file=quantize_file)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=f"openai/whisper-{model_str}",
        chunk_length_s=30,
        device='cpu',
    )
    pipe.model = model_int8
    audio_file = '/root/ProjectStorage/quantize-torch-models/quantized/speech-to-text/examples/[Aussie] 1MnxnpZVRjLGO.wav'
    run_inference(audio_file=audio_file, pipe=pipe)
    # print(pipe(audio_file)['segments'])
    # print(pipe.model)


if __name__ == '__main__':
    main()
