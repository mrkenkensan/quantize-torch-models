import numpy as np
import torch.nn as nn
import torch
import logging
import os
from directory import PROJECT_STORAGE_DIR
SAVE_DIR = os.path.join(PROJECT_STORAGE_DIR, 'quantized')


DYNAMIC_QUANTIZATION_LAYERS = {
    nn.Linear,
    nn.LSTM,
    nn.GRU,
    nn.RNNCell,
    nn.GRUCell,
    nn.LSTMCell,
    nn.EmbeddingBag
}

_DYNAMIC_QUANTIZATION_LAYER_STRS = [str(_l) for _l in DYNAMIC_QUANTIZATION_LAYERS]


def _get_quantizable_layers(model: nn.Module):
    layers = list(model.children())
    out = []
    if len(layers) == 0:
        return [str(type(model))]
    else:
        for child in layers:
            try:
                out.extend(_get_quantizable_layers(child))
            except TypeError:
                out.append(_get_quantizable_layers(child))
    return list(np.intersect1d(out, _DYNAMIC_QUANTIZATION_LAYER_STRS).reshape(-1))


def get_quantizable_layers(model: nn.Module):
    layer_strs = _get_quantizable_layers(model=model)
    layers = [layer for i, layer in enumerate(DYNAMIC_QUANTIZATION_LAYERS) if
              _DYNAMIC_QUANTIZATION_LAYER_STRS[i] in layer_strs]
    return set(layers)


def get_model_size_bytes(model: nn.Module):
    out = 0
    for param in model.parameters():
        out += param.nelement() * param.element_size()
    return out


def get_model_size_megabytes(model: nn.Module):
    out = get_model_size_bytes(model=model)
    return out / (1024 ** 2)


def dynamic_quantize_model(model: nn.Module):
    quantizable_layers = get_quantizable_layers(model=model)
    out = torch.quantization.quantize_dynamic(
        model, quantizable_layers, dtype=torch.qint8
    )
    original_size_mb = get_model_size_megabytes(model=model)
    quantized_size_mb = get_model_size_megabytes(model=out)
    logging.info('quantized size reduction: {0:.1f} MiB -> {1:.1f} MiB (reduced by {2:.1f}%)'
                 .format(original_size_mb, quantized_size_mb,
                         ((original_size_mb - quantized_size_mb) / original_size_mb)* 100))
    return out


def save_model(model: nn.Module, save_file: os.path):
    if os.path.exists(save_file):
        logging.warning(f'save file {save_file} already exists, please delete to save new model')
        return
    logging.info(f'saving model to {save_file}')
    save_dir = os.path.dirname(save_file)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, save_file)


def load_model(save_file: os.path):
    assert os.path.exists(save_file)
    model = torch.load(save_file)
    model.eval()
    return model
