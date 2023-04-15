
## Requirements
```
NVIDIA CUDA 11.8
pytorch 2.0
```

## Setup
```
conda create -n quantize-torch-models python==3.10
conda activate quantize-torch-models
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements.txt
```