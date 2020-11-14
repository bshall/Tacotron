<p align="center">
    <a href="https://colab.research.google.com/github/bshall/Tacotron/blob/main/tacotron-demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</p>

# Tacotron (with Dynamic Convolution Attention)

A PyTorch implementation of [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288). Audio samples can be found [here](bshall.github.io/tacotron/). Colab demo can be found [here](https://colab.research.google.com/github/bshall/Tacotron/blob/main/tacotron-demo.ipynb).

<div align="center">
    <img width="655" height="390" alt="Tacotron (with Dynamic Convolution Attention)" 
      src="https://raw.githubusercontent.com/bshall/Tacotron/main/tacotron.png"><br>
    <sup><strong>Fig 1:</strong>Tacotron (with Dynamic Convolution Attention).</sup>
</div>

<div align="center">
    <img width="897" height="154" alt="Example Mel-spectrogram and attention plot" 
      src="https://raw.githubusercontent.com/bshall/Tacotron/main/example.png"><br>
    <sup><strong>Fig 2:</strong>Example Mel-spectrogram and attention plot.</sup>
</div>

## Quick Start

Ensure you have Python 3.6 and PyTorch 1.7 or greater installed. Then install this package with:
```
pip install tacotron
```

## Example Usage

<p align="center">
    <a href="https://colab.research.google.com/github/bshall/Tacotron/blob/main/tacotron-demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</p>


```python
import torch
import soundfile as sf
from univoc import Vocoder
from tacotron import load_cmudict, text_to_id, Tacotron

# download pretrained weights for the vocoder (and optionally move to GPU)
vocoder = Vocoder.from_pretrained(
    "https://github.com/bshall/UniversalVocoding/releases/download/v0.2/univoc-ljspeech-7mtpaq.pt"
).cuda()

# download pretrained weights for tacotron (and optionally move to GPU)
tacotron = Tacotron.from_pretrained(
    "https://github.com/bshall/Tacotron/releases/download/v0.1/tacotron-ljspeech-yspjx3.pt"
).cuda()

# load cmudict and add pronunciation of PyTorch
cmudict = load_cmudict()
cmudict["PYTORCH"] = "P AY1 T AO2 R CH"

text = "A PyTorch implementation of Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis."

# convert text to phone ids
x = torch.LongTensor(text_to_id(text, cmudict)).unsqueeze(0).cuda()

# synthesize audio
with torch.no_grad():
    mel, _ = tacotron.generate(x)
    wav, sr = vocoder.generate(mel.transpose(1, 2))

# save output
sf.write("location_relative_attention.wav", wav, sr)
```

## Train from Scatch

1. Clone the repo:
```
git clone https://github.com/bshall/Tacotron
cd ./Tacotron
```
2. Install requirements:
```
pip install -r requirements.txt
```
3. Download and extract the [LJ-Speech dataset](https://keithito.com/LJ-Speech-Dataset/):
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvjf LJSpeech-1.1.tar.bz2
```
4. Download the train split [here](https://github.com/bshall/Tacotron/releases/tag/v0.1) and extract it in the root directory of the repo.  
5. Extract Mel spectrograms and preprocess audio:
```
python preprocess.py in_dir=path/to/LJSpeech-1.1 out_dir=datasets/LJSpeech-1.1
```
6. Train the model:
```
python train.py checkpoint_dir=ljspeech dataset_dir=datasets/LJSpeech-1.1 text_dir=path/to/LJSpeech-1.1/metadata.csv
```

## Pretrained Models

Pretrained weights for the LJSpeech model are available [here](https://github.com/bshall/Tacotron/releases/tag/v0.1).

## Notable Differences from the Paper

1. Trained using a batch size of 64 on a single GPU (using automatic mixed precision).
2. Used a gradient clipping threshold of 0.05 as it seems to stabilize the alignment with the smaller batch size.
3. Used a different learning rate schedule (again to deal with smaller batch size).
4. Used 80-bin (instead of 128 bin) log-Mel spectrograms.

## Acknowlegements

- https://github.com/keithito/tacotron
- https://github.com/PetrochukM/PyTorch-NLP
- https://github.com/fatchord/WaveRNN