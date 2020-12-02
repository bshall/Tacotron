from pathlib import Path

import soundfile as sf
import torch
from tacotron import Tacotron, load_cmudict, text_to_id
from univoc import Vocoder

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
with open("potterdict.txt") as file:
    potterdict = (line.strip().split("  ") for line in file)
    potterdict = {word: pronunciation for word, pronunciation in potterdict}
cmudict.update(potterdict)

samples_path = Path("samples")
samples_path.mkdir(exist_ok=True)

with open("sentences.txt") as file:
    for i, line in enumerate(file):
        # convert text to phone ids
        x = torch.LongTensor(text_to_id(line.strip(), cmudict)).unsqueeze(0).cuda()

        # synthesize audio
        with torch.no_grad():
            mel, _ = tacotron.generate(x)
            wav, sr = vocoder.generate(mel.transpose(1, 2))

        # save output
        sf.write(samples_path / f"dca_{i}.wav", wav, sr)
