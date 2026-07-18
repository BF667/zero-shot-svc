<p align="center">
  <h1 align="center">Zero-Shot Singing Voice Conversion</h1>
  <p align="center">
    <strong>RVC-architecture &bull; RMVPE pitch extraction &bull; No training required</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch">
    <img src="https://img.shields.io/badge/F0-RMVPE-green.svg" alt="RMVPE">
    <img src="https://img.shields.io/badge/License-MIT%20%2B%20Custom-yellow.svg" alt="License">
  </p>
  <!-- Stargazer -->
  <p align="center">
    <a href="https://github.com/BF667/zero-shot-svc/stargazers">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=BF667/zero-shot-svc&type=Date&theme=dark" />
        <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=BF667/zero-shot-svc&type=Date" />
        <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=BF667/zero-shot-svc&type=Date" />
      </picture>
    </a>
  </p>
  <!-- Stargazers button -->
  <p align="center">
    <a href="https://github.com/BF667/zero-shot-svc">
      <img src="https://img.shields.io/github/stars/BF667/zero-shot-svc?style=social" alt="GitHub Stars">
    </a>
    &nbsp;
    <a href="https://github.com/BF667/zero-shot-svc/fork">
      <img src="https://img.shields.io/github/forks/BF667/zero-shot-svc?style=social" alt="GitHub Forks">
    </a>
    &nbsp;
    <a href="https://github.com/BF667/zero-shot-svc/watchers">
      <img src="https://img.shields.io/github/watchers/BF667/zero-shot-svc?style=social" alt="GitHub Watchers">
    </a>
  </p>
  <p align="center">
    <a href="https://github.com/BF667/zero-shot-svc/stargazers">
      <img src="https://reporoster.com/stars/dark/BF667/zero-shot-svc" alt="Stargazers" width="400" />
    </a>
  </p>
</p>

---

Convert any singing voice to sound like a target speaker using only a short reference audio clip (5-15 seconds). **No training, no fine-tuning** — just provide a source singing track and a reference voice sample.

This system implements the [RVC (Retrieval-based Voice Conversion) v2](https://github.com/RVC-Boss/Retrieval-based-Voice-Conversion-WebUI) pipeline in a clean, modular Python package with **RMVPE** as the default F0 pitch extraction method.

---

## How It Works

```
Source Singing Audio + Reference Voice Clip
         │
         ▼
  ┌──────────────┐
  │  ContentVec   │ ──► Content features (WHAT is sung — lyrics, phonemes)
  │  (HuBERT var) │     256-dim speaker-invariant representation
  └──────────────┘
  ┌──────────────┐
  │    RMVPE      │ ──► F0 pitch contour (the melody)
  │  (Pitch F0)   │     Frame-level fundamental frequency
  └──────────────┘
  ┌──────────────┐
  │   CAM++      │ ──► Speaker embedding (WHO to sound like)
  │  (Speaker ID) │     192-dim voice characteristics vector
  └──────────────┘
         │
         ▼
  ┌──────────────┐
  │  VITS Gen.   │ ──► Mel-spectrogram (content + F0 + speaker → mel)
  │  (FiLM cond.) │     Conditional generation with normalizing flows
  └──────────────┘
  ┌──────────────┐
  │   HiFi-GAN   │ ──► Waveform (mel → audio)
  │  (Vocoder)   │     High-fidelity 32kHz output
  └──────────────┘
         │
         ▼
  Converted Audio (singing in target voice, preserving melody & lyrics)
```

### Architecture Components

| Component | Model | Purpose | Dim |
|-----------|-------|---------|-----|
| **Content Encoder** | ContentVec (HuBERT-large variant) | Extract speaker-invariant content features | 256 |
| **F0 Extractor** | RMVPE | Robust vocal pitch estimation with Viterbi smoothing | 1 |
| **Speaker Encoder** | CAM++ | Multi-scale conformer speaker verification | 192 |
| **Generator** | VITS (Posterior Encoder + Flow + Transformer Decoder) | Conditional mel-spectrogram generation with FiLM | 128 mel |
| **Vocoder** | HiFi-GAN | Mel-to-waveform synthesis | 32kHz |

### Key Features

- **Zero-shot**: No training or fine-tuning needed — works with any voice
- **RMVPE F0**: State-of-the-art pitch extraction, robust to noise and harmonics
- **Consonant protection**: Smooth F0 transitions near unvoiced regions to avoid artifacts
- **Chunked processing**: Handles arbitrarily long audio by processing in 30s chunks
- **F0 transposition**: Shift pitch by any number of semitones
- **Modular design**: Each component is independent and swappable

---

## Installation

```bash
# Clone the repository
git clone https://github.com/BF667/zero-shot-svc.git
cd zero-shot-svc

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `torch >= 2.1.0` — Deep learning framework
- `torchaudio >= 2.1.0` — Audio I/O and transforms
- `librosa >= 0.10.0` — Audio analysis
- `transformers >= 4.36.0` — HuggingFace model hub access
- `huggingface_hub >= 0.20.0` — Weight downloading
- `scipy`, `numpy`, `soundfile` — Numerical and audio utilities

---

## Quick Start

```bash
# 1. Download pretrained weights (first run only, ~2GB)
python main.py download

# 2. Convert singing voice
python main.py convert \
  --source singing.wav \
  --reference target_voice.wav \
  -o converted.wav

# With pitch shift (+12 semitones = 1 octave up)
python main.py convert \
  -s singing.wav \
  -r target_voice.wav \
  --transpose 12 \
  -o converted_high.wav
```

### Python API

```python
from pipeline.voice_converter import ZeroShotSVC

# Initialize (downloads models on first run)
svc = ZeroShotSVC()
svc.load_models()

# Convert voice
output_path = svc.convert(
    source_path="singing.wav",
    reference_path="target_voice.wav",
    output_path="converted.wav",
    f0_transpose=0,       # semitones (+/-)
    noise_scale=0.4,      # generation diversity
)

# Extract speaker embedding alone
embedding = svc.extract_speaker_embedding("target_voice.wav")
print(f"Embedding shape: {embedding.shape}")  # (192,)

# Analyze audio features
features = svc.extract_features("singing.wav")
print(f"F0 range: {features['f0'][features['f0']>0].min():.0f} - "
      f"{features['f0'][features['f0']>0].max():.0f} Hz")
```

---

## CLI Reference

```
python main.py <command> [options]

Commands:
  convert       Convert singing voice from source to target speaker
  download      Download pretrained model weights
  check         Check which weights are available
  features      Extract and display audio features (F0, content, etc.)
  embedding     Extract speaker embedding from reference audio

Convert Options:
  -s, --source         Source audio path (required)
  -r, --reference      Reference audio path — target voice (required)
  -o, --output         Output audio path (auto-generated if omitted)
  --transpose INT      Pitch shift in semitones (default: 0)
  --f0-curve FLOAT     F0 curve scaling factor (default: 1.0)
  --noise-scale FLOAT  Generation noise scale (default: 0.4)

Global Options:
  --device STR         Device: 'cpu' or 'cuda' (auto-detect)
  --config STR         Path to config YAML file
```

---

## Project Structure

```
zero-shot-svc/
├── main.py                          # CLI entry point
├── configs/
│   └── default.yaml                 # Default configuration
├── models/
│   ├── __init__.py
│   ├── content_encoder.py           # ContentVec (HuBERT) — content features
│   ├── f0_extractor.py              # RMVPE — pitch extraction
│   ├── speaker_encoder.py           # CAM++ — speaker embedding
│   ├── generator.py                 # VITS generator — mel synthesis
│   └── vocoder.py                   # HiFi-GAN vocoder — waveform synthesis
├── pipeline/
│   ├── __init__.py
│   └── voice_converter.py           # Main ZeroShotSVC pipeline
├── weights/
│   └── download_weights.py          # HuggingFace weight downloader
├── utils/
│   ├── __init__.py
│   ├── audio.py                     # Audio I/O, mel spec, resampling
│   └── hparams.py                   # Config dataclass loader
├── scripts/
│   └── test_pipeline.py             # Component test suite
├── requirements.txt
└── README.md
```

---

## Tips for Best Results

1. **Reference audio quality**: Use clean, dry (no reverb/effects) recordings. 5-15 seconds is ideal — enough to capture voice characteristics but not so long that noise accumulates.

2. **Source audio**: Works best with isolated vocals. If your source has instrumental accompaniment, consider using a vocal separator first (e.g., `audio-separator`).

3. **Pitch matching**: If the source singer has a significantly different vocal range than the target, use `--transpose` to adjust. Male→Female typically needs +5 to +12 semitones; Female→Male needs -5 to -12.

4. **Reference selection**: The reference should contain natural speech or singing in the target voice. Multiple references can be concatenated for better speaker representation.

5. **Long audio**: The pipeline automatically chunks audio into 30-second segments. Overlaps are handled to avoid artifacts at chunk boundaries.

---

## Technical Details

### F0 Extraction (RMVPE)
- Residual CNN backbone with 6 blocks
- Centroid-based Viterbi decoding for continuous pitch tracking
- Viterbi smoothing to eliminate octave jumps
- Frequency range: 50-1100 Hz (covers bass to soprano singing)

### Content Encoding (ContentVec)
- 7-layer CNN frontend with causal convolutions
- 4-layer Transformer encoder (12 heads, 768-dim)
- 256-dim linear projection output
- Trained on 10,000+ hours of speech (speaker-invariant by design)

### Speaker Embedding (CAM++)
- Multi-scale CNN feature extraction
- 6 Conformer blocks with self-attention
- Attentive temporal pooling
- 192-dim L2-normalized embedding
- State-of-the-art on VoxCeleb speaker verification

### Generation (VITS + HiFi-GAN)
- Posterior Encoder: 16-layer WaveNet with dilated convolutions
- Normalizing Flow: 4 affine coupling layers for latent normalization
- Decoder: 6-layer Transformer with FiLM (Feature-wise Linear Modulation) conditioning
- HiFi-GAN: 4 upsampling blocks (×8, ×8, ×2, ×2) with multi-receptive-field fusion

---

## Colab

Open [**zero_shot_svc_colab.ipynb**](https://colab.research.google.com/github/BF667/zero-shot-svc/blob/main/zero_shot_svc_colab.ipynb) in Google Colab for a ready-to-run notebook with GPU support. No local installation needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667/zero-shot-svc/blob/main/zero_shot_svc_colab.ipynb)

---

## Related Work

- [RVC (Retrieval-based Voice Conversion) WebUI](https://github.com/RVC-Boss/Retrieval-based-Voice-Conversion-WebUI) — Original RVC implementation
- [RMVPE: A Robust Model for Vocal Pitch Estimation](https://arxiv.org/abs/2306.15412) — Pitch extraction method
- [VITS: Conditional Variational Autoencoder](https://arxiv.org/abs/2106.06103) — Generative backbone
- [HiFi-GAN: Generative Adversarial Networks](https://arxiv.org/abs/2010.05646) — Vocoder
- [CAM++: A Fast and Efficient Speaker Verification Model](https://arxiv.org/abs/2210.16711) — Speaker encoder
- [HuBERT: Self-Supervised Speech Representation](https://arxiv.org/abs/2106.07447) — Content encoder basis

---

## License

This project is dual-licensed under:

### MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### BF667 Guide Licence

```
BF667 Guide Licence

Copyright (c) 2024 BF667

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions, and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions, and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Acknowledgments

- RVC community for the architecture design and pre-trained models
- ContentVec weights from [lengyue233](https://huggingface.co/lengyue233/content-vec-best)
- RMVPE weights from [lj1995](https://huggingface.co/lj1995/VoiceConversionWebUI)
- CAM++ weights from [FunASR](https://huggingface.co/funasr/cam%2B%2B)