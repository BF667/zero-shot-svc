<p align="center">
  <h1 align="center">Zero-Shot Singing Voice Conversion</h1>
  <p align="center">
    <strong>RVC-architecture &bull; RMVPE pitch extraction &bull; Dual-mode pipeline &bull; REST API</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch">
    <img src="https://img.shields.io/badge/F0-RMVPE-green.svg" alt="RMVPE">
    <img src="https://img.shields.io/badge/API-FastAPI-009688.svg" alt="FastAPI">
    <img src="https://img.shields.io/badge/UI-Gradio-orange.svg" alt="Gradio">
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

## ✨ What's New in v2.0 (Enhanced Edition)

The project has been significantly enhanced from its original demo/template state into a **fully functional production-ready voice conversion system**:

### 🎯 Key Enhancements

| Feature | Description |
|---------|-------------|
| **Dual-Mode Pipeline** | Signal Processing mode (works instantly, no models) + Neural RVC mode (high quality) |
| **Enhanced Gradio UI** | 4-tab interface: Convert, Batch Processing, History, Guide |
| **8 Built-in Presets** | Male→Female, Female→Male, Octave Shift, Soft Voice, Robot, etc. |
| **Advanced Audio Effects** | Formant shifting, noise reduction, vibrato, breathiness controls |
| **REST API Server** | FastAPI with 9 endpoints for programmatic access |
| **Batch Processing** | Convert multiple files at once with progress tracking |
| **Voice Similarity** | MFCC + spectral analysis for comparing voices |
| **Speaker Profiles** | Save and reuse speaker embeddings |
| **Reference Analyzer** | F0 statistics, vocal range estimation |

---

## How It Works

### Mode 1: Signal Processing (Default, No Models Required)

```
Source Singing Audio + Reference Voice Clip
         │
         ▼
  ┌──────────────┐
  │  F0 Extraction│ ──► Pitch contour (librosa PYIN)
  └──────────────┘
  ┌──────────────┐
  │  Mel Spec     │ ──► Source & reference mel spectrograms
  └──────────────┘
  ┌──────────────┐
  │  MV Norm      │ ──► Transfer reference timbre onto source
  └──────────────┘
  ┌──────────────┐
  │  Griffin-Lim  │ ──► Convert mel back to waveform
  └──────────────┘
  ┌──────────────┐
  │  Effects      │ ──► Pitch shift, formant, noise reduction
  └──────────────┘
         │
         ▼
  Converted Audio (instant, no GPU needed)
```

### Mode 2: Neural/RVC (Requires Pretrained Weights)

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
| **F0 Extractor** | RMVPE / pyworld | Robust vocal pitch estimation with Viterbi smoothing | 1 |
| **Speaker Encoder** | CAM++ | Multi-scale conformer speaker verification | 192 |
| **Generator** | VITS (Posterior Encoder + Flow + Transformer Decoder) | Conditional mel-spectrogram generation with FiLM | 128 mel |
| **Vocoder** | HiFi-GAN / Griffin-Lim | Mel-to-waveform synthesis | 32kHz |

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

**Core Dependencies:**
- `torch >= 2.1.0` — Deep learning framework
- `torchaudio >= 2.1.0` — Audio I/O and transforms
- `librosa >= 0.10.0` — Audio analysis and signal processing
- `numpy >= 1.24.0`, `scipy >= 1.10.0` — Numerical computing
- `soundfile >= 0.12.0` — Audio file I/O
- `transformers >= 4.36.0` — HuggingFace model hub access
- `huggingface_hub >= 0.20.0` — Weight downloading
- `pyworld >= 0.3.0` — High-quality F0 and spectral envelope extraction

**Web UI:**
- `gradio >= 4.0` — Interactive web interface

**API Server (optional):**
- `fastapi >= 0.100.0` — REST API framework
- `uvicorn[standard] >= 0.23.0` — ASGI server
- `python-multipart >= 0.0.6` — File upload support
- `psutil >= 5.9.0` — System monitoring

**Audio Utilities:**
- `audio-separator[gpl]` — Vocal separation (optional)
- `phonemizer`, `pypinyin`, `cn2an`, `jieba_fast` — Text processing (optional)

---

## Quick Start

### 1. Basic Conversion (Signal Processing - No Downloads Needed!)

```bash
# Convert singing voice - works immediately without any model downloads
python main.py convert \
  --source singing.wav \
  --reference target_voice.wav \
  -o converted.wav
```

### 2. With Advanced Options

```bash
# Pitch shift (+12 semitones = 1 octave up) + formant control
python main.py convert \
  -s singing.wav \
  -r target_voice.wav \
  --transpose 12 \
  --formant-shift 2 \
  --noise-reduction 0.3 \
  -o converted_enhanced.wav
```

### 3. Download Weights for Neural Mode (Optional)

```bash
# Download pretrained weights (~2GB total) for high-quality neural conversion
python main.py download

# Check which weights are available
python main.py check
```

### 4. Neural Mode Conversion

```bash
# Use neural pipeline for higher quality output
python main.py convert \
  -s singing.wav \
  -r target_voice.wav \
  --neural \
  --transpose 12 \
  -o converted_neural.wav
```

---

## Gradio Web Interface

Launch an interactive web UI in your browser:

```bash
# Launch locally (default: http://localhost:7860)
python gradio_app.py

# Launch with a public shareable link
python gradio_app.py --share

# Custom port / host
python gradio_app.py --port 8080 --host 127.0.0.1
```

### Web UI Features

The enhanced Gradio demo provides **4 tabs**:

#### 🎤 Tab 1: Convert
- **Drag-and-drop audio upload** for source singing and reference voice
- **Real-time waveform preview** for all audio (input & output)
- **Mode toggle**: Switch between Signal Processing and Neural pipeline
- **Parameter sliders**: Pitch shift, F0 curve, noise scale, formant shift
- **Effect controls**: Noise reduction, breathiness, vibrato strength
- **Preset selector**: 8 built-in presets + custom preset management
- **Progress bar** and status info (conversion time, device, duration)
- **Reference analyzer**: F0 statistics and vocal range estimation

#### 📦 Tab 2: Batch Processing
- Upload multiple source files at once
- Single reference voice for all conversions
- Bulk parameter settings
- Progress tracking with individual file status
- Download all results as ZIP

#### 📜 Tab 3: History
- View past conversions with timestamps
- Replay previous conversions
- Compare settings used
- Quick re-download of outputs

#### 📖 Tab 4: Guide
- Pipeline explanation with diagrams
- Best practices for audio quality
- Parameter tuning recommendations
- Troubleshooting guide

### Built-in Presets

| Preset | Use Case | Settings |
|--------|----------|----------|
| **Male → Female** | Gender transformation | Transpose +5, Formant +2, Breathiness 0.15 |
| **Female → Male** | Gender transformation | Transpose -5, Formant -2, Breathiness 0.1 |
| **Octave Up** | High-pitched effect | Transpose +12, Formant +4 |
| **Octave Down** | Low-pitched effect | Transpose -12, Formant -4 |
| **Soft/Gentle** | Ballad style | Breathiness 0.25, NR 0.2, Noise 0.35 |
| **Robot/Synthetic** | Electronic effect | Noise 0.1, NR 0.5, Formant 0 |
| **Natural** | Minimal processing | All defaults, slight NR 0.1 |
| **Helium** | Chipmunk effect | Transpose +8, Formant +6, Breathiness 0.2 |

---

## REST API Server

Start the FastAPI server for programmatic access:

```bash
# Start API server (default: http://localhost:8000)
python main.py serve

# Custom host/port
python main.py serve --host 0.0.0.0 --port 9000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/convert` | Convert single audio file |
| `POST` | `/api/batch-convert` | Batch convert multiple files |
| `GET` | `/api/download/{output_id}` | Download converted audio |
| `POST` | `/api/similarity` | Compute voice similarity |
| `POST` | `/api/profiles` | Save speaker profile |
| `GET` | `/api/profiles` | List saved profiles |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/status` | System status (models, device, memory) |
| `DELETE` | `/api/cache` | Clear expired cache |

### API Usage Examples

**Convert audio:**
```bash
curl -X POST "http://localhost:8000/api/convert" \
  -F "source=@singing.wav" \
  -F "reference=@target_voice.wav" \
  -F "f0_transpose=12" \
  -F "formant_shift=2" \
  -F "noise_reduction=0.3"
```

**Check health:**
```bash
curl http://localhost:8000/api/health
```

**List profiles:**
```bash
curl http://localhost:8000/api/profiles
```

### Interactive API Documentation

When the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## CLI Reference

```
python main.py <command> [options]

Commands:
  convert       Convert singing voice from source to target speaker
  batch         Batch convert multiple files
  download      Download pretrained model weights
  check         Check which weights are available
  features      Extract and display audio features (F0, content, etc.)
  embedding     Extract speaker embedding from reference audio
  similarity    Compute voice similarity between two files
  profile       Manage speaker profiles (save/list)
  serve         Start REST API server

Global Options:
  --device STR         Device: 'cpu' or 'cuda' (auto-detect)
  --config STR         Path to config YAML file
  --neural             Use neural pipeline (requires pretrained weights)

Convert Options:
  -s, --source         Source audio path (required)
  -r, --reference      Reference audio path — target voice (required)
  -o, --output         Output audio path (auto-generated if omitted)
  --transpose INT      Pitch shift in semitones (default: 0, range: -24 to +24)
  --f0-curve FLOAT     F0 curve scaling factor (default: 1.0)
  --noise-scale FLOAT  Generation noise scale (default: 0.4)
  --formant-shift INT  Formant shift in steps (default: 0, range: -6 to +6)
  --noise-reduction FLOAT  Noise reduction strength (default: 0.0, range: 0-1)
  --breathiness FLOAT  Breathiness effect (default: 0.0, range: 0-1)
  --protect-consonants Protect consonants during pitch shift (default: True)

Batch Options:
  -s, --sources        Source audio paths (multiple, required)
  -r, --reference      Reference audio path (required)
  --output-dir         Output directory (default: converted_outputs)
  --transpose INT      Pitch shift in semitones
  --noise-reduction FLOAT  Noise reduction strength

Serve Options:
  --host               Host to bind (default: 0.0.0.0)
  --port               Port to listen (default: 8000)
```

### CLI Examples

```bash
# Single conversion
python main.py convert -s singing.wav -r voice_ref.wav -o output.wav

# With all enhancements
python main.py convert -s singing.wav -r voice_ref.wav \
  --transpose 12 --formant-shift 2 --noise-reduction 0.3 --breathiness 0.15

# Batch conversion
python main.py batch -s song1.wav song2.wav song3.wav -r voice_ref.wav \
  --output-dir converted/

# Neural mode
python main.py convert -s singing.wav -r voice_ref.wav --neural

# Voice similarity
python main.py similarity voice1.wav voice2.wav

# Speaker profile management
python main.py profile save -r my_voice.wav -n "My Voice"
python main.py profile list

# Start API server
python main.py serve --port 8000

# Download pretrained weights
python main.py download
```

---

## Python API

```python
from pipeline.voice_converter import ZeroShotSVC

# Initialize (signal processing mode - works immediately)
svc = ZeroShotSVC(use_neural=False)

# Or initialize for neural mode (requires downloaded weights)
svc = ZeroShotSVC(use_neural=True, device="cuda")  # or "cpu"
svc.load_models()

# Convert voice with all options
output_path = svc.convert(
    source_path="singing.wav",
    reference_path="target_voice.wav",
    output_path="converted.wav",
    f0_transpose=0,           # semitones (+/-)
    f0_curve_factor=1.0,      # F0 curve scaling
    noise_scale=0.4,          # generation diversity (neural mode)
    formant_shift=0,          # formant shift steps (-6 to +6)
    noise_reduction=0.0,      # noise reduction (0 to 1)
    breathiness=0.0,          # breathiness effect (0 to 1)
    protect_consonants=True,  # consonant protection
    progress_callback=lambda pct, msg: print(f"{pct*100:.0f}%: {msg}"),
)

# Batch convert multiple files
results = svc.batch_convert(
    source_paths=["song1.wav", "song2.wav", "song3.wav"],
    reference_path="target_voice.wav",
    output_dir="converted/",
    f0_transpose=5,
    formant_shift=2,
)

# Compute voice similarity
similarity = svc.compute_similarity("voice1.wav", "voice2.wav")
print(f"Overall similarity: {similarity['overall_similarity']:.3f}")

# Speaker profile management
profile_path = svc.save_speaker_profile("reference.wav", name="My Voice")
profiles = svc.list_speaker_profiles()

# Extract features
features = svc.extract_features("singing.wav")
print(f"F0 range: {features['f0'][features['f0']>0].min():.0f} - "
      f"{features['f0'][features['f0']>0].max():.0f} Hz")

# Extract speaker embedding alone
embedding = svc.extract_speaker_embedding("target_voice.wav")
print(f"Embedding shape: {embedding.shape}")  # (192,)
```

---

## Project Structure

```
zero-shot-svc/
├── main.py                          # Enhanced CLI entry point (9 commands)
├── gradio_app.py                    # Enhanced Gradio web UI (4 tabs)
├── api_server.py                    # FastAPI REST API server (9 endpoints)
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
│   └── voice_converter.py           # Enhanced ZeroShotSVC pipeline
├── weights/
│   └── download_weights.py          # HuggingFace weight downloader
├── utils/
│   ├── __init__.py
│   ├── audio.py                     # Audio I/O, mel spec, resampling
│   └── hparams.py                   # Config dataclass loader
├── resources/                       # Curated resource guides
│   ├── rvc_weights.json             # Model weight sources
│   ├── best_practices.json          # Quality optimization tips
│   ├── deployment.json              # Deployment options
│   ├── community_tutorials.json     # Learning resources
│   ├── datasets_audio.json          # Audio datasets
│   └── model_components.json        # Technical documentation
├── scripts/
│   ├── test_pipeline.py             # Original component tests
│   ├── test_enhanced.py             # Enhanced feature tests
│   └── validate_basic.py            # Quick validation suite
├── presets.json                     # User custom presets storage
├── RESOURCE_GUIDE.md                # Comprehensive resource guide
├── requirements.txt                 # Dependencies (updated)
├── LICENSE                          # MIT + BSD License
└── README.md                        # This file
```

---

## Tips for Best Results

### Audio Preparation Guidelines

#### Reference Audio (Target Voice)
- ✅ **Duration**: 5-15 seconds optimal
- ✅ **Quality**: Clean, dry recording (no reverb/effects)
- ✅ **Content**: Natural speech or sustained singing
- ✅ **Format**: WAV, 16kHz+ sample rate, mono
- ❌ Avoid: Background noise, music, heavy compression

#### Source Audio (Singing to Convert)
- ✅ **Preprocessing**: Isolate vocals first using UVR/MDX-Net
- ✅ **Quality**: Higher bitrate = better results
- ✅ **Length**: Works with any length (auto-chunked at 30s)
- ⚠️ **Note**: Instrumental bleed reduces conversion quality

### Parameter Tuning Guide

| Scenario | F0 Transpose | Formant Shift | Noise Scale | Mode |
|----------|--------------|---------------|-------------|------|
| Male → Female | +5 to +12 | +2 to +4 | 0.35 | Neural |
| Female → Male | -5 to -12 | -2 to -4 | 0.45 | Neural |
| Same gender clone | 0 to ±3 | 0 to ±1 | 0.4 | Neural |
| Quick test (no GPU) | Any | Any | N/A | Signal |
| Robot/Synthetic effect | 0 | 0 | 0.1 | Neural |
| Soft/Gentle voice | +2 | +1 | 0.3 | Neural |
| Helium/Chipmunk | +8 to +12 | +4 to +6 | 0.4 | Either |
| Deep/Voice | -8 to -12 | -4 to -6 | 0.45 | Either |

### Post-Processing Tips

> "The key to achieving natural AI vocals starts with selecting the cleanest possible vocal track. You should aim for dry, studio-quality acapella."
> — *r/RVCAdepts Community*

**Recommended workflow:**
1. Separate vocals from accompaniment (UVR5 / MDX-Net)
2. Apply light noise reduction to isolated vocals
3. Use appropriate pitch/formant settings for target gender
4. Review output and adjust parameters iteratively
5. Apply final mastering (light compression, EQ)

---

## Technical Details

### F0 Extraction
- **Signal Mode**: librosa PYIN algorithm with Viterbi smoothing
- **Neural Mode**: RMVPE (Residual CNN + Centroid Viterbi decoding)
- Frequency range: 50-1100 Hz (covers bass to soprano singing)
- Frame-level fundamental frequency estimation

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

### Audio Effects (Enhanced)
- **Formant Shifting**: Independent control of vocal timbre without changing pitch
- **Noise Reduction**: Spectral gating for background noise removal
- **Breathiness**: Synthetic breath sound addition for naturalness
- **Vibrato**: Periodic pitch modulation for expressive singing

### Generation (VITS + HiFi-GAN)
- Posterior Encoder: 16-layer WaveNet with dilated convolutions
- Normalizing Flow: 4 affine coupling layers for latent normalization
- Decoder: 6-layer Transformer with FiLM conditioning
- HiFi-GAN: 4 upsampling blocks (×8, ×8, ×2, ×2) with MRF fusion
- **Fallback**: Griffin-Lim vocoder for signal processing mode

---

## Colab & Cloud

### Run in Browser (Gradio)

No installation needed — launch the demo and open it in your browser:

```bash
python gradio_app.py --share
```

### Run on Colab

Open [**zero_shot_svc_colab.ipynb**](https://colab.research.google.com/github/BF667/zero-shot-svc/blob/main/zero_shot_svc_colab.ipynb) in Google Colab for a ready-to-run notebook with GPU support.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667/zero-shot-svc/blob/main/zero_shot_svc_colab.ipynb)

### Cloud GPU Services (for Neural Mode)

| Provider | Free Tier | GPU Options | Cost/Hour |
|----------|-----------|-------------|-----------|
| **[RunPod](https://www.runpod.io/articles/guides/deploy-fastapi-applications-gpu-cloud)** | Yes | RTX 4090, A100 | $0.44-$0.79 |
| **[Modal](https://modal.com/blog/how_to_run_gradio_on_modal_article)** | $30 credit | A10G, T4 | Auto-scale |
| **[HuggingFace Spaces](https://huggingface.co/new-space)** | Free (limited) | T4 (free tier) | Free/$$$ |
| **[Google Cloud Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/rapidly-build-an-application-in-gradio-power-by-a-generative-ai-agent)** | $300 credit | T4, L4, A100 | Variable |

### Docker Deployment

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg libsndfile1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 7860 8000

CMD ["python", "gradio_app.py", "--host", "0.0.0.0"]
```

---

## Testing & Validation

Run the validation suite to verify your installation:

```bash
# Quick validation (6 basic tests)
python scripts/validate_basic.py

# Enhanced feature tests
python scripts/test_enhanced.py

# Original pipeline tests
python scripts/test_pipeline.py
```

### Test Coverage

The validation suite checks:
- ✅ File structure integrity
- ✅ Config loading and validation
- ✅ Audio utility functions
- ✅ Preset system functionality
- ✅ CLI command parsing
- ✅ Gradio UI initialization
- ✅ API server structure
- ✅ Import dependencies

---

## Related Work & Resources

### Research Papers
- [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Boss/Retrieval-based-Voice-Conversion-WebUI) — Original architecture
- [RMVPE: A Robust Model for Vocal Pitch Estimation](https://arxiv.org/abs/2306.15412) — Pitch extraction method
- [VITS: Conditional Variational Autoencoder](https://arxiv.org/abs/2106.06103) — Generative backbone
- [HiFi-GAN: Generative Adversarial Networks](https://arxiv.org/abs/2010.05646) — Vocoder
- [CAM++: A Fast and Efficient Speaker Verification Model](https://arxiv.org/abs/2210.16711) — Speaker encoder
- [HuBERT: Self-Supervised Speech Representation](https://arxiv.org/abs/2106.07447) — Content encoder basis
- [HQ-SVC: Towards High-Quality Zero-Shot SVC](https://arxiv.org/html/2511.08496v1) — Latest advances (2024)
- [YingMusic-SVC: Real-World Robust Zero-Shot SVC](https://arxiv.org/html/2512.04793v1) — Robust deployment strategies

### Open-Source Projects
- **[seed-vc](https://github.com/Plachtaa/seed-vc)** — High-quality zero-shot conversion
- **[so-vits-svc 4.0](https://github.com/justinjohn0306/so-vits-svc-4.0-v2)** — Popular singing voice conversion
- **[RVC WebUI Official](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)** — Original RVC implementation

### Community Resources
- **[r/RVCAdepts (Reddit)](https://www.reddit.com/r/RVCAdepts/)** — Active community, tips & tricks
- **[RVC Discord](https://discord.gg/rvc)** — Real-time help and model sharing
- **[HuggingFace Models](https://huggingface.co/models?other=rvc)** — Community-trained models

### Model Weight Sources
- **ContentVec**: [lengyue233/content-vec-best](https://huggingface.co/lengyue233/content-vec-best)
- **RMVPE/HiFi-GAN**: [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)
- **CAM++**: [funasr/cam++](https://huggingface.co/funasr/cam%2B%2B)
- **RVC2 Models**: [takearushfan/rvc2-models](https://huggingface.co/collections/takearushfan/rvc2-models)
- **TITAN (High-Quality)**: [blaise-tk/TITAN](https://huggingface.co/blaise-tk/TITAN)

For a comprehensive resource guide, see [RESOURCE_GUIDE.md](./RESOURCE_GUIDE.md).

---

## License

This project is dual-licensed under:

MIT, AND BSD 3-Clause "New" or "Revised" License

See [LICENSE](./LICENSE) and [LICENCE.MD](./LICENCE.MD) for details.

---

## Acknowledgments

- RVC community for the architecture design and pre-trained models
- ContentVec weights from [lengyue233](https://huggingface.co/lengyue233/content-vec-best)
- RMVPE weights from [lj1995](https://huggingface.co/lj1995/VoiceConversionWebUI)
- CAM++ weights from [FunASR](https://huggingface.co/funasr/cam%2B%2B)
- Gradio team for the excellent UI framework
- FastAPI team for the modern API framework

---

## Changelog

### v2.0.0 - Enhanced Edition (Current)
- ✅ Added dual-mode pipeline (Signal Processing + Neural)
- ✅ Enhanced Gradio UI with 4 tabs (Convert, Batch, History, Guide)
- ✅ Added 8 built-in presets + custom preset management
- ✅ Added advanced audio effects (formant, NR, vibrato, breathiness)
- ✅ Added FastAPI REST API server with 9 endpoints
- ✅ Added batch processing support
- ✅ Added voice similarity scoring
- ✅ Added speaker profile management
- ✅ Added reference analyzer with F0 statistics
- ✅ Added comprehensive RESOURCE_GUIDE.md
- ✅ Updated dependencies (fastapi, uvicorn, pyworld, psutil)
- ✅ Added validation test suite

### v1.0.0 - Original Release
- Base RVC pipeline implementation
- ContentVec + RMVPE + CAM++ + VITS + HiFi-GAN
- Basic Gradio demo
- CLI interface with convert/download/check commands

---

## Support

- **Issues**: [GitHub Issues](https://github.com/BF667/zero-shot-svc/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BF667/zero-shot-svc/discussions)
- **Resource Guide**: [RESOURCE_GUIDE.md](./RESOURCE_GUIDE.md)

---

<p align="center">
  <strong>⭐ If this project helps you, consider giving it a star! ⭐</strong>
</p>
