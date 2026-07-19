# 🚀 Zero-Shot SVC - Resource Guide for Success

This document contains curated resources to help you maximize the success of your Zero-Shot Singing Voice Conversion project.

---

## 📦 **1. Pretrained Model Weights & Downloads**

### **Essential Model Components**

| Component | Source | Download Link | Size |
|-----------|--------|---------------|------|
| **RMVPE** (Pitch Extraction) | lj1995/VoiceConversionWebUI | [HuggingFace](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) | ~200MB |
| **ContentVec** (Content Encoder) | lengyue233/content-vec-best | [HuggingFace](https://huggingface.co/lengyue233/content-vec-best) | ~500MB |
| **CAM++** (Speaker Encoder) | funasr/cam++ | [HuggingFace](https://huggingface.co/funasr/cam%2B%2B) | ~100MB |
| **HiFi-GAN Vocoder** | lj1995/VoiceConversionWebUI | [HuggingFace](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hifigan_v2.pt) | ~180MB |

### **RVC v2 Pretrained Base Models**
- **RVC2 Models Collection**: [takearushfan/rvc2-models](https://huggingface.co/collections/takearushfan/rvc2-models)
- **TITAN (High-Quality RVC)**: [blaise-tk/TITAN](https://huggingface.co/blaise-tk/TITAN)
- **Complete RVC Resources**: [Politrees/RVC_resources](https://huggingface.co/Politrees/RVC_resources)

### **Quick Download Command**
```bash
# Download all weights at once
python main.py download

# Or manually download specific components
pip install huggingface_hub
huggingface-cli download lj1995/VoiceConversionWebUI rmvpe.pt --local-dir ./weights/
huggingface-cli download lengyue233/content-vec-best pytorch_model.bin --local-dir ./weights/
```

---

## 📚 **2. Research Papers & State-of-the-Art Methods**

### **Key Papers for Understanding**

1. **[RMVPE: A Robust Model for Vocal Pitch Estimation](https://huggingface.co/papers/2306.15412)** 
   - The pitch extraction method used in this project
   - Superior to CREPE and PYIN for polyphonic music

2. **[HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion](https://arxiv.org/html/2511.08496v1)**
   - Latest advances in zero-shot SVC (2024)
   - Techniques for improving quality

3. **[YingMusic-SVC: Real-World Robust Zero-Shot SVC](https://arxiv.org/html/2512.04793v1)**
   - Robust deployment strategies
   - Flow-matching approaches

4. **[Vec-Tok-VC+: Residual-enhanced Robust Zero-shot VC](https://www.isca-archive.org/interspeech_2024/ma24e_interspeech.pdf)**
   - Advanced voice conversion techniques
   - Residual vector quantization

### **Related Open-Source Projects to Study**
- **[seed-vc (Zero-shot VC)](https://github.com/Plachtaa/seed-vc)** - High-quality zero-shot conversion
- **[so-vits-svc 4.0](https://github.com/justinjohn0306/so-vits-svc-4.0-v2)** - Popular singing voice conversion
- **[RVC WebUI Official](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)** - Original RVC implementation

---

## 🎯 **3. Best Practices for Quality Results**

### **Audio Preparation Guidelines**

#### **Reference Audio (Target Voice)**
- ✅ **Duration**: 5-15 seconds optimal
- ✅ **Quality**: Clean, dry recording (no reverb/effects)
- ✅ **Content**: Natural speech or sustained singing
- ✅ **Format**: WAV, 16kHz+ sample rate, mono
- ❌ Avoid: Background noise, music, heavy compression

#### **Source Audio (Singing to Convert)**
- ✅ **Preprocessing**: Isolate vocals first using UVR/MDX-Net
- ✅ **Quality**: Higher bitrate = better results
- ✅ **Length**: Works with any length (auto-chunked at 30s)
- ⚠️ **Note**: Instrumental bleed reduces conversion quality

### **Parameter Tuning Guide**

| Scenario | F0 Transpose | Formant Shift | Noise Scale | Mode |
|----------|--------------|---------------|-------------|------|
| Male → Female | +5 to +12 | +2 to +4 | 0.35 | Neural |
| Female → Male | -5 to -12 | -2 to -4 | 0.45 | Neural |
| Same gender clone | 0 to ±3 | 0 to ±1 | 0.4 | Neural |
| Quick test (no GPU) | Any | Any | N/A | Signal |
| Robot/Synthetic effect | 0 | 0 | 0.1 | Neural |
| Soft/Gentle voice | +2 | +1 | 0.3 | Neural |

### **Post-Processing Tips from Community**
> "The key to achieving natural AI vocals starts with selecting the cleanest possible vocal track. You should aim for dry, studio-quality acapella."
> — *r/RVCAdepts Community*

---

## 🔧 **4. Audio Preprocessing Tools**

### **Vocal Separation (Recommended Before Conversion)**

| Tool | Type | Difficulty | Quality |
|------|------|------------|---------|
| **[UVR5 (Ultimate Vocal Remover)](https://github.com/Anjok07/ultimatevocalremovergui)** | GUI | Easy | ★★★★★ |
| **MDX-Net Models** | Python | Medium | ★★★★☆ |
| **Spleeter (Deezer)** | Python/CLI | Easy | ★★★☆☆ |
| **audio-separator** | Python | Easy | ★★★★☆ |

### **Installation Example**
```bash
# Install UVR5 for vocal isolation
pip install ultimatevocalremovergui

# Or use Python library directly
pip install audio-separator[gpl]
from audio_separator import Separator
separator = Separator()
separator.separate("song_with_vocals.mp3")
```

### **Audio Datasets for Testing**
- **[AMAAI Audio Datasets List](https://github.com/AMAAI-Lab/ai-audio-datasets-list)** - Comprehensive collection
- **[UR-Sing Dataset](https://zenodo.org/records/6404999)** - Solo singing performances
- **[Sonovox Ultimate Vocal Dataset](https://sonovox.ai/products/the-ultimate-vocal-dataset)** - Professional vocals (licensed)

---

## ☁️ **5. Deployment Options**

### **Cloud GPU Services (for Neural Mode)**

| Provider | Free Tier | GPU Options | Cost/Hour |
|----------|-----------|-------------|-----------|
| **[RunPod](https://www.runpod.io/articles/guides/deploy-fastapi-applications-gpu-cloud)** | Yes | RTX 4090, A100 | $0.44-$0.79 |
| **[Modal](https://modal.com/blog/how_to_run_gradio_on_modal_article)** | $30 credit | A10G, T4 | Auto-scale |
| **[GigaGPU](https://gigagpu.com/gradio-ai-demo-dedicated-gpu)** | No | Various | Competitive |
| **[Google Cloud Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/rapidly-build-an-application-in-gradio-power-by-a-generative-ai-agent)** | $300 credit | T4, L4, A100 | Variable |
| **[HuggingFace Spaces](https://huggingface.co/new-space)** | Free (limited) | T4 (free tier) | Free/$$$ |

### **Docker Deployment Template**
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose ports
EXPOSE 7860 8000

# Run Gradio by default
CMD ["python", "gradio_app.py", "--host", "0.0.0.0"]
```

### **FastAPI + Gradio Combined Deployment**
```python
# Deploy both API and UI together
from fastapi import FastAPI
import gradio as gr

app = FastAPI()

@app.get("/api/health")
def health():
    return {"status": "healthy"}

# Mount Gradio as a subpath
demo = gr.mount_gradio_app(app, blocks=build_ui(), path="/gradio")
```

---

## 🎓 **6. Learning Resources & Tutorials**

### **Video Tutorials**
- **[RVC V2 Tutorial - YouTube](https://www.youtube.com/watch?v=5i_Pyw0gH-M)** - Complete RVC walkthrough
- **[Updated RVC Real-Time Tutorial 2024](https://www.youtube.com/watch?v=wkvHJ6LebD4)** - Latest features
- **[UVR5 Vocal Separation Tutorial](https://www.youtube.com/watch?v=9kzlr6otFqU)** - Audio preprocessing

### **Written Guides**
- **[What is RVC? - HuggingFace Blog](https://huggingface.co/blog/Blane187/what-is-rvc)** - Conceptual overview
- **[RVC on Grokipedia](https://grokipedia.com/page/retrieval_based_voice_conversion)** - Community knowledge base
- **[SoftVC VITS Deep Dive](https://techshinobi.org/posts/voice-vits)** - Technical details
- **[Gradio Deployment Guide](https://gradio.app/guides/deploying-gradio-with-modal)** - Hosting instructions

### **Community Forums**
- **[r/RVCAdepts (Reddit)](https://www.reddit.com/r/RVCAdepts/)** - Active RVC community, tips & tricks
- **[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1gj14oa/best_open_source_voice_cloning_if_you_have_lots)** - Voice cloning discussions
- **[GitHub Discussions - RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)** - Official support

---

## 🔬 **7. Research & Development Resources**

### **Papers on Hugging Face**
- Browse latest: [huggingface.co/papers?q=Voice+conversion](https://huggingface.co/papers?q=Voice%20conversion)
- Daily papers feed available

### **Benchmark Datasets**
- **MUSDB18** - Music separation benchmark
- **DSynth100** - Singing voice dataset
- **VCTK** - Multi-speaker corpus (for testing speaker encoder)

### **Evaluation Metrics to Track**
```python
# Recommended metrics for your project:
metrics = {
    "quality": ["MOS (Mean Opinion Score)", "PESQ", "STOI"],
    "similarity": ["Speaker Embedding Cosine Similarity", "MFCC Distance"],
    "pitch": ["F0 RMSE", "V/UV Error Rate"],
    "speed": ["RTF (Real-Time Factor)", "Conversion time per second"],
}
```

---

## 💡 **8. Success Strategies**

### **For Production Use**
1. **Start with Signal Processing mode** - No downloads needed, instant results
2. **Use GPU for Neural mode** - 10-50x faster inference
3. **Always preprocess audio** - Separate vocals first for best quality
4. **Batch process when possible** - Use the batch conversion feature
5. **Cache speaker profiles** - Save reference embeddings for reuse

### **For Development/Research**
1. **Study the pipeline components** - Each module is independently swappable
2. **Experiment with parameters** - Use presets as starting points
3. **Compare with baselines** - Test against original RVC implementation
4. **Contribute improvements** - The modular design makes it easy to add features

### **Performance Optimization Tips**
```bash
# Enable CUDA optimizations
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"  # For various GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use half-precision for faster inference
# In config or code:
model.half()  # Reduces memory usage by ~50%
```

---

## 🌐 **9. Community & Support**

### **Official Channels**
- **GitHub Issues**: [BF667/zero-shot-svc](https://github.com/BF667/zero-shot-svc/issues)
- **RVC Discord**: Search "RVC Discord" for active community
- **HuggingFace Models**: Share your fine-tuned models

### **Model Sharing**
Once you have good results, consider:
1. Sharing speaker profiles on HuggingFace
2. Publishing demo recordings
3. Contributing preset configurations back to the project

---

## 📊 **10. Quick Reference Card**

```
┌─────────────────────────────────────────────────────────────┐
│           ZERO-SHOT SVC QUICK REFERENCE                      │
├─────────────────────────────────────────────────────────────┤
│  START:     python gradio_app.py                            │
│  API:       python main.py serve --port 8000               │
│  CLI:       python main.py convert -s src.wav -r ref.wav   │
│  Batch:     python main.py batch -s *.wav -r ref.wav        │
│  Weights:   python main.py download                         │
│  Test:      python scripts/validate_basic.py                │
├─────────────────────────────────────────────────────────────┤
│  PRESETS:                                                   │
│    Male→Female:  +5st, formant +2                          │
│    Female→Male:  -5st, formant -2                          │
│    Octave Up:    +12st, formant +4                         │
│    Soft Voice:   breathiness 0.25, NR 0.3                  │
│    Robot:        noise_scale 0.1, NR 0.5                   │
├─────────────────────────────────────────────────────────────┤
│  FILE STRUCTURE:                                            │
│    gradio_app.py      → Web UI                             │
│    api_server.py      → REST API                           │
│    main.py            → CLI interface                      │
│    pipeline/          → Core logic                         │
│    weights/           → Model files (~2GB total)           │
│    speaker_profiles/  → Saved embeddings                    │
│    presets.json       → Custom settings                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 **Notes**

This resource guide was compiled from web research conducted on **2025-01-19**. For the most up-to-date links and resources, check the official repositories mentioned above.

**Last Updated**: 2025-01-19  
**Project Version**: 2.0.0 (Enhanced Edition)

---

*💡 Tip: Bookmark this file for quick reference during development!*
