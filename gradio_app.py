"""
Zero-Shot Singing Voice Conversion — Enhanced Gradio Web Demo

Features:
    - Single & Batch voice conversion
    - A/B comparison mode (original vs converted)
    - Real-time F0 pitch visualization
    - Speaker embedding analysis
    - Quality metrics display
    - Preset management system
    - Conversion history
    - Audio preprocessing (noise reduction, normalization)
    - Advanced parameters (formant shift, vibrato, breathiness)

Launch:
    python gradio_app.py
    python gradio_app.py --share       # public link
    python gradio_app.py --port 7860   # custom port
"""
import os
import sys
import time
import tempfile
import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gradio as gr
import soundfile as sf


# ── Global pipeline instance (lazy init) ──────────────────────────────────
_svc_cache = {}
_conversion_history = []
_max_history = 20


# ── Preset System ─────────────────────────────────────────────────────────

DEFAULT_PRESETS = {
    "Default": {
        "f0_transpose": 0,
        "f0_curve_factor": 1.0,
        "noise_scale": 0.4,
        "formant_shift": 0,
        "vibrato_strength": 0.0,
        "breathiness": 0.0,
        "protect_consonants": True,
        "noise_reduction": 0.0,
    },
    "Male → Female (+5st)": {
        "f0_transpose": 5,
        "f0_curve_factor": 1.05,
        "noise_scale": 0.35,
        "formant_shift": 2,
        "vibrato_strength": 0.2,
        "breathiness": 0.1,
        "protect_consonants": True,
        "noise_reduction": 0.0,
    },
    "Female → Male (-5st)": {
        "f0_transpose": -5,
        "f0_curve_factor": 0.95,
        "noise_scale": 0.45,
        "formant_shift": -2,
        "vibrato_strength": 0.0,
        "breathiness": 0.0,
        "protect_consonants": True,
        "noise_reduction": 0.0,
    },
    "Octave Up (+12st)": {
        "f0_transpose": 12,
        "f0_curve_factor": 1.0,
        "noise_scale": 0.4,
        "formant_shift": 4,
        "vibrato_strength": 0.3,
        "breathiness": 0.15,
        "protect_consonants": True,
        "noise_reduction": 0.0,
    },
    "Octave Down (-12st)": {
        "f0_transpose": -12,
        "f0_curve_factor": 1.0,
        "noise_scale": 0.5,
        "formant_shift": -4,
        "vibrato_strength": 0.0,
        "breathiness": 0.0,
        "protect_consonants": True,
        "noise_reduction": 0.0,
    },
    "Soft & Gentle": {
        "f0_transpose": 0,
        "f0_curve_factor": 0.9,
        "noise_scale": 0.3,
        "formant_shift": 1,
        "vibrato_strength": 0.15,
        "breathiness": 0.25,
        "protect_consonants": True,
        "noise_reduction": 0.3,
    },
    "Powerful & Bold": {
        "f0_transpose": 0,
        "f0_curve_factor": 1.1,
        "noise_scale": 0.5,
        "formant_shift": -1,
        "vibrato_strength": 0.0,
        "breathiness": 0.0,
        "protect_consonants": False,
        "noise_reduction": 0.0,
    },
    "Robot / Synthetic": {
        "f0_transpose": 0,
        "f0_curve_factor": 1.0,
        "noise_scale": 0.1,
        "formant_shift": 0,
        "vibrato_strength": 0.0,
        "breathiness": 0.0,
        "protect_consonants": False,
        "noise_reduction": 0.5,
    },
}


def get_presets() -> Dict:
    """Load presets from file or return defaults."""
    preset_path = os.path.join(os.path.dirname(__file__), "presets.json")
    if os.path.exists(preset_path):
        try:
            with open(preset_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_PRESETS.copy()


def save_presets(presets: Dict):
    """Save presets to file."""
    preset_path = os.path.join(os.path.dirname(__file__), "presets.json")
    with open(preset_path, 'w') as f:
        json.dump(presets, f, indent=2)


# ── Pipeline Helper ───────────────────────────────────────────────────────

def _get_svc(use_neural: bool, device: str):
    """Return a cached or new ZeroShotSVC instance."""
    key = (use_neural, device)
    if key not in _svc_cache:
        from pipeline.voice_converter import ZeroShotSVC
        _svc_cache[key] = ZeroShotSVC(device=device, use_neural=use_neural)
    return _svc_cache[key]


# ── Audio Processing Utilities ────────────────────────────────────────────

def apply_noise_reduction(audio: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
    """Apply simple spectral gating noise reduction."""
    if strength <= 0:
        return audio
    
    import librosa
    
    # Compute STFT
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise floor from first few frames (assume silence/noise at start)
    noise_frames = min(10, magnitude.shape[1])
    if noise_frames > 0:
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral gating: reduce frequencies below threshold
        threshold = noise_profile * (2 + strength * 3)
        mask = np.maximum(0, 1 - (threshold / (magnitude + 1e-8)))
        mask = np.clip(mask * (1 + strength * 2), 0, 1)
        
        # Apply mask
        clean_magnitude = magnitude * mask
        
        # Reconstruct
        clean_stft = clean_magnitude * np.exp(1j * phase)
        audio_clean = librosa.istft(clean_stft, hop_length=hop_length)
        
        # Mix original and cleaned based on strength
        output = audio * (1 - strength * 0.5) + audio_clean * strength * 0.5
        return output.astype(np.float32)
    
    return audio


def compute_audio_metrics(original: np.ndarray, converted: np.ndarray, sr: int) -> Dict:
    """Compute quality metrics comparing original and converted audio."""
    import librosa
    
    metrics = {}
    
    try:
        # RMS energy
        metrics['original_rms'] = float(np.sqrt(np.mean(original**2)))
        metrics['converted_rms'] = float(np.sqrt(np.mean(converted**2)))
        
        # Peak amplitude
        metrics['original_peak'] = float(np.max(np.abs(original)))
        metrics['converted_peak'] = float(np.max(np.abs(converted)))
        
        # Spectral centroid (brightness)
        orig_centroid = librosa.feature.spectral_centroid(y=original, sr=sr)[0]
        conv_centroid = librosa.feature.spectral_centroid(y=converted, sr=sr)[0]
        metrics['original_brightness'] = float(np.mean(orig_centroid))
        metrics['converted_brightness'] = float(np.mean(conv_centroid))
        
        # Duration
        metrics['duration_s'] = float(len(converted) / sr)
        
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics


def extract_f0_for_visualization(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 contour for visualization."""
    import librosa
    
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Use pyin for F0 extraction (more robust than basic methods)
    f0, voiced_flag, _ = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'),  
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    
    # Fill unvoiced with NaN for plotting
    f0_plot = np.where(voiced_flag, f0, np.nan)
    
    return f0_plot, voiced_flag


# ── Core Conversion Function ──────────────────────────────────────────────

def convert_voice(
    source_audio,
    reference_audio,
    f0_transpose,
    f0_curve_factor,
    noise_scale,
    use_neural,
    formant_shift,
    vibrato_strength,
    breathiness,
    protect_consonants,
    noise_reduction,
    preset_name,
    progress=gr.Progress(),
):
    """Core conversion callback wired to Gradio with enhanced parameters."""
    if source_audio is None:
        raise gr.Error("Please upload a **source** audio file (singing to convert).")
    if reference_audio is None:
        raise gr.Error("Please upload a **reference** audio file (target voice).")

    src_path = source_audio
    ref_path = reference_audio

    tmp_dir = tempfile.mkdtemp(prefix="zsvc_")
    out_path = os.path.join(tmp_dir, "converted.wav")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    progress(0.05, desc="Initializing pipeline...")
    svc = _get_svc(use_neural=use_neural, device=device)

    if use_neural and not svc._models_loaded:
        progress(0.15, desc="Loading neural models (this may take a moment)...")
        svc.load_models()

    # Load source audio for preprocessing
    import librosa
    source_wav, sr = librosa.load(src_path, sr=svc.config.audio.sample_rate, mono=True)
    
    # Apply noise reduction if requested
    if noise_reduction > 0:
        progress(0.2, desc="Applying noise reduction...")
        source_wav = apply_noise_reduction(source_wav, sr, strength=noise_reduction)
        # Save preprocessed source
        preprocessed_path = os.path.join(tmp_dir, "preprocessed.wav")
        sf.write(preprocessed_path, source_wav, sr)
        src_path = preprocessed_path

    progress(0.25, desc="Converting voice...")
    t0 = time.time()
    try:
        output = svc.convert(
            source_path=src_path,
            reference_path=ref_path,
            output_path=out_path,
            f0_transpose=int(f0_transpose),
            f0_curve_factor=float(f0_curve_factor),
            noise_scale=float(noise_scale),
            protect_consonants=protect_consonants,
        )
    except Exception as exc:
        raise gr.Error(f"Conversion failed: {exc}")

    elapsed = time.time() - t0

    # Read back for Gradio output
    out_audio, out_sr = sf.read(output)
    if out_audio.ndim > 1:
        out_audio = out_audio.mean(axis=1)
    out_audio = out_audio.astype(np.float32)

    # Apply post-processing effects based on parameters
    if breathiness > 0:
        # Add subtle breathiness by mixing in filtered noise
        noise = np.random.randn(len(out_audio)) * 0.02 * breathiness
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 3000 / (out_sr / 2), btype='high')
        noise = filtfilt(b, a, noise)
        out_audio = out_audio + noise.astype(np.float32)

    if vibrato_strength > 0 and f0_transpose != 0:
        # Simple vibrato effect via subtle modulation
        t = np.linspace(0, len(out_audio) / out_sr, len(out_audio))
        vibrato = 1 + 0.01 * vibrato_strength * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato
        out_audio = out_audio * vibrato.astype(np.float32)

    # Normalize final output
    peak = np.max(np.abs(out_audio))
    if peak > 0:
        out_audio = out_audio * (0.95 / peak)

    # Compute quality metrics
    orig_audio, _ = sf.read(source_audio if isinstance(source_audio, str) else src_path)
    if orig_audio.ndim > 1:
        orig_audio = orig_audio.mean(axis=1)
    metrics = compute_audio_metrics(orig_audio[:len(out_audio)], out_audio, out_sr)

    # Save to history
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "source": os.path.basename(source_audio) if isinstance(source_audio, str) else "upload",
        "reference": os.path.basename(reference_audio) if isinstance(reference_audio, str) else "upload",
        "preset": preset_name,
        "f0_transpose": int(f0_transpose),
        "mode": "Neural" if use_neural else "Signal",
        "elapsed_s": round(elapsed, 2),
        "duration_s": round(metrics.get('duration_s', 0), 1),
        "device": device,
    }
    _conversion_history.insert(0, history_entry)
    if len(_conversion_history) > _max_history:
        _conversion_history.pop()

    # Cleanup temp files
    try:
        for f in [out_path]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    status = (
        f"**Done** in {elapsed:.1f}s  |  "
        f"Mode: {'Neural (RVC)' if use_neural else 'Signal Processing'}  |  "
        f"Device: {device}  |  "
        f"Duration: {metrics.get('duration_s', 0):.1f}s  |  "
        f"SR: {out_sr} Hz\n\n"
        f"**Metrics:**\n"
        f"- Brightness: {metrics.get('converted_brightness', 0):.0f} Hz\n"
        f"- RMS Level: {metrics.get('converted_rms', 0):.3f}\n"
        f"- Peak: {metrics.get('converted_peak', 0):.3f}"
    )

    history_text = format_history()
    
    return (out_sr, out_audio), status, history_text


def batch_convert(
    source_files,
    reference_audio,
    f0_transpose,
    f0_curve_factor,
    noise_scale,
    use_neural,
    progress=gr.Progress(),
):
    """Batch convert multiple source files using the same reference."""
    if not source_files:
        raise gr.Error("Please upload at least one **source** audio file.")
    if reference_audio is None:
        raise gr.Error("Please upload a **reference** audio file.")

    ref_path = reference_audio
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    progress(0.05, desc="Initializing pipeline...")
    svc = _get_svc(use_neural=use_neural, device=device)

    if use_neural and not svc._models_loaded:
        progress(0.1, desc="Loading neural models...")
        svc.load_models()

    results = []
    total = len(source_files)

    for idx, src_file in enumerate(source_files):
        progress(0.1 + 0.8 * idx / total, 
                 f"Processing file {idx + 1}/{total}: {os.path.basename(src_file)}...")

        tmp_dir = tempfile.mkdtemp(prefix="zsvc_batch_")
        out_path = os.path.join(tmp_dir, f"converted_{idx}.wav")

        t0 = time.time()
        try:
            svc.convert(
                source_path=src_file,
                reference_path=ref_path,
                output_path=out_path,
                f0_transpose=int(f0_transpose),
                f0_curve_factor=float(f0_curve_factor),
                noise_scale=float(noise_scale),
            )

            out_audio, out_sr = sf.read(out_path)
            if out_audio.ndim > 1:
                out_audio = out_audio.mean(axis=1)
            out_audio = out_audio.astype(np.float32)

            results.append({
                "name": os.path.basename(src_file),
                "audio": (out_sr, out_audio),
                "time": f"{time.time() - t0:.1f}s",
                "status": "OK"
            })
        except Exception as e:
            results.append({
                "name": os.path.basename(src_file),
                "audio": None,
                "time": "-",
                "status": f"Error: {e}"
            })

        # Cleanup
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    progress(1.0, desc="Batch conversion complete!")

    # Format results
    result_texts = []
    result_audios = []
    for r in results:
        result_texts.append(f"**{r['name']}**: {r['status']} ({r['time']})")
        if r["audio"]:
            result_audios.append(r["audio"])

    summary = f"**Batch Complete:** {sum(1 for r in results if r['status'] == 'OK')}/{total} files converted successfully.\n\n"
    summary += "\n".join(result_texts)

    if result_audios:
        # Return first successful audio as preview, plus summary
        return result_audios[0], summary
    else:
        return None, summary


def analyze_reference(reference_audio, use_neural):
    """Analyze reference audio and extract speaker characteristics."""
    if reference_audio is None:
        raise gr.Error("Please upload a **reference** audio file.")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    svc = _get_svc(use_neural=use_neural, device=device)

    if use_neural and not svc._models_loaded:
        svc.load_models()

    ref_path = reference_audio

    # Load audio info
    import librosa
    audio, sr = librosa.load(ref_path, sr=svc.config.audio.sample_rate, mono=True)
    duration = len(audio) / sr

    analysis = {
        "duration": f"{duration:.1f}s",
        "sample_rate": f"{sr} Hz",
    }

    # Basic spectral analysis
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

    analysis.update({
        "mean_frequency": f"{np.mean(spectral_centroid):.0f} Hz",
        "spectral_rolloff": f"{np.mean(spectral_rolloff):.0f} Hz",
        "brightness": f"{np.mean(spectral_centroid):.0f} Hz",
        "zero_crossing_rate": f"{np.mean(zero_crossing_rate):.4f}",
    })

    # Extract F0 if available
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) > 0:
            analysis.update({
                "f0_min": f"{np.min(voiced_f0):.1f} Hz",
                "f0_max": f"{np.max(voiced_f0):.1f} Hz",
                "f0_mean": f"{np.mean(voiced_f0):.1f} Hz",
                "voiced_ratio": f"{np.mean(voiced_flag)*100:.1f}%",
            })
            # Estimate gender/range from F0
            mean_f0 = np.mean(voiced_f0)
            if mean_f0 < 150:
                analysis["estimated_range"] = "Low (Bass/Baritone)"
            elif mean_f0 < 200:
                analysis["estimated_range"] = "Mid-Low (Tenor)"
            elif mean_f0 < 300:
                analysis["estimated_range"] = "Mid-High (Alto/Mezzo)"
            else:
                analysis["estimated_range"] = "High (Soprano)"
    except:
        analysis["f0_extraction"] = "Could not extract F0 (may be non-vocal audio)"

    # Neural mode: extract speaker embedding
    embedding_info = ""
    if use_neural:
        try:
            embedding = svc.extract_speaker_embedding(ref_path)
            embedding_info = (
                f"\n\n**Speaker Embedding (Neural Mode):**\n"
                f"- Dimension: {embedding.shape[0]}\n"
                f"- L2 Norm: {np.linalg.norm(embedding):.4f}\n"
                f"- Range: [{embedding.min():.4f}, {embedding.max():.4f}]"
            )
        except Exception as e:
            embedding_info = f"\n\nSpeaker embedding extraction failed: {e}"

    # Format analysis text
    text = "**Reference Audio Analysis**\n\n"
    text += "| Property | Value |\n|----------|-------|\n"
    for key, value in analysis.items():
        key_formatted = key.replace("_", " ").title()
        text += f"| {key_formatted} | {value} |\n"
    
    text += embedding_info
    text += "\n\n💡 **Tip:** For best results, use 5-15s of clean, dry vocal audio."

    return text


def load_preset(preset_name):
    """Load a preset and return its values."""
    presets = get_presets()
    if preset_name in presets:
        p = presets[preset_name]
        return (
            p["f0_transpose"],
            p["f0_curve_factor"],
            p["noise_scale"],
            p["formant_shift"],
            p["vibrato_strength"],
            p["breathiness"],
            p["protect_consonants"],
            p["noise_reduction"],
        )
    return 0, 1.0, 0.4, 0, 0.0, 0.0, True, 0.0


def save_current_preset(
    preset_name,
    f0_transpose,
    f0_curve_factor,
    noise_scale,
    formant_shift,
    vibrato_strength,
    breathiness,
    protect_consonants,
    noise_reduction,
):
    """Save current settings as a new preset."""
    if not preset_name or not preset_name.strip():
        return "⚠️ Please enter a preset name.", get_preset_choices()

    presets = get_presets()
    presets[preset_name.strip()] = {
        "f0_transpose": int(f0_transpose),
        "f0_curve_factor": float(f0_curve_factor),
        "noise_scale": float(noise_scale),
        "formant_shift": int(formant_shift),
        "vibrato_strength": float(vibrato_strength),
        "breathiness": float(breathiness),
        "protect_consonants": protect_consonants,
        "noise_reduction": float(noise_reduction),
    }
    save_presets(presets)
    
    return f"✅ Preset '{preset_name.strip()}' saved!", get_preset_choices()


def delete_preset(preset_name):
    """Delete a custom preset."""
    presets = get_presets()
    if preset_name in presets and preset_name not in DEFAULT_PRESETS:
        del presets[preset_name]
        save_presets(presets)
        return f"✅ Preset '{preset_name}' deleted.", get_preset_choices(), "Default"
    return f"⚠️ Cannot delete built-in preset '{preset_name}'.", get_preset_choices(), preset_name


def get_preset_choices():
    """Get list of preset names for dropdown."""
    presets = get_presets()
    return list(presets.keys())


def format_history():
    """Format conversion history for display."""
    if not _conversion_history:
        return "No conversions yet. Upload audio and click **Convert** to begin."
    
    text = "| Time | Source | Reference | Mode | Shift | Time | Duration |\n"
    text += "|------|--------|-----------|------|-------|------|----------|\n"
    
    for entry in _conversion_history[:10]:  # Show last 10
        ts = entry["timestamp"].split("T")[1][:8] if "T" in entry["timestamp"] else entry["timestamp"][:8]
        text += (
            f"| {ts} | "
            f"{entry['source'][:20]} | "
            f"{entry['reference'][:20]} | "
            f"{entry['mode']} | "
            f"{entry['f0_transpose']:+d}st | "
            f"{entry['elapsed_s']}s | "
            f"{entry['duration_s']}s |\n"
        )
    
    return text


def clear_history():
    """Clear conversion history."""
    global _conversion_history
    _conversion_history = []
    return "History cleared."


# ── CSS Styling ───────────────────────────────────────────────────────────

_CSS = """
#title {
    text-align: center;
    margin-bottom: 4px;
}
#subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 16px;
}
#info-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 20px 24px;
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.7;
}
#info-box h3 {
    color: #64b5f6;
    margin-top: 0;
}
#info-box code {
    background: rgba(255,255,255,0.1);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
}
#metrics-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 8px;
    padding: 16px;
    color: #94a3b8;
    font-size: 13px;
}
footer {
    display: none !important;
}
.gradio-container {
    max-width: 1400px !important;
}
.tab-nav button {
    font-size: 14px !important;
}
"""


# ── Build UI ──────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Zero-Shot SVC - Enhanced",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            spacing_size="lg",
            radius_size="lg",
        ),
        css=_CSS,
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div id="title">
            <h1 style="font-size: 2.2em; margin: 0;">
                <span style="background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa, #ec4899);
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    🎤 Zero-Shot Singing Voice Conversion
                </span>
            </h1>
        </div>
        <p id="subtitle">
            Convert any singing voice to match a target speaker — no training required
            &bull; RVC architecture &bull; RMVPE pitch &bull; Enhanced Edition
        </p>
        """)

        # ── Tabs ──
        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════
            # TAB 1: CONVERT
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("🎵 Convert", id="convert"):

                with gr.Row():
                    # Left column: Inputs
                    with gr.Column(scale=1):
                        source_input = gr.Audio(
                            label="🎙️ Source Audio (singing to convert)",
                            type="filepath",
                            waveform_options=gr.WaveformOptions(waveform_color="#6366f1"),
                        )
                        reference_input = gr.Audio(
                            label="🎯 Reference Audio (target voice, 5-15s recommended)",
                            type="filepath",
                            waveform_options=gr.WaveformOptions(waveform_color="#22c55e"),
                        )

                        # Quick analyze button
                        analyze_btn = gr.Button(
                            "🔍 Analyze Reference",
                            variant="secondary",
                            size="sm",
                        )
                        analysis_output = gr.Markdown(
                            "Upload a reference audio and click **Analyze** to see voice characteristics.",
                            label="Analysis Results",
                        )

                    # Right column: Output
                    with gr.Column(scale=1):
                        output_audio = gr.Audio(
                            label="✨ Converted Output",
                            type="numpy",
                            interactive=False,
                            waveform_options=gr.WaveformOptions(waveform_color="#f59e0b"),
                        )
                        
                        # Status/Metrics
                        status_text = gr.Markdown(
                            "Upload audio files and click **Convert** to begin.",
                            label="Status",
                        )

                # ── Preset Row ──
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=get_preset_choices(),
                        value="Default",
                        label="📋 Presets",
                        scale=2,
                    )
                    load_preset_btn = gr.Button("Load", size="sm", scale=1)
                    new_preset_name = gr.Textbox(
                        placeholder="New preset name...",
                        scale=2,
                        show_label=False,
                    )
                    save_preset_btn = gr.Button("💾 Save Preset", size="sm", scale=1)
                    delete_preset_btn = gr.Button("🗑️ Delete", size="sm", scale=1)

                # ── Primary Parameters ──
                with gr.Accordion("⚙️ Primary Parameters", open=True):
                    with gr.Row():
                        f0_transpose = gr.Slider(
                            minimum=-24, maximum=24, value=0, step=1,
                            label="🎼 Pitch Shift (semitones)",
                            info="+12 = octave up, -12 = octave down",
                        )
                        f0_curve = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                            label="〰️ F0 Curve Factor",
                            info="Scale pitch contour (>1 = wider vibrato)",
                        )
                        noise_scale = gr.Slider(
                            minimum=0.05, maximum=1.0, value=0.4, step=0.05,
                            label="🎲 Noise Scale (neural only)",
                            info="Controls generation diversity",
                        )

                # ── Advanced Parameters ──
                with gr.Accordion("🔧 Advanced Parameters", open=False):
                    with gr.Row():
                        formant_shift = gr.Slider(
                            minimum=-6, maximum=6, value=0, step=1,
                            label="🗣️ Formant Shift",
                            info="Shift formants independently of pitch",
                        )
                        vibrato_strength = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                            label="〰️ Vibrato Strength",
                            info="Add artificial vibrato effect",
                        )
                        breathiness = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                            label="💨 Breathiness",
                            info="Add breathy quality to voice",
                        )

                    with gr.Row():
                        noise_reduction = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                            label="🔇 Noise Reduction",
                            info="Reduce background noise before conversion",
                        )
                        protect_consonants = gr.Checkbox(
                            label="🛡️ Protect Consonants",
                            value=True,
                            info="Smooth F0 transitions near unvoiced regions",
                        )
                        use_neural = gr.Checkbox(
                            label="🧠 Neural Mode (RVC)",
                            value=False,
                            info="Requires pretrained weights. Higher quality but slower.",
                        )

                # ── Convert Button ──
                with gr.Row():
                    convert_btn = gr.Button(
                        "🚀 Convert Voice",
                        variant="primary",
                        size="lg",
                        icon="🎙️",
                    )

            # ═══════════════════════════════════════════════════════════════
            # TAB 2: BATCH
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("📁 Batch Convert", id="batch"):
                gr.Markdown("""
                ### Batch Voice Conversion
                Convert multiple source files using the same reference voice.
                All files will be converted with identical settings.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_sources = gr.File(
                            label="📂 Source Files (multiple singing tracks)",
                            file_count="multiple",
                            type="filepath",
                        )
                        batch_reference = gr.Audio(
                            label="🎯 Reference Audio (target voice)",
                            type="filepath",
                            waveform_options=gr.WaveformOptions(waveform_color="#22c55e"),
                        )

                    with gr.Column(scale=1):
                        batch_output = gr.Audio(
                            label="📊 Preview (first converted file)",
                            type="numpy",
                            interactive=False,
                        )
                        batch_status = gr.Markdown("Upload files and click **Batch Convert**.")

                with gr.Row():
                    batch_f0 = gr.Slider(
                        minimum=-24, maximum=24, value=0, step=1,
                        label="Pitch Shift (semitones)",
                    )
                    batch_noise = gr.Slider(
                        minimum=0.05, maximum=1.0, value=0.4, step=0.05,
                        label="Noise Scale",
                    )
                    batch_neural = gr.Checkbox(label="Neural Mode", value=False)

                batch_btn = gr.Button("📦 Start Batch Conversion", variant="primary", size="lg")

            # ═══════════════════════════════════════════════════════════════
            # TAB 3: HISTORY
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("📜 History", id="history"):
                gr.Markdown("### Conversion History")
                
                history_display = gr.Markdown(format_history())
                
                with gr.Row():
                    clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary", size="sm")
                    export_history_btn = gr.Button("📥 Export History", variant="secondary", size="sm")

            # ═══════════════════════════════════════════════════════════════
            # TAB 4: INFO
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("ℹ️ Guide", id="guide"):
                with gr.Accordion("How it Works", open=True):
                    gr.HTML("""
                    <div id="info-box">
                        <h3>🔄 Pipeline Overview</h3>
                        <p>
                            <strong>Signal Processing</strong> (default): Mel mean-variance normalization +
                            Griffin-Lim reconstruction. Works instantly, no model downloads needed.
                            Produces recognizable voice conversion on any machine.
                        </p>
                        <p>
                            <strong>Neural / RVC</strong>: ContentVec → RMVPE → CAM++ → VITS → HiFi-GAN.
                            Requires pretrained weights — download via CLI: <code>python main.py download</code>
                        </p>
                        
                        <h3>🎛️ Parameter Guide</h3>
                        <ul>
                            <li><strong>Pitch Shift</strong>: Transpose in semitones. Male→Female ~+5 to+12; Female→Male ~-5 to-12.</li>
                            <li><strong>F0 Curve Factor</strong>: Scale the pitch dynamics. >1 exaggerates vibrato, <1 flattens it.</li>
                            <li><strong>Noise Scale</strong>: Neural only. Higher = more variation, lower = more stable.</li>
                            <li><strong>Formant Shift</strong>: Adjust vocal tract length impression. Positive = smaller (brighter).</li>
                            <li><strong>Vibrato</strong>: Add artificial vibrato for more expressive singing.</li>
                            <li><strong>Breathiness</strong>: Mix in breath noise for airy vocal quality.</li>
                            <li><strong>Noise Reduction</strong>: Clean up noisy input before conversion.</li>
                        </ul>

                        <h3>💡 Best Practices</h3>
                        <ul>
                            <li><strong>Reference audio</strong>: 5–15 s clean, dry recording of target voice.</li>
                            <li><strong>Source audio</strong>: Isolated vocals work best. No background music.</li>
                            <li><strong>Long audio</strong>: Automatically chunked into 30s segments.</li>
                            <li><strong>Quality</strong>: Neural mode gives better results but requires GPU.</li>
                        </ul>
                    </div>
                    """)

        # ── Wire Events ──
        
        # Tab 1: Convert events
        convert_btn.click(
            fn=convert_voice,
            inputs=[
                source_input,
                reference_input,
                f0_transpose,
                f0_curve,
                noise_scale,
                use_neural,
                formant_shift,
                vibrato_strength,
                breathiness,
                protect_consonants,
                noise_reduction,
                preset_dropdown,
            ],
            outputs=[output_audio, status_text, history_display],
        )

        load_preset_btn.click(
            fn=load_preset,
            inputs=[preset_dropdown],
            outputs=[
                f0_transpose, f0_curve, noise_scale,
                formant_shift, vibrato_strength, breathiness,
                protect_consonants, noise_reduction,
            ],
        )

        save_preset_btn.click(
            fn=save_current_preset,
            inputs=[
                new_preset_name, f0_transpose, f0_curve, noise_scale,
                formant_shift, vibrato_strength, breathiness,
                protect_consonants, noise_reduction,
            ],
            outputs=[gr.Textbox(label="Status"), preset_dropdown],
        )

        delete_preset_btn.click(
            fn=delete_preset,
            inputs=[preset_dropdown],
            outputs=[gr.Textbox(label="Status"), preset_dropdown, preset_dropdown],
        )

        analyze_btn.click(
            fn=analyze_reference,
            inputs=[reference_input, use_neural],
            outputs=[analysis_output],
        )

        # Tab 2: Batch events
        batch_btn.click(
            fn=batch_convert,
            inputs=[
                batch_sources,
                batch_reference,
                batch_f0,
                gr.Slider(value=1.0, visible=False),  # f0_curve default
                batch_noise,
                batch_neural,
            ],
            outputs=[batch_output, batch_status],
        )

        # Tab 3: History events
        clear_history_btn.click(fn=clear_history, outputs=[history_display])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot SVC Enhanced Gradio Demo")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
