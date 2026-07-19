"""
Zero-Shot Singing Voice Conversion — Gradio Web Demo

Launch:
    python gradio_app.py
    python gradio_app.py --share       # public link
    python gradio_app.py --port 7860   # custom port

The demo exposes the full ZeroShotSVC pipeline through a browser UI
with real-time audio preview, parameter controls, and progress feedback.
"""
import os
import sys
import time
import tempfile
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gradio as gr

# ── Global pipeline instance (lazy init) ──────────────────────────────────
_svc_cache = {}  # keyed by (use_neural, device)


def _get_svc(use_neural: bool, device: str):
    """Return a cached or new ZeroShotSVC instance."""
    key = (use_neural, device)
    if key not in _svc_cache:
        from pipeline.voice_converter import ZeroShotSVC
        _svc_cache[key] = ZeroShotSVC(device=device, use_neural=use_neural)
    return _svc_cache[key]


def convert_voice(
    source_audio,
    reference_audio,
    f0_transpose,
    f0_curve_factor,
    noise_scale,
    use_neural,
    progress=gr.Progress(),
):
    """Core conversion callback wired to Gradio."""
    if source_audio is None or reference_audio is None:
        raise gr.Error("Please upload both a **source** and a **reference** audio file.")

    # Gradio with type="filepath" gives us the uploaded file path directly.
    src_path = source_audio
    ref_path = reference_audio

    # Auto-generate output path in a temp dir
    tmp_dir = tempfile.mkdtemp(prefix="zsvc_")
    out_path = os.path.join(tmp_dir, "converted.wav")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    progress(0.1, desc="Initializing pipeline...")
    svc = _get_svc(use_neural=use_neural, device=device)

    if use_neural and not svc._models_loaded:
        progress(0.15, desc="Loading neural models (this may take a moment)...")
        svc.load_models()

    progress(0.3, desc="Converting voice...")
    t0 = time.time()
    try:
        output = svc.convert(
            source_path=src_path,
            reference_path=ref_path,
            output_path=out_path,
            f0_transpose=int(f0_transpose),
            f0_curve_factor=float(f0_curve_factor),
            noise_scale=float(noise_scale),
        )
    except Exception as exc:
        raise gr.Error(f"Conversion failed: {exc}")

    elapsed = time.time() - t0

    # Read back for Gradio output (type="numpy" expects (sr, data))
    import soundfile as sf
    out_audio, out_sr = sf.read(output)
    # Ensure mono float32
    if out_audio.ndim > 1:
        out_audio = out_audio.mean(axis=1)
    out_audio = out_audio.astype(np.float32)

    # Cleanup temp output (source/ref are Gradio-managed)
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    status = (
        f"**Done** in {elapsed:.1f}s  |  "
        f"Mode: {'Neural (RVC)' if use_neural else 'Signal Processing'}  |  "
        f"Device: {device}  |  "
        f"Duration: {len(out_audio) / out_sr:.1f}s  |  "
        f"SR: {out_sr} Hz"
    )
    return (out_sr, out_audio), status


# ── Build UI ──────────────────────────────────────────────────────────────

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
footer {
    display: none !important;
}
"""

def build_ui():
    with gr.Blocks(
        title="Zero-Shot SVC",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
        css=_CSS,
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div id="title">
            <h1 style="font-size: 2em; margin: 0;">
                <span style="background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Zero-Shot Singing Voice Conversion
                </span>
            </h1>
        </div>
        <p id="subtitle">
            Convert any singing voice to match a target speaker &mdash;
            no training required &bull; RVC architecture &bull; RMVPE pitch
        </p>
        """)

        # ── Info banner ──
        with gr.Accordion("How it works & tips", open=False):
            gr.HTML("""
            <div id="info-box">
                <h3>Pipeline</h3>
                <p>
                    <strong>Signal Processing</strong> (default): Mel mean-variance normalization +
                    Griffin-Lim reconstruction. Works instantly, no model downloads needed.
                    Produces recognizable voice conversion on any machine.
                </p>
                <p>
                    <strong>Neural / RVC</strong> (toggle below): ContentVec &rarr; RMVPE &rarr;
                    CAM++ &rarr; VITS &rarr; HiFi-GAN. Requires <code>--neural</code> pretrained
                    weights &mdash; run <code>python main.py download</code> first.
                </p>
                <h3>Tips</h3>
                <ul>
                    <li><strong>Reference audio</strong>: 5&ndash;15 s clean, dry recording of the
                        target voice (speech or singing).</li>
                    <li><strong>Source audio</strong>: Isolated vocals work best. Use a vocal
                        separator first if the track has instruments.</li>
                    <li><strong>Pitch shift</strong>: Male &rarr; Female ~ +5 to +12 st;
                        Female &rarr; Male ~ &minus;5 to &minus;12 st.</li>
                </ul>
            </div>
            """)

        # ── Main row ──
        with gr.Row():
            with gr.Column(scale=1):
                source_input = gr.Audio(
                    label="Source Audio (singing to convert)",
                    type="filepath",
                    waveform_options=gr.WaveformOptions(waveform_color="#6366f1"),
                )
                reference_input = gr.Audio(
                    label="Reference Audio (target voice, 5-15 s)",
                    type="filepath",
                    waveform_options=gr.WaveformOptions(waveform_color="#22c55e"),
                )

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Converted Output",
                    type="numpy",
                    interactive=False,
                    waveform_options=gr.WaveformOptions(waveform_color="#f59e0b"),
                )
                status_text = gr.Markdown("Upload audio files and click **Convert** to begin.")

        # ── Parameters ──
        with gr.Row():
            f0_transpose = gr.Slider(
                minimum=-24, maximum=24, value=0, step=1,
                label="Pitch Shift (semitones)",
                info="+12 = 1 octave up, -12 = 1 octave down",
            )
            f0_curve = gr.Slider(
                minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                label="F0 Curve Factor",
                info="Scale the pitch contour (>1 = wider vibrato)",
            )
            noise_scale = gr.Slider(
                minimum=0.05, maximum=1.0, value=0.4, step=0.05,
                label="Noise Scale (neural only)",
                info="Controls generation diversity",
            )

        with gr.Row():
            use_neural = gr.Checkbox(
                label="Neural mode (requires pretrained weights)",
                value=False,
                info="Signal processing mode is used by default and works without downloads.",
            )
            convert_btn = gr.Button(
                "Convert Voice",
                variant="primary",
                size="lg",
                icon="🎙️",
            )

        # ── Examples ──
        gr.Examples(
            examples=[
                [None, None, 0, 1.0, 0.4, False],
                [None, None, 5, 1.0, 0.4, False],
                [None, None, -7, 1.0, 0.4, False],
            ],
            inputs=[source_input, reference_input, f0_transpose, f0_curve, noise_scale, use_neural],
            label="Quick presets (upload your own audio to use these)",
        )

        # ── Wire events ──
        convert_btn.click(
            fn=convert_voice,
            inputs=[
                source_input,
                reference_input,
                f0_transpose,
                f0_curve,
                noise_scale,
                use_neural,
            ],
            outputs=[output_audio, status_text],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot SVC Gradio Demo")
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