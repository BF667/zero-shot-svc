"""
Zero-Shot Singing Voice Conversion - Enhanced CLI Entry Point

Convert singing voice from source audio to match a reference speaker's voice.

Two modes:
  DEFAULT (signal processing): Works immediately, no pretrained weights needed.
    Uses mel mean-variance normalization + Griffin-Lim vocoder.
  NEURAL (--neural flag): Requires pretrained RVC weights.
    Uses ContentVec + RMVPE + CAM++ + VITS + HiFi-GAN.

Enhanced Features:
  - Batch conversion: Process multiple files at once
  - Voice similarity scoring
  - Speaker profile management
  - Formant shifting
  - Noise reduction
  - Progress display
  - Quality metrics
  - API server mode

Usage Examples:

  # Basic conversion (signal processing, no models needed)
  python main.py convert -s singing.wav -r target_voice.wav

  # With pitch shift and formant control
  python main.py convert -s singing.wav -r voice.wav --transpose 12 --formant-shift 2

  # Neural mode with noise reduction
  python main.py convert -s singing.wav -r voice.wav --neural --noise-reduction 0.3

  # Batch convert multiple files
  python main.py batch -s file1.wav file2.wav file3.wav -r target_voice.wav

  # Compute voice similarity
  python main.py similarity audio1.wav audio2.wav

  # Save speaker profile for later use
  python main.py profile save -r reference.wav -n "My Voice"

  # Start REST API server
  python main.py serve --port 8000

  # Download pretrained weights
  python main.py download
"""
import os
import sys
import argparse
import time
from pathlib import Path


def cmd_convert(args):
    """Run single voice conversion."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pipeline.voice_converter import ZeroShotSVC
    from utils.hparams import Config

    config = Config.from_yaml(args.config)

    # Signal processing by default, neural if --neural
    svc = ZeroShotSVC(config=config, device=args.device, use_neural=args.neural)

    output_path = svc.convert(
        source_path=args.source,
        reference_path=args.reference,
        output_path=args.output,
        f0_transpose=args.transpose,
        f0_curve_factor=args.f0_curve,
        noise_scale=args.noise_scale,
        formant_shift=getattr(args, 'formant_shift', 0),
        noise_reduction=getattr(args, 'noise_reduction', 0.0),
        breathiness=getattr(args, 'breathiness', 0.0),
        protect_consonants=getattr(args, 'protect_consonants', True),
        progress_callback=lambda pct, msg: print(f"  [{pct*100:.0f}%] {msg}"),
    )

    return output_path


def cmd_batch(args):
    """Batch convert multiple files."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pipeline.voice_converter import ZeroShotSVC
    from utils.hparams import Config

    config = Config.from_yaml(args.config)
    svc = ZeroShotSVC(config=config, device=args.device, use_neural=args.neural)

    source_paths = args.sources
    if not source_paths:
        print("Error: No source files specified. Use -s file1.wav file2.wav ...")
        return

    # Validate all source files exist
    for src in source_paths:
        if not os.path.exists(src):
            print(f"Error: Source file not found: {src}")
            return

    results = svc.batch_convert(
        source_paths=source_paths,
        reference_path=args.reference,
        output_dir=args.output_dir,
        f0_transpose=args.transpose,
        f0_curve_factor=args.f0_curve,
        noise_scale=args.noise_scale,
        formant_shift=getattr(args, 'formant_shift', 0),
        noise_reduction=getattr(args, 'noise_reduction', 0.0),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH CONVERSION SUMMARY")
    print("=" * 60)
    
    success_count = 0
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        filename = os.path.basename(r["source"])
        
        if r["status"] == "success":
            print(f"  {status_icon} {filename} -> {os.path.basename(r['output'])} ({r['time']:.1f}s)")
            success_count += 1
        else:
            print(f"  {status_icon} {filename}: {r['status']}")

    print(f"\nTotal: {success_count}/{len(results)} files converted successfully")
    print(f"Output directory: {args.output_dir}")
    
    return results


def cmd_download(args):
    """Download pretrained weights."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from weights.download_weights import download_all
    download_all(args.dir)


def cmd_check(args):
    """Check weight availability."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from weights.download_weights import check_weights
    status = check_weights(args.dir)
    print("Weight Availability:")
    all_ok = True
    for name, path in status.items():
        if path:
            print(f"  [OK]   {name}: {path}")
        else:
            print(f"  [MISS] {name}")
            all_ok = False
    if all_ok:
        print("\nAll weights available! Ready for conversion.")
    else:
        print("\nSome weights missing. Run: python main.py download")
    return all_ok


def cmd_features(args):
    """Extract and display audio features."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline.voice_converter import ZeroShotSVC
    from utils.hparams import Config

    config = Config.from_yaml(args.config)
    svc = ZeroShotSVC(config=config, device=args.device, use_neural=True)
    svc.load_models()

    features = svc.extract_features(args.audio)

    print(f"\nFeature Analysis for: {args.audio}")
    print(f"  Duration:  {features['duration']:.1f}s")
    print(f"  Content:   shape={features['content'].shape}")
    print(f"  F0:        shape={features['f0'].shape}")

    f0 = features['f0']
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) > 0:
        print(f"  F0 range:  {voiced_f0.min():.1f} - {voiced_f0.max():.1f} Hz")
        print(f"  F0 mean:   {voiced_f0.mean():.1f} Hz")
        print(f"  Voiced:    {len(voiced_f0)}/{len(f0)} "
              f"({100 * len(voiced_f0) / len(f0):.0f}%)")


def cmd_extract_embedding(args):
    """Extract speaker embedding from reference audio."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline.voice_converter import ZeroShotSVC
    from utils.hparams import Config

    config = Config.from_yaml(args.config)
    svc = ZeroShotSVC(config=config, device=args.device, use_neural=True)
    svc.load_models()

    embedding = svc.extract_speaker_embedding(args.reference)
    print(f"\nSpeaker Embedding: shape={embedding.shape}")
    print(f"  L2 norm: {sum(embedding**2)**0.5:.4f}")
    print(f"  Min:     {embedding.min():.4f}")
    print(f"  Max:     {embedding.max():.4f}")
    print(f"  Mean:    {embedding.mean():.4f}")

    if args.save:
        import numpy as np
        np.save(args.save, embedding)
        print(f"  Saved to: {args.save}")


def cmd_similarity(args):
    """Compute voice similarity between two audio files."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline.voice_converter import ZeroShotSVC
    from utils.hparams import Config

    config = Config.from_yaml(args.config)
    svc = ZeroShotSVC(config=config, device=args.device, use_neural=False)

    print(f"\nComputing voice similarity...")
    print(f"  Audio 1: {args.audio1}")
    print(f"  Audio 2: {args.audio2}")

    t0 = time.time()
    similarity = svc.compute_similarity(args.audio1, args.audio2)
    elapsed = time.time() - t0

    print(f"\n{'=' * 50}")
    print("SIMILARITY RESULTS")
    print(f"{'=' * 50}")
    print(f"  Overall Similarity:      {similarity['overall_similarity']:.3f}")
    print(f"  MFCC Cosine Similarity:  {similarity['mfcc_cosine_similarity']:.3f}")
    print(f"  Spectral Centroid Corr:   {similarity['spectral_centroid_correlation']:.3f}")
    print(f"  RMS Correlation:          {similarity['rms_correlation']:.3f}")
    print(f"\nComputed in {elapsed:.2f}s")

    # Interpretation
    overall = similarity['overall_similarity']
    if overall > 0.8:
        interpretation = "Very similar voices"
    elif overall > 0.6:
        interpretation = "Moderately similar"
    elif overall > 0.4:
        interpretation = "Somewhat different"
    else:
        interpretation = "Very different voices"

    print(f"  Interpretation: {interpretation}")


def cmd_profile(args):
    """Manage speaker profiles."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline.voice_converter import ZeroShotSVC
    from utils.hparams import Config

    config = Config.from_yaml(args.config)
    svc = ZeroShotSVC(config=config, device=args.device, use_neural=args.neural)

    if args.action == "save":
        if not args.reference:
            print("Error: Please specify a reference audio file with -r/--reference")
            return
        
        profile_path = svc.save_speaker_profile(
            reference_path=args.reference,
            name=args.name
        )
        print(f"Speaker profile saved to: {profile_path}")

    elif args.action == "list":
        profiles = svc.list_speaker_profiles()
        
        if not profiles:
            print("No saved profiles found.")
            return
        
        print(f"\n{'=' * 50}")
        print("SAVED SPEAKER PROFILES")
        print(f"{'=' * 50}")
        
        for p in profiles:
            print(f"\n  Name:       {p.get('name', 'Unknown')}")
            print(f"  Duration:   {p.get('duration', 0):.1f}s")
            print(f"  Created:    {p.get('created_at', 'Unknown')[:19]}")
            
            if 'f0_mean' in p:
                print(f"  F0 Mean:    {p['f0_mean']:.1f} Hz")
            if 'spectral_centroid_mean' in p:
                print(f"  Brightness: {p['spectral_centroid_mean']:.0f} Hz")
            if 'embedding_dim' in p:
                print(f"  Embedding:  {p['embedding_dim']}-dim vector stored")

    else:
        print("Error: Unknown profile action. Use 'save' or 'list'.")
        print("Examples:")
        print("  python main.py profile save -r ref.wav -n 'My Voice'")
        print("  python main.py profile list")


def cmd_serve(args):
    """Start REST API server."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import here to avoid requiring fastapi unless explicitly requested
    try:
        import uvicorn
    except ImportError:
        print("Error: FastAPI and Uvicorn are required to run the server.")
        print("Install them with: pip install fastapi uvicorn python-multipart psutil")
        return
    
    from api_server import app
    
    print("\nStarting Zero-Shot SVC API Server...")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Singing Voice Conversion - Enhanced CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single conversion
  python main.py convert -s singing.wav -r voice_ref.wav -o output.wav
  
  # With enhancements
  python main.py convert -s singing.wav -r voice_ref.wav --transpose 12 --formant-shift 2 --noise-reduction 0.3
  
  # Batch conversion
  python main.py batch -s song1.wav song2.wav -r voice_ref.wav --output-dir converted/
  
  # Neural mode
  python main.py convert -s singing.wav -r voice_ref.wav --neural
  
  # Voice similarity
  python main.py similarity voice1.wav voice2.wav
  
  # Speaker profiles
  python main.py profile save -r my_voice.wav -n "My Voice"
  python main.py profile list
  
  # Start API server
  python main.py serve --port 8000
  
  # Download weights
  python main.py download
        """,
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu' or 'cuda' (auto-detect)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    parser.add_argument("--neural", action="store_true",
                        help="Use neural pipeline (requires pretrained weights). "
                             "Default: signal processing (works without models).")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Convert command ──
    p_convert = subparsers.add_parser("convert", help="Convert singing voice")
    p_convert.add_argument("-s", "--source", required=True, help="Source audio path")
    p_convert.add_argument("-r", "--reference", required=True, 
                          help="Reference audio (target voice)")
    p_convert.add_argument("-o", "--output", default=None, help="Output audio path")
    p_convert.add_argument("--transpose", type=int, default=0,
                          help="Pitch shift in semitones (+/-)")
    p_convert.add_argument("--f0-curve", type=float, default=1.0,
                          help="F0 curve scaling factor")
    p_convert.add_argument("--noise-scale", type=float, default=0.4,
                          help="Generation noise scale (neural mode only)")
    p_convert.add_argument("--formant-shift", type=int, default=0,
                          help="Formant shift in steps (-6 to +6)")
    p_convert.add_argument("--noise-reduction", type=float, default=0.0,
                          help="Noise reduction strength (0 to 1)")
    p_convert.add_argument("--breathiness", type=float, default=0.0,
                          help="Breathiness effect (0 to 1)")
    p_convert.add_argument("--protect-consonants", action="store_true", default=True,
                          help="Protect consonants during pitch shift")

    # ── Batch command ──
    p_batch = subparsers.add_parser("batch", help="Batch convert multiple files")
    p_batch.add_argument("-s", "--sources", nargs="+", required=True,
                        help="Source audio paths (multiple)")
    p_batch.add_argument("-r", "--reference", required=True,
                        help="Reference audio (target voice)")
    p_batch.add_argument("--output-dir", type=str, default="converted_outputs",
                        help="Output directory for converted files")
    p_batch.add_argument("--transpose", type=int, default=0,
                        help="Pitch shift in semitones (+/-)")
    p_batch.add_argument("--f0-curve", type=float, default=1.0,
                        help="F0 curve scaling factor")
    p_batch.add_argument("--noise-scale", type=float, default=0.4,
                        help="Generation noise scale")
    p_batch.add_argument("--formant-shift", type=int, default=0,
                        help="Formant shift in steps")
    p_batch.add_argument("--noise-reduction", type=float, default=0.0,
                        help="Noise reduction strength")

    # ── Download command ──
    p_download = subparsers.add_parser("download", help="Download pretrained weights")
    p_download.add_argument("--dir", type=str, default=None, help="Output directory")

    # ── Check command ──
    p_check = subparsers.add_parser("check", help="Check weight availability")
    p_check.add_argument("--dir", type=str, default=None, help="Weights directory")

    # ── Features command ──
    p_features = subparsers.add_parser("features", help="Extract audio features")
    p_features.add_argument("-a", "--audio", required=True, help="Audio file path")

    # ── Extract embedding command ──
    p_embed = subparsers.add_parser("embedding", help="Extract speaker embedding")
    p_embed.add_argument("-r", "--reference", required=True, help="Reference audio path")
    p_embed.add_argument("--save", type=str, default=None, help="Save embedding to .npy file")

    # ── Similarity command ──
    p_sim = subparsers.add_parser("similarity", help="Compute voice similarity")
    p_sim.add_argument("audio1", help="First audio file path")
    p_sim.add_argument("audio2", help="Second audio file path")

    # ── Profile command ──
    p_profile = subparsers.add_parser("profile", help="Manage speaker profiles")
    p_profile.add_argument("action", choices=["save", "list"], 
                          help="Action: save or list profiles")
    p_profile.add_argument("-r", "--reference", default=None,
                          help="Reference audio (for save action)")
    p_profile.add_argument("-n", "--name", default=None,
                          help="Profile name (for save action)")

    # ── Serve command ──
    p_serve = subparsers.add_parser("serve", help="Start REST API server")
    p_serve.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    p_serve.add_argument("--port", type=int, default=8000,
                        help="Port to listen on")

    args = parser.parse_args()

    # Route to appropriate handler
    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "features":
        cmd_features(args)
    elif args.command == "embedding":
        cmd_extract_embedding(args)
    elif args.command == "similarity":
        cmd_similarity(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        print("\nAvailable commands: convert, batch, download, check, features, "
              "embedding, similarity, profile, serve")


if __name__ == "__main__":
    main()
