"""
Zero-Shot Singing Voice Conversion - CLI Entry Point

Convert singing voice from source audio to match a reference speaker's voice,
using RVC-inspired architecture with RMVPE pitch extraction. NO TRAINING REQUIRED.

Usage Examples:

  # Basic conversion (auto-download models on first run)
  python main.py convert --source singing.wav --reference target_voice.wav

  # With pitch shift
  python main.py convert --source singing.wav --reference target_voice.wav --transpose 12

  # Specify output path
  python main.py convert --source singing.wav --reference target_voice.wav -o output.wav

  # Download pretrained weights only
  python main.py download

  # Check weight status
  python main.py check

  # Extract features for analysis
  python main.py features --audio test.wav
"""
import os
import sys
import argparse


def cmd_convert(args):
    """Run voice conversion."""
    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pipeline.voice_converter import ZeroShotSVC
    from utils.hparams import Config

    # Load config
    config = Config.from_yaml(args.config)

    # Create converter
    svc = ZeroShotSVC(config=config, device=args.device)

    # Load models
    svc.load_models()

    # Convert
    output_path = svc.convert(
        source_path=args.source,
        reference_path=args.reference,
        output_path=args.output,
        f0_transpose=args.transpose,
        f0_curve_factor=args.f0_curve,
        noise_scale=args.noise_scale,
    )

    return output_path


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
    svc = ZeroShotSVC(config=config, device=args.device)
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
    svc = ZeroShotSVC(config=config, device=args.device)
    svc.load_models()

    embedding = svc.extract_speaker_embedding(args.reference)
    print(f"\nSpeaker Embedding: shape={embedding.shape}")
    print(f"  L2 norm: {sum(embedding**2)**0.5:.4f}")
    print(f"  Min:     {embedding.min():.4f}")
    print(f"  Max:     {embedding.max():.4f}")
    print(f"  Mean:    {embedding.mean():.4f}")

    # Save if requested
    if args.save:
        import numpy as np
        np.save(args.save, embedding)
        print(f"  Saved to: {args.save}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Singing Voice Conversion (RVC-based, RMVPE F0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py convert -s singing.wav -r voice_ref.wav -o output.wav
  python main.py convert -s singing.wav -r voice_ref.wav --transpose 12
  python main.py download
  python main.py check
  python main.py features -a test.wav
        """,
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu' or 'cuda' (auto-detect)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert command
    p_convert = subparsers.add_parser("convert", help="Convert singing voice")
    p_convert.add_argument("-s", "--source", required=True, help="Source audio path")
    p_convert.add_argument("-r", "--reference", required=True, help="Reference audio (target voice)")
    p_convert.add_argument("-o", "--output", default=None, help="Output audio path")
    p_convert.add_argument("--transpose", type=int, default=0,
                           help="Pitch shift in semitones (+ = up, - = down)")
    p_convert.add_argument("--f0-curve", type=float, default=1.0,
                           help="F0 curve scaling factor")
    p_convert.add_argument("--noise-scale", type=float, default=0.4,
                           help="Generation noise scale (higher = more variation)")

    # Download command
    p_download = subparsers.add_parser("download", help="Download pretrained weights")
    p_download.add_argument("--dir", type=str, default=None, help="Output directory")

    # Check command
    p_check = subparsers.add_parser("check", help="Check weight availability")
    p_check.add_argument("--dir", type=str, default=None, help="Weights directory")

    # Features command
    p_features = subparsers.add_parser("features", help="Extract audio features")
    p_features.add_argument("-a", "--audio", required=True, help="Audio file path")

    # Extract embedding command
    p_embed = subparsers.add_parser("embedding", help="Extract speaker embedding")
    p_embed.add_argument("-r", "--reference", required=True, help="Reference audio path")
    p_embed.add_argument("--save", type=str, default=None, help="Save embedding to .npy file")

    args = parser.parse_args()

    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "features":
        cmd_features(args)
    elif args.command == "embedding":
        cmd_extract_embedding(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()