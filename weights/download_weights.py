"""
Pretrained Weight Downloader

Downloads all required pre-trained models for Zero-Shot SVC.
Models are cached locally and only downloaded once.
"""
import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download


WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")


MODELS = {
    "rmvpe": {
        "description": "RMVPE F0 Extractor (Robust Model for Vocal Pitch Estimation)",
        "repo_id": "lj1995/VoiceConversionWebUI",
        "files": ["rmvpe.pt"],
        "paper": "arXiv:2306.15412",
    },
    "contentvec": {
        "description": "ContentVec Content Encoder (HuBERT variant for voice conversion)",
        "repo_id": "lengyue233/content-vec-best",
        "files": ["pytorch_model.bin"],
        "paper": "Based on HuBERT (Hsu et al., 2021)",
    },
    "cam++": {
        "description": "CAM++ Speaker Encoder (speaker verification model)",
        "repo_id": "funasr/cam++",
        "files": ["cam++.pth"],
        "paper": "Chen et al., 2022 - CAM++ for Speaker Verification",
    },
    "rvc_base_generator": {
        "description": "RVC v2 Base Generator (pre-trained VITS generator)",
        "repo_id": "RVC-Boss/Retrieval-based-Voice-Conversion-WebUI",
        "files": [],
        "notes": "Generator weights are trained per-speaker. For zero-shot, "
                 "use the pre-trained base model or a community universal model.",
    },
    "hifigan_vocoder": {
        "description": "HiFi-GAN Vocoder (mel-to-waveform synthesis)",
        "repo_id": "lj1995/VoiceConversionWebUI",
        "files": ["hifigan_v2.pt"],
        "paper": "Kong et al., 2020 - HiFi-GAN",
    },
}


def download_all(output_dir: str = None, verbose: bool = True):
    """Download all pre-trained model weights.

    Args:
        output_dir: Directory to save weights. Default: ./weights/
        verbose: Print progress information.
    """
    output_dir = output_dir or WEIGHTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("Zero-Shot SVC - Pretrained Weight Downloader")
        print("=" * 70)
        print(f"\nWeights will be saved to: {output_dir}\n")

    results = {}

    for model_name, info in MODELS.items():
        if verbose:
            print(f"\n[{model_name.upper()}]")
            print(f"  Description: {info['description']}")
            print(f"  Source:      huggingface.co/{info['repo_id']}")
            if 'paper' in info:
                print(f"  Paper:       {info['paper']}")

        try:
            for filename in info["files"]:
                if verbose:
                    print(f"  Downloading {filename}...", end=" ")

                path = hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=filename,
                    cache_dir=output_dir,
                    local_dir=output_dir,
                )
                results[model_name] = {"status": "success", "path": path}

                if verbose:
                    print(f"OK")
                    print(f"  Saved to: {path}")

        except Exception as e:
            results[model_name] = {"status": "error", "error": str(e)}
            if verbose:
                print(f"  ERROR: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("Download Summary")
        print("=" * 70)
        for name, result in results.items():
            status = result["status"]
            if status == "success":
                print(f"  [OK]   {name}: {result['path']}")
            else:
                print(f"  [FAIL] {name}: {result['error']}")
        print("=" * 70)

    return results


def check_weights(output_dir: str = None) -> dict:
    """Check which model weights are available locally.

    Args:
        output_dir: Directory where weights are stored.

    Returns:
        Dictionary mapping model names to their local paths (or None if missing).
    """
    output_dir = output_dir or WEIGHTS_DIR
    status = {}

    for model_name, info in MODELS.items():
        found = False
        if os.path.exists(output_dir):
            for filename in info["files"]:
                path = os.path.join(output_dir, filename)
                if os.path.exists(path):
                    status[model_name] = path
                    found = True
                    break

        # Also check HuggingFace cache
        if not found:
            from huggingface_hub import try_to_load_from_cache
            for filename in info["files"]:
                cache_info = try_to_load_from_cache(
                    info["repo_id"], filename, cache_dir=output_dir
                )
                if cache_info is not None:
                    status[model_name] = cache_info
                    found = True
                    break

        if not found:
            status[model_name] = None

    return status


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download pretrained weights")
    parser.add_argument("--dir", type=str, default=None, help="Output directory")
    parser.add_argument("--check", action="store_true", help="Check existing weights")
    args = parser.parse_args()

    if args.check:
        status = check_weights(args.dir)
        print("Weight Status:")
        for name, path in status.items():
            if path:
                print(f"  [OK]   {name}: {path}")
            else:
                print(f"  [MISS] {name}")
    else:
        download_all(args.dir)