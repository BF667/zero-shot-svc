#!/usr/bin/env python3
"""
Zero-Shot SVC - Resource Setup & Validation Script

This script helps you:
1. Check system requirements
2. Download pretrained model weights
3. Validate installation
4. Provide optimization tips
5. Test the pipeline

Run: python setup_resources.py
"""
import os
import sys
import json
import time
import platform
import subprocess
from pathlib import Path


def print_banner():
    """Print ASCII banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🎤  ZERO-SHOT SVC - RESOURCE SETUP WIZARD  🎤             ║
║                                                              ║
║   Let's get your voice conversion system ready!              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check Python version compatibility."""
    print("\n📋 Checking Python version...")
    version = sys.version_info
    print(f"   Current: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("   ✅ Python version compatible (3.9+)")
        return True
    else:
        print("   ⚠️  Warning: Python 3.9+ recommended")
        return False


def check_system_resources():
    """Check available system resources."""
    print("\n💻 Checking system resources...")
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   RAM: {ram_gb:.1f} GB {'✅' if ram_gb >= 8 else '⚠️  (8GB+ recommended)'}")
    except ImportError:
        print("   RAM: Unknown (install psutil to check)")
    
    # Check disk space
    disk = psutil.disk_usage('/') if 'psutil' in dir() else None
    if disk:
        free_gb = disk.free / (1024**3)
        print(f"   Disk Free: {free_gb:.1f} GB {'✅' if free_gb >= 10 else '⚠️  (10GB+ recommended for weights)'}")
    
    # Check GPU (CUDA)
    gpu_available = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"   GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) ✅")
            gpu_available = True
        else:
            print("   GPU: Not detected (CPU mode will be used)")
    except ImportError:
        print("   GPU: PyTorch not installed yet")
    
    return gpu_available


def check_dependencies():
    """Check required Python packages."""
    print("\n📦 Checking dependencies...")
    
    required = {
        'numpy': 'Numerical computing',
        'scipy': 'Scientific computing',
        'librosa': 'Audio processing',
        'soundfile': 'Audio I/O',
        'torch': 'Deep learning framework',
        'torchaudio': 'Audio for PyTorch',
    }
    
    optional = {
        'gradio': 'Web UI',
        'fastapi': 'REST API server',
        'uvicorn': 'API server',
        'transformers': 'HuggingFace models',
        'huggingface_hub': 'Model downloads',
        'psutil': 'System monitoring',
    }
    
    installed = {}
    missing_required = []
    
    # Check required packages
    for package, desc in required.items():
        try:
            __import__(package)
            installed[package] = True
            print(f"   ✅ {package}: {desc}")
        except ImportError:
            installed[package] = False
            missing_required.append(package)
            print(f"   ❌ {package}: {desc} - MISSING")
    
    # Check optional packages
    print("\n   Optional packages:")
    for package, desc in optional.items():
        try:
            __import__(package)
            print(f"   ✅ {package}: {desc}")
        except ImportError:
            print(f"   ⚪ {package}: {desc} - not installed")
    
    return len(missing_required) == 0, missing_required


def download_weights():
    """Download pretrained model weights."""
    print("\n⬇️  Checking model weights...")
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    weight_files = {
        "rmvpe.pt": "RMVPE Pitch Extractor (~200MB)",
        "pytorch_model.bin": "ContentVec Content Encoder (~500MB)",
        "cam++.pth": "CAM++ Speaker Encoder (~100MB)",
        "hifigan_v2.pt": "HiFi-GAN Vocoder (~180MB)",
    }
    
    existing = []
    missing = []
    
    for filename, description in weight_files.items():
        filepath = weights_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024*1024)
            existing.append(filename)
            print(f"   ✅ {filename}: {description} ({size_mb:.0f} MB)")
        else:
            missing.append((filename, description))
    
    if missing:
        print(f"\n   Missing {len(missing)} weight file(s):")
        for filename, desc in missing:
            print(f"   ⬇️  {filename}: {desc}")
        
        print("\n   To download weights, run:")
        print("   python main.py download")
        print("\n   Or manually from HuggingFace:")
        print("   https://huggingface.co/lj1995/VoiceConversionWebUI")
        print("   https://huggingface.co/lengyue233/content-vec-best")
        print("   https://huggingface.co/funasr/cam%2B%2B")
    else:
        print("\n   🎉 All model weights present!")
    
    return len(missing) == 0


def validate_installation():
    """Run validation tests."""
    print("\n🧪 Running validation tests...")
    
    test_script = Path("scripts/validate_basic.py")
    if test_script.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(test_script)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("   ✅ All basic validation tests passed!")
                return True
            else:
                print("   ⚠️  Some validation tests failed:")
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                return False
        except subprocess.TimeoutExpired:
            print("   ⏱️  Validation timed out")
            return False
        except Exception as e:
            print(f"   ❌ Error running tests: {e}")
            return False
    else:
        print("   ⚪ Validation script not found (skipping)")
        return None


def provide_optimization_tips(gpu_available):
    """Provide context-specific optimization tips."""
    print("\n💡 Optimization Tips:")
    
    if gpu_available:
        print("""
   For GPU Acceleration:
   ─────────────────────
   • Use --neural flag for best quality
   • Enable CUDA optimizations:
     export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   • Consider half-precision for faster inference:
     Add model.half() before inference
   
   Recommended GPU Memory:
   • 4GB+: Basic neural mode
   • 8GB+: Full quality, longer audio
   • 16GB+: Batch processing, real-time possible
""")
    else:
        print("""
   CPU-Only Mode:
   ──────────────────
   • Signal processing mode works without downloads
   • For neural mode, consider cloud GPU options:
     - RunPod (runpod.io): $0.44/hr RTX 4090
     - Modal (modal.com): $30 free credit
     - HuggingFace Spaces: Free T4 tier
   • Keep audio under 30s for faster processing
""")
    
    print("""
   Audio Quality Tips:
   ────────────────────
   • Always separate vocals first (UVR5 recommended)
   • Use clean, dry reference audio (5-15s)
   • WAV format preferred over MP3
   • 16kHz+ sample rate for source audio
   
   Quick Start Commands:
   ──────────────────────
   # Launch web UI
   python gradio_app.py
   
   # Start API server  
   python main.py serve
   
   # Quick conversion
   python main.py convert -s input.wav -r reference.wav
   
   # Batch process
   python main.py batch -s *.wav -r target.wav
""")


def generate_report(results):
    """Generate a summary report."""
    report_path = Path("setup_report.json")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "platform": platform.platform(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "os": platform.system(),
        },
        "checks": results,
        "recommendations": []
    }
    
    # Add recommendations based on results
    if not results.get('deps_ok'):
        report["recommendations"].append(
            "Install missing dependencies: pip install -r requirements.txt"
        )
    
    if not results.get('weights_ok'):
        report["recommendations"].append(
            "Download model weights: python main.py download"
        )
    
    if not results.get('gpu_available'):
        report["recommendations"].append(
            "Consider using a cloud GPU for neural mode (RunPod, Modal)"
        )
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_path}")


def main():
    """Main setup wizard."""
    print_banner()
    
    results = {}
    
    # Step 1: Check Python
    results['python_ok'] = check_python_version()
    
    # Step 2: Check system
    results['gpu_available'] = check_system_resources()
    
    # Step 3: Check dependencies
    results['deps_ok'], missing_deps = check_dependencies()
    
    # Step 4: Check weights
    results['weights_ok'] = download_weights()
    
    # Step 5: Validate
    results['validation_ok'] = validate_installation()
    
    # Step 6: Tips
    provide_optimization_tips(results.get('gpu_available', False))
    
    # Generate report
    generate_report(results)
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 SETUP SUMMARY")
    print("="*60)
    
    all_good = all([
        results.get('python_ok', False),
        results.get('deps_ok', False),
        results.get('validation_ok', False) is not False,
    ])
    
    if all_good:
        print("""
   ✨ Your system is ready for Zero-Shot SVC!
   
   Next steps:
   1. Launch the web UI:  python gradio_app.py
   2. Or use the CLI:      python main.py convert -s src.wav -r ref.wav
   3. Read the full guide: cat RESOURCE_GUIDE.md
""")
    else:
        print("""
   ⚠️  Some setup steps need attention.
   
   Review the items marked with ❌ above,
   then run this script again to verify.
   
   For help, see: RESOURCE_GUIDE.md
""")
    
    return all_good


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
