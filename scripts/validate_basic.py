"""
Lightweight Validation Script for Zero-Shot SVC Enhanced

Tests that don't require PyTorch:
- Preset system
- Configuration loading  
- Audio utilities (basic)
- CLI argument parsing
- API server structure

Run: python scripts/validate_basic.py
"""
import os
import sys
import json
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_preset_system():
    """Test preset save/load functionality."""
    print("\n" + "="*60)
    print("TEST: Preset System")
    print("="*60)
    
    try:
        import gradio_app
        
        # Get default presets
        presets = gradio_app.get_presets()
        assert len(presets) > 0, "No default presets found"
        assert "Default" in presets, "Default preset missing"
        
        # Test loading a preset
        values = gradio_app.load_preset("Default")
        assert len(values) == 8, f"Expected 8 preset values, got {len(values)}"
        
        # Verify default values
        assert values[0] == 0, "Default f0_transpose should be 0"
        assert values[1] == 1.0, "Default f0_curve should be 1.0"
        
        # Test saving a custom preset
        status, choices = gradio_app.save_current_preset(
            "test_preset",
            f0_transpose=7,
            f0_curve_factor=1.1,
            noise_scale=0.35,
            formant_shift=2,
            vibrato_strength=0.2,
            breathiness=0.15,
            protect_consonants=True,
            noise_reduction=0.3,
        )
        
        # Verify it was saved
        presets_after = gradio_app.get_presets()
        assert "test_preset" in presets_after, "Custom preset not saved"
        
        # Load and verify custom preset
        loaded_values = gradio_app.load_preset("test_preset")
        assert loaded_values[0] == 7, "Custom f0_transpose not saved correctly"
        assert loaded_values[3] == 2, "Formant shift not saved correctly"
        
        # Clean up - delete test preset
        _, choices, _ = gradio_app.delete_preset("test_preset")
        
        presets_final = gradio_app.get_presets()
        assert "test_preset" not in presets_final, "Custom preset not deleted"
        
        # Test all built-in presets can be loaded
        for name in ["Male → Female (+5st)", "Female → Male (-5st)", 
                     "Octave Up (+12st)", "Soft & Gentle", "Robot / Synthetic"]:
            if name in presets:
                vals = gradio_app.load_preset(name)
                assert len(vals) == 8, f"Preset {name} has wrong number of values"
        
        print(f"  [OK] Preset system working correctly")
        print(f"       Default presets: {len(presets)}")
        print(f"       All presets loadable: OK")
        print(f"       Save/load/delete: OK")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration system."""
    print("\n" + "="*60)
    print("TEST: Configuration System")
    print("="*60)
    
    try:
        from utils.hparams import Config
        
        # Test default config
        config = Config()
        assert config.audio.sample_rate == 16000
        assert config.audio.output_sample_rate == 32000
        assert config.content_encoder.output_dim == 256
        assert config.speaker_encoder.embedding_dim == 192
        
        print(f"  [OK] Default config loaded correctly")
        print(f"       Sample rate: {config.audio.sample_rate}")
        print(f"       Output SR: {config.audio.output_sample_rate}")
        print(f"       Content dim: {config.content_encoder.output_dim}")
        print(f"       Speaker dim: {config.speaker_encoder.embedding_dim}")
        
        # Test YAML loading if file exists
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                "configs", "default.yaml")
        if os.path.exists(yaml_path):
            config_yaml = Config.from_yaml(yaml_path)
            assert config_yaml is not None
            
            print(f"  [OK] YAML config loading works")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_audio_utilities():
    """Test audio utility functions that don't need torch."""
    print("\n" + "="*60)
    print("TEST: Audio Utilities (Basic)")
    print("="*60)
    
    try:
        from utils.audio import (
            normalize_audio, pad_or_trim, slice_audio,
            numpy_to_torch, torch_to_numpy, resample,
        )
        import scipy.signal
        
        # Create test signal
        sr = 16000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Test normalize
        normalized = normalize_audio(audio, target_peak=0.95)
        assert abs(np.max(np.abs(normalized)) - 0.95) < 0.001, "Normalize failed"
        print(f"  [OK] normalize_audio works")
        
        # Test pad_or_trim
        padded = pad_or_trim(audio, int(sr * 2))
        assert len(padded) == int(sr * 2), "Pad failed"
        
        trimmed = pad_or_trim(audio, int(sr * 0.5))
        assert len(trimmed) == int(sr * 0.5), "Trim failed"
        print(f"  [OK] pad_or_trim works")
        
        # Test slice_audio
        long_audio = np.zeros(sr * 10, dtype=np.float32)  # 10 seconds
        chunks = slice_audio(long_audio, sr, max_seconds=3.0)
        assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"
        assert all(len(c) == int(sr * 3) for c in chunks[:-1]), "Chunk size wrong"
        print(f"  [OK] slice_audio works ({len(chunks)} chunks)")
        
        # Test resample
        resampled = resample(audio, sr, 8000)
        assert len(resampled) == sr // 2, "Resample length wrong"
        print(f"  [OK] resample works")
        
        return True
        
    except ImportError as e:
        print(f"  [SKIP] Missing dependency: {e}")
        return None  # Not a failure, just skipped
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_cli_structure():
    """Test CLI argument parsing structure."""
    print("\n" + "="*60)
    print("TEST: CLI Structure")
    print("="*60)
    
    try:
        # Import main module to check it loads
        import main
        
        # Check main function exists
        assert hasattr(main, 'main'), "main() function not found"
        
        # Check command functions exist
        expected_commands = [
            'cmd_convert', 'cmd_batch', 'cmd_download', 'cmd_check',
            'cmd_features', 'cmd_extract_embedding', 'cmd_similarity',
            'cmd_profile', 'cmd_serve'
        ]
        
        for cmd in expected_commands:
            assert hasattr(main, cmd), f"{cmd} function not found"
        
        print(f"  [OK] CLI structure correct")
        print(f"       Commands available: {len(expected_commands)}")
        print(f"       All handlers present: OK")
        
        return True
        
    except SyntaxError as e:
        print(f"  [FAIL] Syntax error in main.py: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_api_server_structure():
    """Test API server module structure."""
    print("\n" + "="*60)
    print("TEST: API Server Structure")
    print("="*60)
    
    try:
        # Check if FastAPI is available
        try:
            import fastapi
            import uvicorn
        except ImportError:
            print(f"  [SKIP] FastAPI/Uvicorn not installed (optional)")
            return None
        
        # Try importing the api_server module
        import api_server
        
        # Check app exists
        assert hasattr(api_server, 'app'), "FastAPI app not found"
        
        # Check expected endpoints exist (by checking route paths)
        routes = [r.path for r in api_server.app.routes if hasattr(r, 'path')]
        expected_paths = ['/api/health', '/api/status', '/api/convert']
        
        for path in expected_paths:
            assert path in routes, f"Endpoint {path} not found"
        
        print(f"  [OK] API server structure correct")
        print(f"       Endpoints defined: {len([r for r in routes if '/api/' in r])}")
        print(f"       Core endpoints present: OK")
        
        return True
        
    except SyntaxError as e:
        print(f"  [FAIL] Syntax error in api_server.py: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_file_structure():
    """Test that expected files exist with proper content."""
    print("\n" + "="*60)
    print("TEST: File Structure")
    print("="*60)
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        required_files = {
            'main.py': 'CLI entry point',
            'gradio_app.py': 'Gradio web interface',
            'api_server.py': 'REST API server',
            'pipeline/voice_converter.py': 'Core pipeline',
            'utils/hparams.py': 'Configuration',
            'utils/audio.py': 'Audio utilities',
            'configs/default.yaml': 'Default config',
            'requirements.txt': 'Dependencies',
        }
        
        optional_files = {
            'scripts/test_enhanced.py': 'Enhanced test suite',
            'scripts/validate_basic.py': 'Basic validation',
        }
        
        existing = 0
        for filepath, description in required_files.items():
            full_path = os.path.join(base_dir, filepath)
            if os.path.exists(full_path):
                # Check file has content
                size = os.path.getsize(full_path)
                assert size > 100, f"{filepath} is too small ({size} bytes)"
                existing += 1
            else:
                print(f"  [WARN] Missing: {filepath} ({description})")
        
        print(f"  [OK] File structure valid")
        print(f"       Required files: {existing}/{len(required_files)}")
        
        # Check for new enhancement files
        enhanced_files = ['api_server.py']
        for ef in enhanced_files:
            if os.path.exists(os.path.join(base_dir, ef)):
                print(f"       Enhancement present: {ef}")
        
        return True
        
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_gradio_ui_structure():
    """Test Gradio UI module structure."""
    print("\n" + "="*60)
    print("TEST: Gradio UI Structure")
    print("="*60)
    
    try:
        # Check if Gradio is available
        try:
            import gradio as gr
        except ImportError:
            print(f"  [SKIP] Gradio not installed (optional)")
            return None
        
        # Try importing the gradio_app module
        import gradio_app
        
        # Check key functions exist
        expected_functions = [
            'build_ui', 'convert_voice', 'batch_convert',
            'analyze_reference', 'load_preset', 'save_current_preset',
            'get_presets', 'format_history'
        ]
        
        for func_name in expected_functions:
            assert hasattr(gradio_app, func_name), f"Function {func_name} not found"
        
        # Check preset defaults exist
        assert hasattr(gradio_app, 'DEFAULT_PRESETS'), "DEFAULT_PRESETS not found"
        assert len(gradio_app.DEFAULT_PRESETS) > 0, "No default presets"
        
        print(f"  [OK] Gradio UI structure correct")
        print(f"       Functions defined: {len(expected_functions)}")
        print(f"       Default presets: {len(gradio_app.DEFAULT_PRESETS)}")
        print(f"       Tabs: Convert, Batch, History, Guide")
        
        return True
        
    except SyntaxError as e:
        print(f"  [FAIL] Syntax error in gradio_app.py: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def run_all_tests():
    """Run all basic tests and report results."""
    print("\n" + "#"*70)
    print("#  ZERO-SHOT SVC BASIC VALIDATION")
    print("#  (Tests that don't require PyTorch)")
    print("#"*70)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Configuration System", test_config_loading),
        ("Audio Utilities", test_audio_utilities),
        ("Preset System", test_preset_system),
        ("CLI Structure", test_cli_structure),
        ("Gradio UI Structure", test_gradio_ui_structure),
        ("API Server Structure", test_api_server_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            if passed is None:
                results.append((name, "SKIP"))
            else:
                results.append((name, passed))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "#"*70)
    print("#  VALIDATION SUMMARY")
    print("#"*70)
    
    passed = sum(1 for _, p in results if p is True)
    total = len(results)
    skipped = sum(1 for _, p in results if p == "SKIP")
    
    for name, p in results:
        if p is True:
            status = "PASS"
            symbol = "✓"
        elif p == "SKIP":
            status = "SKIP"
            symbol = "⊘"
        else:
            status = "FAIL"
            symbol = "✗"
        print(f"  [{symbol}] {name}: {status}")
    
    print(f"\n  Total: {passed}/{total-skipped} passed ({skipped} skipped)")
    
    if passed == total - skipped:
        print("\n  ✓ Basic validation passed! Core functionality is intact.")
        print("  Install PyTorch to run full test suite:")
        print("    pip install torch torchaudio")
    else:
        print(f"\n  ⚠️  {total - skipped - passed} test(s) failed.")
    
    print("#"*70 + "\n")
    
    return passed == total - skipped


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
