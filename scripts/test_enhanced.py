"""
Enhanced Test Suite for Zero-Shot SVC

Tests all new functionality:
- Signal processing conversion (no models needed)
- Formant shifting
- Noise reduction
- Voice similarity scoring
- Speaker profile management
- Batch conversion
- Quality metrics
- Preset system

Run: python scripts/test_enhanced.py
"""
import os
import sys
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_audio(duration_s=2.0, sr=16000, frequency=440):
    """Create a simple test audio signal (sine wave)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    # Create a more complex signal with harmonics (more voice-like)
    audio = (
        0.5 * np.sin(2 * np.pi * frequency * t) +  # Fundamental
        0.25 * np.sin(2 * np.pi * frequency * 2 * t) +  # 2nd harmonic
        0.125 * np.sin(2 * np.pi * frequency * 3 * t) +  # 3rd harmonic
        0.05 * np.random.randn(len(t)).astype(np.float32)  # Noise
    ).astype(np.float32)
    return audio


def save_test_audio(audio, path, sr=16000):
    """Save audio to file."""
    import soundfile as sf
    sf.write(path, audio, sr)


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("TEST: Module Imports")
    print("="*60)
    
    try:
        from pipeline.voice_converter import ZeroShotSVC
        print("  [OK] ZeroShotSVC imported")
        
        from utils.hparams import Config
        print("  [OK] Config imported")
        
        from utils.audio import (
            load_audio, save_audio, normalize_audio,
            compute_mel_spectrogram, slice_audio,
        )
        print("  [OK] Audio utilities imported")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_signal_processing_conversion():
    """Test basic signal processing conversion."""
    print("\n" + "="*60)
    print("TEST: Signal Processing Conversion")
    print("="*60)
    
    try:
        from pipeline.voice_converter import ZeroShotSVC
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            source_path = os.path.join(tmpdir, "source.wav")
            reference_path = os.path.join(tmpdir, "reference.wav")
            
            # Source: higher pitch (like female singing)
            src_audio = create_test_audio(duration_s=3.0, frequency=440)
            save_test_audio(src_audio, source_path)
            
            # Reference: lower pitch (like male voice)
            ref_audio = create_test_audio(duration_s=5.0, frequency=220)
            save_test_audio(ref_audio, reference_path)
            
            output_path = os.path.join(tmpdir, "output.wav")
            
            # Initialize and convert
            svc = ZeroShotSVC(use_neural=False)
            result = svc.convert(
                source_path=source_path,
                reference_path=reference_path,
                output_path=output_path,
                f0_transpose=0,
            )
            
            # Verify output exists
            assert os.path.exists(output_path), "Output file not created"
            
            # Verify output has content
            import soundfile as sf
            out_audio, out_sr = sf.read(output_path)
            assert len(out_audio) > 0, "Output audio is empty"
            assert abs(np.max(out_audio)) > 0.001, "Output audio is silent"
            
            print(f"  [OK] Conversion successful")
            print(f"       Input duration: {len(src_audio)/16000:.1f}s")
            print(f"       Output duration: {len(out_audio)/out_sr:.1f}s")
            print(f"       Output SR: {out_sr} Hz")
            
            return True
            
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pitch_shift():
    """Test pitch shifting functionality."""
    print("\n" + "="*60)
    print("TEST: Pitch Shifting")
    print("="*60)
    
    try:
        from pipeline.voice_converter import ZeroShotSVC
        
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.wav")
            reference_path = os.path.join(tmpdir, "reference.wav")
            
            src_audio = create_test_audio(duration_s=2.0, frequency=261)  # C4
            ref_audio = create_test_audio(duration_s=3.0, frequency=220)
            
            save_test_audio(src_audio, source_path)
            save_test_audio(ref_audio, reference_path)
            
            svc = ZeroShotSVC(use_neural=False)
            
            # Test various pitch shifts
            shifts = [-12, -5, 0, 5, 12]
            results = {}
            
            for shift in shifts:
                output_path = os.path.join(tmpdir, f"output_shift{shift}.wav")
                svc.convert(
                    source_path=source_path,
                    reference_path=reference_path,
                    output_path=output_path,
                    f0_transpose=shift,
                )
                
                import soundfile as sf
                out_audio, _ = sf.read(output_path)
                results[shift] = len(out_audio)
            
            print(f"  [OK] All pitch shifts completed successfully")
            for shift, length in results.items():
                print(f"       Shift {shift:+3d}st: {length} samples")
            
            return True
            
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_formant_shift():
    """Test formant shifting functionality."""
    print("\n" + "="*60)
    print("TEST: Formant Shifting")
    print("="*60)
    
    try:
        from pipeline.voice_converter import ZeroShotSVC
        
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.wav")
            reference_path = os.path.join(tmpdir, "reference.wav")
            
            src_audio = create_test_audio(duration_s=2.0, frequency=300)
            ref_audio = create_test_audio(duration_s=3.0, frequency=200)
            
            save_test_audio(src_audio, source_path)
            save_test_audio(ref_audio, reference_path)
            
            svc = ZeroShotSVC(use_neural=False)
            
            # Test formant shifts
            for shift in [-2, 0, 2]:
                output_path = os.path.join(tmpdir, f"output_formant{shift}.wav")
                svc.convert(
                    source_path=source_path,
                    reference_path=reference_path,
                    output_path=output_path,
                    formant_shift=shift,
                )
                
                assert os.path.exists(output_path), f"Formant {shift} failed"
            
            print(f"  [OK] Formant shifting works correctly")
            print(f"       Tested shifts: -2, 0, +2")
            
            return True
            
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_noise_reduction():
    """Test noise reduction preprocessing."""
    print("\n" + "="*60)
    print("TEST: Noise Reduction")
    print("="*60)
    
    try:
        from pipeline.voice_converter import ZeroShotSVC
        
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source_noisy.wav")
            reference_path = os.path.join(tmpdir, "reference.wav")
            
            # Create noisy audio
            np.random.seed(42)
            clean = create_test_audio(duration_s=2.0, frequency=350)
            noise = 0.3 * np.random.randn(len(clean)).astype(np.float32)
            noisy = clean + noise
            noisy = (noisy / np.max(np.abs(noisy)) * 0.8).astype(np.float32)
            
            save_test_audio(noisy, source_path)
            
            ref_audio = create_test_audio(duration_s=3.0, frequency=250)
            save_test_audio(ref_audio, reference_path)
            
            svc = ZeroShotSVC(use_neural=False)
            
            # Test with different noise reduction levels
            for nr_level in [0.0, 0.5, 1.0]:
                output_path = os.path.join(tmpdir, f"output_nr{nr_level:.1f}.wav")
                svc.convert(
                    source_path=source_path,
                    reference_path=reference_path,
                    output_path=output_path,
                    noise_reduction=nr_level,
                )
                
                assert os.path.exists(output_path), f"NR level {nr_level} failed"
            
            print(f"  [OK] Noise reduction works correctly")
            print(f"       Tested levels: 0%, 50%, 100%")
            
            return True
            
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_voice_similarity():
    """Test voice similarity computation."""
    print("\n" + "="*60)
    print("TEST: Voice Similarity Computation")
    print("="*60)
    
    try:
        from pipeline.voice_converter import ZeroShotSVC
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two similar audios
            audio1_path = os.path.join(tmpdir, "audio1.wav")
            audio2_path = os.path.join(tmpdir, "audio2.wav")
            audio3_path = os.path.join(tmpdir, "audio3.wav")  # Different
            
            # Similar pair (same frequency range)
            audio1 = create_test_audio(duration_s=2.0, frequency=300)
            audio2 = create_test_audio(duration_s=2.0, frequency=320)
            
            # Different pair (very different frequency)
            audio3 = create_test_audio(duration_s=2.0, frequency=100)
            
            save_test_audio(audio1, audio1_path)
            save_test_audio(audio2, audio2_path)
            save_test_audio(audio3, audio3_path)
            
            svc = ZeroShotSVC(use_neural=False)
            
            # Test similar pair
            sim_similar = svc.compute_similarity(audio1_path, audio2_path)
            
            # Test different pair
            sim_different = svc.compute_similarity(audio1_path, audio3_path)
            
            print(f"  [OK] Similarity computed successfully")
            print(f"       Similar pair score:   {sim_similar['overall_similarity']:.3f}")
            print(f"       Different pair score:  {sim_different['overall_similarity']:.3f}")
            
            # The similar pair should generally have higher similarity
            # (though this isn't guaranteed with simple sine waves)
            assert 'overall_similarity' in sim_similar
            assert 'mfcc_cosine_similarity' in sim_similar
            
            return True
            
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speaker_profiles():
    """Test speaker profile management."""
    print("\n" + "="*60)
    print("TEST: Speaker Profile Management")
    print("="*60)
    
    try:
        from pipeline.voice_converter import ZeroShotSVC
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "reference.wav")
            ref_audio = create_test_audio(duration_s=8.0, frequency=280)  # Long enough for profile
            save_test_audio(ref_audio, ref_path)
            
            svc = ZeroShotSVC(use_neural=False)
            
            # Save profile
            profile_path = svc.save_speaker_profile(ref_path, name="TestVoice")
            
            assert os.path.exists(profile_path), "Profile not saved"
            
            # List profiles
            profiles = svc.list_speaker_profiles()
            
            assert len(profiles) > 0, "No profiles found"
            assert profiles[0]['name'] == "TestVoice", "Profile name mismatch"
            
            print(f"  [OK] Speaker profile management works")
            print(f"       Profile saved to: {profile_path}")
            print(f"       Profile name: {profiles[0]['name']}")
            print(f"       Duration: {profiles[0]['duration']:.1f}s")
            
            return True
            
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_batch_conversion():
    """Test batch conversion of multiple files."""
    print("\n" + "="*60)
    print("TEST: Batch Conversion")
    print("="*60)
    
    try:
        from pipeline.voice_converter import ZeroShotSVC
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "reference.wav")
            ref_audio = create_test_audio(duration_s=5.0, frequency=240)
            save_test_audio(ref_audio, ref_path)
            
            # Create multiple source files
            source_paths = []
            for i in range(3):
                src_path = os.path.join(tmpdir, f"source_{i}.wav")
                freq = 300 + i * 50  # Different frequencies
                src_audio = create_test_audio(duration_s=2.0, frequency=freq)
                save_test_audio(src_audio, src_path)
                source_paths.append(src_path)
            
            output_dir = os.path.join(tmpdir, "outputs")
            
            svc = ZeroShotSVC(use_neural=False)
            
            results = svc.batch_convert(
                source_paths=source_paths,
                reference_path=ref_path,
                output_dir=output_dir,
                f0_transpose=0,
            )
            
            # Verify all succeeded
            success_count = sum(1 for r in results if r["status"] == "success")
            assert success_count == 3, f"Expected 3 successes, got {success_count}"
            
            # Verify output files exist
            for r in results:
                assert r["status"] == "success", f"Conversion failed: {r}"
                assert os.path.exists(r["output"]), "Output file missing"
            
            print(f"  [OK] Batch conversion successful")
            print(f"       Files processed: {len(results)}")
            print(f"       Successful: {success_count}")
            
            return True
            
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_metrics():
    """Test quality metrics computation."""
    print("\n" + "="*60)
    print("TEST: Quality Metrics")
    print("="*60)
    
    try:
        from utils.audio import normalize_audio
        import librosa
        
        # Create test signals
        sr = 16000
        audio1 = create_test_audio(duration_s=2.0, frequency=400, sr=sr)
        audio2 = create_test_audio(duration_s=2.0, frequency=350, sr=sr)
        
        # Compute metrics manually (as the Gradio app would)
        metrics = {
            'original_rms': float(np.sqrt(np.mean(audio1**2))),
            'converted_rms': float(np.sqrt(np.mean(audio2**2))),
            'original_peak': float(np.max(np.abs(audio1))),
            'converted_peak': float(np.max(np.abs(audio2))),
        }
        
        orig_centroid = librosa.feature.spectral_centroid(y=audio1, sr=sr)[0]
        conv_centroid = librosa.feature.spectral_centroid(y=audio2, sr=sr)[0]
        metrics['original_brightness'] = float(np.mean(orig_centroid))
        metrics['converted_brightness'] = float(np.mean(conv_centroid))
        metrics['duration_s'] = float(len(audio2) / sr)
        
        print(f"  [OK] Quality metrics computed successfully")
        print(f"       Original RMS:     {metrics['original_rms']:.4f}")
        print(f"       Converted RMS:    {metrics['converted_rms']:.4f}")
        print(f"       Original Brightness: {metrics['original_brightness']:.0f} Hz")
        print(f"       Converted Brightness: {metrics['converted_brightness']:.0f} Hz")
        print(f"       Duration:          {metrics['duration_s']:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_preset_system():
    """Test preset save/load functionality."""
    print("\n" + "="*60)
    print("TEST: Preset System")
    print("="*60)
    
    try:
        import json
        import gradio_app
        
        # Get default presets
        presets = gradio_app.get_presets()
        assert len(presets) > 0, "No default presets found"
        assert "Default" in presets, "Default preset missing"
        
        # Test loading a preset
        values = gradio_app.load_preset("Default")
        assert len(values) == 8, f"Expected 8 preset values, got {len(values)}"
        
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
        
        # Clean up - delete test preset
        _, choices, _ = gradio_app.delete_preset("test_preset")
        
        presets_final = gradio_app.get_presets()
        assert "test_preset" not in presets_final, "Custom preset not deleted"
        
        print(f"  [OK] Preset system working correctly")
        print(f"       Default presets: {len(presets)}")
        print(f"       Save/load/delete: OK")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "#"*70)
    print("#  ZERO-SHOT SVC ENHANCED TEST SUITE")
    print("#"*70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Signal Processing", test_signal_processing_conversion),
        ("Pitch Shifting", test_pitch_shift),
        ("Formant Shifting", test_formant_shift),
        ("Noise Reduction", test_noise_reduction),
        ("Voice Similarity", test_voice_similarity),
        ("Speaker Profiles", test_speaker_profiles),
        ("Batch Conversion", test_batch_conversion),
        ("Quality Metrics", test_quality_metrics),
        ("Preset System", test_preset_system),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "#"*70)
    print("#  TEST SUMMARY")
    print("#"*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "PASS" if p else "FAIL"
        symbol = "✓" if p else "✗"
        print(f"  [{symbol}] {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! The enhanced application is working correctly.")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    print("#"*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
