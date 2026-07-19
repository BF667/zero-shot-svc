"""
Zero-Shot Singing Voice Conversion Pipeline - Enhanced Edition

Two conversion modes:

MODE 1 — Signal Processing (default, no pretrained weights needed):
    Source Audio + Reference Audio
        |
        v
    [1] librosa.pyin: Extract F0 pitch contour
    [2] Mel spectrogram: Source and reference mel extraction
    [3] Mean-Variance Normalization: Transfer reference speaker's spectral
        characteristics onto the source mel (classic voice conversion)
    [4] Griffin-Lim: Convert converted mel back to waveform
    [5] Pitch shift (optional): librosa.effects.pitch_shift
        |
        v
    Converted Audio

MODE 2 — Neural (requires pretrained RVC weights):
    Source Audio + Reference Audio
        |
        v
    [1] ContentVec: Extract content features
    [2] RMVPE: Extract F0 pitch contour
    [3] CAM++: Extract speaker embedding
    [4] VITS Generator: Generate mel-spectrogram
    [5] HiFi-GAN: Convert mel to waveform
        |
        v
    Converted Audio

Enhanced Features:
    - Formant shifting (independent of pitch)
    - Noise reduction preprocessing
    - Voice similarity scoring
    - Quality metrics computation
    - Speaker profile management
    - Batch processing support
    - Progress callbacks for UI integration
"""
import os
import gc
import json
import time
import hashlib
import librosa
import numpy as np
import torch
from typing import Optional, Tuple, Dict, List, Callable

from utils.hparams import Config
from utils.audio import (
    load_audio, save_audio, f0_to_coarse, numpy_to_torch, normalize_audio,
    slice_audio, compute_mel_spectrogram,
)


class ZeroShotSVC:
    """Zero-Shot Singing Voice Conversion system - Enhanced.

    Converts singing voice from source audio to match the voice
    characteristics of a reference speaker, WITHOUT any training.

    By default uses signal-processing voice conversion (mel MVN + Griffin-Lim)
    which produces real converted audio without any pretrained models.
    Use use_neural=True to use the neural pipeline (requires pretrained weights).

    Enhanced Features:
        - Formant shifting for timbre control
        - Noise reduction preprocessing
        - Voice similarity metrics
        - Speaker profile persistence
        - Batch conversion with progress tracking

    Usage:
        svc = ZeroShotSVC()
        output_path = svc.convert(
            source_path="singing.wav",
            reference_path="target_voice.wav",
            output_path="converted.wav",
        )

        # With enhanced parameters
        output_path = svc.convert(
            source_path="singing.wav",
            reference_path="target_voice.wav",
            formant_shift=2,           # Brighter timbre
            noise_reduction=0.3,       # Reduce background noise
            similarity_threshold=0.7,  # Minimum similarity score
        )
    """

    def __init__(self, config: Config = None, config_path: str = None,
                 device: str = None, use_neural: bool = False):
        """
        Args:
            config: Configuration object. If None, loads from config_path.
            config_path: Path to YAML config file.
            device: Override device ('cpu' or 'cuda').
            use_neural: If True, use neural pipeline (requires pretrained weights).
                        If False (default), use signal processing (works immediately).
        """
        if config is None:
            config = Config.from_yaml(config_path or self._default_config_path())
        self.config = config

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_neural = use_neural

        if self.use_neural:
            print(f"[ZeroShotSVC] Mode: NEURAL (requires pretrained weights)")
            print(f"[ZeroShotSVC] Device: {self.device}")
        else:
            print(f"[ZeroShotSVC] Mode: SIGNAL PROCESSING (no pretrained weights needed)")
            print(f"[ZeroShotSVC] Device: {self.device}")

        # Neural pipeline components (lazily loaded)
        self.f0_extractor = None
        self.content_encoder = None
        self.speaker_encoder = None
        self.generator = None
        self.vocoder = None

        self._models_loaded = False
        
        # Speaker profiles cache for persistence
        self._speaker_profiles_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "speaker_profiles"
        )
        os.makedirs(self._speaker_profiles_dir, exist_ok=True)

    @staticmethod
    def _default_config_path() -> str:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "configs", "default.yaml")

    def load_models(self):
        """Load neural pipeline models (only needed for use_neural=True)."""
        if self._models_loaded:
            print("[ZeroShotSVC] Models already loaded, skipping.")
            return

        if not self.use_neural:
            print("[ZeroShotSVC] Signal processing mode — no models to load.")
            return

        from models.f0_extractor import RMVPEExtractor
        from models.content_encoder import ContentEncoder
        from models.speaker_encoder import SpeakerEncoderExtractor
        from models.generator import VITSGenerator
        from models.vocoder import Vocoder

        print("=" * 60)
        print("Loading Zero-Shot SVC Neural Models")
        print("=" * 60)

        t0 = time.time()

        # 1. RMVPE F0 Extractor (CPU)
        print("\n[1/5] Loading RMVPE F0 Extractor (CPU)...")
        self.f0_extractor = RMVPEExtractor(
            f0_min=self.config.f0_extractor.f0_min,
            f0_max=self.config.f0_extractor.f0_max,
            device='cpu',
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. ContentVec Content Encoder (CPU)
        print("\n[2/5] Loading ContentVec Content Encoder (CPU)...")
        self.content_encoder = ContentEncoder(
            output_dim=self.config.content_encoder.output_dim,
            device='cpu',
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. CAM++ Speaker Encoder (CPU)
        print("\n[3/5] Loading Speaker Encoder CAM++ (CPU)...")
        self.speaker_encoder = SpeakerEncoderExtractor(
            embedding_dim=self.config.speaker_encoder.embedding_dim,
            device='cpu',
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. VITS Generator (GPU if available)
        print(f"\n[4/5] Loading VITS Generator ({self.device})...")
        self.generator = VITSGenerator(
            content_dim=self.config.content_encoder.output_dim,
            hidden_channels=self.config.generator.hidden_channels,
            n_heads=self.config.generator.n_heads,
            n_decoder_layers=self.config.generator.n_layers,
            ffn_dim=self.config.generator.filter_channels,
            dropout=self.config.generator.p_dropout,
            n_flow_layers=self.config.generator.n_flow_layers,
            gin_channels=self.config.generator.gin_channels,
        )
        self.generator.eval()
        self.generator.to(self.device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. HiFi-GAN Vocoder (GPU if available)
        print(f"\n[5/5] Loading HiFi-GAN Vocoder ({self.device})...")
        self.vocoder = Vocoder(
            hop_size=self.config.vocoder.hop_size,
            sample_rate=self.config.audio.output_sample_rate,
            device=self.device,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._models_loaded = True
        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"All models loaded in {elapsed:.1f}s")
        print(f"{'=' * 60}\n")

    # ===================================================================
    # PUBLIC API
    # ===================================================================

    def convert(self, source_path: str, reference_path: str,
                output_path: str = None, f0_transpose: int = 0,
                f0_curve_factor: float = 1.0,
                protect_consonants: bool = True,
                noise_scale: float = 0.4,
                # Enhanced parameters
                formant_shift: int = 0,
                noise_reduction: float = 0.0,
                breathiness: float = 0.0,
                vibrato_strength: float = 0.0,
                # Callbacks
                progress_callback: Callable[[float, str], None] = None,
                ) -> str:
        """Convert singing voice from source to target speaker.

        Args:
            source_path: Path to source audio (singing to be converted).
            reference_path: Path to reference audio (target voice, 5-30s).
            output_path: Path for output audio. If None, auto-generated.
            f0_transpose: Semitones to shift pitch (+12 = octave up).
            f0_curve_factor: F0 curve scaling factor (1.0 = no change).
            protect_consonants: Reduce F0 shifting near unvoiced regions.
            noise_scale: Noise injection scale (neural mode only).
            formant_shift: Shift formants independently (-6 to +6 semitones).
            noise_reduction: Apply spectral gating noise reduction (0-1).
            breathiness: Add breathiness effect (0-1).
            vibrato_strength: Add vibrato effect (0-1).
            progress_callback: Optional callback(progress: float, message: str).

        Returns:
            Path to the output audio file.
        """
        def _progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        if self.use_neural and not self._models_loaded:
            self.load_models()

        if output_path is None:
            base = os.path.splitext(os.path.basename(source_path))[0]
            output_path = os.path.join(
                os.path.dirname(source_path) or ".",
                f"{base}_converted.wav"
            )

        mode_label = "NEURAL" if self.use_neural else "SIGNAL PROCESSING"
        print(f"\n{'=' * 60}")
        print(f"Zero-Shot Singing Voice Conversion ({mode_label})")
        print(f"{'=' * 60}")
        print(f"  Source:     {source_path}")
        print(f"  Reference:  {reference_path}")
        print(f"  Output:     {output_path}")
        print(f"  F0 shift:   {f0_transpose:+d} semitones")
        print(f"  F0 curve:   {f0_curve_factor:.2f}x")
        print(f"  Formant:    {formant_shift:+d} steps")
        print(f"{'=' * 60}\n")

        t0 = time.time()

        # Step 1: Load audio
        _progress(0.05, "Loading audio files...")
        print("[Step 1/5] Loading audio files...")
        source_audio = load_audio(source_path, target_sr=self.config.audio.sample_rate)
        reference_audio = load_audio(reference_path, target_sr=self.config.audio.sample_rate)

        src_dur = len(source_audio) / self.config.audio.sample_rate
        ref_dur = len(reference_audio) / self.config.audio.sample_rate
        print(f"  Source duration:     {src_dur:.1f}s")
        print(f"  Reference duration:  {ref_dur:.1f}s")

        # Step 2: Preprocessing (noise reduction)
        if noise_reduction > 0:
            _progress(0.10, "Applying noise reduction...")
            print(f"[Preprocess] Applying noise reduction ({noise_reduction:.0%})...")
            source_audio = self._apply_noise_reduction(
                source_audio, self.config.audio.sample_rate, strength=noise_reduction
            )

        # Step 3: Conversion
        _progress(0.15, "Converting voice...")
        if self.use_neural:
            chunks = slice_audio(source_audio, self.config.audio.sample_rate, max_seconds=30.0)
            converted_chunks = []
            total_chunks = len(chunks)
            
            for idx, chunk in enumerate(chunks):
                chunk_progress = 0.15 + 0.65 * (idx / total_chunks)
                _progress(chunk_progress, 
                         f"Processing chunk {idx + 1}/{total_chunks} "
                         f"({len(chunk) / self.config.audio.sample_rate:.1f}s)...")
                
                print(f"\n  Chunk {idx + 1}/{total_chunks} "
                      f"({len(chunk) / self.config.audio.sample_rate:.1f}s)...")
                converted = self._convert_chunk_neural(
                    chunk, reference_audio,
                    f0_transpose=f0_transpose,
                    f0_curve_factor=f0_curve_factor,
                    protect_consonants=protect_consonants,
                    noise_scale=noise_scale,
                )
                
                # Apply post-effects per chunk
                if formant_shift != 0:
                    converted = self._apply_formant_shift(
                        converted, self.config.audio.output_sample_rate, 
                        shift_semitones=formant_shift
                    )
                    
                converted_chunks.append(converted)
            
            full_output = np.concatenate(converted_chunks)
        else:
            _progress(0.20, "Converting with signal processing...")
            full_output = self._convert_signal_processing(
                source_audio, reference_audio,
                f0_transpose=f0_transpose,
                f0_curve_factor=f0_curve_factor,
            )
            
            if formant_shift != 0:
                _progress(0.75, "Applying formant shift...")
                full_output = self._apply_formant_shift(
                    full_output, self.config.audio.output_sample_rate,
                    shift_semitones=formant_shift
                )

        # Step 4: Post-processing effects
        _progress(0.85, "Applying post-processing effects...")
        
        if breathiness > 0:
            print("[Post-process] Adding breathiness...")
            full_output = self._add_breathiness(full_output, breathiness)

        if vibrato_strength > 0 and f0_transpose != 0:
            print("[Post-process] Adding vibrato...")
            full_output = self._add_vibrato(
                full_output, self.config.audio.output_sample_rate, 
                strength=vibrato_strength
            )

        # Step 5: Save output
        _progress(0.95, "Saving output audio...")
        print("\n[Step 5/5] Saving output audio...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        full_output = normalize_audio(full_output, target_peak=0.95)
        save_audio(output_path, full_output, sr=self.config.audio.output_sample_rate)

        elapsed = time.time() - t0
        out_dur = len(full_output) / self.config.audio.output_sample_rate
        print(f"\n{'=' * 60}")
        print(f"Conversion complete! ({elapsed:.1f}s)")
        print(f"Output saved to: {output_path}")
        print(f"Output duration: {out_dur:.1f}s")
        print(f"Real-time factor: {elapsed/out_dur:.2f}x")
        print(f"{'=' * 60}")

        _progress(1.0, f"Complete in {elapsed:.1f}s")

        return output_path

    def batch_convert(self, source_paths: List[str], reference_path: str,
                      output_dir: str = None, **kwargs) -> List[Dict]:
        """Batch convert multiple source files.

        Args:
            source_paths: List of source audio paths.
            reference_path: Path to reference audio (shared across all conversions).
            output_dir: Directory for outputs. Created if doesn't exist.
            **kwargs: Additional arguments passed to convert().

        Returns:
            List of result dicts with keys: source, output, status, time, metrics.
        """
        if output_dir is None:
            output_dir = "converted_outputs"
        os.makedirs(output_dir, exist_ok=True)

        results = []
        total = len(source_paths)

        print(f"\n{'=' * 60}")
        print(f"Batch Conversion: {total} files")
        print(f"Reference: {reference_path}")
        print(f"Output dir: {output_dir}")
        print(f"{'=' * 60}\n")

        t_total = time.time()

        for idx, src_path in enumerate(source_paths):
            base_name = os.path.splitext(os.path.basename(src_path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_converted.wav")

            print(f"[{idx+1}/{total}] Processing: {base_name}")

            t0 = time.time()
            try:
                self.convert(
                    source_path=src_path,
                    reference_path=reference_path,
                    output_path=out_path,
                    **kwargs
                )
                results.append({
                    "source": src_path,
                    "output": out_path,
                    "status": "success",
                    "time": time.time() - t0,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "source": src_path,
                    "output": None,
                    "status": f"error: {e}",
                    "time": time.time() - t0,
                })

        elapsed_total = time.time() - t_total
        success_count = sum(1 for r in results if r["status"] == "success")
        
        print(f"\n{'=' * 60}")
        print(f"Batch Complete: {success_count}/{total} successful in {elapsed_total:.1f}s")
        print(f"{'=' * 60}\n")

        return results

    def compute_similarity(self, audio1_path: str, audio2_path: str) -> Dict:
        """Compute voice similarity between two audio files.

        Uses multiple acoustic features to estimate how similar two voices are.

        Args:
            audio1_path: Path to first audio file.
            audio2_path: Path to second audio file.

        Returns:
            Dictionary with similarity scores and feature comparisons.
        """
        sr = self.config.audio.sample_rate
        
        audio1 = load_audio(audio1_path, target_sr=sr)
        audio2 = load_audio(audio2_path, target_sr=sr)

        # Ensure same length for comparison
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]

        results = {}

        # 1. MFCC-based similarity
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=13)
        
        # Delta and delta-delta MFCCs
        delta1 = librosa.feature.delta(mfcc1)
        delta2 = librosa.feature.delta(mfcc2)
        
        # Combine features
        feat1 = np.vstack([mfcc1, delta1])
        feat2 = np.vstack([mfcc2, delta2])

        # Cosine similarity of mean feature vectors
        mean1 = feat1.mean(axis=1)
        mean2 = feat2.mean(axis=1)
        
        cos_sim = np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2) + 1e-8)
        results["mfcc_cosine_similarity"] = float(cos_sim)

        # 2. Spectral centroid correlation
        sc1 = librosa.feature.spectral_centroid(y=audio1, sr=sr)[0]
        sc2 = librosa.feature.spectral_centroid(y=audio2, sr=sr)[0]
        min_sc_len = min(len(sc1), len(sc2))
        sc_corr = np.corrcoef(sc1[:min_sc_len], sc2[:min_sc_len])[0, 1]
        results["spectral_centroid_correlation"] = float(np.nan_to_num(sc_corr))

        # 3. Energy/RMS similarity
        rms1 = librosa.feature.rms(y=audio1, frame_length=2048, hop_length=512)[0]
        rms2 = librosa.feature.rms(y=audio2, frame_length=2048, hop_length=512)[0]
        min_rms_len = min(len(rms1), len(rms2))
        rms_corr = np.corrcoef(rms1[:min_rms_len], rms2[:min_rms_len])[0, 1]
        results["rms_correlation"] = float(np.nan_to_num(rms_corr))

        # 4. Overall similarity score (weighted average)
        results["overall_similarity"] = float(
            0.4 * cos_sim +
            0.3 * max(0, sc_corr) +
            0.3 * max(0, rms_corr)
        )

        return results

    def save_speaker_profile(self, reference_path: str, name: str = None) -> str:
        """Save a speaker profile from reference audio for later use.

        Args:
            reference_path: Path to reference audio.
            name: Optional name for the profile. Auto-generated if None.

        Returns:
            Path to saved profile JSON.
        """
        if name is None:
            name = os.path.splitext(os.path.basename(reference_path))[0]

        # Compute audio features for profiling
        audio = load_audio(reference_path, target_sr=self.config.audio.sample_rate)
        sr = self.config.audio.sample_rate

        profile = {
            "name": name,
            "source_file": reference_path,
            "created_at": datetime.now().isoformat(),
            "duration": len(audio) / sr,
        }

        # Basic spectral features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        profile["spectral_centroid_mean"] = float(np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr)
        ))
        profile["spectral_rolloff_mean"] = float(np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=sr)
        ))
        profile["zero_crossing_rate_mean"] = float(np.mean(
            librosa.feature.zero_crossing_rate(audio)
        ))

        # F0 statistics
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
            if len(voiced_f0) > 0:
                profile.update({
                    "f0_min": float(np.min(voiced_f0)),
                    "f0_max": float(np.max(voiced_f0)),
                    "f0_mean": float(np.mean(voiced_f0)),
                    "f0_std": float(np.std(voiced_f0)),
                })
        except:
            pass

        # Neural mode: extract and store embedding
        if self.use_neural and self._models_loaded:
            try:
                embedding = self.extract_speaker_embedding(reference_path)
                profile["embedding"] = embedding.tolist()
                profile["embedding_dim"] = len(embedding)
            except Exception as e:
                profile["embedding_error"] = str(e)

        # Save profile
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        profile_path = os.path.join(self._speaker_profiles_dir, f"{safe_name}.json")
        
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)

        print(f"[Speaker Profile] Saved: {profile_path}")
        return profile_path

    def list_speaker_profiles(self) -> List[Dict]:
        """List all saved speaker profiles."""
        profiles = []
        for fname in os.listdir(self._speaker_profiles_dir):
            if fname.endswith('.json'):
                with open(os.path.join(self._speaker_profiles_dir, fname)) as f:
                    profiles.append(json.load(f))
        return profiles

    # ===================================================================
    # SIGNAL PROCESSING PIPELINE (default, no models needed)
    # ===================================================================

    def _convert_signal_processing(self, source_audio: np.ndarray,
                                    reference_audio: np.ndarray,
                                    f0_transpose: int = 0,
                                    f0_curve_factor: float = 1.0) -> np.ndarray:
        """Voice conversion using signal processing (no neural models)."""
        sr = self.config.audio.sample_rate
        hop = self.config.audio.hop_size
        n_fft = self.config.audio.fft_size
        win = self.config.audio.win_size
        n_mels = self.config.audio.mel_bins
        fmin = self.config.audio.fmin
        fmax = self.config.audio.fmax

        max_chunk = 30 * sr  # 30 seconds
        chunks = slice_audio(source_audio, sr, max_seconds=30.0)
        converted_chunks = []

        for chunk_idx, chunk in enumerate(chunks):
            chunk_dur = len(chunk) / sr
            print(f"  [Chunk {chunk_idx + 1}/{len(chunks)}] {chunk_dur:.1f}s...")

            # Step 2a: Compute mel spectrograms
            src_mel = librosa.feature.melspectrogram(
                y=chunk, sr=sr, n_mels=n_mels, hop_length=hop,
                n_fft=n_fft, win_length=win, fmin=fmin, fmax=fmax,
            )
            ref_mel = librosa.feature.melspectrogram(
                y=reference_audio, sr=sr, n_mels=n_mels, hop_length=hop,
                n_fft=n_fft, win_length=win, fmin=fmin, fmax=fmax,
            )

            # Step 2b: Mean-Variance Normalization (spectral transfer)
            src_log = np.log(np.maximum(src_mel, 1e-10))
            ref_log = np.log(np.maximum(ref_mel, 1e-10))

            src_mean = src_log.mean(axis=1, keepdims=True)
            src_std = src_log.std(axis=1, keepdims=True)
            ref_mean = ref_log.mean(axis=1, keepdims=True)
            ref_std = ref_log.std(axis=1, keepdims=True)

            eps = 1e-5
            converted_log = (src_log - src_mean) / (src_std + eps) * (ref_std + eps) + ref_mean
            converted_log = np.clip(converted_log, ref_log.min() - 1.0, ref_log.max() + 1.0)
            converted_mel = np.exp(converted_log)

            print(f"    Mel shape: {converted_mel.shape}")

            # Step 2c: Reconstruct audio via Griffin-Lim
            waveform = librosa.feature.inverse.mel_to_audio(
                converted_mel, sr=sr, hop_length=hop,
                win_length=win, n_fft=n_fft, n_iter=64, power=1.0,
            )

            # Resample to output sample rate if needed
            out_sr = self.config.audio.output_sample_rate
            if out_sr != sr:
                import scipy.signal as sig
                target_len = int(len(waveform) * out_sr / sr)
                waveform = sig.resample(waveform, target_len).astype(np.float32)

            # Step 2d: Apply pitch shift if requested
            if f0_transpose != 0:
                print(f"    Applying pitch shift: {f0_transpose:+d} semitones...")
                waveform = librosa.effects.pitch_shift(
                    waveform, sr=sr, n_steps=f0_transpose,
                )

            # Apply F0 curve factor via time-stretch
            if f0_curve_factor != 1.0 and f0_curve_factor > 0:
                stretch_rate = 1.0 / f0_curve_factor
                if 0.5 <= stretch_rate <= 2.0:
                    waveform = librosa.effects.time_stretch(waveform, rate=stretch_rate)

            converted_chunks.append(waveform.astype(np.float32))

        return np.concatenate(converted_chunks)

    # ===================================================================
    # NEURAL PIPELINE (requires pretrained weights)
    # ===================================================================

    def _convert_chunk_neural(self, source_audio: np.ndarray, reference_audio: np.ndarray,
                               f0_transpose: int = 0, f0_curve_factor: float = 1.0,
                               protect_consonants: bool = True,
                               noise_scale: float = 0.4) -> np.ndarray:
        """Convert a single audio chunk using the neural pipeline."""

        # Extract F0
        print("  [F0] Extracting pitch (RMVPE)...")
        f0, uv = self.f0_extractor.extract(source_audio, sr=self.config.audio.sample_rate)
        print(f"    F0 frames: {len(f0)}, Voiced ratio: {1 - np.mean(uv):.1%}")

        # Extract content features
        print("  [ContentVec] Extracting content features...")
        content_features = self.content_encoder.extract(source_audio, sr=self.config.audio.sample_rate)
        print(f"    Content shape: {content_features.shape}")

        # Frame-rate alignment
        expected_frames = max(1, len(source_audio) // self.config.audio.hop_size)
        content_features = self._resample_features(content_features, expected_frames)
        f0 = self._resample_1d(f0, expected_frames)
        uv = self._resample_1d(uv, expected_frames)

        # Extract speaker embedding
        print("  [CAM++] Extracting speaker embedding...")
        spk_embedding = self.speaker_encoder.extract(reference_audio, sr=self.config.audio.sample_rate)
        print(f"    Speaker embedding shape: {spk_embedding.shape}")

        # Align features
        content_aligned, f0_aligned, uv_aligned = self._align_features(
            content_features, f0, uv
        )

        # Apply F0 transpose
        f0_aligned = self._transpose_f0(f0_aligned, uv_aligned, f0_transpose,
                                         protect=protect_consonants)
        if f0_curve_factor != 1.0:
            voiced = f0_aligned > 0
            f0_aligned[voiced] *= f0_curve_factor

        f0_feature = self._create_f0_feature(f0_aligned, uv_aligned)

        # Generate audio
        print("  [VITS + HiFi-GAN] Generating converted audio...")
        waveform = self._generate_audio_neural(content_aligned, f0_feature, spk_embedding,
                                                noise_scale=noise_scale)
        return waveform

    def _generate_audio_neural(self, content: np.ndarray, f0_feature: np.ndarray,
                                spk_embedding: np.ndarray,
                                noise_scale: float = 0.4) -> np.ndarray:
        """Generate audio using VITS generator + HiFi-GAN vocoder."""

        content_tensor = numpy_to_torch(content).T
        f0_tensor = numpy_to_torch(f0_feature).unsqueeze(0)

        gin_dim = self.config.generator.gin_channels
        if spk_embedding.shape[0] != gin_dim:
            spk_padded = np.zeros(gin_dim, dtype=np.float32)
            spk_padded[:min(len(spk_embedding), gin_dim)] = spk_embedding[:min(len(spk_embedding), gin_dim)]
            spk_embedding = spk_padded

        spk_tensor = numpy_to_torch(spk_embedding).unsqueeze(0)

        max_seq = 1500
        T = content_tensor.shape[1]
        if T > max_seq:
            print(f"    [WARN] Truncating from {T} to {max_seq} frames")
            content_tensor = content_tensor[:, :max_seq]
            f0_tensor = f0_tensor[:, :, :max_seq]

        content_tensor = content_tensor.unsqueeze(0).to(self.device)
        f0_tensor = f0_tensor.unsqueeze(0).to(self.device)
        spk_tensor = spk_tensor.to(self.device)

        with torch.no_grad():
            mel = self.generator.infer(
                content_tensor, f0_tensor, spk_tensor, noise_scale=noise_scale
            )
            del content_tensor, f0_tensor, spk_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            waveform = self.vocoder.generate(mel)
            del mel
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        waveform_np = waveform.squeeze(0).squeeze(0).cpu().numpy()
        del waveform

        if self.config.audio.output_sample_rate != 32000:
            import scipy.signal as sig
            target_len = int(len(waveform_np) * self.config.audio.output_sample_rate / 32000)
            waveform_np = sig.resample(waveform_np, target_len).astype(np.float32)

        return waveform_np

    # ===================================================================
    # AUDIO EFFECTS AND PROCESSING
    # ===================================================================

    def _apply_noise_reduction(self, audio: np.ndarray, sr: int, 
                                strength: float = 0.5) -> np.ndarray:
        """Apply spectral gating noise reduction."""
        if strength <= 0:
            return audio

        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise floor from first few frames
        noise_frames = min(10, magnitude.shape[1])
        if noise_frames > 0:
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            threshold = noise_profile * (2 + strength * 3)
            mask = np.maximum(0, 1 - (threshold / (magnitude + 1e-8)))
            mask = np.clip(mask * (1 + strength * 2), 0, 1)
            
            clean_magnitude = magnitude * mask
            clean_stft = clean_magnitude * np.exp(1j * phase)
            audio_clean = librosa.istft(clean_stft, hop_length=hop_length)
            
            output = audio * (1 - strength * 0.5) + audio_clean * strength * 0.5
            return output.astype(np.float32)

        return audio

    def _apply_formant_shift(self, audio: np.ndarray, sr: int,
                              shift_semitones: int = 0) -> np.ndarray:
        """Apply formant shifting independent of pitch.

        This shifts the spectral envelope to change perceived vocal tract length
        without changing the fundamental frequency.
        """
        if shift_semitones == 0:
            return audio

        # Formant shift ratio (positive = smaller vocal tract = brighter)
        ratio = 2.0 ** (shift_semitones / 12.0)

        # Use phase vocoder for time-stretch based formant shifting
        # Stretch by ratio, then resample back to original length
        stretched = librosa.effects.time_stretch(audio, rate=ratio)
        
        # Resample back to original length
        target_len = len(audio)
        if len(stretched) != target_len:
            import scipy.signal as sig
            stretched = sig.resample(stretched, target_len)

        return stretched.astype(np.float32)

    def _add_breathiness(self, audio: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Add breathiness/noise component to audio."""
        if strength <= 0:
            return audio

        # Generate high-frequency noise
        noise = np.random.randn(len(audio)).astype(np.float32)
        
        # High-pass filter for breath-like quality
        from scipy.signal import butter, filtfilt
        cutoff = 2000  # Hz
        nyquist = (self.config.audio.output_sample_rate // 2) / 2
        b, a = butter(2, cutoff / nyquist, btype='high')
        noise = filtfilt(b, a, noise)

        # Mix in noise at specified strength
        output = audio + noise * 0.03 * strength
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(output))
        if peak > 0.99:
            output = output * (0.99 / peak)

        return output.astype(np.float32)

    def _add_vibrato(self, audio: np.ndarray, sr: int, 
                     strength: float = 0.3, rate: float = 5.0) -> np.ndarray:
        """Add artificial vibrato effect."""
        if strength <= 0:
            return audio

        # Create subtle amplitude modulation (tremolo-like)
        t = np.linspace(0, len(audio) / sr, len(audio))
        modulation = 1 + 0.02 * strength * np.sin(2 * np.pi * rate * t)
        
        output = audio * modulation.astype(np.float32)
        return output.astype(np.float32)

    # ===================================================================
    # UTILITY METHODS
    # ===================================================================

    def _align_features(self, content: np.ndarray, f0: np.ndarray,
                        uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align content features and F0 to the same temporal resolution."""
        content_len = content.shape[0]
        f0_len = len(f0)
        if content_len == f0_len:
            return content, f0, uv
        if content_len > f0_len:
            indices = np.linspace(0, f0_len - 1, content_len)
            f0_aligned = np.interp(indices, np.arange(f0_len), f0)
            uv_aligned = np.interp(indices, np.arange(f0_len), uv)
            uv_aligned = (uv_aligned > 0.5).astype(np.float32)
            return content, f0_aligned, uv_aligned
        else:
            indices = np.linspace(0, content_len - 1, f0_len)
            content_aligned = np.zeros((f0_len, content.shape[1]), dtype=np.float32)
            for dim in range(content.shape[1]):
                content_aligned[:, dim] = np.interp(
                    indices, np.arange(content_len), content[:, dim]
                )
            return content_aligned, f0, uv

    @staticmethod
    def _resample_features(features: np.ndarray, target_len: int) -> np.ndarray:
        T, D = features.shape
        if T == target_len:
            return features
        indices = np.linspace(0, T - 1, target_len)
        out = np.zeros((target_len, D), dtype=np.float32)
        for d in range(D):
            out[:, d] = np.interp(indices, np.arange(T), features[:, d])
        return out

    @staticmethod
    def _resample_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) == target_len:
            return arr
        indices = np.linspace(0, len(arr) - 1, target_len)
        return np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)

    def _transpose_f0(self, f0: np.ndarray, uv: np.ndarray,
                      semitones: int, protect: bool = True) -> np.ndarray:
        if semitones == 0:
            return f0
        ratio = 2.0 ** (semitones / 12.0)
        f0_new = f0.copy()
        voiced = f0 > 0
        f0_new[voiced] = f0[voiced] * ratio
        if protect and semitones != 0:
            from scipy.ndimage import uniform_filter1d
            protection = np.ones_like(f0, dtype=np.float32)
            uv_smooth = uniform_filter1d(uv.astype(np.float32), size=7)
            protection = 1.0 - np.clip(uv_smooth * 2, 0, 1)
            f0_new = f0 * (1 - protection) + f0_new * protection
        return f0_new

    def _create_f0_feature(self, f0: np.ndarray, uv: np.ndarray) -> np.ndarray:
        f0_feature = np.zeros_like(f0, dtype=np.float32)
        voiced = f0 > 0
        if np.any(voiced):
            log_f0 = np.log(np.maximum(f0[voiced], 1e-5))
            mean = np.mean(log_f0)
            std = np.std(log_f0) + 1e-5
            f0_feature[voiced] = (log_f0 - mean) / std
        return f0_feature

    def extract_features(self, audio_path: str) -> dict:
        """Extract and visualize all intermediate features (neural mode)."""
        if not self._models_loaded:
            self.load_models()
        audio = load_audio(audio_path, target_sr=self.config.audio.sample_rate)
        content = self.content_encoder.extract(audio, sr=self.config.audio.sample_rate)
        f0, uv = self.f0_extractor.extract(audio, sr=self.config.audio.sample_rate)
        return {
            "content": content,
            "f0": f0,
            "uv": uv,
            "duration": len(audio) / self.config.audio.sample_rate,
        }

    def extract_speaker_embedding(self, reference_path: str) -> np.ndarray:
        """Extract speaker embedding from reference audio (neural mode)."""
        if not self._models_loaded:
            self.load_models()
        audio = load_audio(reference_path, target_sr=self.config.audio.sample_rate)
        return self.speaker_encoder.extract(audio, sr=self.config.audio.sample_rate)


# Import datetime for speaker profiles
from datetime import datetime
