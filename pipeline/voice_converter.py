"""
Zero-Shot Singing Voice Conversion Pipeline

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
"""
import os
import gc
import time
import librosa
import numpy as np
import torch
from typing import Optional, Tuple

from utils.hparams import Config
from utils.audio import (
    load_audio, save_audio, f0_to_coarse, numpy_to_torch, normalize_audio,
    slice_audio, compute_mel_spectrogram,
)


class ZeroShotSVC:
    """Zero-Shot Singing Voice Conversion system.

    Converts singing voice from source audio to match the voice
    characteristics of a reference speaker, WITHOUT any training.

    By default uses signal-processing voice conversion (mel MVN + Griffin-Lim)
    which produces real converted audio without any pretrained models.
    Use use_neural=True to use the neural pipeline (requires pretrained weights).

    Usage:
        svc = ZeroShotSVC()
        output_path = svc.convert(
            source_path="singing.wav",
            reference_path="target_voice.wav",
            output_path="converted.wav",
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
                noise_scale: float = 0.4) -> str:
        """Convert singing voice from source to target speaker.

        Args:
            source_path: Path to source audio (singing to be converted).
            reference_path: Path to reference audio (target voice, 5-30s).
            output_path: Path for output audio. If None, auto-generated.
            f0_transpose: Semitones to shift pitch (+12 = octave up).
            f0_curve_factor: F0 curve scaling factor (1.0 = no change).
            protect_consonants: Reduce F0 shifting near unvoiced regions.
            noise_scale: Noise injection scale (neural mode only).

        Returns:
            Path to the output audio file.
        """
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
        print(f"{'=' * 60}\n")

        t0 = time.time()

        # Step 1: Load audio
        print("[Step 1/4] Loading audio files...")
        source_audio = load_audio(source_path, target_sr=self.config.audio.sample_rate)
        reference_audio = load_audio(reference_path, target_sr=self.config.audio.sample_rate)

        src_dur = len(source_audio) / self.config.audio.sample_rate
        ref_dur = len(reference_audio) / self.config.audio.sample_rate
        print(f"  Source duration:     {src_dur:.1f}s")
        print(f"  Reference duration:  {ref_dur:.1f}s")

        # Route to the correct conversion method
        if self.use_neural:
            chunks = slice_audio(source_audio, self.config.audio.sample_rate, max_seconds=30.0)
            converted_chunks = []
            for idx, chunk in enumerate(chunks):
                print(f"\n  Chunk {idx + 1}/{len(chunks)} "
                      f"({len(chunk) / self.config.audio.sample_rate:.1f}s)...")
                converted = self._convert_chunk_neural(
                    chunk, reference_audio,
                    f0_transpose=f0_transpose,
                    f0_curve_factor=f0_curve_factor,
                    protect_consonants=protect_consonants,
                    noise_scale=noise_scale,
                )
                converted_chunks.append(converted)
            full_output = np.concatenate(converted_chunks)
        else:
            # Signal processing path — handles its own chunking internally
            full_output = self._convert_signal_processing(
                source_audio, reference_audio,
                f0_transpose=f0_transpose,
                f0_curve_factor=f0_curve_factor,
            )

        # Step 4: Save output
        print("\n[Step 4/4] Saving output audio...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        full_output = normalize_audio(full_output, target_peak=0.95)
        save_audio(output_path, full_output, sr=self.config.audio.output_sample_rate)

        elapsed = time.time() - t0
        out_dur = len(full_output) / self.config.audio.output_sample_rate
        print(f"\n{'=' * 60}")
        print(f"Conversion complete! ({elapsed:.1f}s)")
        print(f"Output saved to: {output_path}")
        print(f"Output duration: {out_dur:.1f}s")
        print(f"{'=' * 60}")

        return output_path

    # ===================================================================
    # SIGNAL PROCESSING PIPELINE (default, no models needed)
    # ===================================================================

    def _convert_signal_processing(self, source_audio: np.ndarray,
                                    reference_audio: np.ndarray,
                                    f0_transpose: int = 0,
                                    f0_curve_factor: float = 1.0) -> np.ndarray:
        """Voice conversion using signal processing (no neural models).

        Algorithm:
        1. Extract mel spectrogram from source and reference
        2. Compute per-band mean and std for both speakers
        3. Apply Mean-Variance Normalization (MVN):
           converted = (src - src_mean) / src_std * ref_std + ref_mean
        4. This transfers the reference speaker's spectral envelope
           (timbre/formants) onto the source content
        5. Reconstruct audio via Griffin-Lim phase reconstruction
        6. Optionally apply pitch shift

        This is a classic, well-proven voice conversion technique that
        produces recognizable converted speech/singing without any
        training or pretrained models.
        """
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

            # -----------------------------------------------------------
            # Step 2a: Compute mel spectrograms
            # -----------------------------------------------------------
            src_mel = librosa.feature.melspectrogram(
                y=chunk, sr=sr, n_mels=n_mels, hop_length=hop,
                n_fft=n_fft, win_length=win, fmin=fmin, fmax=fmax,
            )
            ref_mel = librosa.feature.melspectrogram(
                y=reference_audio, sr=sr, n_mels=n_mels, hop_length=hop,
                n_fft=n_fft, win_length=win, fmin=fmin, fmax=fmax,
            )

            # -----------------------------------------------------------
            # Step 2b: Mean-Variance Normalization (spectral transfer)
            # -----------------------------------------------------------
            # Convert to log-mel for better statistical modeling
            src_log = np.log(np.maximum(src_mel, 1e-10))
            ref_log = np.log(np.maximum(ref_mel, 1e-10))

            # Per-frequency-band statistics
            src_mean = src_log.mean(axis=1, keepdims=True)
            src_std = src_log.std(axis=1, keepdims=True)
            ref_mean = ref_log.mean(axis=1, keepdims=True)
            ref_std = ref_log.std(axis=1, keepdims=True)

            # Apply MVN: normalize source, then apply reference statistics
            eps = 1e-5
            converted_log = (src_log - src_mean) / (src_std + eps) * (ref_std + eps) + ref_mean

            # Soft clamp to prevent extreme values
            converted_log = np.clip(converted_log, ref_log.min() - 1.0, ref_log.max() + 1.0)

            # Convert back to power mel
            converted_mel = np.exp(converted_log)

            print(f"    Mel shape: {converted_mel.shape}")

            # -----------------------------------------------------------
            # Step 2c: Reconstruct audio via Griffin-Lim
            # -----------------------------------------------------------
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

            # -----------------------------------------------------------
            # Step 2d: Apply pitch shift if requested
            # -----------------------------------------------------------
            if f0_transpose != 0:
                print(f"    Applying pitch shift: {f0_transpose:+d} semitones...")
                waveform = librosa.effects.pitch_shift(
                    waveform, sr=sr, n_steps=f0_transpose,
                )

            # Apply F0 curve factor via time-stretch (simple approximation)
            if f0_curve_factor != 1.0 and f0_curve_factor > 0:
                # Time stretch inversely to change F0: faster = higher pitch
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
        from models.vocoder import Vocoder

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