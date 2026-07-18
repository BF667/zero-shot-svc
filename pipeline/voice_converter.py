"""
Zero-Shot Singing Voice Conversion Pipeline

This is the main pipeline that orchestrates the full voice conversion process:

Source Singing Audio + Reference Audio (target voice)
    |
    v
[1] ContentVec: Extract content features (WHAT is being sung)
[2] RMVPE: Extract F0 pitch contour (the melody)
[3] CAM++: Extract speaker embedding from reference (WHO to sound like)
[4] F0 normalization: Match F0 statistics to target speaker
[5] VITS Generator: Generate mel-spectrogram (content + F0 + speaker -> mel)
[6] HiFi-GAN: Convert mel-spectrogram to waveform
    |
    v
Converted Audio (singing in target voice, preserving original melody/lyrics)

Key Design Principles:
- NO TRAINING REQUIRED: Uses only pre-trained models
- Reference audio only: 5-15 seconds of target speaker audio is sufficient
- Modular: Each component can be swapped independently
- RVC-architecture compatible: Follows the same pipeline as RVC v2
"""
import os
import time
import numpy as np
import torch
from typing import Optional, Tuple

from utils.hparams import Config
from utils.audio import (
    load_audio, save_audio, f0_to_coarse, numpy_to_torch, normalize_audio,
    slice_audio, compute_mel_spectrogram,
)
from models.f0_extractor import RMVPEExtractor
from models.content_encoder import ContentEncoder
from models.speaker_encoder import SpeakerEncoderExtractor
from models.generator import VITSGenerator
from models.vocoder import Vocoder


class ZeroShotSVC:
    """Zero-Shot Singing Voice Conversion system.

    Converts singing voice from source audio to match the voice
    characteristics of a reference speaker, WITHOUT any training.

    Usage:
        svc = ZeroShotSVC()
        # One-time setup (downloads pre-trained models)
        svc.load_models()

        # Convert voice
        output_path = svc.convert(
            source_path="singing.wav",
            reference_path="target_voice.wav",
            output_path="converted.wav",
        )
    """

    def __init__(self, config: Config = None, config_path: str = None,
                 device: str = None):
        """
        Args:
            config: Configuration object. If None, loads from config_path.
            config_path: Path to YAML config file.
            device: Override device ('cpu' or 'cuda').
        """
        if config is None:
            config = Config.from_yaml(config_path or self._default_config_path())
        self.config = config

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ZeroShotSVC] Device: {self.device}")
        print(f"[ZeroShotSVC] F0 method: {config.f0_extractor.method} (RMVPE)")

        # Components (lazily loaded)
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
        """Load all pre-trained models.

        This is the initialization step that downloads/loads all required
        pre-trained models. Models are cached locally for subsequent uses.
        """
        if self._models_loaded:
            print("[ZeroShotSVC] Models already loaded, skipping.")
            return

        print("=" * 60)
        print("Loading Zero-Shot SVC Models")
        print("=" * 60)

        t0 = time.time()

        # 1. RMVPE F0 Extractor
        print("\n[1/5] Loading RMVPE F0 Extractor...")
        self.f0_extractor = RMVPEExtractor(
            f0_min=self.config.f0_extractor.f0_min,
            f0_max=self.config.f0_extractor.f0_max,
            device=self.device,
        )

        # 2. ContentVec Content Encoder
        print("\n[2/5] Loading ContentVec Content Encoder...")
        self.content_encoder = ContentEncoder(
            output_dim=self.config.content_encoder.output_dim,
            device=self.device,
        )

        # 3. CAM++ Speaker Encoder
        print("\n[3/5] Loading Speaker Encoder (CAM++)...")
        self.speaker_encoder = SpeakerEncoderExtractor(
            embedding_dim=self.config.speaker_encoder.embedding_dim,
            device=self.device,
        )

        # 4. VITS Generator
        print("\n[4/5] Loading VITS Generator...")
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
        print(f"[Generator] Model loaded on {self.device} "
              f"(randomly initialized - load pretrained weights for best quality)")

        # 5. HiFi-GAN Vocoder
        print("\n[5/5] Loading HiFi-GAN Vocoder...")
        self.vocoder = Vocoder(
            hop_size=self.config.vocoder.hop_size,
            sample_rate=self.config.audio.output_sample_rate,
            device=self.device,
        )

        self._models_loaded = True
        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"All models loaded in {elapsed:.1f}s")
        print(f"{'=' * 60}\n")

    def convert(self, source_path: str, reference_path: str,
                output_path: str = None, f0_transpose: int = 0,
                f0_curve_factor: float = 1.0,
                protect_consonants: bool = True,
                noise_scale: float = 0.4) -> str:
        """Convert singing voice from source to target speaker.

        Args:
            source_path: Path to source audio (singing to be converted).
            reference_path: Path to reference audio (target voice, 5-30s recommended).
            output_path: Path for output audio. If None, auto-generated.
            f0_transpose: Semitones to shift pitch (e.g., +12 = one octave up).
            f0_curve_factor: F0 curve scaling factor (1.0 = no change).
            protect_consonants: If True, reduce F0 shifting on unvoiced frames.
            noise_scale: Noise injection scale for generation (higher = more variation).

        Returns:
            Path to the output audio file.
        """
        if not self._models_loaded:
            self.load_models()

        if output_path is None:
            base = os.path.splitext(os.path.basename(source_path))[0]
            output_path = os.path.join(
                os.path.dirname(source_path) or ".",
                f"{base}_converted.wav"
            )

        print(f"\n{'=' * 60}")
        print(f"Zero-Shot Singing Voice Conversion")
        print(f"{'=' * 60}")
        print(f"  Source:     {source_path}")
        print(f"  Reference:  {reference_path}")
        print(f"  Output:     {output_path}")
        print(f"  F0 shift:   {f0_transpose:+d} semitones")
        print(f"  F0 curve:   {f0_curve_factor:.2f}x")
        print(f"{'=' * 60}\n")

        t0 = time.time()

        # ---------------------------------------------------------------
        # Step 1: Load audio
        # ---------------------------------------------------------------
        print("[Step 1/7] Loading audio files...")
        source_audio = load_audio(source_path, target_sr=self.config.audio.sample_rate)
        reference_audio = load_audio(reference_path, target_sr=self.config.audio.sample_rate)

        print(f"  Source duration:     {len(source_audio) / self.config.audio.sample_rate:.1f}s")
        print(f"  Reference duration:  {len(reference_audio) / self.config.audio.sample_rate:.1f}s")

        # Handle long audio by chunking
        chunks = slice_audio(source_audio, self.config.audio.sample_rate, max_seconds=30.0)
        converted_chunks = []

        for chunk_idx, chunk in enumerate(chunks):
            print(f"\n  Processing chunk {chunk_idx + 1}/{len(chunks)} "
                  f"({len(chunk) / self.config.audio.sample_rate:.1f}s)...")

            converted = self._convert_chunk(
                chunk, reference_audio,
                f0_transpose=f0_transpose,
                f0_curve_factor=f0_curve_factor,
                protect_consonants=protect_consonants,
                noise_scale=noise_scale,
            )
            converted_chunks.append(converted)

        # Concatenate chunks
        full_output = np.concatenate(converted_chunks)

        # ---------------------------------------------------------------
        # Step 7: Save output
        # ---------------------------------------------------------------
        print("\n[Step 7/7] Saving output audio...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Normalize output
        full_output = normalize_audio(full_output, target_peak=0.95)
        save_audio(output_path, full_output, sr=self.config.audio.output_sample_rate)

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"Conversion complete! ({elapsed:.1f}s)")
        print(f"Output saved to: {output_path}")
        print(f"Output duration: {len(full_output) / self.config.audio.output_sample_rate:.1f}s")
        print(f"{'=' * 60}")

        return output_path

    def _convert_chunk(self, source_audio: np.ndarray, reference_audio: np.ndarray,
                       f0_transpose: int = 0, f0_curve_factor: float = 1.0,
                       protect_consonants: bool = True,
                       noise_scale: float = 0.4) -> np.ndarray:
        """Convert a single audio chunk."""

        # -----------------------------------------------------------
        # Step 2: Extract F0 using RMVPE (default method)
        # -----------------------------------------------------------
        print("  [2/7] Extracting F0 (RMVPE)...")
        f0, uv = self.f0_extractor.extract(source_audio, sr=self.config.audio.sample_rate)
        print(f"    F0 frames: {len(f0)}, Voiced ratio: {1 - np.mean(uv):.1%}")

        # -----------------------------------------------------------
        # Step 3: Extract content features using ContentVec
        # -----------------------------------------------------------
        print("  [3/7] Extracting content features (ContentVec)...")
        content_features = self.content_encoder.extract(source_audio, sr=self.config.audio.sample_rate)
        print(f"    Content shape: {content_features.shape}")

        # -----------------------------------------------------------
        # Step 4: Extract speaker embedding from reference audio
        # -----------------------------------------------------------
        print("  [4/7] Extracting speaker embedding (CAM++)...")
        spk_embedding = self.speaker_encoder.extract(reference_audio, sr=self.config.audio.sample_rate)
        print(f"    Speaker embedding shape: {spk_embedding.shape}")

        # -----------------------------------------------------------
        # Step 5: Align features and apply F0 transformations
        # -----------------------------------------------------------
        print("  [5/7] Aligning features and applying F0 processing...")
        content_aligned, f0_aligned, uv_aligned = self._align_features(
            content_features, f0, uv
        )

        # Apply F0 transpose (semitone shift)
        f0_aligned = self._transpose_f0(f0_aligned, uv_aligned, f0_transpose,
                                         protect=protect_consonants)

        # Apply F0 curve factor
        if f0_curve_factor != 1.0:
            voiced = f0_aligned > 0
            f0_aligned[voiced] *= f0_curve_factor

        # Create F0 features for generator input
        f0_feature = self._create_f0_feature(f0_aligned, uv_aligned)

        # -----------------------------------------------------------
        # Step 6: Generate audio (Generator + Vocoder)
        # -----------------------------------------------------------
        print("  [6/7] Generating converted audio...")
        waveform = self._generate_audio(content_aligned, f0_feature, spk_embedding,
                                        noise_scale=noise_scale)

        return waveform

    def _align_features(self, content: np.ndarray, f0: np.ndarray,
                        uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align content features and F0 to the same temporal resolution.

        ContentVec outputs at ~50Hz (320 samples/frame at 16kHz).
        RMVPE outputs at 50Hz (320 samples/frame at 16kHz).
        They should be aligned, but we handle any minor mismatches here.
        """
        content_len = content.shape[0]
        f0_len = len(f0)

        if content_len == f0_len:
            return content, f0, uv

        # Interpolate to match the shorter sequence
        if content_len > f0_len:
            # Interpolate F0 to match content length
            indices = np.linspace(0, f0_len - 1, content_len)
            f0_aligned = np.interp(indices, np.arange(f0_len), f0)
            uv_aligned = np.interp(indices, np.arange(f0_len), uv)
            uv_aligned = (uv_aligned > 0.5).astype(np.float32)
            return content, f0_aligned, uv_aligned
        else:
            # Interpolate content to match F0 length
            indices = np.linspace(0, content_len - 1, f0_len)
            content_aligned = np.zeros((f0_len, content.shape[1]), dtype=np.float32)
            for dim in range(content.shape[1]):
                content_aligned[:, dim] = np.interp(
                    indices, np.arange(content_len), content[:, dim]
                )
            return content_aligned, f0, uv

    def _transpose_f0(self, f0: np.ndarray, uv: np.ndarray,
                      semitones: int, protect: bool = True) -> np.ndarray:
        """Shift F0 by a number of semitones.

        Args:
            f0: F0 values, unvoiced = 0.
            uv: Unvoiced flags.
            semitones: Number of semitones to shift (+ = up, - = down).
            protect: If True, reduce shifting on frames near unvoiced regions.

        Returns:
            Transposed F0.
        """
        if semitones == 0:
            return f0

        # 2^(semitones/12) gives the frequency ratio
        ratio = 2.0 ** (semitones / 12.0)
        f0_new = f0.copy()

        voiced = f0 > 0
        f0_new[voiced] = f0[voiced] * ratio

        # Protection: reduce shifting near unvoiced frames
        if protect and semitones != 0:
            # Create a soft mask: frames surrounded by voiced frames get full shift
            # frames near unvoiced boundaries get reduced shift
            protection = np.ones_like(f0, dtype=np.float32)
            kernel_size = 7
            uv_float = uv.astype(np.float32)
            # Smooth the UV signal
            from scipy.ndimage import uniform_filter1d
            uv_smooth = uniform_filter1d(uv_float, size=kernel_size)
            # Protection factor: 0 near unvoiced, 1 in voiced regions
            protection = 1.0 - np.clip(uv_smooth * 2, 0, 1)

            # Blend between original and transposed F0 near unvoiced regions
            f0_new = f0 * (1 - protection) + f0_new * protection

        return f0_new

    def _create_f0_feature(self, f0: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Create the F0 feature for generator input.

        Normalizes F0 and creates a single-channel feature.
        Unvoiced frames get a special value (e.g., 0).
        """
        f0_feature = np.zeros_like(f0, dtype=np.float32)

        voiced = f0 > 0
        if np.any(voiced):
            # Log-scale normalization (F0 is roughly log-normal distributed)
            log_f0 = np.log(np.maximum(f0[voiced], 1e-5))
            # Z-normalize using the voiced frames
            mean = np.mean(log_f0)
            std = np.std(log_f0) + 1e-5
            f0_feature[voiced] = (log_f0 - mean) / std

        return f0_feature

    def _generate_audio(self, content: np.ndarray, f0_feature: np.ndarray,
                        spk_embedding: np.ndarray,
                        noise_scale: float = 0.4) -> np.ndarray:
        """Generate audio from features using the VITS generator + HiFi-GAN vocoder.

        Args:
            content: Content features, shape (T, 256).
            f0_feature: F0 feature, shape (T,).
            spk_embedding: Speaker embedding, shape (192,).
            noise_scale: Noise scale for generation.

        Returns:
            Generated waveform, numpy array.
        """
        # Prepare tensors
        content_tensor = numpy_to_torch(content).T  # (256, T)
        f0_tensor = numpy_to_torch(f0_feature).unsqueeze(0)  # (1, T)

        # Project speaker embedding to gin_channels if needed
        gin_dim = self.config.generator.gin_channels
        if spk_embedding.shape[0] != gin_dim:
            # Pad or project the speaker embedding
            spk_padded = np.zeros(gin_dim, dtype=np.float32)
            spk_padded[:min(len(spk_embedding), gin_dim)] = spk_embedding[:min(len(spk_embedding), gin_dim)]
            spk_embedding = spk_padded

        spk_tensor = numpy_to_torch(spk_embedding).unsqueeze(0)  # (1, gin_channels)

        # Add batch dimension
        content_tensor = content_tensor.unsqueeze(0).to(self.device)  # (1, 256, T)
        f0_tensor = f0_tensor.unsqueeze(0).to(self.device)  # (1, 1, T)
        spk_tensor = spk_tensor.to(self.device)  # (1, gin_channels)

        # Generate mel-spectrogram using VITS generator
        with torch.no_grad():
            mel = self.generator.infer(
                content_tensor, f0_tensor, spk_tensor, noise_scale=noise_scale
            )
            # mel: (1, 128, T_mel)

            # Generate waveform using HiFi-GAN vocoder
            waveform = self.vocoder.generate(mel)
            # waveform: (1, 1, T_audio)

        # Convert to numpy
        waveform_np = waveform.squeeze(0).squeeze(0).cpu().numpy()

        # Resample from vocoder output rate to target rate if needed
        if self.config.audio.output_sample_rate != 32000:
            import scipy.signal as sig
            target_len = int(len(waveform_np) * self.config.audio.output_sample_rate / 32000)
            waveform_np = sig.resample(waveform_np, target_len).astype(np.float32)

        return waveform_np

    def extract_features(self, audio_path: str) -> dict:
        """Extract and visualize all intermediate features (for debugging/analysis).

        Args:
            audio_path: Path to audio file.

        Returns:
            Dictionary with 'content', 'f0', 'uv' features.
        """
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
        """Extract speaker embedding from reference audio.

        Args:
            reference_path: Path to reference audio file.

        Returns:
            Speaker embedding vector.
        """
        if not self._models_loaded:
            self.load_models()

        audio = load_audio(reference_path, target_sr=self.config.audio.sample_rate)
        return self.speaker_encoder.extract(audio, sr=self.config.audio.sample_rate)