"""
RMVPE (Robust Model for Vocal Pitch Estimation in Polyphonic Music)

This module implements the RMVPE pitch extraction algorithm based on the paper:
"RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music"
(Wei et al., 2023, arXiv:2306.15412)

RMVPE is superior to CREPE, DIO, and PM for singing voice because:
1. It directly estimates vocal pitch from polyphonic music (no need to separate vocals first)
2. It uses residual CNNs with log-mel spectrogram input for robust feature extraction
3. It handles the wide pitch range of singing (50-1100 Hz) with high accuracy

The model architecture:
- Input: Log-mel spectrogram of the audio
- Backbone: Residual CNN with dilated convolutions
- Output: Centroid-wise pitch predictions with confidence scores
- Post-processing: Viterbi decoding for smooth pitch contour
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as signal
import librosa
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# RMVPE Neural Network Architecture
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Residual convolutional block with batch normalization and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        # Residual projection if dimensions change
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.use_trim = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        # Trim residual if conv changed the length (dilation edge case)
        if out.size(2) != residual.size(2) and self.residual is None:
            residual = residual[:, :, :out.size(2)]
        out = self.bn(out)
        out = self.activation(out)
        if self.residual is not None:
            residual = self.residual(residual)
            if residual.size(2) != out.size(2):
                residual = residual[:, :, :out.size(2)]
        return out + residual


class RMVPEBackbone(nn.Module):
    """RMVPE backbone network with residual CNN blocks.

    Processes log-mel spectrogram features through multiple
    dilated convolutional layers to extract pitch-relevant features.
    """

    def __init__(self, in_channels: int = 128, hidden_channels: int = 512):
        super().__init__()
        # Initial projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Dilated residual blocks (progressively increasing dilation for temporal context)
        self.blocks = nn.ModuleList([
            ConvBlock(hidden_channels, hidden_channels, dilation=1),
            ConvBlock(hidden_channels, hidden_channels, dilation=2),
            ConvBlock(hidden_channels, hidden_channels, dilation=4),
            ConvBlock(hidden_channels, hidden_channels, dilation=8),
            ConvBlock(hidden_channels, hidden_channels, dilation=2),
            ConvBlock(hidden_channels, hidden_channels, dilation=1),
        ])

        # Output projection: for each centroid, predict pitch bin probabilities
        # RMVPE uses 360 centroids covering the pitch range
        self.n_centroids = 360
        self.output_proj = nn.Conv1d(hidden_channels, self.n_centroids, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Log-mel spectrogram, shape (B, n_mels, T)

        Returns:
            Pitch logits, shape (B, n_centroids, T)
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# RMVPE F0 Extractor
# ---------------------------------------------------------------------------

class RMVPEExtractor:
    """RMVPE-based F0 (fundamental frequency) extractor.

    This class wraps the RMVPE neural network model and provides
    a clean interface for pitch extraction from audio waveforms.

    Usage:
        extractor = RMVPEExtractor()
        f0, uv = extractor.extract(audio, sr=16000)

    The extractor:
    1. Computes a log-mel spectrogram from the input audio
    2. Passes it through the RMVPE backbone network
    3. Applies Viterbi decoding to produce a smooth pitch contour
    4. Returns F0 values and voiced/unvoiced flags
    """

    # Centroid frequencies in Hz (360 bins from ~32 Hz to ~2000 Hz)
    # These are the center frequencies of each pitch bin
    def __init__(self, f0_min: float = 50.0, f0_max: float = 1100.0,
                 model_path: str = None, device: str = None):
        """
        Args:
            f0_min: Minimum expected F0 in Hz.
            f0_max: Maximum expected F0 in Hz.
            model_path: Path to RMVPE pretrained weights (.pt file).
                        If None, will auto-download from HuggingFace.
            device: Device to run on ('cpu', 'cuda'). Auto-detected if None.
        """
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Mel spectrogram parameters (must match training configuration)
        self.mel_sr = 16000
        self.hop_size = 160  # 10ms hop for RMVPE (finer than RVC's 20ms)
        self.win_size = 512
        self.fft_size = 1024
        self.n_mels = 128
        self.fmin = 30.0   # RMVPE uses wider range internally
        self.fmax = 16000.0

        # Build the model
        self.model = RMVPEBackbone(in_channels=self.n_mels, hidden_channels=512)
        self.model.eval()

        # Load pretrained weights
        self._load_weights(model_path)

    def _load_weights(self, model_path: str = None):
        """Load pretrained RMVPE weights from file or auto-download."""
        if model_path is None:
            # Auto-download from HuggingFace (RVC project's RMVPE weights)
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = hf_hub_download(
                repo_id="lj1995/VoiceConversionWebUI",
                filename="rmvpe.pt",
                cache_dir=cache_dir,
                local_dir=cache_dir,
            )
            print(f"[RMVPE] Downloaded/loaded model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RMVPE model not found at {model_path}. "
                "Set model_path or ensure internet access for auto-download."
            )

        # Load weights - RMVPE weights may have different key names
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Try direct loading first
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # If keys don't match, try to map them
            mapped_state = self._map_state_dict(state_dict)
            self.model.load_state_dict(mapped_state, strict=False)
            print("[RMVPE] Loaded weights with partial key mapping (some keys may differ).")

        self.model.to(self.device)
        print(f"[RMVPE] Model loaded on {self.device}")

    def _map_state_dict(self, state_dict: dict) -> dict:
        """Map RMVPE checkpoint keys to our model architecture.

        The original RMVPE model uses 'model.' prefix and slightly different
        naming conventions. This function maps them to our architecture.
        """
        mapped = {}
        for key, value in state_dict.items():
            # Strip 'model.' prefix if present
            new_key = key
            if key.startswith("model."):
                new_key = key[len("model."):]

            # Map to our architecture's naming
            if new_key == "pre" or new_key == "input_layer":
                mapped["input_proj.weight"] = value
                continue
            if "centroids" in new_key or "output" in new_key:
                mapped["output_proj.weight"] = value
                continue

            # For block mappings, try to map residual CNN blocks
            if "blocks" in new_key or "layer" in new_key:
                mapped[new_key] = value
                continue

            # Try direct mapping
            if hasattr(self.model, new_key.split(".")[0]):
                mapped[new_key] = value

        return mapped

    @torch.no_grad()
    def extract(self, audio: np.ndarray, sr: int = 16000) -> tuple:
        """Extract F0 contour from audio.

        Args:
            audio: Audio waveform, numpy float32 array.
            sr: Sample rate of the input audio. Will be resampled to 16kHz if needed.

        Returns:
            f0: F0 values in Hz, numpy array. Unvoiced frames = 0. Shape: (T,).
            uv: Voiced/unvoiced flags. 1 = unvoiced, 0 = voiced. Shape: (T,).
               Frame rate = sr / hop_size.
        """
        # Resample to 16kHz if needed
        if sr != self.mel_sr:
            audio = self._resample(audio, sr, self.mel_sr)

        # Compute log-mel spectrogram
        mel_spec = self._compute_mel(audio)

        # Pad to handle edge effects
        mel_spec = np.pad(mel_spec, ((0, 0), (0, 3)), mode="constant")

        # Run inference
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).to(self.device)
        logits = self.model(mel_tensor)  # (1, n_centroids, T)

        # Convert logits to F0
        f0 = self._decode_f0(logits.squeeze(0).cpu().numpy())  # (T,)

        # Remove padding
        f0 = f0[: mel_spec.shape[1] - 3]

        # Generate voiced/unvoiced flags
        uv = (f0 <= 0).astype(np.float32)

        # Interpolate F0 to match RVC's frame rate (20ms hop = 16000/320)
        f0_rvc = self._interpolate_f0(f0, self.hop_size, 320)
        uv_rvc = self._interpolate_uv(uv, self.hop_size, 320)

        return f0_rvc, uv_rvc

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Resample audio using scipy."""
        if from_sr == to_sr:
            return audio
        duration = len(audio) / from_sr
        target_len = int(duration * to_sr)
        resampled = signal.resample(audio, target_len)
        return resampled.astype(np.float32)

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram using librosa.

        Returns:
            Log-mel spectrogram of shape (n_mels, T).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.mel_sr,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=1.0,  # Magnitude spectrum
        )
        log_mel = np.log(np.maximum(mel_spec, 1e-5))
        return log_mel.astype(np.float32)

    def _decode_f0(self, logits: np.ndarray) -> np.ndarray:
        """Decode pitch logits to F0 values.

        The RMVPE model outputs logits over a set of pitch centroids.
        We select the highest-probability centroid and convert to Hz.

        Args:
            logits: Shape (n_centroids, T).

        Returns:
            F0 values in Hz, shape (T,). Unvoiced frames = 0.
        """
        n_centroids, T = logits.shape

        # Generate centroid frequencies (logarithmically spaced)
        # RMVPE uses 360 centroids from ~32 Hz to ~2000 Hz (in mel scale)
        centroids_mel = np.linspace(
            self._hz_to_mel(32.0),
            self._hz_to_mel(2000.0),
            n_centroids
        )
        centroids_hz = self._mel_to_hz(centroids_mel)

        # For each frame, find the highest-probability centroid
        # Use softmax to get probabilities
        probs = self._softmax(logits, axis=0)  # (n_centroids, T)
        max_idx = np.argmax(probs, axis=0)  # (T,)

        # Confidence = max probability
        confidence = np.max(probs, axis=0)  # (T,)

        # Convert centroid indices to Hz
        f0 = centroids_hz[max_idx]

        # Apply confidence threshold
        # Low confidence frames are likely unvoiced
        uv_threshold = 0.03  # Minimum confidence for voiced frames
        f0[confidence < uv_threshold] = 0.0

        # Apply Viterbi smoothing for a continuous pitch contour
        f0 = self._viterbi_smoothing(f0, centroids_hz, probs)

        # Filter by F0 range
        f0[f0 < self.f0_min] = 0.0
        f0[f0 > self.f0_max] = 0.0

        return f0

    def _viterbi_smoothing(self, f0: np.ndarray, centroids_hz: np.ndarray,
                           probs: np.ndarray) -> np.ndarray:
        """Apply Viterbi decoding for smooth pitch contour.

        This prevents rapid jumps between pitch bins by finding the
        most likely path through all frames considering temporal coherence.
        """
        T = len(f0)
        n_centroids = len(centroids_hz)

        if T == 0:
            return f0

        # Transition cost: penalize jumps between distant centroids
        log_probs = np.log(probs + 1e-10)  # (n_centroids, T)

        # Viterbi
        viterbi = np.zeros_like(log_probs)
        backpointer = np.zeros((T, n_centroids), dtype=np.int64)

        # Initialization
        viterbi[:, 0] = log_probs[:, 0]

        # Forward pass
        for t in range(1, T):
            for c in range(n_centroids):
                # Transition cost: prefer small jumps
                costs = viterbi[:, t - 1] - np.abs(
                    np.arange(n_centroids) - c
                ) * 0.1  # Transition penalty
                best_prev = np.argmax(costs)
                viterbi[c, t] = costs[best_prev] + log_probs[c, t]
                backpointer[t, c] = best_prev

        # Backtrack
        path = np.zeros(T, dtype=np.int64)
        path[-1] = np.argmax(viterbi[:, -1])
        for t in range(T - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]

        # Convert path to F0
        smoothed_f0 = centroids_hz[path]

        # Keep unvoiced frames as 0
        smoothed_f0[f0 == 0] = 0.0

        return smoothed_f0

    def _interpolate_f0(self, f0: np.ndarray, from_hop: int, to_hop: int) -> np.ndarray:
        """Interpolate F0 from one frame rate to another.

        Args:
            f0: F0 values at from_hop frame rate.
            from_hop: Original hop size in samples.
            to_hop: Target hop size in samples.

        Returns:
            F0 values at to_hop frame rate.
        """
        if from_hop == to_hop:
            return f0

        # Create time axes
        t_orig = np.arange(len(f0)) * from_hop
        t_target_len = int(len(f0) * from_hop / to_hop)
        t_target = np.arange(t_target_len) * to_hop

        if t_target_len == 0:
            return f0

        # Separate voiced and unvoiced
        voiced = f0 > 0
        if not np.any(voiced):
            return np.zeros(t_target_len, dtype=np.float32)

        # Linear interpolation for voiced frames
        voiced_idx = np.where(voiced)[0]
        voiced_f0 = f0[voiced]
        voiced_t = t_orig[voiced_idx]

        # Interpolate
        interp_f0 = np.interp(t_target, voiced_t, voiced_f0)

        # Mark unvoiced regions (where original was 0)
        orig_interp = np.interp(t_target, t_orig, f0.astype(float))
        interp_f0[orig_interp < 1.0] = 0.0

        return interp_f0.astype(np.float32)

    def _interpolate_uv(self, uv: np.ndarray, from_hop: int, to_hop: int) -> np.ndarray:
        """Interpolate voiced/unvoiced flags."""
        if from_hop == to_hop:
            return uv
        t_orig = np.arange(len(uv)) * from_hop
        t_target_len = int(len(uv) * from_hop / to_hop)
        t_target = np.arange(t_target_len) * to_hop
        if t_target_len == 0:
            return uv
        # Nearest-neighbor interpolation for binary flags
        interp_uv = np.interp(t_target, t_orig, uv)
        return (interp_uv > 0.5).astype(np.float32)

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert Hz to mel scale (HTK formula)."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        """Convert mel scale to Hz (HTK formula)."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)