"""
Audio utility functions for loading, resampling, preprocessing,
and feature extraction.
"""
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy import signal


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file (.wav, .flac, .mp3, etc.)
        target_sr: Target sample rate (default 16kHz for HuBERT/RMVPE).

    Returns:
        Audio waveform as float32 numpy array at target_sr.
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sr: int = 32000):
    """Save audio to file.

    Args:
        path: Output file path.
        audio: Waveform array (float32, mono).
        sr: Sample rate.
    """
    sf.write(path, audio, sr)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a new sample rate using scipy.

    Args:
        audio: Input waveform.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled waveform.
    """
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    return resampled.astype(np.float32)


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad (with silence) or trim audio to a fixed length.

    Args:
        audio: Input waveform.
        target_length: Desired number of samples.

    Returns:
        Padded/trimmed waveform.
    """
    if len(audio) > target_length:
        # Trim from center
        start = (len(audio) - target_length) // 2
        return audio[start : start + target_length]
    elif len(audio) < target_length:
        pad_width = target_length - len(audio)
        return np.pad(audio, (0, pad_width), mode="constant")
    return audio


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    hop_size: int = 320,
    win_size: int = 640,
    fft_size: int = 1280,
    n_mels: int = 128,
    fmin: float = 50.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """Compute log-mel spectrogram from audio waveform.

    Args:
        audio: Waveform (float32).
        sr: Sample rate.
        hop_size: Hop length in samples.
        win_size: Window size in samples.
        fft_size: FFT size.
        n_mels: Number of mel filter banks.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz).

    Returns:
        Log-mel spectrogram of shape (n_mels, T).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_size,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0,
    )
    # Convert to log scale (add small epsilon to avoid log(0))
    log_mel = np.log(np.maximum(mel_spec, 1e-5))
    return log_mel.astype(np.float32)


def f0_to_coarse(f0: np.ndarray, f0_bin: int = 256, f0_max: float = 1100.0, f0_min: float = 50.0) -> np.ndarray:
    """Convert continuous F0 values to coarse (quantized) F0 indices.

    In RVC, F0 is quantized into discrete bins for conditioning the generator.

    Args:
        f0: F0 array in Hz, shape (T,). Unvoiced frames should be 0.
        f0_bin: Number of quantization bins.
        f0_max: Maximum F0 for quantization.
        f0_min: Minimum F0 for quantization.

    Returns:
        Coarse F0 indices as integer array, shape (T,). Unvoiced = 0.
    """
    f0_mel_min = 1127.0 * np.log(1.0 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + f0_max / 700.0)

    is_voiced = f0 > 0
    coarse = np.zeros_like(f0, dtype=np.int64)

    if np.any(is_voiced):
        f0_mel = 1127.0 * np.log(1.0 + f0[is_voiced] / 700.0)
        # Normalize to [0, f0_bin - 1]
        coarse[is_voiced] = np.round(
            (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min) * (f0_bin - 1)
        ).astype(np.int64)
        # Clamp to valid range
        coarse[is_voiced] = np.clip(coarse[is_voiced], 1, f0_bin - 1)

    return coarse


def numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor with appropriate shape."""
    return torch.from_numpy(arr).float()


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize audio to target peak amplitude."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def slice_audio(audio: np.ndarray, sr: int, max_seconds: float = 60.0) -> list:
    """Slice long audio into manageable chunks for processing.

    Args:
        audio: Full waveform.
        sr: Sample rate.
        max_seconds: Maximum chunk length in seconds.

    Returns:
        List of audio chunks.
    """
    max_samples = int(sr * max_seconds)
    chunks = []
    for i in range(0, len(audio), max_samples):
        chunk = audio[i : i + max_samples]
        chunks.append(chunk)
    return chunks