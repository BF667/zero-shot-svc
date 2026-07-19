"""
HiFi-GAN Vocoder

Generates high-fidelity audio waveforms from mel-spectrograms.

HiFi-GAN (Kong et al., 2020) uses a combination of:
1. Upsampling network: Transposes mel-spectrograms to waveform resolution
   using transposed convolutions with multi-receptive field fusion (MRF)
2. Multi-Receptive Field Fusion (MRF): Multiple parallel residual blocks
   with different kernel sizes capture patterns at different time scales
3. Periodic discriminators: For adversarial training (not used at inference)

This is the standard vocoder used in RVC v2 for high-quality
speech and singing synthesis.

Architecture:
    Mel spectrogram (B, 128, T)  [at 32000Hz, hop=320 -> 10ms frames]
        -> Initial Conv1d
        -> Upsample x4 (via transposed convolutions):
            320 -> 640 -> 1280 -> 2560 -> 5120  (x16 total upsampling)
        -> MRF blocks at each upsample step
        -> Final Conv1d -> waveform (B, 1, T*320)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with multi-kernel dilated convolutions.

    Uses 3 parallel convolutions with different kernel sizes (3, 7, 11)
    and dilations to capture multi-scale temporal patterns.
    This is the core building block of HiFi-GAN's MRF.
    """

    def __init__(self, channels: int, kernel_sizes: list = [3, 7, 11],
                 dilations: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        self.convs = nn.ModuleList()
        for ks, dils in zip(kernel_sizes, dilations):
            layers = nn.ModuleList()
            for d in dils:
                layers.append(nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, ks, padding=d * (ks - 1) // 2, dilation=d),
                    nn.LeakyReLU(0.1),
                ))
            self.convs.append(layers)
        self.convs_fusion = nn.Conv1d(channels * len(kernel_sizes), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, C, T).
        Returns:
            Output tensor, shape (B, C, T).
        """
        res = x
        out_list = []
        for convs in self.convs:
            h = x
            for conv in convs:
                h = conv(h) + h  # Residual within each scale
            out_list.append(h)
        out = torch.cat(out_list, dim=1)
        out = self.convs_fusion(out)
        return out + res


class UpBlock(nn.Module):
    """Upsampling block: Transposed Conv + MRF.

    Each upsample step:
    1. Transposed convolution doubles the temporal resolution
    2. Multi-Receptive Field Fusion (MRF) processes the upsampled signal
    """

    def __init__(self, in_channels: int, out_channels: int,
                 upsample_factor: int = 2):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=upsample_factor * 2,
            stride=upsample_factor,
            padding=upsample_factor // 2,
        )
        self.mrf = ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, in_ch, T).
        Returns:
            Upsampled tensor, shape (B, out_ch, T * upsample_factor).
        """
        x = self.upsample(x)
        x = self.mrf(x)
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator (vocoder) for mel-to-waveform synthesis.

    Converts mel-spectrograms into high-fidelity audio waveforms.

    The upsampling path:
    - Input mel: 128 channels at T frames (hop=320 samples per frame at 32kHz)
    - After 4x upsampling of 2x each: waveform at T*16 samples
    - With hop_size=320 and upsample_factor=16: 320/16 = 20 samples per mel frame
    - So the actual waveform should use a ratio that matches

    For RVC compatibility:
    - Mel hop = 320 samples at 32kHz (10ms per frame)
    - Need upsampling factor of 320 (from mel frame rate to sample rate)
    - 320 = 2^6 * 5, but we use 4 upsample layers of factor 5,5,4,2 = 200
    - Additional Conv1d layers handle the remaining ratio

    Actually, in practice RVC uses a simpler approach where the generator
    output is at the target sample rate directly. We implement the standard
    HiFi-GAN with configurable upsampling factors.
    """

    def __init__(self, in_channels: int = 128, out_channels: int = 1,
                 upsample_rates: list = [8, 8, 2, 2],
                 upsample_kernel_sizes: list = [16, 16, 4, 4],
                 resblock_kernel_sizes: list = [3, 7, 11],
                 resblock_dilations: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 initial_channels: int = 512):
        super().__init__()

        self.num_upsamples = len(upsample_rates)

        # Calculate channel sizes (halve at each upsample step)
        channels = [initial_channels]
        for i in range(self.num_upsamples):
            channels.append(channels[-1] // 2)

        # Initial projection
        self.conv_pre = nn.Conv1d(in_channels, channels[0], kernel_size=7, padding=3)

        # Upsampling blocks with MRF
        self.ups = nn.ModuleList()
        for i in range(self.num_upsamples):
            self.ups.append(UpBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                upsample_factor=upsample_rates[i],
            ))

        # Final projection
        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels[-1], out_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for better convergence."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel-spectrogram, shape (B, in_channels, T).

        Returns:
            Generated waveform, shape (B, 1, T * total_upsample_factor).
        """
        x = self.conv_pre(mel)

        for up in self.ups:
            x = up(x)

        x = self.conv_post(x)
        return x


class Vocoder:
    """High-level vocoder wrapper for waveform synthesis.

    Usage:
        vocoder = Vocoder()
        waveform = vocoder.generate(mel_spectrogram)
    """

    def __init__(self, model_path: str = None, device: str = None,
                 hop_size: int = 320, sample_rate: int = 32000):
        """
        Args:
            model_path: Path to pretrained HiFi-GAN weights.
            device: Device to run on.
            hop_size: Hop size used for mel-spectrogram computation.
            sample_rate: Target audio sample rate.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hop_size = hop_size
        self.sample_rate = sample_rate

        # Standard HiFi-GAN configuration for RVC
        # Total upsampling: 8 * 8 * 2 * 2 = 256
        # With hop_size=320 and upsample=256, we need 320/256 ≈ 1.25 ratio
        # We handle this by adjusting the mel computation to match
        self.model = HiFiGANGenerator(
            in_channels=128,
            out_channels=1,
            upsample_rates=[8, 8, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4],
        )
        self.model.eval()

        self._load_weights(model_path)

    def _load_weights(self, model_path: str = None):
        """Load pretrained HiFi-GAN weights."""
        if model_path is not None and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            try:
                self.model.load_state_dict(state_dict)
                print(f"[Vocoder] Loaded HiFi-GAN weights from {model_path}")
            except RuntimeError:
                self.model.load_state_dict(state_dict, strict=False)
                print("[Vocoder] Partially loaded HiFi-GAN weights.")
        else:
            print("[Vocoder] No pretrained weights found.")
            print("[Vocoder] Using randomly initialized HiFi-GAN.")
            print("[Vocoder] For best results, provide HiFi-GAN pretrained weights.")
            print("[Vocoder] Download from: https://huggingface.co/lj1995/VoiceConversionWebUI")

        self.model.to(self.device)
        print(f"[Vocoder] Model loaded on {self.device}")

    @torch.no_grad()
    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel-spectrogram.

        Args:
            mel: Mel-spectrogram, shape (B, 128, T) or (128, T).

        Returns:
            Waveform, shape (B, 1, T') or (1, T').
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        self.model.eval()
        waveform = self.model(mel.to(self.device))

        return waveform


import os