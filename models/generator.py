"""
VITS-style Generator for Voice Conversion

Based on VITS (Variational Inference with adversarial learning for end-to-end
Text-to-Speech, Kim et al., 2021), adapted for singing voice conversion.

In the RVC/SVC context, the generator:
1. Takes content features (from ContentVec), F0, and speaker embedding as input
2. Uses a Posterior Encoder to convert content features + F0 into a latent representation
3. Uses a Flow module for distribution normalization
4. Uses a Transformer-based decoder to generate mel-spectrogram
5. The mel-spectrogram is then passed to a vocoder (HiFi-GAN) for waveform synthesis

Key adaptation for ZERO-SHOT:
- Instead of using a learned speaker lookup table (which requires training per speaker),
  we condition the generator directly on the speaker embedding from the reference audio.
- The speaker embedding is injected via FiLM (Feature-wise Linear Modulation) layers
  and added to the transformer decoder's cross-attention.

Architecture flow:
    Content (T, 256) + F0 (T, 1) + SpkEmb (192,)
        -> Posterior Encoder -> z (latent)
        -> Normalization Flows -> z_norm
        -> Transformer Decoder (conditioned on spk_emb) -> mel spectrogram
        -> HiFi-GAN Vocoder -> waveform
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) or (B, C, T)
        if x.dim() == 3 and x.size(1) > x.size(2):
            x = x.transpose(1, 2)
        return self.norm(x)


class Conv1dGLU(nn.Module):
    """Conv1d with Gated Linear Unit activation.

    Used in VITS for the posterior encoder and decoder.
    Splits channels into two halves, applies sigmoid to one half and
    multiplies with the other (gating mechanism).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels * 2, kernel_size,
            padding=padding, dilation=dilation
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Split into value and gate
        val, gate = x.chunk(2, dim=1)
        return val * torch.sigmoid(gate)


class WaveNetBlock(nn.Module):
    """Dilated causal convolution block with residual connection.

    Used in the VITS posterior encoder to process content features + F0.
    Each block has:
    - Dilated conv -> tanh
    - 1x1 conv -> sigmoid (gate)
    - Element-wise product
    - 1x1 conv -> residual + skip connection
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            channels, channels * 2, kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor, shape (B, C, T).
        Returns:
            Tuple of (output, skip_connection).
        """
        residual = x
        h = self.dilated_conv(x)
        val, gate = h.chunk(2, dim=1)
        h = torch.tanh(val) * torch.sigmoid(gate)
        h = self.residual_conv(h)
        skip = self.skip_conv(h)
        return (h + residual, skip)


# ---------------------------------------------------------------------------
# Posterior Encoder
# ---------------------------------------------------------------------------

class PosteriorEncoder(nn.Module):
    """Posterior encoder that converts content features + F0 into latent z.

    In the RVC context:
    - Input: Content features (256-dim) concatenated with F0 features
    - Processing: WaveNet-style dilated convolutions
    - Output: Mean and std of the latent distribution, then reparameterization

    The posterior encoder learns to capture the acoustic details beyond
    just content (e.g., prosody, breathiness, vocal effort) while being
    guided by the F0 contour.
    """

    def __init__(self, in_channels: int = 257, hidden_channels: int = 192,
                 out_channels: int = 192, n_layers: int = 16,
                 kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels  # 256 (content) + 1 (F0)
        self.hidden_channels = hidden_channels

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        # WaveNet dilated convolutions with increasing dilation
        self.wavenet_blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)  # Cycle through 1, 2, 4, 8
            self.wavenet_blocks.append(
                WaveNetBlock(hidden_channels, kernel_size, dilation)
            )

        # Output projections for mean and log_std
        self.out_proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input features, shape (B, in_channels, T).
               Typically content features (256) + F0 (1) = 257 channels.

        Returns:
            z: Latent representation, shape (B, out_channels, T).
            log_q: Log probability of the latent distribution.
            m: Mean of the distribution, shape (B, out_channels, T).
            logs: Log std of the distribution, shape (B, out_channels, T).
        """
        x = self.input_proj(x)

        # Process through WaveNet blocks
        skip_connections = 0
        for block in self.wavenet_blocks:
            x, skip = block(x)
            skip_connections = skip_connections + skip

        x = F.relu(skip_connections)

        # Project to mean and log_std
        stats = self.out_proj(x)  # (B, out_channels * 2, T)
        m, logs = stats.chunk(2, dim=1)

        # Reparameterization trick
        z = m + torch.randn_like(m) * torch.exp(logs)

        # Log probability
        log_q = torch.sum(
            -0.5 * (math.log(2 * math.pi) + 2 * logs + ((z - m) ** 2) / torch.exp(2 * logs)),
            dim=[1, 2],
        ).mean()

        return z, log_q, m, logs


# ---------------------------------------------------------------------------
# Normalizing Flow
# ---------------------------------------------------------------------------

class Flow(nn.Module):
    """Normalizing flow for transforming the latent distribution.

    Uses affine coupling layers (similar to RealNVP) to transform
    the posterior distribution into a simpler (Gaussian) distribution.
    This enables better sampling during generation.

    In RVC, flows help normalize the latent space so the decoder
    can generate more naturally.
    """

    def __init__(self, channels: int = 192, hidden_channels: int = 192,
                 n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_layers):
            self.flows.append(AffineCouplingLayer(channels, hidden_channels, kernel_size))
            # Reverse the permutation between layers for better expressivity
            self.flows.append(InvertibleConv1d(channels))

    def forward(self, z: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Args:
            z: Latent representation, shape (B, channels, T).
            reverse: If True, run the flow in reverse (for generation).

        Returns:
            Transformed latent representation.
        """
        log_det_total = 0
        if not reverse:
            for flow in self.flows:
                if isinstance(flow, AffineCouplingLayer):
                    z, log_det = flow(z)
                    log_det_total += log_det
                else:
                    z = flow(z)
        else:
            for flow in reversed(self.flows):
                if isinstance(flow, AffineCouplingLayer):
                    z, log_det = flow(z, reverse=True)
                    log_det_total += log_det
                else:
                    z = flow(z, reverse=True)
        return z


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow.

    Splits the input into two halves:
    - One half is transformed using a neural network
    - The other half is scaled and shifted based on the first half's output
    """

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        half_ch = channels // 2

        # Network to predict scale (s) and shift (t)
        self.net = nn.Sequential(
            nn.Conv1d(half_ch, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, half_ch * 2, 1),  # s and t
        )

        # Initialize the last layer to near-zero (start as identity)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple:
        """
        Args:
            x: Input, shape (B, C, T).
            reverse: Reverse direction.

        Returns:
            Transformed output, log determinant.
        """
        x0, x1 = x.chunk(2, dim=1)
        h = self.net(x0)
        s, t = h.chunk(2, dim=1)

        # Clamp scale to prevent numerical issues
        s = torch.clamp(s, min=-5.0, max=5.0)

        if not reverse:
            y1 = x1 * torch.exp(s) + t
            log_det = torch.sum(s, dim=[1, 2])
        else:
            y1 = (x1 - t) * torch.exp(-s)
            log_det = torch.sum(-s, dim=[1, 2])

        return torch.cat([x0, y1], dim=1), log_det


class InvertibleConv1d(nn.Module):
    """Invertible 1x1 convolution for channel permutation.

    Used between coupling layers to allow all dimensions to be transformed.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Initialize with a random orthogonal matrix
        w = torch.linalg.qr(torch.randn(channels, channels), mode='reduced')[0]
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            return F.conv1d(x, self.weight.unsqueeze(-1))
        else:
            # Inverse using LU decomposition
            w_inv = torch.inverse(self.weight)
            return F.conv1d(x, w_inv.unsqueeze(-1))


# ---------------------------------------------------------------------------
# FiLM Conditioning
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Conditions a feature map on a speaker embedding by learning
    scale (gamma) and shift (beta) from the embedding.

    output = gamma * input + beta

    This is the key mechanism for zero-shot speaker conditioning:
    the speaker embedding modulates the decoder's internal representations.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int):
        super().__init__()
        self.gamma = nn.Linear(conditioning_dim, feature_dim)
        self.beta = nn.Linear(conditioning_dim, feature_dim)
        # Initialize to identity
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor, shape (B, T, feature_dim).
            cond: Conditioning vector, shape (B, conditioning_dim).

        Returns:
            Modulated features, shape (B, T, feature_dim).
        """
        gamma = self.gamma(cond).unsqueeze(1)  # (B, 1, feature_dim)
        beta = self.beta(cond).unsqueeze(1)  # (B, 1, feature_dim)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Transformer Decoder
# ---------------------------------------------------------------------------

class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for the transformer decoder.

    Uses sinusoidal encoding based on relative position, which
    works better than absolute positions for variable-length sequences.
    """

    def __init__(self, dim: int, max_len: int = 3000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with speaker conditioning via FiLM.

    Modified from standard transformer to include:
    1. FiLM conditioning on speaker embedding
    2. Cross-attention between text content and acoustic features (not used
       in zero-shot mode, but kept for architecture compatibility)
    """

    def __init__(self, d_model: int, nhead: int, d_ff: int = 768,
                 dropout: float = 0.1, gin_channels: int = 256):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FiLM conditioning from speaker embedding
        self.film = FiLMLayer(d_model, gin_channels) if gin_channels > 0 else None

    def forward(self, x: torch.Tensor, spk_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (B, T, d_model).
            spk_emb: Speaker embedding, shape (B, gin_channels).

        Returns:
            Output features, shape (B, T, d_model).
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))

        # Apply FiLM speaker conditioning
        if self.film is not None and spk_emb is not None:
            x = self.film(x, spk_emb)

        return x


class GeneratorDecoder(nn.Module):
    """VITS-style Transformer decoder for mel-spectrogram generation.

    Takes the normalized latent z from the flow module and generates
    a mel-spectrogram, conditioned on the speaker embedding.

    Architecture:
    1. Initial projection to transformer dimension
    2. Relative positional encoding
    3. Stack of TransformerDecoderLayers with FiLM conditioning
    4. Output projection to mel-spectrogram dimension
    """

    def __init__(self, in_channels: int = 192, out_channels: int = 128,
                 hidden_channels: int = 192, n_heads: int = 2,
                 n_layers: int = 6, ffn_dim: int = 768,
                 dropout: float = 0.1, gin_channels: int = 256):
        super().__init__()
        self.pos_enc = RelativePositionalEncoding(hidden_channels)

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=hidden_channels,
                nhead=n_heads,
                d_ff=ffn_dim,
                dropout=dropout,
                gin_channels=gin_channels,
            )
            for _ in range(n_layers)
        ])

        # Output projection to mel-spectrogram
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, z: torch.Tensor, spk_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: Normalized latent, shape (B, in_channels, T).
            spk_emb: Speaker embedding, shape (B, gin_channels).

        Returns:
            Mel-spectrogram, shape (B, out_channels, T).
        """
        # Transpose for transformer: (B, T, C)
        h = z.transpose(1, 2)
        h = self.input_proj(h)

        # Add positional encoding
        h = self.pos_enc(h)

        # Process through transformer layers
        for layer in self.layers:
            h = layer(h, spk_emb)

        # Output projection
        mel = self.output_proj(h)  # (B, T, out_channels)
        return mel.transpose(1, 2)  # (B, out_channels, T)


# ---------------------------------------------------------------------------
# Full Generator
# ---------------------------------------------------------------------------

class VITSGenerator(nn.Module):
    """Complete VITS-style generator for singing voice conversion.

    Combines:
    1. Posterior Encoder: content + F0 -> latent z
    2. Normalizing Flows: z -> normalized z
    3. Transformer Decoder: normalized z + speaker embedding -> mel spectrogram

    For zero-shot: speaker embedding is extracted from reference audio
    (no per-speaker training required).
    """

    def __init__(self, content_dim: int = 256, f0_bin: int = 256,
                 hidden_channels: int = 192, out_channels: int = 128,
                 n_flow_layers: int = 4, n_decoder_layers: int = 6,
                 n_heads: int = 2, ffn_dim: int = 768,
                 gin_channels: int = 256, dropout: float = 0.1):
        super().__init__()

        # Posterior encoder (content + F0 -> latent)
        self.posterior_encoder = PosteriorEncoder(
            in_channels=content_dim + 1,  # 256 content + 1 F0
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            n_layers=16,
            kernel_size=5,
        )

        # Normalizing flows
        self.flow = Flow(
            channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_flow_layers,
        )

        # Transformer decoder (latent + speaker -> mel)
        self.decoder = GeneratorDecoder(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            gin_channels=gin_channels,
        )

    def forward(self, content: torch.Tensor, f0: torch.Tensor,
                spk_emb: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: content + F0 + speaker -> mel spectrogram.

        Args:
            content: Content features from ContentVec, shape (B, 256, T).
            f0: F0 values (normalized), shape (B, 1, T).
            spk_emb: Speaker embedding, shape (B, gin_channels).

        Returns:
            Generated mel-spectrogram, shape (B, out_channels, T).
        """
        # Concatenate content and F0
        x = torch.cat([content, f0], dim=1)  # (B, 257, T)

        # Posterior encoding
        z, log_q, m, logs = self.posterior_encoder(x)

        # Normalizing flow (forward)
        z_norm = self.flow(z)

        # Generate mel-spectrogram
        mel = self.decoder(z_norm, spk_emb)

        return mel

    @torch.no_grad()
    def infer(self, content: torch.Tensor, f0: torch.Tensor,
              spk_emb: torch.Tensor, noise_scale: float = 0.667) -> torch.Tensor:
        """Inference mode with noise injection for naturalness.

        Args:
            content: Content features, shape (B, 256, T).
            f0: F0 values, shape (B, 1, T).
            spk_emb: Speaker embedding, shape (B, gin_channels).
            noise_scale: Amount of noise to inject (controls variance).

        Returns:
            Generated mel-spectrogram, shape (B, out_channels, T).
        """
        # Concatenate content and F0
        x = torch.cat([content, f0], dim=1)

        # Posterior encoding (deterministic at inference: use mean only)
        z, _, m, logs = self.posterior_encoder(x)

        # Add noise for naturalness
        z = m + noise_scale * torch.randn_like(m) * torch.exp(logs)

        # Flow forward
        z_norm = self.flow(z)

        # Decode
        mel = self.decoder(z_norm, spk_emb)
        return mel