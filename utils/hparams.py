"""
Hyperparameters and configuration loader for Zero-Shot SVC.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    output_sample_rate: int = 32000
    hop_size: int = 320
    win_size: int = 640
    fft_size: int = 1280
    mel_bins: int = 128
    fmin: float = 50.0
    fmax: float = 8000.0


@dataclass
class ContentEncoderConfig:
    model_name: str = "contentvec"
    output_dim: int = 256
    hop_length: int = 320
    repo_id: str = "lengyue233/content-vec-best"


@dataclass
class F0ExtractorConfig:
    method: str = "rmvpe"
    f0_min: float = 50.0
    f0_max: float = 1100.0
    rmvpe_model_repo: str = "lj1995/VoiceConversionWebUI"
    rmvpe_model_file: str = "rmvpe.pt"


@dataclass
class SpeakerEncoderConfig:
    model_name: str = "cam++"
    embedding_dim: int = 192
    sample_rate: int = 16000
    repo_id: str = "funasr/cam++"


@dataclass
class GeneratorConfig:
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_flow_layers: int = 4
    n_speakers: int = 1
    gin_channels: int = 256
    use_spk_conditioning: bool = True


@dataclass
class VocoderConfig:
    name: str = "hifigan"
    hop_size: int = 320


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    content_encoder: ContentEncoderConfig = field(default_factory=ContentEncoderConfig)
    f0_extractor: F0ExtractorConfig = field(default_factory=F0ExtractorConfig)
    speaker_encoder: SpeakerEncoderConfig = field(default_factory=SpeakerEncoderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        cfg = cls()
        if "audio" in d:
            for k, v in d["audio"].items():
                if hasattr(cfg.audio, k):
                    setattr(cfg.audio, k, v)
        if "content_encoder" in d:
            for k, v in d["content_encoder"].items():
                if hasattr(cfg.content_encoder, k):
                    setattr(cfg.content_encoder, k, v)
        if "f0_extractor" in d:
            for k, v in d["f0_extractor"].items():
                if hasattr(cfg.f0_extractor, k):
                    setattr(cfg.f0_extractor, k, v)
        if "speaker_encoder" in d:
            for k, v in d["speaker_encoder"].items():
                if hasattr(cfg.speaker_encoder, k):
                    setattr(cfg.speaker_encoder, k, v)
        if "generator" in d:
            for k, v in d["generator"].items():
                if hasattr(cfg.generator, k):
                    setattr(cfg.generator, k, v)
        if "vocoder" in d:
            for k, v in d["vocoder"].items():
                if hasattr(cfg.vocoder, k):
                    setattr(cfg.vocoder, k, v)
        return cfg