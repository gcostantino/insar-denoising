from dataclasses import dataclass, field
from typing import List, Tuple, Union

from kito.config.moduleconfig import ModelConfig, KitoModuleConfig


@dataclass
class InSARDenoiserModelConfig(ModelConfig):
    max_encoder_features: int = 512
    n_attention_heads: int = 8
    transformer_dropout_rate: float = 0.1
    enc_conv_dropout_rate: float = 0.1
    dec_conv_dropout_rate: float = 0.1
    weight_decay: float = 1e-4

    segmentation_task: bool = False
    regress_all_noise_sources: bool = False
    multi_task: bool = False
    multi_loss: bool = False
    supervise_stratified_turbulent: bool = False
    freeze_encoder: bool = False
    subtract_first_frame: bool = False

    mask_loss_weight: float = 1.0
    structural_similarity_loss_weight: float = 0.0
    reconstruction_loss_weight: float = 1.0
    temporal_evolution_loss_weight: float = 0.0

    topography_size: Union[Tuple, List] = (1, 128, 128, 1)
    img_size: int = 128
    num_temporal_positions: int = 9


@dataclass
class InSARDenoiserConfig(KitoModuleConfig):
    """
    Base configuration container for InSARDenoiser.

    Contains all configuration sections:
    - training: Training parameters
    - model: Model architecture and settings
    - workdir: Output directories
    - data: Dataset and preprocessing
    """
    model: InSARDenoiserModelConfig = field(default_factory=InSARDenoiserModelConfig)
    # all other fields are inherited
