from dataclasses import dataclass
from typing import Tuple

import gin


@gin.configurable
@dataclass
class FPOConfig:
    """Data class for storing FPO hyperparameters."""
    wandb_project: str = ""
    wandb_entity: str = "" #
    actor_class: str = "DiffusionPolicy" # ["DiffusionPolicy", "FeedForwardNN"]
    critic_class: str = "FeedForwardNN" # ["DiffusionPolicy", "FeedForwardNN"]
    use_rnn: bool = False  # specific to rsl_rl
    num_timesteps: int = 200_000_000
    num_evals: int = 100
    episode_length: int = 1000
    unroll_length: int = 20
    num_updates_per_batch: int = 4
    discounting: float = 0.995 # gamma
    gae_lambda: float = 0.98
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    normalize_observation: bool = False
    learning_rate: float = 3e-4
    entropy_cost: float = 5e-4
    clipping_epsilon: float = 0.005 #0.025
    cfm_diff_clip_epsilon: float = 1.0
    num_envs: int = 1024
    render_nums: int = 20
    batch_size: int = 256
    num_minibatches: int = 4
    seed: int = 0
    num_fpo_samples: int = 25
    positive_advantage: bool = False

    # diffusion policy config
    dp_num_steps: int = 10
    dp_fixed_noise_inference: bool = False
    scheduler_type: str = "linear"  # ["linear", "cosine", "step", "exponential", None]
    schedule_dict: str = "start_factor:1.0,end_factor:0.5"
