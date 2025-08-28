import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam

import wandb

from .diffusion_policy import DiffusionPolicy
from .network import FeedForwardNN


@dataclass
class FpoActionInfo:
    x_t_path: torch.Tensor         # (*, flow_steps, action_dim)
    loss_eps: torch.Tensor         # (*, sample_dim, action_dim)
    loss_t: torch.Tensor           # (*, sample_dim, 1)
    initial_cfm_loss: torch.Tensor # (*,)

    def detach(self):
        self.x_t_path = self.x_t_path.detach()
        self.loss_eps = self.loss_eps.detach()
        self.loss_t = self.loss_t.detach()
        self.initial_cfm_loss = self.initial_cfm_loss.detach()
        return self

class FPOActorCritic(nn.Module):
    def __init__(
            self, 
            num_actor_obs, 
            num_critic_obs, 
            num_actions, 
            actor_class="DiffusionPolicy",
            critic_class="FeedForwardNN",
            **kwargs):
        super().__init__()
        assert actor_class in ["DiffusionPolicy", "FeedForwardNN"], "Unsupported actor class"
        assert critic_class in ["FeedForwardNN"], "Unsupported critic class"
        actor_class = eval(actor_class)
        critic_class = eval(critic_class)

        self.obs_dim_actor = num_actor_obs + num_actions + 1
        self.obs_dim_critic = num_critic_obs
        self.act_dim = num_actions

        
        self.actor = actor_class(self.obs_dim_actor, self.act_dim, **kwargs)
        self.num_train_samples = kwargs['num_fpo_samples']

        print(f"training FPO with {self.num_train_samples} samples")


        # Critic: regular feedforward
        self.critic = critic_class(self.obs_dim_critic, 1)

        # Just for compatibility, this is not used by FPO's actor
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # self._current_action_info_cache = None

    def act(self, obs, **kwargs):
        with torch.no_grad():
            action, x_t_path, eps, t, initial_cfm_loss = self.actor.sample_action_with_info(obs, self.num_train_samples)
        
        action_info = FpoActionInfo(
            x_t_path=x_t_path,
            loss_eps=eps,
            loss_t=t,
            initial_cfm_loss=initial_cfm_loss
        )

        return action, action_info

    def act_inference(self, obs, **kwargs):
        with torch.no_grad():
            action = self.actor.sample_action(obs)
        return action
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def reset(self, dones=None):
        pass
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
