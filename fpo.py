import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.storage import RolloutStorage
from torch.distributions import MultivariateNormal
from torch.optim import Adam

import wandb

from .fpo_actor_critic import FPOActorCritic
from .fpo_rollout_storage import FPORolloutStorage


class FPO:
    """
    Flow Policy Optimization (https://arxiv.org/abs/2507.21053)
    This implementation is an reimplementation for FPO based on Jax and rsl-rl.
    This implementation is originally for the toddlerbot project (https://toddlerbot.github.io/).
    Author: Yao He
    """

    policy: FPOActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy: FPOActorCritic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.95,
        lam=0.98,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.02,
        device="gpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.kwargs = kwargs
        # Multi-GPU parameters
        # Not tested yet
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy
        self.policy.to(self.device)

        self.optimizer = Adam(self.policy.parameters(), learning_rate)
        self.scheduler_type = kwargs.get("scheduler_type", None)
        self.num_training_steps = kwargs["num_training_steps"]

        if self.scheduler_type is not None:
            self.scheduler_config = kwargs.get("schedule_dict", {})
            print(
                f"Using {self.scheduler_type} scheduler with config: {self.scheduler_config}"
            )
        if self.scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.scheduler_config.get("start_factor", 1.0),
                end_factor=self.scheduler_config.get("end_factor", 0.1),
                total_iters=self.num_training_steps,
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_config.get("T_max", 5000),
                eta_min=self.scheduler_config.get("eta_min", 0.1),
            )
        elif self.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_config.get("step_size", 1000),
                gamma=self.scheduler_config.get("gamma", 0.9),
            )

        self.cfm_diff_clip = kwargs.get("cfm_diff_clip_epsilon", 1.0)
        self.storage: FPORolloutStorage = None  # type: ignore
        self.transition = FPORolloutStorage.Transition()

        self.symmetry = None
        # FPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.positive_advantage = kwargs.get("positive_advantage", False)

        print(f"positive_advantage = {self.positive_advantage}")

        self.rnd = None

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        # create rollout storage
        # refer to DiffusionPolicy
        eps_shape = (
            self.policy.actor.init_noise.shape[1]
            if self.policy.actor.fixed_noise_inference
            else torch.Size([self.policy.actor.out_dim])
        )
        x_t_path_shape = torch.Size([self.policy.actor.num_steps + 1, *eps_shape])
        loss_eps_shape = torch.Size(
            [self.policy.num_train_samples, self.policy.actor.out_dim]
        )
        loss_t_shape = torch.Size([self.policy.num_train_samples, 1])
        initial_cfm_loss_shape = torch.Size([self.policy.num_train_samples])

        self.storage = FPORolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            x_t_path_shape,
            loss_eps_shape,
            loss_t_shape,
            initial_cfm_loss_shape,
            # rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        action, action_info = self.policy.act(obs)
        # compute the actions and values
        self.transition.actions = action.detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        # dummy value for compatibility
        self.transition.actions_log_prob = torch.zeros(
            torch.Size([self.transition.actions.shape[0], 1]), device=self.device
        )
        self.transition.action_mean = torch.zeros_like(self.transition.actions)
        self.transition.action_sigma = torch.zeros_like(self.transition.actions)
        self.transition.x_t_path = action_info.x_t_path.detach()
        self.transition.loss_eps = action_info.loss_eps.detach()
        self.transition.loss_t = action_info.loss_t.detach()
        self.transition.initial_cfm_loss = action_info.initial_cfm_loss.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            x_t_pathes_batch,
            loss_eps_batch,
            loss_ts_batch,
            initial_cfm_losses_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                    obs_type="policy",
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch,
                    actions=None,
                    env=self.symmetry["_env"],
                    obs_type="critic",
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor

                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

                x_t_pathes_batch = x_t_pathes_batch.repeat(num_aug, 1)
                loss_eps_batch = loss_eps_batch.repeat(num_aug, 1)
                loss_ts_batch = loss_ts_batch.repeat(num_aug, 1)
                initial_cfm_losses_batch = initial_cfm_losses_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions

            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )

            # -- entropy

            entropy_batch = torch.tensor(0.0).to(
                self.device
            )  # or pull from actor if defined


            B, N, D = loss_eps_batch.shape  # batch size, num train samples, action dim
            expanded_obs = obs_batch.unsqueeze(1).expand(B, N, -1)  # [B, N, D_s]
            expanded_acts = actions_batch.unsqueeze(1).expand(B, N, -1)  # [B, N, D_a]

            # cfm_loss
            cfm_loss = self.policy.actor.compute_cfm_loss(
                expanded_obs, expanded_acts, loss_eps_batch, loss_ts_batch
            )
            # Surrogate loss
            cfm_difference = initial_cfm_losses_batch - cfm_loss

            cfm_difference = torch.clamp(
                cfm_difference, -self.cfm_diff_clip, self.cfm_diff_clip
            ) 


            rho_s = torch.exp(torch.clamp(cfm_difference.mean(dim=1), -self.cfm_diff_clip, self.cfm_diff_clip))

            surrogate = torch.squeeze(advantages_batch) * rho_s

            surrogate_clipped = torch.squeeze(advantages_batch) * torch.clamp(
                rho_s, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            # surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            surrogate_loss = (-torch.min(surrogate, surrogate_clipped)).mean()

            # surrogate_loss -= self.entropy_coef * entropy_batch.mean()

            # # Value function loss
            value_loss = nn.MSELoss()(value_batch, returns_batch)

            # -- For FPO
            loss = (
                surrogate_loss
                + value_loss * self.value_loss_coef
                - entropy_batch.mean() * self.entropy_coef
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For FPO
        self.scheduler.step()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "clipped_ratio_mean": (torch.abs(rho_s - 1.0) > self.clip_param)
            .float()
            .mean()
            .item(),
            "initial_cfm_loss": initial_cfm_losses_batch.mean().item(),
            "cfm_loss": cfm_loss.mean().item(),
            "cfm_difference": cfm_difference.mean().item(),
            "policy_ratio_mean": rho_s.mean().item(),
            "policy_ratio_min": rho_s.min().item(),
            "policy_ratio_max": rho_s.max().item(),
            # "policy_loss": (-torch.min(surrogate, surrogate_clipped)).mean().item(),
            "adv": advantages_batch.mean().item(),
            "surrogate_loss1_mean": surrogate.mean().item(),
            "surrogate_loss2_mean": surrogate_clipped.mean().item(),
            "action_min": actions_batch.min().item(),
            "action_max": actions_batch.max().item(),
        }

        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
