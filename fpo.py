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

        # create optimizer
        # split optimizer
        # self.actor_optim = Adam(self.policy.actor.parameters(), learning_rate)
        # self.critic_optim = Adam(self.policy.critic.parameters(), learning_rate)
        # use combined one

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
        # self.t_so_far = 0
        # self.i_so_far = 0

        # for rsl_rl compatible, no use
        # self.value_loss_coef = 1.0
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
        # rnd_state_shape = None
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

        # generator for mini batches
        # if self.policy.is_recurrent:
        #     generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # else:
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
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            # self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            # actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # # -- critic
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )

            # # -- entropy
            # # we only keep the entropy of the first augmentation (the original one)
            # mu_batch = self.policy.action_mean[:original_batch_size]
            # sigma_batch = self.policy.action_std[:original_batch_size]
            # entropy_batch = self.policy.entropy[:original_batch_size]
            entropy_batch = torch.tensor(0.0).to(
                self.device
            )  # or pull from actor if defined

            # # KL
            # if self.desired_kl is not None and self.schedule == "adaptive":
            #     with torch.inference_mode():
            #         kl = torch.sum(
            #             torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            #             + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            #             / (2.0 * torch.square(sigma_batch))
            #             - 0.5,
            #             axis=-1,
            #         )
            #         kl_mean = torch.mean(kl)

            #         # Reduce the KL divergence across all GPUs
            #         if self.is_multi_gpu:
            #             torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            #             kl_mean /= self.gpu_world_size

            #         # Update the learning rate
            #         # Perform this adaptation only on the main process
            #         # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
            #         #       then the learning rate should be the same across all GPUs.
            #         if self.gpu_global_rank == 0:
            #             if kl_mean > self.desired_kl * 2.0:
            #                 self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            #             elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            #                 self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            #         # Update the learning rate for all GPUs
            #         if self.is_multi_gpu:
            #             lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            #             torch.distributed.broadcast(lr_tensor, src=0)
            #             self.learning_rate = lr_tensor.item()

            #         # Update the learning rate for all parameter groups
            #         for param_group in self.optimizer.param_groups:
            #             param_group["lr"] = self.learning_rate

            B, N, D = loss_eps_batch.shape  # batch size, num train samples, action dim
            expanded_obs = obs_batch.unsqueeze(1).expand(B, N, -1)  # [B, N, D_s]
            expanded_acts = actions_batch.unsqueeze(1).expand(B, N, -1)  # [B, N, D_a]
            # flat_eps = loss_eps_batch                                         # [B, N, D_a]
            # flat_t = loss_ts_batch                                            # [B, N, 1]
            # flat_init_loss = initial_cfm_losses_batch                         # [B, N]
            # we do the flat inside compute_cfm_loss, unlike the original fpo implementation
            # cfm_loss
            cfm_loss = self.policy.actor.compute_cfm_loss(
                expanded_obs, expanded_acts, loss_eps_batch, loss_ts_batch
            )
            # Surrogate loss
            # cfm_difference = flat_init_loss - cfm_loss
            cfm_difference = initial_cfm_losses_batch - cfm_loss
            # cfm_difference = cfm_difference.view(B, N)
            cfm_difference = torch.clamp(
                cfm_difference, -self.cfm_diff_clip, self.cfm_diff_clip
            ) 
            # ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

            rho_s = torch.exp(torch.clamp(cfm_difference.mean(dim=1), -self.cfm_diff_clip, self.cfm_diff_clip))
            # surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate = torch.squeeze(advantages_batch) * rho_s
            # surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            #     ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            # )
            surrogate_clipped = torch.squeeze(advantages_batch) * torch.clamp(
                rho_s, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            # surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            surrogate_loss = (-torch.min(surrogate, surrogate_clipped)).mean()

            # surrogate_loss -= self.entropy_coef * entropy_batch.mean()

            # # Value function loss
            # if self.use_clipped_value_loss:
            #     value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            #         -self.clip_param, self.clip_param
            #     )
            #     value_losses = (value_batch - returns_batch).pow(2)
            #     value_losses_clipped = (value_clipped - returns_batch).pow(2)
            #     value_loss = torch.max(value_losses, value_losses_clipped).mean()
            # else:
            #     value_loss = (returns_batch - value_batch).pow(2).mean()
            value_loss = nn.MSELoss()(value_batch, returns_batch)

            # loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # # Symmetry loss
            # if self.symmetry:
            #     # obtain the symmetric actions
            #     # if we did augmentation before then we don't need to augment again
            #     if not self.symmetry["use_data_augmentation"]:
            #         data_augmentation_func = self.symmetry["data_augmentation_func"]
            #         obs_batch, _ = data_augmentation_func(
            #             obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
            #         )
            #         # compute number of augmentations per sample
            #         num_aug = int(obs_batch.shape[0] / original_batch_size)

            #     # actions predicted by the actor for symmetrically-augmented observations
            #     mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

            #     # compute the symmetrically augmented actions
            #     # note: we are assuming the first augmentation is the original one.
            #     #   We do not use the action_batch from earlier since that action was sampled from the distribution.
            #     #   However, the symmetry loss is computed using the mean of the distribution.
            #     action_mean_orig = mean_actions_batch[:original_batch_size]
            #     _, actions_mean_symm_batch = data_augmentation_func(
            #         obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
            #     )

            #     # compute the loss (we skip the first augmentation as it is the original one)
            #     mse_loss = torch.nn.MSELoss()
            #     symmetry_loss = mse_loss(
            #         mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
            #     )
            #     # add the loss to the total loss
            #     if self.symmetry["use_mirror_loss"]:
            #         loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
            #     else:
            #         symmetry_loss = symmetry_loss.detach()

            # Compute the gradients
            # -- For FPO
            # split optim version
            # self.optimizer.zero_grad()
            # self.actor_optim.zero_grad()
            # surrogate_loss.backward(retain_graph=True)

            # # Collect gradients from all GPUs
            # if self.is_multi_gpu:
            #     self.reduce_parameters()

            # # Apply the gradients
            # # -- For FPO
            # # nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            # nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            # # self.optimizer.step()
            # self.actor_optim.step()

            # self.critic_optim.zero_grad()
            # value_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            # self.critic_optim.step()

            # -- For FPO
            # one optimizer
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
