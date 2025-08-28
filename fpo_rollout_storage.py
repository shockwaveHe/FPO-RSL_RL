import torch
from rsl_rl.storage import RolloutStorage


class FPORolloutStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        """
        Transition class for FPO Rollout Storage.
        It extends the base Transition class to include additional fields specific to FPO.
        """
        def __init__(self):
            super().__init__()
            self.x_t_path = None         # (*, flow_steps, action_dim)
            self.loss_eps = None         # (*, sample_dim, action_dim)
            self.loss_t = None           # (*, sample_dim, 1)
            self.initial_cfm_loss = None # (*,)

        def clear(self):
           self.__init__()


    def __init__(self,         
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        x_t_path_shape,
        loss_eps_shape,
        loss_t_shape,
        initial_cfm_loss_shape,
        device="cpu",
        **kwargs):

        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            device=device,
            **kwargs
        )

        self.x_t_path_shape = x_t_path_shape
        self.loss_eps_shape = loss_eps_shape
        self.loss_t_shape = loss_t_shape
        self.initial_cfm_losses_shape = initial_cfm_loss_shape

        self.x_t_pathes = torch.zeros(num_transitions_per_env, num_envs, *x_t_path_shape, device=self.device)
        self.loss_eps = torch.zeros(num_transitions_per_env, num_envs, *loss_eps_shape, device=self.device)
        self.loss_ts = torch.zeros(num_transitions_per_env, num_envs, *loss_t_shape, device=self.device)
        self.initial_cfm_losses = torch.zeros(num_transitions_per_env, num_envs, *initial_cfm_loss_shape, device=self.device)

    def add_transitions(self, transition: Transition):
        """
        Add a transition to the storage.
        """
        super().add_transitions(transition)
        # fall back to account for the last line in the super function
        self.step -= 1

        self.x_t_pathes[self.step].copy_(transition.x_t_path)
        self.loss_eps[self.step].copy_(transition.loss_eps)
        self.loss_ts[self.step].copy_(transition.loss_t)
        self.initial_cfm_losses[self.step].copy_(transition.initial_cfm_loss)
        # track back
        self.step += 1

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For FPO
        # old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        # old_mu = self.mu.flatten(0, 1)
        # old_sigma = self.sigma.flatten(0, 1)
        x_t_pathes = self.x_t_pathes.flatten(0, 1)
        loss_eps = self.loss_eps.flatten(0, 1)
        loss_ts = self.loss_ts.flatten(0, 1)
        initial_cfm_losses = self.initial_cfm_losses.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For FPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                # old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                # old_mu_batch = old_mu[batch_idx]
                # old_sigma_batch = old_sigma[batch_idx]
                x_t_pathes_batch = x_t_pathes[batch_idx]
                loss_eps_batch = loss_eps[batch_idx]
                loss_ts_batch = loss_ts[batch_idx]
                initial_cfm_losses_batch = initial_cfm_losses[batch_idx]

                # yield the mini-batch
                yield obs_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, x_t_pathes_batch, loss_eps_batch, loss_ts_batch, initial_cfm_losses_batch, (# old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, None # , rnd_state_batch