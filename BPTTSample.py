from .BPTT import BPTT, CaD, CaDu
from stable_baselines3.common.off_policy_algorithm import SelfOffPolicyAlgorithm, MaybeCallback
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule, TrainFreq
import torch as th
from stable_baselines3.common.utils import polyak_update
from typing import Optional
from tqdm import tqdm


class BPTTSample(BPTT):
    def __init__(
        self, 
        train_env,
        actor_batch_size,
        initial_sample: bool = True,
        *args,**kwargs
    ):
        self.initial_sample = initial_sample
        self.train_env = train_env
        self.actor_batch_size = actor_batch_size
        super().__init__(*args, **kwargs)
    
    def _setup_model(self):
        super()._setup_model()
        self.train_env.reset()
        self.train_env.set_requires_grad(True)
        self.env.set_requires_grad(False)
        
    def _set_name(self):
        self.name = "BPTTSample"
        
    def learn(
        self: SelfOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            self.name if self.comment is None else f"{self.name}_{self.comment}",
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()
        with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
            try:
                while self.num_timesteps < total_timesteps:
                    rollout = self.collect_rollouts(
                        self.env,
                        train_freq=self.train_freq,
                        action_noise=self.action_noise,
                        callback=callback,
                        learning_starts=self.learning_starts,
                        replay_buffer=self.replay_buffer,
                        log_interval=log_interval,
                    )

                    if not rollout.continue_training:
                        break

                    if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                        # If no `gradient_steps` is specified,
                        # do as many gradients steps as steps performed during the rollout
                        gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                        # Special case when the user passes `gradient_steps=0`
                        if gradient_steps > 0:
                            self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                    pbar.update(self.num_timesteps - pbar.n)

                callback.on_training_end()
                
            except KeyboardInterrupt:
                # print(f"Training interrupted by user, saving current model at {self.policy_save_path}")
                # self.save(self.policy_save_path)
                print("Training interrupted by user, stopping training.")
                
        return self
    
    # def train()
    def train_actor(self, log_interval = None):
        # assert self.H >= 1, "horizon must be greater than 1"
        batch_state = self.replay_buffer.sample(self.actor_batch_size)[-1]
        
        ent_coef_loss = None
        env = self.train_env
        env.reset(state=batch_state)
        env.detach()
        reward_loss, entropy_loss = 0., 0.
        # pre_active = th.ones((self.actor_batch_size,), device=self.device, dtype=th.bool)
        discount_factor = th.ones((env.num_envs,), dtype=th.float32, device=self.device)
        episode_done = th.zeros((env.num_envs,), device=self.device, dtype=th.bool)
        obs = env.get_observation()
        for inner_step in range(self.H):
            # dream a horizon of experience
            pre_obs = obs.clone()
            # iteration
            actions, entropy = self.policy.actor.action_and_entropy(pre_obs)
            # step
            obs, reward, done, info = env.step(actions)
            # done = th.zeros_like(done, dtype=th.bool)  # debug
            for i in range(len(episode_done)):
                episode_done[i] = info[i]["episode_done"]

            reward, done = reward.to(self.device), done.to(self.device)

            # compute the temporal difference
            with th.no_grad():
                next_actions = self.policy.actor(obs)
                next_actions = next_actions if not isinstance(next_actions, tuple) else next_actions[0]
                next_values, _ = th.cat(self.policy.critic_target(obs.detach(), next_actions.detach()), dim=-1).min(dim=-1)

            # compute the loss
            reward_loss = reward_loss - reward * discount_factor
            entropy_loss = entropy_loss - entropy * discount_factor * self.ent_coef
            done_but_not_episode_end = (done | (inner_step == self.H - 1)) & ~episode_done
            # done_but_not_episode_end = th.ones_like(done, dtype=th.bool)
            if done_but_not_episode_end.any() and self._end_value:
                reward_loss = reward_loss - \
                             next_values * discount_factor * self.gamma * done_but_not_episode_end

            discount_factor = discount_factor * self.gamma * ~done + done

            self.rollout_buffer.add(obs=CaD(pre_obs),
                                    reward=CaD(reward),
                                    action=CaD(actions),
                                    next_obs=CaD(obs),
                                    done=CaD(done),
                                    episode_done=CaD(episode_done),
                                    value=CaD(next_values),
                                    )

        # update
        actor_loss = (reward_loss+entropy_loss).mean()  # average of value and accumlative rewards
        self.policy.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=False)
        th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
        self.policy.actor.optimizer.step()
        # polyak_update(params=self.policy.actor.parameters(),
        #               target_params=self.policy.actor.parameters(), tau=self.tau)

        self.rollout_buffer.compute_returns()
        env.detach()

        # # update critic
        for i in range(self.gradient_steps):
            values, _ = th.cat(self.policy.critic(self.rollout_buffer.obs, self.rollout_buffer.action), dim=-1).min(dim=-1)
            target = self.rollout_buffer.returns
            critic_loss = th.nn.functional.mse_loss(target, values)
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
            self.policy.critic.optimizer.step()

            polyak_update(params=self.policy.critic.parameters(), target_params=self.policy.critic_target.parameters(), tau=self.tau)
            polyak_update(params=self.critic_batch_norm_stats, target_params=self.critic_batch_norm_stats_target, tau=1.)

        self.rollout_buffer.clear()

        self._logger.record("train/reward_loss", reward_loss.mean().item())
        self._logger.record("train/entropy_loss", entropy_loss.mean().item())
        self._logger.record("train/actor_loss", actor_loss.mean().item())
        self._logger.record("train/critic_loss", critic_loss.item() if isinstance(critic_loss, th.Tensor) else critic_loss)
        self.logger.record("train/ent_coef_loss", (ent_coef_loss.item() if isinstance(ent_coef_loss, th.Tensor) else ent_coef_loss))
