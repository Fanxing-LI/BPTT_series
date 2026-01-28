import copy
import io
import pathlib
import time
import warnings
from collections import deque
from typing import Type, Optional, Dict, ClassVar, Any, Union, List, Iterable, Tuple

import os, sys
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from VisFly.utils.policies.td_policies import BasePolicy, MultiInputPolicy
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback, TrainFreq, TrainFrequencyUnit, RolloutReturn, GymEnv
from tqdm import tqdm
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name, safe_mean, should_collect_more_steps
from torch.nn import functional as F
from VisFly.utils.algorithms.lr_scheduler import transfer_schedule
from VisFly.utils.test.debug import get_network_statistics
from copy import deepcopy
from VisFly.utils.common import set_seed

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.sac.sac import SAC, SelfSAC
from .common import FullDictReplayBuffer, DictReplayBuffer, compute_td_returns, DataBuffer3, SimpleRolloutBuffer, RequiresGrad
from .policy import Policy as SimplePolicy


CaD = lambda x: x.clone().detach()
CaDu = lambda x: x.clone().detach().cpu()


# from stable_baselines3.common.buffers import DictReplayBuffer

class BPTT(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiInputPolicy": MultiInputPolicy,
        "SimplePolicy": SimplePolicy

    }
    observation_space: spaces.Space
    action_space: spaces.Space
    num_envs: int
    lr_schedule: Schedule

    def __init__(
            self,
            env,
            policy: Union[Type, str],
            train_env=None,
            policy_kwargs: Optional[Dict] = None,
            learning_rate: Union[float, Schedule] = 1e-3,
            comment: Optional[str] = None,
            save_path: Optional[str] = None,
            horizon: float = 1,
            tau: float = 0.005,
            gamma: float = 0.99,
            gradient_steps: int = 1,
            train_freq: Tuple[int, str] = (100, "step"),
            buffer_size: int = 1_000_000,
            batch_size: int = 256,
            pre_stop: float = 0.1,
            device: Optional[str] = "auto",
            seed: int = 42,
            ent_coef: Union[str, float] = 0.0001,
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: Optional[int] = None,
            tensorboard_log: Optional[str] = None,
            verbose: int = 1,
            replay_buffer_class: Optional[Type[DictReplayBuffer]] = FullDictReplayBuffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            actor_gradient_steps: Optional[int] = None,
            _init_setup_model: bool = True,
            scene_freq: Optional[TrainFreq] = None,
            buffer_reset: bool = False,
    ):
        root = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.save_path = f"{root}/saved" if save_path is None else save_path
        self.H = horizon
        self.tau = tau
        self.gamma = gamma
        self.comment = comment
        self.env = env
        self.env = train_env
        self.num_envs = env.num_envs

        self.learning_rate = learning_rate

        self.ent_coef = ent_coef

        self.pre_stop = pre_stop
        self._seed = seed
        self._set_seed()

        self.actor_gradient_steps = actor_gradient_steps
        self._actor_n_updates = 0

        stats_window_size = stats_window_size if stats_window_size else self.num_envs

        self.scene_freq = scene_freq
        if self.scene_freq and not isinstance(self.scene_freq, TrainFreq):
            Warning(f"scene_freq should be a TrainFreq, got {self.scene_freq}, converting to TrainFreq(1000000, TrainFrequencyUnit.STEP)")
            self.scene_freq = TrainFreq(self.scene_freq, TrainFrequencyUnit.STEP)

        self._end_value = False

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=transfer_schedule(learning_rate),
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log if tensorboard_log is not None else self.save_path,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            batch_size=batch_size,
            buffer_size=buffer_size,
            use_sde_at_warmup=use_sde_at_warmup,
            tau=tau,
            gamma=gamma,
            gradient_steps=gradient_steps,
            train_freq=train_freq,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            support_multi_env=True,
            sde_support=False,
        )
        if _init_setup_model:
            self._setup_model()

        if buffer_reset:
            self.enable_reset_from_buffer()

    def enable_reset_from_buffer(self):
        self.env.set_replay_buffer(self.replay_buffer)

    def _set_seed(self):
        set_seed(self._seed)

    def enable_end_value(self):
        self._end_value = True

    def _set_name(self):
        self.name = "BPTT"

    def _setup_model(self):
        super()._setup_model()

        self._set_name()

        self.env.reset()
        if self.env:
            self.env.reset()
            self.env.set_requires_grad(True)

        self.policy.critic_bp = self.policy.critic

        self.create_save_path()

        self.rollout_buffer = SimpleRolloutBuffer(gamma=self.gamma)

        # self.actor_batch_norm_stats_target = get_parameters_by_name(self.policy.actor, ["running_"])
        self.actor_batch_norm_stats = get_parameters_by_name(self.policy.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.policy.critic, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.policy.critic_target, ["running_"])

    def create_save_path(self, comment=None):
        self.tensorboard_log = self.save_path
        self.comment = self.comment if comment is None else comment
        self.comment = "std" if self.comment is None else self.comment
        index = 1
        path = f"{self.save_path}/{self.name}_{self.comment}_{index}"
        while os.path.exists(path):
            index += 1
            # path = f"{self.save_path}/{self.name}_{comment}_{index}" if comment is not None \
            #     else f"{self.save_path}/{self.name}_{index}"
            path = f"{self.save_path}/{self.name}_{self.comment}_{index}"
        self.policy_save_path = path if path.endswith(".zip") else f"{path}.zip"

    def train_actor(self, log_interval: Optional[int] = None) -> None:
        # assert self.H >= 1, "horizon must be greater than 1"
        ent_coef_loss = None
        self.env.detach()
        reward_loss, entropy_loss = 0., 0.
        discount_factor = th.ones((self.env.num_envs,), dtype=th.float32, device=self.device)
        episode_done = th.zeros((self.env.num_envs,), device=self.device, dtype=th.bool)
        pre_start = th.ones((self.env.num_envs,), device=self.device, dtype=th.bool)  # is this step the first step
        obs = self.env.get_observation()
        for inner_step in range(self.H):
            # dream a horizon of experience
            pre_obs = obs.clone()
            # iteration
            actions, entropy = self.policy.action_and_entropy(pre_obs, deterministic=False)
            # step
            obs, reward, done, info = self.env.step(actions)
            for i in range(len(episode_done)):
                episode_done[i] = info[i]["episode_done"]

            # self.train_num_timesteps += self.env.num_envs
            # if done.any():
            #     self._update_train_info_buffer(info, done)

            reward, done = reward.to(self.device), done.to(self.device)

            # compute the temporal difference
            with th.no_grad():
                next_actions = self.policy.predict(obs)
                next_actions = next_actions if not isinstance(next_actions, tuple) else next_actions[0]
                f = th.min if reward.mean() >= 0 else th.max
                # next_values, _ = th.cat(self.policy.critic_target(obs.detach(), next_actions.detach()), dim=-1).min(dim=-1)
                next_values, _ = f(th.cat(self.policy.critic_target(obs.detach(), next_actions.detach()), dim=-1), dim=-1)

            # compute the loss
            reward_loss = reward_loss - reward * discount_factor
            entropy_loss = entropy_loss - entropy * discount_factor * self.ent_coef
            done_but_not_episode_end = (done | (inner_step == self.H - 1)) & ~episode_done
            if done_but_not_episode_end.any() and self._end_value:
                reward_loss = reward_loss - \
                             next_values * discount_factor * self.gamma * done_but_not_episode_end

            discount_factor = discount_factor * self.gamma * ~done + done
            # pre_active = pre_active & ~done

            self.rollout_buffer.add(obs=CaD(pre_obs),
                                    reward=CaD(reward),
                                    action=CaD(actions),
                                    next_obs=CaD(obs),
                                    done=CaD(done),
                                    episode_done=CaD(episode_done),
                                    value=CaD(next_values),
                                    )
            self._store_transition(self.replay_buffer,
                                   buffer_action=CaDu(actions),
                                   new_obs=CaDu(obs),
                                   reward=CaDu(reward),
                                   dones=CaDu(done),
                                   infos=info,
                                   states=CaDu(self.env.full_state)
                                   )
            self._update_info_buffer(infos=info, dones=CaDu(done))
            self.check_whether_dump(log_interval=log_interval, dones=CaDu(done))

        # update
        actor_loss = (reward_loss+entropy_loss).mean()  # average of value and accumlative rewards
        self.policy.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=False)
        th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
        self.policy.actor.optimizer.step()
        # polyak_update(params=self.policy.actor.parameters(),
        #               target_params=self.policy.actor.parameters(), tau=self.tau)

        self.rollout_buffer.compute_returns()
        self.env.detach()

        # # update critic
        for i in range(self.gradient_steps):
            values = self.policy.critic(self.rollout_buffer.obs, self.rollout_buffer.action)
            # values, _ = th.cat(self.policy.critic(self.rollout_buffer.obs, self.rollout_buffer.action), dim=-1).min(dim=-1)
            target = self.rollout_buffer.returns
            critic_loss = 0
            for value in values:
                critic_loss += th.nn.functional.mse_loss(target, value[...,0])
            # critic_loss = th.nn.functional.mse_loss(target, values)
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

    def check_and_reset_scene(self) -> None:
        if not hasattr(self, "_pre_scene_fresh_step"):
            self._pre_scene_fresh_step = 0
        if self.scene_freq:
            if self.scene_freq.unit == TrainFrequencyUnit.EPISODE:
                if self._episode_num - self._pre_scene_fresh_step >= self.scene_freq.frequency:
                    print(f"Resetting scene at episode {self._episode_num}")
                    self.env.reset_env_by_id()
                    self._pre_scene_fresh_step = self._episode_num
            elif self.scene_freq.unit == TrainFrequencyUnit.STEP:
                if self.num_timesteps - self._pre_scene_fresh_step >= self.scene_freq.frequency:
                    print(f"Resetting scene at step {self.num_timesteps}")
                    self.env.reset_env_by_id()
                    self._pre_scene_fresh_step = self.num_timesteps



    def learn(
            self: SelfSAC,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: Optional[int] = None,
            tb_log_name: str = "Algorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfSAC:
        log_interval = log_interval if log_interval else self.n_envs
        # add tqdm
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
                    self.check_and_reset_scene()

                    self._update_current_progress_remaining(num_timesteps=self.num_timesteps, total_timesteps=total_timesteps)

                    self.train(
                        gradient_steps=self.gradient_steps,
                        batch_size=self.batch_size,
                        log_interval=log_interval,
                    )

                    # Update the progress bar
                    pbar.update(self.num_timesteps - pbar.n)
            except KeyboardInterrupt:
                # print(f"Training interrupted by user, saving current model at {self.policy_save_path}")
                # self.save(self.policy_save_path)
                print("Training interrupted by user, stopping training.")

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int = 64, log_interval=None) -> None:
        actor_gradient_steps = gradient_steps if self.actor_gradient_steps is None else self.actor_gradient_steps
        self.policy.set_training_mode(True)

        optimizers = [self.policy.actor.optimizer, self.policy.critic.optimizer]
        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        for j in range(actor_gradient_steps):
            self.train_actor(log_interval=log_interval)
            self.num_timesteps += self.num_envs * self.H
            pass

        self._actor_n_updates += self.actor_gradient_steps
        self._n_updates += self.gradient_steps
        # self.num_timesteps += self.num_envs * self.H

        self.logger.record("train/actor_n_updates", self._actor_n_updates)
        self.logger.record("train/n_updates", self._n_updates)

        self.policy.set_training_mode(False)

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase] = None,
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        path = self.policy_save_path if path is None else path
        print(f"Saving model to {path}")
        super().save(
            path,
            exclude=exclude,
            include=include,
        )

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.policy.eval()
        action = self.policy.predict(observation, deterministic=deterministic)

        return action

    def check_whether_dump(self, dones, log_interval: Optional[int] = None) -> None:
        for idx, done in enumerate(dones):
            if done:
                # Update stats
                # num_collected_episodes += 1
                self._episode_num += 1

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

            if len(self.ep_info_buffer[0]["extra"]) >= 0:
                for key in self.ep_info_buffer[0]["extra"].keys():
                    self.logger.record(
                        f"rollout/ep_{key}_mean",
                        safe_mean(
                            [ep_info["extra"][key] for ep_info in self.ep_info_buffer]
                        ),
                    )
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/ep_success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _sample_action(
            self,
            learning_starts: int,
            action_noise: Optional[ActionNoise] = None,
            n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:

        action = self.predict(self._last_obs, deterministic=False)
        action = action.cpu().numpy() if not isinstance(action, tuple) else action[0]
        buffer_action = copy.deepcopy(action)
        return action, buffer_action

    def load_parameters(self, path):
        data, params, pytorch_variables = load_from_zip_file(
            path,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"
        self.set_parameters(params, exact_match=True)


    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        # if print_system_info:
        #     print("== CURRENT SYSTEM INFO ==")
        #     get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            # check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if ("pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e)) :
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            elif  "frozen_part" in str(e):
                warnings.warn(
                    "You are probably loading a StagePolicy model with frozen part"
                )
                model.set_parameters(params, exact_match=False, device=device)
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    @classmethod
    def load_policy(
            self,
            path,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            print_system_info: bool = False,
    ):
        data, params, pytorch_variables = load_from_zip_file(
            path,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        policy = data["policy_class"](
            data["observation_space"],
            data["action_space"],
            data["lr_schedule"],
            **data["policy_kwargs"])
        # policy.set_parameters(params, exact_match=True, device=device)
        state_dict = params.get("policy")
        policy.load_state_dict(state_dict, strict=False)
        return policy

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # actions = np.random.rand(*actions.shape).astype(dtype=actions.dtype)# debug
            # buffer_actions = actions # debug
            # Rescale and perform action
            states = env.full_state.clone().detach()
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            # new_stoch, new_deter = self.world_step(new_obs, actions, new_obs["stoch"], new_obs["deter"], deterministic=False)
            # self.env.set_latent(deter=new_deter.detach(), stoch=new_stoch.detach())
            # new_obs["stoch"], new_obs["deter"] = new_stoch.detach(), new_deter.detach()

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos, states)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition(
            self,
            replay_buffer: ReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
            states: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            states,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_