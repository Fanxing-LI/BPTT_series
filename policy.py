import copy

import torch.nn as nn
import torch as th
from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from VisFly.utils.policies.extractors import create_mlp, load_extractor_class
from VisFly.utils.policies.td_policies import obs_as_tensor
from . import tools
# CAP the standard deviation of the actor
import torch.distributions as torchd
from VisFly.utils.type import TensorDict
LOG_STD_MAX = 2
LOG_STD_MIN = -10
MAX_STD = 0.3
MIN_STD = 0.1


def scale_std(std):
    return (MAX_STD - MIN_STD) * th.sigmoid(
        std + 2.0
    ) + MIN_STD


class BaseModel(nn.Module):
    dist_alias = {
        "normal": th.distributions.Normal,
        "mse": tools.MSEDist,
        "bernoulli": tools.Bernoulli,
        "symlog_disc": tools.DiscDist,
        "symlog_mse": tools.SymlogDist,
        "tanh_normal": tools.TanhNormal,
        "trunc_normal": tools.SafeTruncatedNormal
    }
    optimizer_alias = {
        "adam": th.optim.Adam,
        "adamW": th.optim.AdamW,
        "rmsprop": th.optim.RMSprop,
        "sgd": th.optim.SGD
    }
    scheduler_alias = {
        "linear": th.optim.lr_scheduler.LinearLR,
        "step": th.optim.lr_scheduler.StepLR,
        "cosine": th.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_restart": th.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }

    def __init__(
            self,
    ):
        super().__init__()
        self._device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")

    def _build(self, **kwargs):
        raise NotImplementedError

    def _create_optimizer(self, optimizer_class, optimizer_kwargs, lr_scheduler_class, lr_scheduler_kwargs):
        self._scaler = th.cuda.amp.GradScaler(enabled=False)
        self.optimizer = self.optimizer_alias[optimizer_class](self.parameters(), **optimizer_kwargs)
        if lr_scheduler_class:
            self.scheduler = self.scheduler_alias[lr_scheduler_class](self.optimizer, **lr_scheduler_kwargs)

    def optimize(self, loss, clip_value=100., retain_graph=True):
        self.optimizer.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(optimizer=self.optimizer)
        n = th.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
        self._scaler.step(self.optimizer)
        self._scaler.update()
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        return {"grad_norm": n}
        # self.optimizer.zero_grad()
        # loss.backward(retain_graph=retain_graph)
        # th.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
        # self.optimizer.step()

    @property
    def device(self):
        return self._device


class LatentCombineExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Dict,
                 net_arch: Optional[Dict] = {},
                 activation_fn: Type[nn.Module] = nn.ReLU, ):
        super(LatentCombineExtractor, self).__init__()
        _features_dim = 0
        for key in observation_space.keys():
            _features_dim += observation_space[key].shape[0]
        self._features_dim = _features_dim

    def _build(self, observation_space, net_arch, activation_fn):
        pass

    def extract(self, observations) -> th.Tensor:
        if isinstance(observations, dict):
            if len(observations["stoch"].shape) > len(observations["deter"].shape):
                observations["stoch"] = observations["stoch"].reshape(*observations["stoch"].shape[:-2], -1)
            return th.cat([observations["stoch"], observations["deter"]], dim=-1)
        elif isinstance(observations, th.Tensor):
            return observations
        else:
            raise NotImplementedError

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, x):
        return self.extract(x)


class ContinuousCritic(BaseModel):
    def __init__(
            self,
            action_space: spaces.Space,
            net_arch: dict,
            features_extractor: LatentCombineExtractor,
            activation_fn: Type[nn.Module] = nn.ReLU,
            n_critics: int = 2,
            dist: str = "mse",
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            lr_scheduler_class: Optional[Type[th.optim.lr_scheduler.LRScheduler]] = None,
            lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self._dist_type = dist
        self.q_networks = []
        self.features_extractor = features_extractor
        features_dim = self.features_extractor.features_dim
        for idx in range(n_critics):
            net, _ = create_mlp(input_dim=features_dim + action_space.shape[0],
                                output_dim=1 if self._dist_type != "symlog_disc" else 255,
                                activation_fn=activation_fn,
                                **net_arch,
                                )
            q_net = nn.Sequential(net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)
        self._create_optimizer(optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs, lr_scheduler_class=lr_scheduler_class, lr_scheduler_kwargs=lr_scheduler_kwargs)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        dists = self.get_dist(obs, actions)
        return tuple(dist.mean for dist in dists)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features, h = self.extract_features(obs, self.features_extractor)
        return self.dist(self.q_networks[0](features)).mean

    def get_dist(self, obs, actions):
        obs = obs_as_tensor(obs, device=self.device)
        features = self.features_extractor(obs)
        qvalue_input = th.cat([features, actions], dim=-1)
        dists = tuple(self.dist(q_net(qvalue_input)) for q_net in self.q_networks)
        return dists

    def dist(self, mean):
        dist_type = self._dist_type
        if dist_type == "mse":
            mean = mean
            dist = tools.MSEDist(mean, agg="sum")
        elif dist_type == "symlog_mse":
            mean = mean
            dist = tools.SymlogDist(mean, agg="sum")
        elif dist_type == "symlog_disc":
            mean = mean
            dist = tools.DiscDist(logits=mean, device=self.device)
        else:
            raise NotImplementedError

        return dist

    @property
    def device(self):
        return next(self.parameters()).device


class Actor(BaseModel):
    def __init__(
            self,
            action_space: spaces.Space,
            net_arch: dict,
            features_extractor: LatentCombineExtractor,
            activation_fn: Type[nn.Module] = nn.ReLU,
            # squash_output: bool = False,
            dist: str = "normal",
            log_std_init: float = -3,
            optimizer_class: Type[th.optim.Optimizer] = "adam",
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            lr_scheduler_class: Optional[Type[th.optim.lr_scheduler.LRScheduler]] = None,
            lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._dist_type = dist
        self.features_extractor = features_extractor

        self.mu, output_dim = create_mlp(input_dim=self.features_extractor.features_dim,
                                         activation_fn=activation_fn,
                                         **net_arch,
                                         )
        self.sigma = copy.deepcopy(self.mu)
        # self.range = (action_space.low, action_space.high)
        self._mean_layer = nn.Linear(output_dim, action_space.shape[0])
        self._std_layer = copy.deepcopy(self._mean_layer)
        self.dist = self.dist_alias[dist]
        self._create_optimizer(optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs, lr_scheduler_class=lr_scheduler_class, lr_scheduler_kwargs=lr_scheduler_kwargs)

        self._low = th.tensor(action_space.low, dtype=th.float32)
        self._high = th.tensor(action_space.high, dtype=th.float32)
        # self.squash_fn = squash_output if not squash_output else nn.Tanh

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # if self.squash_fn:
        #     return self.squash_fn(self.get_dist(obs).rsample())
        # else:
        #     return self.get_dist(obs)
        return self.get_dist(obs).rsample().clamp(min=self._low, max=self._high)

    def action_and_entropy(self, obs: th.Tensor, deterministic: bool=False) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns the action and the entropy of the distribution.
        """
        dist = self.get_dist(obs)
        if deterministic:
            return dist.mean.clamp(min=self._low, max=self._high), dist.entropy()
        else:
            return dist.rsample().clamp(min=self._low, max=self._high), dist.entropy()

    def get_dist(self, obs):
        obs = obs_as_tensor(obs, device=self.device)
        features = self.features_extractor(obs)
        mu = self.mu(features)
        # sigma = self.sigma(features)
        mean = self._mean_layer(mu)
        std = self._std_layer(mu)
        std = scale_std(std)
        # std = std.exp()
        if self._dist_type == "trunc_normal":
            return torchd.independent.Independent(self.dist(mean, std, self._low, self._high), 1)
        else:
            return torchd.independent.Independent(self.dist(mean, std), 1)

    @th.no_grad()
    def predict(self, obs, deterministic=False):
        if deterministic:
            return self.get_dist(obs).mean.clamp(min=self._low, max=self._high)
        else:
            return self.get_dist(obs).sample().clamp(min=self._low, max=self._high)

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._low = self._low.to(*args, **kwargs)
        self._high = self._high.to(*args, **kwargs)


class Policy(nn.Module):
    act_alias = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    optim_alias = {
        "adam": th.optim.Adam,
        "rmsprop": th.optim.RMSprop,
        "sgd": th.optim.SGD,
        "adagrad": th.optim.Adagrad,
        "adamW": th.optim.AdamW,
        "adamax": th.optim.Adamax,
        "asgd": th.optim.ASGD,
        "lbfgs": th.optim.LBFGS,
        "rprop": th.optim.Rprop,
    }

    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule: Optional[List[float]] = None,
            features_extractor_class: Optional[Union[Type[LatentCombineExtractor], nn.Module]] = "FlexibleExtractor",
            features_extractor_kwargs: Optional[Dict[str, Any]] = {},
            activation_fn: Type[nn.Module] = "relu",
            actor: dict = {},
            critic: dict = {},
            share_features_extractor: bool = False,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            use_sde: bool = False
    ):
        # optimizer_class = self.optim_alias[optimizer_class]
        activation_fn = self.act_alias[activation_fn]
        # super(self).__init__()
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        features_extractor_class = load_extractor_class(features_extractor_class)
        self.features_extractor = features_extractor_class(observation_space, activation_fn=activation_fn, **features_extractor_kwargs)

        actor.update({"optimizer_class": optimizer_class, "optimizer_kwargs": optimizer_kwargs})
        critic.update({"optimizer_class": optimizer_class, "optimizer_kwargs": optimizer_kwargs})

        self.actor = Actor(
            action_space=action_space,
            features_extractor=self.features_extractor,
            activation_fn=activation_fn,
            **actor
        )
        self.critic = ContinuousCritic(
            action_space=action_space,
            features_extractor=self.features_extractor if not share_features_extractor else copy.deepcopy(self.features_extractor),
            activation_fn=activation_fn,
            **critic
        )

        self.critic_target = copy.deepcopy(self.critic)

    def forward(self, obs):
        return self.actor(obs)

    def evaluate(self, obs):
        return self.critic(obs)

    @th.no_grad()
    def predict(self, obs: Dict[str, th.Tensor], deterministic=False, sample=False) -> Tuple[th.Tensor, th.Tensor]:
        if not sample:
            return self.actor.predict(obs, deterministic=deterministic)
        else:
            num_sample = 5
            obs_batch = TensorDict({
                k: v.unsqueeze(0).repeat(num_sample+1, *([1] * (v.dim()))) for k, v in obs.items()
            })
            action_mean = self.actor.predict(obs, deterministic=True)
            action_sample = self.actor.predict(obs_batch[:-1], deterministic=False)
            action_sample = (self.actor.predict(obs_batch[:-1], deterministic=True) \
                            + th.randn_like(action_sample) * 0.05).clamp(-1,1)
            # + (th.rand_like(action_sample)-0.5) * 0.2).clamp(-1, 1)
            actions = th.cat([action_mean.unsqueeze(0), action_sample], dim=0)
            q = self.critic(obs_batch, actions)[0].squeeze()
            max_q, max_q_indices = th.max(q, dim=0)
            # print(q.std(), q.mean())
            # if (max_q - q[0]).abs() <= 0.01:
            #     max_q_indices = th.tensor([0])
            # max_q_indices = th.where(
            #     (max_q - q[0]).abs() <= 0.04,
            #     th.zeros_like(max_q_indices, device=max_q_indices.device),
            #     max_q_indices
            #     )
            mask = th.atleast_1d((max_q - q[0]).abs() <= 0.03)
            max_q_indices = th.where(mask, th.zeros_like(max_q_indices), max_q_indices)
            selected_actions = actions[max_q_indices.squeeze(),th.arange(len(max_q_indices), device=max_q_indices.device),]
            return selected_actions

    def action_and_entropy(self, obs, deterministic=False):
        return self.actor.action_and_entropy(obs, deterministic=deterministic)

    def set_training_mode(self, mode: bool = True):
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(mode)

    # for compatibility
    def scale_action(self, action):
        return action

    def unscale_action(self, action):
        return action

    def to(self, *args, **kwargs):
        # super().to(*args, **kwargs)
        self.actor.to(*args, **kwargs)
        self.critic.to(*args, **kwargs)
        self.critic_target.to(*args, **kwargs)
        return self
