import torch
import torch.nn.functional as F
import torch.distributions as torchd
import numpy as np
import torch.nn as nn

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


to_np = lambda x: x.detach().cpu().numpy()


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        probs = F.softmax(logits, dim=-1)
        if logits is not None and unimix_ratio > 0.0:
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape) #.detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    @property
    def mode(self):
        return symexp(self._mode)

    @property
    def mean(self):
        return self.mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        # distance = (self._mode - value) ** 2
        if len(value.shape) >= 4: # data is image
            merge_indices = list(range(len(distance.shape)))[-3:]
        else: # data is vector
            merge_indices = list(range(len(distance.shape)))[-1]
        if self._agg == "mean":
            loss = distance.mean(merge_indices)
        elif self._agg == "sum":
            loss = distance.sum(merge_indices)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    @property
    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)

class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    @property
    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.sample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)

class DiscDist:
    def __init__(
            self,
            logits,
            low=-20.0,
            high=20.0,
            transfwd=symlog,
            transbwd=symexp,
            device="cuda",
    ):
        wid_len = 255
        assert logits.shape[-1] == wid_len
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=wid_len, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / wid_len
        self.transfwd = transfwd
        self.transbwd = transbwd

    @property
    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    # @property
    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
                F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
                + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def rsample(self, sample_shape):
        event = super().rsample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event

class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    @property
    def mode(self):
        return self._mode

    @property
    def mean(self):
        return self.mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if len(value.shape) >= 4: # data is image
            merge_indices = list(range(len(distance.shape)))[-3:]  
        else: # data is vector
            merge_indices = list(range(len(distance.shape)))[-1]
        if self._agg == "mean":
            loss = distance.mean(merge_indices)
        elif self._agg == "sum":
            loss = distance.sum(merge_indices)
        else:
            raise NotImplementedError(self._agg)
        return -loss

    def rsample(self, sample_shape=()):
        return self._mode

    def sample(self, sample_shape=()):
        return self._mode


class EmptyNet(torch.nn.Module):
    def __init__(self):
        super(EmptyNet, self).__init__()

    def forward(self, x):
        return x


class TanhNormal(torchd.TransformedDistribution):
    """
    A distribution that samples from a Normal and applies a Tanh transformation.
    This wraps PyTorch's TransformedDistribution framework.
    """
    def __init__(self, mean, std, epsilon=1e-6):
        self._mean = mean
        self._std = std
        self.epsilon = epsilon

        base_dist = torchd.Normal(mean, std)
        transforms = [torchd.transforms.TanhTransform(cache_size=1)]
        super().__init__(base_dist, transforms)

    def log_prob(self, value):
        """
        Log probability of the transformed value. Clamp value to avoid numerical instability.
        """
        value_clamped = torch.clamp(value, -1 + self.epsilon, 1 - self.epsilon)
        return super().log_prob(value_clamped)

    @property
    def mean(self):
        """Return tanh-transformed mean of the base distribution."""
        return torch.tanh(self._mean)

    @property
    def mode(self):
        """Return tanh-transformed mode of the base distribution."""
        return torch.tanh(self._mean)

    @property
    def base_mean(self):
        """Return the original Gaussian mean."""
        return self._mean

    @property
    def base_std(self):
        """Return the original Gaussian std."""
        return self._std

    def entropy(self):
        """
        Approximate the entropy of the tanh-transformed distribution.
        Note: This is not the exact entropy.
        """
        h_gaussian = self.base_dist.entropy()
        cosh_mean = torch.cosh(self._mean)
        correction = -2 * (torch.log(cosh_mean) + 0.5 * self._std.pow(2) / cosh_mean.pow(2))
        return (h_gaussian + correction).mean()




# class TanhNormal:
#     """
#     A distribution that samples from a Normal distribution and applies a tanh transformation.
#     Useful in reinforcement learning when actions must be bounded within [-1, 1].
#
#     Args:
#         mean (Tensor): Mean of the Gaussian distribution.
#         std (Tensor): Standard deviation of the Gaussian distribution.
#         epsilon (float): Small value to avoid numerical instability (default: 1e-6).
#     """
#     def __init__(self, mean, std, epsilon=1e-6):
#         self.mean = mean
#         self.std = std
#         self.epsilon = epsilon
#         self.normal_dist = torchd.Normal(mean, std)
#
#     def sample(self, sample_shape=torch.Size()):
#         """
#         Sample from the Gaussian and apply tanh transformation.
#
#         Returns:
#             tanh_z (Tensor): Sample after tanh.
#             z (Tensor): Original Gaussian sample (for backprop).
#         """
#         z = self.normal_dist.sample(sample_shape)
#         tanh_z = torch.tanh(z)
#         return tanh_z
#
#     def rsample(self, sample_shape=torch.Size()):
#         """Alias for sample() method."""
#         z = self.normal_dist.rsample(sample_shape)  # Reparameterized sampling
#         tanh_z = torch.tanh(z)
#         return tanh_z
#
#     def log_prob(self, value):
#         value_clipped = torch.clamp(value, -1 + self.epsilon, 1 - self.epsilon)
#         z = torch.atanh(value_clipped)
#         log_prob_z = self.normal_dist.log_prob(z)
#         correction = torch.log(torch.clamp(1 - value_clipped.pow(2), min=self.epsilon))
#         return log_prob_z - correction
#
#     def entropy(self):
#         """
#         Approximate the entropy of the tanh-transformed distribution.
#         Note: This is a crude approximation and should not be relied upon in high-precision applications.
#         """
#         h_gaussian = self.normal_dist.entropy()
#         cosh_mean = torch.cosh(self.mean)
#         correction = -2 * (torch.log(cosh_mean) + 0.5 * self.std.pow(2) / cosh_mean.pow(2))
#         return (h_gaussian + correction).mean()
#
#     @property
#     def mode(self):
#         """Approximate mode: tanh of the Gaussian mean."""
#         return torch.tanh(self.mean)

def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f