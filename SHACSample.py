from .BPTT import BPTT, CaD, CaDu
from .BPTTSample import BPTTSample
from stable_baselines3.common.off_policy_algorithm import SelfOffPolicyAlgorithm, MaybeCallback
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule, TrainFreq
import torch as th
from stable_baselines3.common.utils import polyak_update
from typing import Optional
from tqdm import tqdm


class SHACSample(BPTTSample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_end_value()

    def _set_name(self):
        self.name = "SHACSample"