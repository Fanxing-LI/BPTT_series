from algorithms.BPTT_series.BPTT import BPTT


class SHAC(BPTT):
    """
    SHAC (Self-Hybrid Actor-Critic) algorithm implementation.
    Inherits from BPTT (Backpropagation Through Time).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_end_value()

    def _set_name(self):
        self.name = "SHAC"
