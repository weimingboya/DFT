from .utils import *
from .typing import *

def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, Sequence):
        b_s = x[0].size(0)
    else:
        b_s = x.size(0)
    return b_s


def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, Sequence):
        device = x[0].device
    else:
        device = x.device
    return device
