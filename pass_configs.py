from dataclasses import dataclass
from typing import Literal

@dataclass
class pass_configs:
    """The configs used in a forward or backwards pass of our transformer"""

    train: bool=False
    pad_token_ind: int=0
    pass_num: int=0
    seq_len: int=0

    forward_pass_mode: Literal['prefill', 'decode'] = None