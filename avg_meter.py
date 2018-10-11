# -*- coding: utf-8 -*-
# 🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐-🐐

from __future__ import division, print_function, generators, unicode_literals
from typing import *

import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from pathlib import Path


class AVGMeter(object):
    """
    EXAMPLE:
        a = AVGMeter(precition=6)
        a.append(2)
        a.append(10)
        a.append(3)
        print('count:\t', a.count)
            >> count:	 3
        print('sum:\t', a.sum)
            >> sum:	 15.0
        print('===>\t', a)
            >> ===>	 5.000000
        print('===>\t', a.avg_of_last(2))
            >> ===>	 6.5
    """

    def __init__(self, precition: int = 6, log_freq=None):
        self.precision = precition
        self.log_freq = log_freq
        self.reset()

    def append(self, x: float):
        self.values.append(x)
        self.sum += x
        self.count += 1

    def reset(self):
        self.sum = 0.0  # type: float
        self.count = 0  # type: int
        self.values = []  # type: List[float]

    def avg_of_last(self, last_size=2) -> float:
        x = np.mean(self.values[-last_size:])  # type: float
        return float(round(x, self.precision))

    @property
    def avg(self) -> float:
        return round(self.sum / self.count, self.precision)

    @property
    def avgl(self) -> float:
        assert self.log_freq is not None
        return self.avg_of_last(self.log_freq)

    @property
    def last(self) -> Optional[float]:
        if len(self.values) > 0:
            return self.values[-1]

    def save_plt(self, path: Path, hex_color: str = '#16a085', linewidth: int = 3):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.grid(color='#c1c6c9', linestyle='-.', linewidth=0.5)
        plt.plot(self.values, label='loss', linewidth=linewidth, color=hex_color)
        plt.xlim(0, len(self.values) - 1)
        plt.savefig(str(path))

    def __str__(self) -> str:
        return '{:.{p}f}'.format(self.avg, p=self.precision)


        # __repr__ = __str__
