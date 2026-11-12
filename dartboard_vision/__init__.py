"""Reusable dartboard alignment and multiview contact detection.

Importing this package never opens cameras, reads recordings or writes outputs.
See the README for the saved-recording replay and calibration limitations.
"""

from .geometry import Board
from .masks import MaskSettings
from .session import DartboardSession, RoundState
from .tracking import PyramidTracker

__all__ = ["Board", "MaskSettings", "DartboardSession", "RoundState", "PyramidTracker"]
