import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Smac_interface import SmacInterface


class SelectorInterface(SmacInterface):
    """Using Smac interface."""
