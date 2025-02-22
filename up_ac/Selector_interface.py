from Smac_interface import SmacInterface
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SelectorInterface(SmacInterface):
    """Using Smac interface."""
