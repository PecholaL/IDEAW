""" Test Only
"""

import sys
import torch

sys.path.append("..")
from models.attackLayer import AttackLayer

att = AttackLayer("config.yaml", "cpu")

# audio tensor batch
data = torch.rand([8, 18000])
