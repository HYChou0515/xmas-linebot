from enum import Enum
from typing import Optional

from pydantic import BaseModel


class UseUpgradedModel(int, Enum):
    OLD_MODEL = 0
    NEW_MODEL = 1


class GamutMapping(int, Enum):
    # If the image is over-saturated, scaling is recommended
    SCALING = 1
    CLIPPING = 2


class WhiteBalanceSetting(BaseModel):
    use_upgraded_model: Optional[UseUpgradedModel]
    gamut_mapping: Optional[GamutMapping]
