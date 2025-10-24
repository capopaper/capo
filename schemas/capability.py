from pydantic import BaseModel
from typing import List
from enum import Enum


class CapabilityCategory(str, Enum):
    VERBINDEN = "verbinden"
    KIEZENENSTUREN = "kiezen_en_sturen"
    HANDELEN = "handelen"
    ONDERSTEUNEND = "ondersteunend"


class Capability(BaseModel):
    name: str
    definition: str
    category: CapabilityCategory


class Capabilities(BaseModel):
    capabilities: List[Capability]
