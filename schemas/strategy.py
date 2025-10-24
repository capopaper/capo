from pydantic import BaseModel
from typing import List
from enum import Enum


class StrategyCategory(str, Enum):
    POLICE_WORKFORCE = "police_workforce"
    OPERATIONS = "operations"
    SOCIAL = "social"
    COOPERATION = "cooperation"
    TECHNOLOGICAL = "technological"
    ECOLOGICAL = "ecological"
    LEGAL = "legal"
    GOVERNANCE = "governance"
    INFRASTRUCTURE = "infrastructure"


class StrategyScope(str, Enum):
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"


class StrategyStakeholders(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class StrategyWithProperties(BaseModel):
    name: str
    description: str
    has_goals: List[str]
    category: StrategyCategory
    scope: StrategyScope
    stakeholders: StrategyStakeholders
    named: bool


class Strategy(BaseModel):
    name: str


class Strategies(BaseModel):
    strategies: List[Strategy]


class StrategyGroup(BaseModel):
    strategy_names: List[str]


class GroupedStrategies(BaseModel):
    grouped_strategies: List[StrategyGroup]
