from pydantic import BaseModel
from typing import List


class StrategyGoalRelation(BaseModel):
    strategy: str
    has_goals: List[str]


class StrategyGoalRelations(BaseModel):
    strategy_goal_relations: List[StrategyGoalRelation]


class StrategyTrendRelation(BaseModel):
    strategy: str
    is_response_to: List[str]


class StrategyTrendRelations(BaseModel):
    strategy_trend_relations: List[StrategyTrendRelation]


class StrategyCapabilityRelation(BaseModel):
    strategy: str
    requires: List[str]


class StrategyCapabilityRelations(BaseModel):
    strategy_capability_relations: List[StrategyCapabilityRelation]
