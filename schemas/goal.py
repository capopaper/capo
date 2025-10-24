from pydantic import BaseModel
from typing import List


class Goal(BaseModel):
    name: str
    strategies: List[str]


class Goals(BaseModel):
    goals: List[Goal]


class GoalGroup(BaseModel):
    goal_names: List[str]


class GroupedGoals(BaseModel):
    grouped_goals: List[GoalGroup]
