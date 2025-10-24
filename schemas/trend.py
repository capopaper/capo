from pydantic import BaseModel
from typing import List


class Trend(BaseModel):
    name: str


class Trends(BaseModel):
    trends: List[Trend]
    

class TrendGroup(BaseModel):
    trend_names: List[str]


class GroupedTrends(BaseModel):
    grouped_trends: List[TrendGroup]

