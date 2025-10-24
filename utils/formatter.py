from typing import Any, Optional, List, Union, Dict
from schemas.strategy import Strategies, GroupedStrategies
from schemas.goal import Goals, GroupedGoals
from schemas.trend import Trends, GroupedTrends
from schemas.ontology import (
    StrategyGoalRelations,
    StrategyTrendRelations,
    StrategyCapabilityRelations,
)


def build_document_annotation(
    document_id: Union[str, int],
    strategies: Optional[Strategies] = None,
    goals: Optional[Goals] = None,
    trends: Optional[Trends] = None,
    grouped_strategies: Optional[GroupedStrategies] = None,
    grouped_goals: Optional[GroupedGoals] = None,
    grouped_trends: Optional[GroupedTrends] = None,
    strategy_goal_relations: Optional[StrategyGoalRelations] = None,
    strategy_trend_relations: Optional[StrategyTrendRelations] = None,
    strategy_capability_relations: Optional[StrategyCapabilityRelations] = None,
) -> dict:

    strategy_list = [s.name for s in (strategies.strategies if strategies else [])]
    goal_list = [g.name for g in (goals.goals if goals else [])]
    trend_list = [t.name for t in (trends.trends if trends else [])]

    grouped_strat = [
        g.strategy_names
        for g in (grouped_strategies.grouped_strategies if grouped_strategies else [])
    ]
    grouped_goal = [
        g.goal_names for g in (grouped_goals.grouped_goals if grouped_goals else [])
    ]
    grouped_trend = [
        g.trend_names for g in (grouped_trends.grouped_trends if grouped_trends else [])
    ]

    # Gather all strategy names seen anywhere:
    names_from_groups = [name for grp in grouped_strat for name in grp]
    names_from_goal_models = [
        name for g in (goals.goals if goals else []) for name in g.strategies
    ]
    names_from_goal_rels = [
        rel.strategy
        for rel in (
            strategy_goal_relations.strategy_goal_relations
            if strategy_goal_relations
            else []
        )
    ]
    names_from_trend_rels = [
        rel.strategy
        for rel in (
            strategy_trend_relations.strategy_trend_relations
            if strategy_trend_relations
            else []
        )
    ]
    names_from_capability_rels = [
        rel.strategy
        for rel in (
            strategy_capability_relations.strategy_capability_relations
            if strategy_capability_relations
            else []
        )
    ]

    all_strategy_names = sorted(
        set(
            strategy_list
            + names_from_groups
            + names_from_goal_models
            + names_from_goal_rels
            + names_from_trend_rels
            + names_from_capability_rels
        )
    )

    def _empty_rel() -> Dict[str, List[str]]:
        return {"has_goal": [], "is_response_to": [], "requires": []}

    rel_map: Dict[str, Dict[str, List[str]]] = {
        s: _empty_rel() for s in all_strategy_names
    }

    # From explicit StrategyGoalRelations
    if strategy_goal_relations:
        for rel in strategy_goal_relations.strategy_goal_relations:
            rel_map.setdefault(rel.strategy, _empty_rel())["has_goal"].extend(
                rel.has_goals
            )

    # Derive from Goals: goal -> strategies
    if goals:
        for g in goals.goals:
            for sname in g.strategies:
                rel_map.setdefault(sname, _empty_rel())["has_goal"].append(g.name)

    # From StrategyTrendRelations
    if strategy_trend_relations:
        for rel in strategy_trend_relations.strategy_trend_relations:
            rel_map.setdefault(rel.strategy, _empty_rel())["is_response_to"].extend(
                rel.is_response_to
            )

    # From StrategyCapabilityRelations
    if strategy_capability_relations:
        for rel in strategy_capability_relations.strategy_capability_relations:
            rel_map.setdefault(rel.strategy, _empty_rel())["requires"].extend(
                rel.requires
            )

    # de-duplicate while preserving order
    def _dedup(seq: List[str]) -> List[str]:
        return list(dict.fromkeys(seq))

    strategy_relations_out = [
        {
            "strategy": s,
            "has_goal": _dedup(rel_map[s]["has_goal"]),
            "is_response_to": _dedup(rel_map[s]["is_response_to"]),
            "requires": _dedup(rel_map[s]["requires"]),
        }
        for s in all_strategy_names
    ]

    return {
        "document_id": document_id,
        "strategies": all_strategy_names,
        "goals": goal_list,
        "trends": trend_list,
        "grouped_strategies": grouped_strat,
        "grouped_goals": grouped_goal,
        "grouped_trends": grouped_trend,
        "strategy_relations": strategy_relations_out,
    }


def append_document_annotation(
    *,
    bundle: Optional[Dict[str, Any]],
    annotator_id: str,
    document_annotation: Dict[str, Any],
    deduplicate_by_document_id: bool = True,
    replace_if_exists: bool = True,
) -> Dict[str, Any]:
    if not bundle:
        return {
            "annotator_id": annotator_id,
            "documents_annotated": [document_annotation],
        }

    if "annotator_id" in bundle and bundle["annotator_id"] != annotator_id:
        raise ValueError(
            f"annotator_id mismatch: bundle has '{bundle['annotator_id']}', got '{annotator_id}'."
        )

    docs = bundle.setdefault("documents_annotated", [])
    if not deduplicate_by_document_id:
        docs.append(document_annotation)
        return bundle

    new_id = document_annotation.get("document_id")
    if new_id is None:
        raise ValueError("document_annotation must include 'document_id'")

    idx = next((i for i, d in enumerate(docs) if d.get("document_id") == new_id), None)
    if idx is None:
        docs.append(document_annotation)
    elif replace_if_exists:
        docs[idx] = document_annotation

    return bundle
