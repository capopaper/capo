import os
import json
from functools import cached_property
from typing import Any, Iterable, Literal, Optional, Protocol, Union, runtime_checkable
from openai import AzureOpenAI
from ollama import Client as OllamaClient
from schemas.strategy import Strategies, GroupedStrategies
from schemas.goal import Goals, GroupedGoals
from schemas.trend import Trends, GroupedTrends
from schemas.capability import Capabilities
from schemas.ontology import StrategyGoalRelations, StrategyTrendRelations, StrategyCapabilityRelations
from prompts import (
    STRATEGY_SIMPLE_SYSTEM_PROMPT,
    GOAL_SYSTEM_PROMPT,
    TREND_SYSTEM_PROMPT,
    GROUP_STRATEGIES_SYSTEM_PROMPT,
    GROUP_GOALS_SYSTEM_PROMPT,
    GROUP_TRENDS_SYSTEM_PROMPT,
    LINK_STRATEGIES_AND_GOALS_SYSTEM_PROMPT,
    LINK_STRATEGIES_AND_TRENDS_SYSTEM_PROMPT,
    LINK_STRATEGIES_AND_CAPABILITIES_SYSTEM_PROMPT
)
from annotation_instructions import ANNOTATION_INSTRUCTIONS_SYSTEM_PROMPT

Provider = Literal["azure", "ollama"]
TIMEOUT = 180


@runtime_checkable
class HasName(Protocol):
    name: str


class LLMExtractor:

    def __init__(
        self,
        *,
        model: str,
        provider: Provider = "azure",
        ollama_host: Optional[str] = None,
        ollama_keep_alive: Optional[str] = None,
    ):

        self.model = model
        self.provider = provider

        self._ollama_url = ollama_host or os.getenv("OLLAMA_ENDPOINT")
        self._ollama_keep_alive = ollama_keep_alive or os.getenv("OLLAMA_KEEP_ALIVE")

        self._azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self._azure_api_version = os.getenv("AZURE_API_VERSION")
        
        self.input_tokens_total: int = 0
        self.output_tokens_total: int = 0


    @cached_property
    def azure_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            azure_endpoint=self._azure_endpoint,
            api_key=self._azure_key,
            api_version=self._azure_api_version,
        )

    @cached_property
    def ollama_client(self) -> OllamaClient:
        return OllamaClient(host=self._ollama_url, timeout=TIMEOUT)
    
    
    def _record_tokens(self, response) -> tuple[int, int]:
        in_tok = 0
        out_tok = 0
        try:
            # OpenAI/Azure
            usage = getattr(response, "usage", None)
            if usage is not None:
                if hasattr(usage, "get"):
                    prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
                    completion = usage.get("completion_tokens") or usage.get("output_tokens")
                else:
                    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                    completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
                in_tok = int(prompt or 0)
                out_tok = int(completion or 0)

            else:
                # Ollama
                if isinstance(response, dict):
                    in_tok = int(response.get("prompt_eval_count") or 0)
                    out_tok = int(response.get("eval_count") or 0)
                else:
                    in_tok = int(getattr(response, "prompt_eval_count", 0) or 0)
                    out_tok = int(getattr(response, "eval_count", 0) or 0)

        except Exception:
            # Swallow token parsing issues; don't break the pipeline
            pass

        self.input_tokens_total += in_tok
        self.output_tokens_total += out_tok
        return in_tok, out_tok


    def _build_messages(
        self,
        complete_annotation_instructions: str,
        specific_task_instructions: str,
        document: str,
    ):
        return [
            {
                "role": "system",
                "content": complete_annotation_instructions,
            },
            {"role": "user", "content": specific_task_instructions},
            {"role": "user", "content": document},
        ]

    def _parse_ollama_response(self, ollama_response):
        response_content = getattr(ollama_response, "message", None)
        raw = ""
        if response_content is not None and hasattr(response_content, "content"):
            raw = (response_content.content or "").strip()
        else:
            # Fallback if response is dict-like
            raw = (ollama_response.get("message", {}).get("content", "")).strip()

        # Guard against accidental code fences if a template changes upstream
        if raw.startswith("```"):
            raw = raw.strip("` \n")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        return raw

    def _serialize_names(self, items: Iterable[HasName]) -> str:
        names = [item.name for item in items]
        return json.dumps(names, ensure_ascii=False, separators=(",", ":"))

    def _concept_items(self, concepts: Any) -> Iterable[HasName]:
        for attr in ("strategies", "goals", "trends", "capabilities"):
            if hasattr(concepts, attr):
                return getattr(concepts, attr)
        raise TypeError(
            f"Unsupported concepts container: {type(concepts)!r}. "
            "Expected an object with one of: .strategies, .goals, .trends, .capabilities"
        )

    def _concept_label(self, concepts: Any) -> str:
        if hasattr(concepts, "strategies"):
            return "strategies"
        if hasattr(concepts, "goals"):
            return "goals"
        if hasattr(concepts, "trends"):
            return "trends"
        if hasattr(concepts, "capabilities"):
            return "capabilities"
        return "concepts"

    def _build_grouping_messages(
        self,
        document: str,
        group_system_prompt,
        concepts_to_group: Union[Strategies, Goals, Trends],
    ):
        messages = self._build_messages(
            ANNOTATION_INSTRUCTIONS_SYSTEM_PROMPT, group_system_prompt, document
        )

        items = self._concept_items(concepts_to_group)
        label = self._concept_label(concepts_to_group)
        messages.insert(
            2,
            {
                "role": "user",
                "content": (
                    f"{label.capitalize()} to group (JSON array of names):\n"
                    f"{self._serialize_names(items)}\n\n"
                ),
            },
        )
        return messages
    
    
    def _format_items_for_prompt(self, container: Any, items: Iterable[Any]) -> str:
        if isinstance(container, Capabilities):
            payload = [cap.model_dump(mode="json", by_alias=True, exclude_none=True) for cap in items]
            return json.dumps(payload, indent=2, ensure_ascii=False)
        else:
            names = [getattr(x, "name", str(x)) for x in items]
            return "\n".join(f"- {n}" for n in names)


    
    def _build_linking_messages(
        self,
        document: str,
        link_system_prompt: str,
        concept_to_link_1: Union[Strategies, Goals, Trends, Capabilities],
        concept_to_link_2: Union[Strategies, Goals, Trends, Capabilities],
    ):
        messages = self._build_messages(
            ANNOTATION_INSTRUCTIONS_SYSTEM_PROMPT, link_system_prompt, document
        )

        items_to_link_1 = self._concept_items(concept_to_link_1)
        label_1 = self._concept_label(concept_to_link_1)

        items_to_link_2 = self._concept_items(concept_to_link_2)
        label_2 = self._concept_label(concept_to_link_2)

        items_str_1 = self._format_items_for_prompt(concept_to_link_1, items_to_link_1)
        items_str_2 = self._format_items_for_prompt(concept_to_link_2, items_to_link_2)

        messages.insert(
            2,
            {
                "role": "user",
                "content": (
                    f"{label_1.capitalize()}:\n{items_str_1}\n\n"
                    f"{label_2.capitalize()}:\n{items_str_2}\n\n"
                ),
            },
        )
        return messages


    def extract_strategies(self, document: str, temperature: float) -> Strategies:
        if self.provider == "azure":
            response = self.azure_client.responses.parse(
                model=self.model,
                input=self._build_messages(
                    ANNOTATION_INSTRUCTIONS_SYSTEM_PROMPT,
                    STRATEGY_SIMPLE_SYSTEM_PROMPT,
                    document,
                ),
                temperature=temperature,
                text_format=Strategies,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=self._build_messages(
                    ANNOTATION_INSTRUCTIONS_SYSTEM_PROMPT,
                    STRATEGY_SIMPLE_SYSTEM_PROMPT,
                    document,
                ),
                format=Strategies.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return Strategies.model_validate(json.loads(parsed_ollama_response))
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_goal_messages(
        self,
        document: str,
        strategies: Strategies,
    ):
        messages = self._build_messages(
            ANNOTATION_INSTRUCTIONS_SYSTEM_PROMPT, GOAL_SYSTEM_PROMPT, document
        )
        messages.insert(
            2,
            {
                "role": "user",
                "content": (
                    "Already extracted strategies "
                    "(these are already extracted as strategies, so they cannot be goals):\n"
                    f"{self._serialize_names(strategies.strategies)}"
                ),
            },
        )

        return messages

    def extract_goals(
        self,
        document: str,
        strategies: Strategies,
        temperature: float = 0.2,
    ) -> Goals:
        if self.provider == "azure":
            messages = self._build_goal_messages(document, strategies)
            response = self.azure_client.responses.parse(
                model=self.model,
                input=messages,
                temperature=temperature,
                text_format=Goals,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=self._build_goal_messages(document, strategies),
                format=Goals.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return Goals.model_validate(json.loads(parsed_ollama_response))
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_trend_messages(
        self,
        document: str,
        strategies: Strategies,
        goals: Goals,
    ):
        messages = self._build_messages(
            ANNOTATION_INSTRUCTIONS_SYSTEM_PROMPT, TREND_SYSTEM_PROMPT, document
        )
        messages.insert(
            2,
            {
                "role": "user",
                "content": (
                    "Already extracted strategies "
                    "(these are already marked as strategies, so they cannot be trends):\n"
                    f"{self._serialize_names(strategies.strategies)}"
                    "\n\nAlready extracted goals "
                    "(these are already marked as goals, so they cannot be trends):\n"
                    f"{self._serialize_names(goals.goals)}"
                ),
            },
        )

        return messages

    def extract_trends(
        self,
        document: str,
        strategies: Strategies,
        goals: Goals,
        temperature: float = 0.2,
    ) -> Trends:
        if self.provider == "azure":
            messages = self._build_trend_messages(
                document, strategies=strategies, goals=goals
            )
            response = self.azure_client.responses.parse(
                model=self.model,
                input=messages,
                temperature=temperature,
                text_format=Trends,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=self._build_trend_messages(document, strategies, goals),
                format=Trends.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return Trends.model_validate(json.loads(parsed_ollama_response))
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def group_strategies(
        self,
        document: str,
        strategies: Strategies,
        temperature: float = 0.2,
    ) -> GroupedStrategies:
        messages = self._build_grouping_messages(
            document, GROUP_STRATEGIES_SYSTEM_PROMPT, strategies
        )
        if self.provider == "azure":
            response = self.azure_client.responses.parse(
                model=self.model,
                input=messages,
                temperature=temperature,
                text_format=GroupedStrategies,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                format=GroupedStrategies.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return GroupedStrategies.model_validate(
                    json.loads(parsed_ollama_response)
                )
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def group_goals(
        self,
        document: str,
        goals: Goals,
        temperature: float = 0.2,
    ) -> GroupedGoals:
        messages = self._build_grouping_messages(
            document, GROUP_GOALS_SYSTEM_PROMPT, goals
        )
        if self.provider == "azure":
            response = self.azure_client.responses.parse(
                model=self.model,
                input=messages,
                temperature=temperature,
                text_format=GroupedGoals,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                format=GroupedGoals.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return GroupedGoals.model_validate(json.loads(parsed_ollama_response))
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def group_trends(
        self,
        document: str,
        trends: Trends,
        temperature: float = 0.2,
    ) -> GroupedTrends:
        messages = self._build_grouping_messages(
            document, GROUP_TRENDS_SYSTEM_PROMPT, trends
        )
        if self.provider == "azure":
            response = self.azure_client.responses.parse(
                model=self.model,
                input=messages,
                temperature=temperature,
                text_format=GroupedTrends,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                format=GroupedTrends.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return GroupedTrends.model_validate(json.loads(parsed_ollama_response))
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def link_strategies_and_goals(
        self,
        document: str,
        strategies: Strategies,
        goals: Goals,
        temperature: float = 0.2,
    ) -> StrategyGoalRelations:
        messages = self._build_linking_messages(
            document, LINK_STRATEGIES_AND_GOALS_SYSTEM_PROMPT, strategies, goals
        )
        if self.provider == "azure":
            response = self.azure_client.responses.parse(
                model=self.model,
                input=messages,
                temperature=temperature,
                text_format=StrategyGoalRelations,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                format=StrategyGoalRelations.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return StrategyGoalRelations.model_validate(
                    json.loads(parsed_ollama_response)
                )
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def link_strategies_and_trends(
        self,
        document: str,
        strategies: Strategies,
        trends: Trends,
        temperature: float = 0.2,
    ) -> StrategyTrendRelations:
        messages = self._build_linking_messages(
            document, LINK_STRATEGIES_AND_TRENDS_SYSTEM_PROMPT, strategies, trends
        )
        if self.provider == "azure":
            response = self.azure_client.responses.parse(
                model=self.model,
                input=messages,
                temperature=temperature,
                text_format=StrategyTrendRelations,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                format=StrategyTrendRelations.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return StrategyTrendRelations.model_validate(
                    json.loads(parsed_ollama_response)
                )
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
    def link_strategies_and_capabilities(
        self,
        document: str,
        strategies: Strategies,
        capabilities: Capabilities,
        temperature: float = 0.2,
    ) -> StrategyCapabilityRelations:
        messages = self._build_linking_messages(
            document, LINK_STRATEGIES_AND_CAPABILITIES_SYSTEM_PROMPT, strategies, capabilities
        )
        if self.provider == "azure":
            response = self.azure_client.responses.parse(
                model=self.model,
                input=messages,
                temperature=temperature,
                text_format=StrategyCapabilityRelations,
            )
            self._record_tokens(response)
            return response.output_parsed

        elif self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                format=StrategyCapabilityRelations.model_json_schema(),
                stream=False,
                options={"temperature": float(temperature)},
                keep_alive=self._ollama_keep_alive,
            )
            self._record_tokens(response)
            parsed_ollama_response = self._parse_ollama_response(
                ollama_response=response
            )

            try:
                return StrategyCapabilityRelations.model_validate(
                    json.loads(parsed_ollama_response)
                )
            except json.JSONDecodeError:
                raise ValueError(
                    f"Ollama response is not valid JSON: {parsed_ollama_response}"
                )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")
