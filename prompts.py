STRATEGY_SIMPLE_SYSTEM_PROMPT = """
From the following text, extract all the strategies you can find.
Your task:
For each strategy you identify in the text, extract:
- Name: Summarize with a few words in Dutch the extracted strategy.
- Description: The sentence from which the name of the strategy has been derived (exact sentence from the text).
""".strip()

GOAL_SYSTEM_PROMPT = """
From the following text, extract all the goals you can find.
Your task:
Use the provided list of strategies to identify the goals these strategies have. For each goal you identify in the text, extract:
- Name: Summarize with a few words in Dutch the extracted goal.
- Description: The sentence from which the name of the goal has been derived (exact sentence from the text).
- Strategies: Strategy or strategies from the given list the goal is a goal of. It MUST be a list of length equal or bigger than 1.
IMPORTANT: A goal cannot be any of the strategies already identified.
If goals cannot be found in the text because they are not explicitly mentioned, return an empty set.
""".strip()


TREND_SYSTEM_PROMPT = """
From the following text, extract all the trends you can find.
Use the provided list of strategies and goals to identify the trends that the strategies are a response to. For each trend you identify in the text, extract:
- Name: Summarize with a few words in Dutch the extracted trend.
- Description: The sentence from which the name of the trend has been derived (exact sentence from the text).
IMPORTANT: A trend cannot be any of the strategies nor any of the goals already identified.
REMEMBER: Trends are EXTERNAL to the police organization.
If trends cannot be found in the text because they are not explicitly mentioned, return an empty set.
""".strip()


GROUP_STRATEGIES_SYSTEM_PROMPT = """
Group semantically equivalent strategies.
Use ONLY the provided strategy names; do not invent new names.
Return ONLY a single valid JSON object matching the schema.
No code fences, no Markdown, no commentary.
""".strip()

GROUP_GOALS_SYSTEM_PROMPT = """
Group semantically equivalent goals.
Use ONLY the provided goal names; do not invent new names.
Return ONLY a single valid JSON object matching the schema.
No code fences, no Markdown, no commentary.
""".strip()

GROUP_TRENDS_SYSTEM_PROMPT = """
Group semantically equivalent trends.
Use ONLY the provided trend names; do not invent new names.
Return ONLY a single valid JSON object matching the schema.
No code fences, no Markdown, no commentary.
""".strip()

LINK_STRATEGIES_AND_GOALS_SYSTEM_PROMPT = """
Link the strategies to the goals according to the text.
Use ONLY the provided strategy and goal names; do not invent new names.
Return ONLY a single valid JSON object matching the schema.
No code fences, no Markdown, no commentary.
""".strip()

LINK_STRATEGIES_AND_TRENDS_SYSTEM_PROMPT = """
Link the strategies to the trends according to the text.
Use ONLY the provided strategy and trend names; do not invent new names.
Return ONLY a single valid JSON object matching the schema.
No code fences, no Markdown, no commentary.
""".strip()

LINK_STRATEGIES_AND_CAPABILITIES_SYSTEM_PROMPT = """
Link the strategies found to the given list of capabilities according to the text.
Make sure that each strategy is linked to AT LEAST ONE capability.
Use ONLY the provided strategy and capabilities names; do not invent new names.
Return ONLY a single valid JSON object matching the schema.
No code fences, no Markdown, no commentary.
""".strip()
