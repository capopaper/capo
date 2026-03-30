import json
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def extract_completed_info(data):
    """
    Extracts all completed annotations and returns a dictionary with annotator IDs and document IDs.

    Args:
        data (list): Parsed JSON data.

    Returns:
        dict: {
            "annotator_id": [list of unique annotator IDs],
            "documents_annotated": [list of unique document IDs]
        }
    """
    annotator_ids = set()
    document_ids = set()

    for doc in data:
        doc_id = doc.get("id")
        for annotation in doc.get("annotations", []):
            if not annotation.get("was_cancelled", False):
                annotator_id = annotation.get("completed_by")
                annotator_ids.add(annotator_id)
                document_ids.add(doc_id)

    return {
        "annotator_id": sorted(list(annotator_ids)),
        "documents_annotated": sorted(list(document_ids))
    }


def extract_filtered_annotations(data, annotation_info, doc_names_dictionary):
    """
    Extracts and structures annotations by annotator, including grouping and relations.

    Returns:
        Tuple of two lists:
        - cleaned_output: only 'text' fields
        - extended_output: includes 'id', 'text', 'parentID'
    """
    from collections import defaultdict

    cleaned_output = []
    extended_output = []
    # document_dictionary = 

    for annotator_id in annotation_info["annotator_id"]:
        annotated_docs_clean = []
        annotated_docs_ext = []

        for doc in data:
            doc_id = doc.get("id")
            if doc_id not in annotation_info["documents_annotated"]:
                continue

            for annotation in doc.get("annotations", []):
                if annotation.get("completed_by") != annotator_id:
                    continue

                strategies, goals, trends = [], [], []
                strategy_map, goal_map, trend_map = {}, {}, {}
                relations = []
                capability_links = []

                for result in annotation.get("result", []):
                    r_type = result.get("type")

                    if r_type == "labels":
                        label_type = result.get("value", {}).get("labels", [])
                        text = result.get("value", {}).get("text", "")
                        quote_id = result.get("id")
                        parent_id = result.get("parentID", None)

                        entry = {
                            "id": quote_id,
                            "text": text,
                            "parentID": parent_id
                        }

                        if "Strategy" in label_type:
                            strategies.append(entry)
                            strategy_map[quote_id] = entry
                        elif "Goal" in label_type:
                            goals.append(entry)
                            goal_map[quote_id] = entry
                        elif "Trend" in label_type:
                            trends.append(entry)
                            trend_map[quote_id] = entry

                    elif r_type == "relation":
                        relations.append({
                            "from_id": result.get("from_id"),
                            "to_id": result.get("to_id")
                        })

                    elif r_type == "choices":
                        capability_links.append({
                            "strategy_id": result.get("id"),
                            "capabilities": result.get("value", {}).get("choices", [])
                        })

                # Grouping logic
                def group_by_parent(entries):
                    grouped = []
                    seen = set()
                    for entry in entries:
                        if entry["id"] in seen:
                            continue
                        group = [entry]
                        seen.add(entry["id"])
                        pid = entry.get("parentID")
                        if pid:
                            for other in entries:
                                if other["id"] == pid or other.get("parentID") == pid:
                                    if other["id"] not in seen:
                                        group.append(other)
                                        seen.add(other["id"])
                        if len(group) > 1:
                            grouped.append([e["text"] for e in group])
                    return grouped

                grouped_strategies = group_by_parent(strategies)
                grouped_goals = group_by_parent(goals)
                grouped_trends = group_by_parent(trends)

                # Relation logic
                strategy_relations = defaultdict(lambda: {
                    "strategy": "",
                    "has_goal": [],
                    "is_response_to": [],
                    "requires": []
                })

                for rel in relations:
                    from_id = rel["from_id"]
                    to_id = rel["to_id"]

                    from_is_strategy = from_id in strategy_map
                    to_is_strategy = to_id in strategy_map

                    if not from_is_strategy and to_is_strategy:
                        from_id, to_id = to_id, from_id
                        from_is_strategy, to_is_strategy = True, False

                    if not from_is_strategy:
                        print(f"Warning: Skipping relation where neither ID is a strategy: {from_id} → {to_id}")
                        continue

                    strategy_text = strategy_map[from_id]["text"]
                    relation_entry = strategy_relations[strategy_text]
                    relation_entry["strategy"] = strategy_text

                    if to_id in goal_map:
                        relation_entry["has_goal"].append(goal_map[to_id]["text"])
                    elif to_id in trend_map:
                        relation_entry["is_response_to"].append(trend_map[to_id]["text"])
                    elif to_id in strategy_map:
                        relation_entry["requires"].append(strategy_map[to_id]["text"])

                # Add capabilities from choices
                for link in capability_links:
                    strategy_id = link["strategy_id"]
                    capabilities = link["capabilities"]
                    if strategy_id in strategy_map:
                        strategy_text = strategy_map[strategy_id]["text"]
                        strategy_relations[strategy_text]["strategy"] = strategy_text
                        strategy_relations[strategy_text]["requires"].extend(capabilities)

                # Cleaned version
                doc_clean = {
                    "document_id": doc_names_dictionary[doc_id],
                    "strategies": [s["text"] for s in strategies],
                    "goals": [g["text"] for g in goals],
                    "trends": [t["text"] for t in trends],
                    "grouped_strategies": grouped_strategies,
                    "grouped_goals": grouped_goals,
                    "grouped_trends": grouped_trends,
                    "strategy_relations": list(strategy_relations.values())
                }

                # Extended version
                doc_ext = {
                    "document_id": doc_names_dictionary[doc_id],
                    "strategies": strategies,
                    "goals": goals,
                    "trends": trends,
                    "grouped_strategies": grouped_strategies,
                    "grouped_goals": grouped_goals,
                    "grouped_trends": grouped_trends,
                    "strategy_relations": list(strategy_relations.values())
                }

                annotated_docs_clean.append(doc_clean)
                annotated_docs_ext.append(doc_ext)

        cleaned_output.append({
            "annotator_id": annotator_id,
            "documents_annotated": annotated_docs_clean
        })

        extended_output.append({
            "annotator_id": annotator_id,
            "documents_annotated": annotated_docs_ext
        })

    return cleaned_output, extended_output

def main():
    import sys
    from pathlib import Path

    def get_project_root() -> Path:
        """Recursively searches for the project root marker '.git' or 'README.md'."""
        current_path = Path.cwd()
        while current_path != current_path.parent:
            if (current_path / ".git").exists() or (current_path / "README.md").exists():
                return current_path
            current_path = current_path.parent
        return Path(__file__).resolve().parents[2]

    PROJECT_ROOT = get_project_root()
    print(f"Project root identified as: {PROJECT_ROOT}")

    human_annotations_path = PROJECT_ROOT / "data" / "human_annotations" / "original_annotations.json"
    llm_annotations_path = PROJECT_ROOT / "data" / "llm_annotations" / "azure_gpt-4.1.json" 
    output_path = PROJECT_ROOT / "output" / "annotations_cleaned.json"

    (PROJECT_ROOT / "output").mkdir(exist_ok=True)

    print(f"Loading human annotations from: {human_annotations_path}")
    with open(human_annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading LLM annotations from: {llm_annotations_path}")
    with open(llm_annotations_path, "r", encoding="utf-8") as f:
        data_LLM = json.load(f)

    annotation_info = extract_completed_info(data)
    doc_names = [doc.get('document_id') for doc in data_LLM.get("documents_annotated")]

    doc_names_dictionary = {
        27: "tutorial", 28: "test", 29: doc_names[0], 30: doc_names[1],
        31: doc_names[2], 32: doc_names[3], 33: doc_names[4], 34: doc_names[5],
        35: doc_names[6], 36: doc_names[7], 37: doc_names[8], 38: doc_names[9]
    }

    excluded_docs = [27, 37, 38, 39]
    excluded_annotators = [1, 3]
    for id in excluded_annotators:
        annotation_info["annotator_id"].remove(id)
    for id in excluded_docs:
        annotation_info["documents_annotated"].remove(id)
    
    [output, extended_output] = extract_filtered_annotations(data, annotation_info, doc_names_dictionary)
    output.append(data_LLM)

    print(f"Saving cleaned output to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()
