# analysis_logic_links.py

from collections import defaultdict
import string
import json
from sentence_transformers import SentenceTransformer, util
import spacy
import numpy as np
import os

# --- Boilerplate and Helper Functions (from previous scripts) ---

print("Loading NLP models...")
nlp = spacy.load("nl_core_news_sm")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def similarity(a, b):
    embeddings = embedding_model.encode([a, b], convert_to_tensor=True, show_progress_bar=False)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return score

def basic_normalize(text_list):
    normalized = []
    if not text_list: return []
    for text in text_list:
        cleaned = text.strip().lower().strip(string.punctuation)
        normalized.append(cleaned)
    return list(dict.fromkeys(normalized))

def merge_concepts(concepts_by_person, threshold):
    all_items = []
    for person, concepts in concepts_by_person.items():
        for concept in concepts:
            all_items.append({'text': concept, 'annotator': person})
    
    n = len(all_items)
    if n == 0: return []
    
    sim_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            score = similarity(all_items[i]['text'], all_items[j]['text'])
            sim_matrix[i, j] = sim_matrix[j, i] = score

    visited = np.zeros(n, dtype=bool)
    groups_of_indices = []
    for i in range(n):
        if not visited[i]:
            q = [i]
            visited[i] = True
            group = {i}
            head = 0
            while head < len(q):
                curr = q[head]
                head += 1
                neighbors = np.where(sim_matrix[curr] >= threshold)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        group.add(neighbor)
                        q.append(neighbor)
            groups_of_indices.append(sorted(list(group)))
    
    merged_output = []
    all_people = set(concepts_by_person.keys())
    for group_indices in groups_of_indices:
        contributions = {p: [] for p in all_people}
        group_texts = set()
        for idx in group_indices:
            item = all_items[idx]
            contributions[item['annotator']].append(item['text'])
            group_texts.add(item['text'])
        
        final_contributions = {p: sorted(list(set(c))) for p, c in contributions.items()}
        merged_output.append({
            'concept_group': sorted(list(group_texts)),
            'contributions': final_contributions
        })
    return merged_output

def create_final_annotation_files(merged_groups_by_doc, all_annotators):
    all_concept_ids = []
    for doc_id, groups in sorted(merged_groups_by_doc.items()):
        for i in range(len(groups)):
            all_concept_ids.append(f"concept{doc_id}_{i+1}")

    human_annotators = sorted([str(ann_id) for ann_id in all_annotators if str(ann_id) != 'LLM'])
    humans_annotations = {
        f"annotator{ann_id}": {concept_id: [False] for concept_id in all_concept_ids}
        for ann_id in human_annotators
    }
    LLM_annotations = {concept_id: [False] for concept_id in all_concept_ids}

    for doc_id, merged_groups in sorted(merged_groups_by_doc.items()):
        for i, group in enumerate(merged_groups):
            concept_id = f"concept{doc_id}_{i+1}"
            for annotator_id, contribution in group['contributions'].items():
                if len(contribution) > 0:
                    if str(annotator_id) == 'LLM':
                        if concept_id in LLM_annotations:
                            LLM_annotations[concept_id] = [True]
                    else:
                        annotator_key = f"annotator{annotator_id}"
                        if annotator_key in humans_annotations and concept_id in humans_annotations[annotator_key]:
                            humans_annotations[annotator_key][concept_id] = [True]
    return humans_annotations, LLM_annotations

# --- NEW ANALYSIS FUNCTIONS FOR ISOLATED LINKING SKILL ---

def analyze_strategy_capability_links_isolated(data, document_ids, threshold):
    """
    Analyzes Strategy -> Capability links.
    An annotator is only evaluated on a link if they identified the source strategy.
    """
    alldocs_merged_groups = {}
    all_annotators = {blob['annotator_id'] for blob in data}

    for doc_id in document_ids:
        # 1. Collect all raw strategies and their linked capabilities by annotator
        concepts_by_person = defaultdict(list)
        strat_to_caps_map = defaultdict(lambda: defaultdict(set))
        for blob in data:
            ann_id = blob['annotator_id']
            for doc in blob['documents_annotated']:
                if doc['document_id'] == doc_id:
                    for rel in doc.get('strategy_relations', []):
                        strategy = rel.get('strategy')
                        if strategy:
                            concepts_by_person[ann_id].append(strategy)
                            caps = basic_normalize(rel.get('requires', []))
                            if caps:
                                strat_to_caps_map[ann_id][strategy].update(caps)

        if not concepts_by_person:
            alldocs_merged_groups[doc_id] = []
            continue

        # 2. Group strategies into "strategy concepts"
        strategy_groups = merge_concepts(concepts_by_person, threshold)
        doc_link_concepts = []

        # 3. For each strategy concept, evaluate the linking skill
        for strat_group in strategy_groups:
            source_strat_texts = set(strat_group['concept_group'])
            
            # Find annotators who contributed to this strategy concept
            contributing_annotators = {
                ann_id for ann_id, contribs in strat_group['contributions'].items() if contribs
            }
            if not contributing_annotators:
                continue

            # 4. Create a master list of all capabilities linked to this strategy concept
            master_caps = set()
            for ann_id in contributing_annotators:
                for strat_text in source_strat_texts:
                    if strat_text in strat_to_caps_map[ann_id]:
                        master_caps.update(strat_to_caps_map[ann_id][strat_text])
            
            # 5. For each capability, create a "link concept" and check who agrees
            for cap in sorted(list(master_caps)):
                link_contributions = {ann_id: [] for ann_id in all_annotators}
                found = False
                for ann_id in contributing_annotators:
                    # Check if this annotator made the link
                    made_link = any(
                        cap in strat_to_caps_map[ann_id].get(strat_text, set())
                        for strat_text in source_strat_texts
                    )
                    if made_link:
                        link_contributions[ann_id].append(f"LINK_EXISTS")
                        found = True
                
                if found:
                    doc_link_concepts.append({
                        'concept_group': [f"{list(source_strat_texts)[0]} -> {cap}"], # Representative name
                        'contributions': link_contributions
                    })
        
        alldocs_merged_groups[doc_id] = doc_link_concepts
    return alldocs_merged_groups


def analyze_strategy_to_concept_links_isolated(data, document_ids, threshold, target_concept_name, link_key):
    """
    Generic function to analyze Strategy -> [Goal/Trend] links.
    An annotator is evaluated only if they identified the source strategy.
    """
    alldocs_merged_groups = {}
    all_annotators = {blob['annotator_id'] for blob in data}

    for doc_id in document_ids:
        # 1. Collect raw strategies, targets, and links
        strats_by_person = defaultdict(list)
        targets_by_person = defaultdict(list)
        strat_to_target_map = defaultdict(lambda: defaultdict(set))

        for blob in data:
            ann_id = blob['annotator_id']
            for doc in blob['documents_annotated']:
                if doc['document_id'] == doc_id:
                    strats_by_person[ann_id].extend(basic_normalize(doc.get('strategies', [])))
                    targets_by_person[ann_id].extend(basic_normalize(doc.get(target_concept_name, [])))
                    for rel in doc.get('strategy_relations', []):
                        strategy = rel.get('strategy')
                        if strategy:
                            targets = basic_normalize(rel.get(link_key, []))
                            if targets:
                                strat_to_target_map[ann_id][strategy].update(targets)
        
        if not strats_by_person:
            alldocs_merged_groups[doc_id] = []
            continue

        # 2. Group source strategies and find contributors
        strategy_groups = merge_concepts(strats_by_person, threshold)
        doc_link_concepts = []

        for strat_group in strategy_groups:
            source_strat_texts = set(strat_group['concept_group'])
            contributing_annotators = {
                ann_id for ann_id, contribs in strat_group['contributions'].items() if contribs
            }
            if not contributing_annotators:
                continue

            # 3. For this group of contributors, find all targets they linked to the source strategies
            linked_targets_by_person = defaultdict(list)
            for ann_id in contributing_annotators:
                for strat_text in source_strat_texts:
                    if strat_text in strat_to_target_map[ann_id]:
                        linked_targets_by_person[ann_id].extend(list(strat_to_target_map[ann_id][strat_text]))
            
            if not linked_targets_by_person:
                continue

            # 4. Group these linked targets into "target concepts"
            target_groups = merge_concepts(linked_targets_by_person, threshold)

            # 5. For each pair of (strategy concept, target concept), evaluate the link
            for target_group in target_groups:
                target_texts = set(target_group['concept_group'])
                link_contributions = {ann_id: [] for ann_id in all_annotators}
                found = False

                for ann_id in contributing_annotators:
                    made_link = any(
                        target in strat_to_target_map[ann_id].get(strat, set())
                        for strat in source_strat_texts for target in target_texts
                    )
                    if made_link:
                        link_contributions[ann_id].append(f"LINK_EXISTS")
                        found = True
                
                if found:
                    doc_link_concepts.append({
                        'concept_group': [f"{list(source_strat_texts)[0]} -> {list(target_texts)[0]}"],
                        'contributions': link_contributions
                    })
        
        alldocs_merged_groups[doc_id] = doc_link_concepts
    return alldocs_merged_groups


def main():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    input_path = PROJECT_ROOT / "output" / "annotations_cleaned.json"
    output_dir = PROJECT_ROOT / "output" / "alt_test_input" / "links"

    print(f"Loading cleaned data file '{input_path}'...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_ids = sorted([doc.get('document_id') for doc in data[0].get("documents_annotated")])
    all_annotators = {blob['annotator_id'] for blob in data}
    
    # Using the optimal threshold for links we found earlier
    threshold = 0.8
    print(f"\n>>> Running ISOLATED LINK analysis at a fixed threshold of {threshold} <<<\n")

    # --- Run analysis for each of the three link types ---
    print("Analyzing Strategy -> Capability links...")
    s_cap_links = analyze_strategy_capability_links_isolated(data, document_ids, threshold)
    
    print("Analyzing Strategy -> Goal links...")
    s_goal_links = analyze_strategy_to_concept_links_isolated(data, document_ids, threshold, 'goals', 'has_goal')
    
    print("Analyzing Strategy -> Trend links...")
    s_trend_links = analyze_strategy_to_concept_links_isolated(data, document_ids, threshold, 'trends', 'is_response_to')

    # --- Create final files for ALT-test ---
    print("\nCreating final dense annotation files for links...")
    h_scap, l_scap = create_final_annotation_files(s_cap_links, all_annotators)
    h_sgoal, l_sgoal = create_final_annotation_files(s_goal_links, all_annotators)
    h_strend, l_strend = create_final_annotation_files(s_trend_links, all_annotators)

    files_to_save = {
        "strategy_capability_links": (h_scap, l_scap),
        "strategy_goal_links": (h_sgoal, l_sgoal),
        "strategy_trend_links": (h_strend, l_strend),
    }

    print("\n" + "="*50)
    print("Number of Unique Link Concepts ('n') Generated:")
    print(f"  - Strategy -> Capability: {len(l_scap)}")
    print(f"  - Strategy -> Goal:       {len(l_sgoal)}")
    print(f"  - Strategy -> Trend:      {len(l_strend)}")
    print("="*50 + "\n")

    # --- Save files to a new, dedicated directory ---
    print("Saving annotation and concept group files...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving files to '{output_dir}' directory...")
    for name, (h_data, l_data) in files_to_save.items():
        with open(os.path.join(output_dir, f"humans_annotations_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(h_data, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, f"llm_annotations_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(l_data, f, indent=2, ensure_ascii=False)
            
    print("\nAll files generated successfully.")
    print("You are now ready to run the ALT-test on the files in the new directory.")

if __name__ == "__main__":
    main()