# analysis_logic_reverted.py (for Threshold 0.8)

from collections import defaultdict
import string
import json
from sentence_transformers import SentenceTransformer, util
import spacy
import numpy as np

# Load models
print("Loading NLP models...")
nlp = spacy.load("nl_core_news_sm")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def similarity(a, b):
    # Added show_progress_bar=False to prevent terminal display issues
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

def create_final_annotation_files(merged_groups_by_doc, all_annotators):
    all_concept_ids = []
    for doc_id, groups in sorted(merged_groups_by_doc.items()):
        for i in range(len(groups)):
            all_concept_ids.append(f"concept{doc_id}_{i+1}")

    human_annotators = sorted([ann_id for ann_id in all_annotators if ann_id != 'LLM'])
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
                    if annotator_id == 'LLM':
                        if concept_id in LLM_annotations:
                            LLM_annotations[concept_id] = [True]
                    else:
                        annotator_key = f"annotator{annotator_id}"
                        if annotator_key in humans_annotations and concept_id in humans_annotations[annotator_key]:
                            humans_annotations[annotator_key][concept_id] = [True]
    return humans_annotations, LLM_annotations

# --- REVERTED TO THE ORIGINAL MERGE_CONCEPTS LOGIC ---
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
            group = np.where(sim_matrix[i] >= threshold)[0]
            q = list(group)
            visited[q] = True
            head = 0
            while head < len(q):
                curr = q[head]
                head += 1
                neighbors = np.where(sim_matrix[curr] >= threshold)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
            groups_of_indices.append(sorted(list(set(q))))
    
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

def create_data_set(data, concept_name, document_ids, threshold):
    alldocs_merged_groups = {}
    for doc_id in document_ids:
        concepts_by_person = defaultdict(list)
        for annotation in data:
            annot_id = annotation.get("annotator_id")
            for doc in annotation.get("documents_annotated"):
                if doc.get("document_id") == doc_id:
                    concepts_found = basic_normalize(doc.get(concept_name, []))
                    if concepts_found:
                        concepts_by_person[annot_id].extend(concepts_found)
        if concepts_by_person:
            merged_groups = merge_concepts(concepts_by_person, threshold)
            alldocs_merged_groups[doc_id] = merged_groups
        else:
            alldocs_merged_groups[doc_id] = []
    return alldocs_merged_groups


def main():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # --- End of snippet ---

    input_path = PROJECT_ROOT / "output" / "annotations_cleaned.json"
    output_dir = PROJECT_ROOT / "output" / "alt_test_input" / "entities"

    print(f"Loading cleaned data file '{input_path}'...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_ids = sorted([doc.get('document_id') for doc in data[0].get("documents_annotated")])
    all_annotators = sorted([str(blob['annotator_id']) for blob in data])
    
    # --- SETTING THE NEW, STRICTER THRESHOLD ---
    threshold = 0.8
    print(f"\n>>> Running analysis for ENTITIES (Strategies, Goals, Trends) at a fixed threshold of {threshold} <<<\n")

    print("Grouping concepts for 'strategies'...")
    strategies_by_doc = create_data_set(data, "strategies", document_ids, threshold)
    print("Grouping concepts for 'goals'...")
    goals_by_doc = create_data_set(data, "goals", document_ids, threshold)
    print("Grouping concepts for 'trends'...")
    trends_by_doc = create_data_set(data, "trends", document_ids, threshold)
    
    print("\nCreating final dense annotation files...")
    h_strat, l_strat = create_final_annotation_files(strategies_by_doc, all_annotators)
    h_goal, l_goal = create_final_annotation_files(goals_by_doc, all_annotators)
    h_trend, l_trend = create_final_annotation_files(trends_by_doc, all_annotators)

    # --- ADDED: PRINT THE NUMBER OF INSTANCES (n) FOR EACH CONCEPT ---
    print("\n" + "="*50)
    print("Number of Unique Concepts (Instances 'n') Generated:")
    print(f"  - Strategies: {len(l_strat)}")
    print(f"  - Goals:      {len(l_goal)}")
    print(f"  - Trends:     {len(l_trend)}")
    print("="*50 + "\n")


    files_to_save = {
        "strategies": (h_strat, l_strat, strategies_by_doc),
        "goals": (h_goal, l_goal, goals_by_doc),
        "trends": (h_trend, l_trend, trends_by_doc),
    }

    print("Saving annotation and concept group files...")
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, (h_data, l_data, concept_data) in files_to_save.items():
        with open(os.path.join(output_dir, f"humans_annotations_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(h_data, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, f"llm_annotations_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(l_data, f, indent=2, ensure_ascii=False)
        
        all_concepts_flat = [group for doc_groups in concept_data.values() for group in doc_groups]
        with open(os.path.join(output_dir, f"concept_groups_all_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(all_concepts_flat, f, indent=2, ensure_ascii=False)
            
    print(f"\nAll files generated successfully in the '{output_dir}' directory.")
    print("You are now ready to run the ALT-test script on the files in this directory.")

if __name__ == "__main__":
    main()