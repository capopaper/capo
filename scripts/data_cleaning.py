
from collections import defaultdict
import time
import string
import json
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.lang.nl.stop_words import STOP_WORDS as DUTCH_STOP_WORDS

# Load Dutch language model and sentence embedding model
nlp = spacy.load("nl_core_news_sm")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Grouping function with file output
def group_similar_strings(strings, threshold=0.7, list_name="List", output_file="indirect_links.txt", append=True):
    n = len(strings)
    similarity_matrix = [[0.0] * n for _ in range(n)]
    groups = []
    visited = [False] * n
    indirect_links_info = []

    # Step 1: Compute similarity matrix
    for i in range(n):
        for j in range(i + 1, n):
            score = similarity(strings[i], strings[j])
            similarity_matrix[i][j] = similarity_matrix[j][i] = score

    # Step 2: Build groups using DFS
    def dfs(i, group, direct_links):
        visited[i] = True
        group.append(i)
        for j in range(n):
            if not visited[j] and similarity_matrix[i][j] >= threshold:
                direct_links.add((i, j))
                dfs(j, group, direct_links)

    #Code to check indirect links, 
    for i in range(n):
        if not visited[i]:
            group = []
            direct_links = set()
            dfs(i, group, direct_links)

            # Check for true indirect links
            all_pairs = {(a, b) for a in group for b in group if a < b}
            indirect = all_pairs - direct_links
            for a, b in indirect:
                score = similarity_matrix[a][b]
                if score < threshold:
                    indirect_links_info.append((strings[a], strings[b], score))

            groups.append([strings[idx] for idx in group])

    # Write results to file
    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        f.write(f"{list_name}\n")
        f.write("Groups:\n")
        for g in groups:
            f.write(f"  {g}\n")
        if indirect_links_info:
            f.write("True Indirect Links:\n")
            for a, b, score in indirect_links_info:
                f.write(f"  {a} <-> {b} (score: {score:.2f})\n")
        else:
            f.write("No true indirect links.\n")
        f.write("\n")

    return groups

from collections import defaultdict

def merge_groups_highest_match(grouped_lists_by_person, threshold=0.7):
    # Step 1: Assign unique IDs to each group
    group_map = {}
    group_ids = []
    for person, groups in grouped_lists_by_person.items():
        for idx, group in enumerate(groups):
            gid = (person, idx)
            group_ids.append(gid)
            group_map[gid] = group

    # Step 2: Compute pairwise similarities between groups from different people
    raw_matches = []
    for i, gid1 in enumerate(group_ids):
        for j in range(i + 1, len(group_ids)):
            gid2 = group_ids[j]
            person1, person2 = gid1[0], gid2[0]
            if person1 == person2:
                continue  # skip intra-person comparisons

            group1 = group_map[gid1]
            group2 = group_map[gid2]
            max_score = max(similarity(a, b) for a in group1 for b in group2)

            if max_score >= threshold:
                raw_matches.append((gid1, gid2, max_score))

    # Step 3: Keep only highest match per person per group
    best_matches = {}
    for gid1, gid2, score in raw_matches:
        p1, p2 = gid1[0], gid2[0]

        key1 = (gid1, p2)
        if key1 not in best_matches or best_matches[key1][2] < score:
            best_matches[key1] = (gid1, gid2, score)

        key2 = (gid2, p1)
        if key2 not in best_matches or best_matches[key2][2] < score:
            best_matches[key2] = (gid2, gid1, score)

    # Step 4: Build graph of connected groups using union-find
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for gid1, gid2, _ in best_matches.values():
        union(gid1, gid2)

    # Step 5: Collect merged groups
    clusters = defaultdict(list)
    for gid in group_ids:
        root = find(gid)
        clusters[root].append(gid)

    # Step 6: Build final output
    all_people = set(grouped_lists_by_person.keys())
    merged_output = []
    for cluster in clusters.values():
        all_strings = set()
        contributions = {person: [] for person in all_people}
        for person, idx in cluster:
            group = grouped_lists_by_person[person][idx]
            for s in group:
                all_strings.add(s)
                contributions[person].append(s)
        merged_output.append({
            'concept_group': sorted(all_strings),
            'contributions': contributions
        })

    return merged_output

#create input in format of alt test
def update_alt_test_input(merged_groups, doc_id, LLM_annotations, humans_annotations):

    if humans_annotations == {}: #create file
        for annotator in merged_groups[0]['contributions'].keys(): 
            if annotator == "LLM":
                continue
            humans_annotations.update({f"annotator{annotator}": {}})
    
    for item in range(len(merged_groups)):
            conceptNr = f"{doc_id}_{item+1}"
            for person, contribution in merged_groups[item]['contributions'].items():
                if person == "LLM": #this should contain the annotator ID code of the LLM
                    LLM_annotations.update({
                        f"concept{conceptNr}" : [contribution != []]
                        })
                else:
                    humans_annotations[f"annotator{person}"].update({
                         f"concept{conceptNr}" : [contribution != []] 
                    })
    return humans_annotations, LLM_annotations

def create_data_set(data, concept_nr, document_ids, threshold):
    import json
    humans_annotations = {}
    LLM_annotations = {}
    concepts = ["strategies", "goals", "trends"]
    concept = concepts[concept_nr]
    alldocs_merged_groups = []
    
    for doc_id in document_ids:
        concept_groups_by_person = {}
        for annotation in data:
            annot_id = annotation.get("annotator_id")
            for doc in annotation.get("documents_annotated"):
                if doc.get("document_id") == doc_id:
                    concepts_found = basic_normalize(doc.get(concept))
                    grouped_concepts = group_similar_strings(concepts_found)
                    concept_groups_by_person.update({
                        annot_id: grouped_concepts})
        merged_groups = merge_groups_highest_match(concept_groups_by_person, threshold)
        [humans_annotations, LLM_annotations] = update_alt_test_input(merged_groups, doc_id, LLM_annotations, humans_annotations)
        
        filename = f"grouped_concepts_doc_{doc_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(merged_groups, f, indent=2, ensure_ascii=False)

        alldocs_merged_groups.append(merged_groups)
    
                                             
    return alldocs_merged_groups, humans_annotations, LLM_annotations

# def make_result_matrix(alldocs_merged_groups):
#     result_matrix = []
#     for concept_group, contributions in alldocs_merged_groups:
#         for person in contributions:
#             if person 


import json


#clean data fro
def basic_normalize(text_list):
    import string

    normalized = []
    for text in text_list:
        cleaned = text.strip().lower().strip(string.punctuation)
        normalized.append(cleaned)
    return list(dict.fromkeys(normalized))

#similarity score
def similarity(a, b):
    embeddings = embedding_model.encode([a, b], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return score

def main():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1] 
    output_dir = PROJECT_ROOT / "output" / "data_cleaning_experimental"
    input_path = PROJECT_ROOT / "output" / "annotations_cleaned.json"

    output_dir.mkdir(exist_ok=True)
    print(f"Experimental outputs will be saved to: {output_dir}")

    print(f"Loading data from {input_path}...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}. Please run data_formating.py first.")
        return

    threshold = 0.8
    document_ids = sorted(list(set(doc.get('document_id') for doc in data[0].get("documents_annotated"))))

    print("Creating ALT-test input data (experimental)...")
    [all_strategie_instances, h_strat, l_strat] = create_data_set(data, 0, document_ids, threshold)  
    [all_goals_instances, h_goals, l_goals] = create_data_set(data, 1, document_ids, threshold) 
    [all_trends_instances, h_trends, l_trends] = create_data_set(data, 2, document_ids, threshold) 

    print("Saving output files...")
    
    with open(output_dir / "concept_groups_strategy.txt", 'w', encoding='utf-8') as f:
        f.write("List of all concepts found:\n")
        for i, doc_id in enumerate(document_ids):
            f.write(f"document_id: {doc_id}\n")
            for concept in all_strategie_instances[i]:
                f.write("  Concept:\n")
                for var in concept['concept_group']:
                    f.write(f"    {var}\n")
            f.write("\n")

    with open(output_dir / "humans_annotations_strategies.json", "w", encoding="utf-8") as f:
        json.dump(h_strat, f, indent=2, ensure_ascii=False)
    with open(output_dir / "llm_annotations_strategies.json", "w", encoding="utf-8") as f:
        json.dump(l_strat, f, indent=2, ensure_ascii=False)
    with open(output_dir / "concept_groups_all_strategy.json", "w", encoding="utf-8") as f:
        json.dump(all_strategie_instances, f, indent=2, ensure_ascii=False)

    # Save goal files
    with open(output_dir / "humans_annotations_goals.json", "w", encoding="utf-8") as f:
        json.dump(h_goals, f, indent=2, ensure_ascii=False)
    with open(output_dir / "llm_annotations_goals.json", "w", encoding="utf-8") as f:
        json.dump(l_goals, f, indent=2, ensure_ascii=False)
    with open(output_dir / "concept_groups_all_goals.json", "w", encoding="utf-8") as f:
        json.dump(all_goals_instances, f, indent=2, ensure_ascii=False)

    # Save trend files
    with open(output_dir / "humans_annotations_trends.json", "w", encoding="utf-8") as f:
        json.dump(h_trends, f, indent=2, ensure_ascii=False)
    with open(output_dir / "llm_annotations_trends.json", "w", encoding="utf--8") as f:
        json.dump(l_trends, f, indent=2, ensure_ascii=False)
    with open(output_dir / "concept_groups_all_trends.json", "w", encoding="utf-8") as f:
        json.dump(all_trends_instances, f, indent=2, ensure_ascii=False)
        
    print("Done.")

if __name__ == "__main__":
    main()