import os
import json

def save_json_to_output(filename: str, json_str: str, foldername:str = "output"):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(base_dir, "..", foldername)

    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{filename}.json")

    try:
        data = json.loads(json_str)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"JSON saved successfully to {file_path}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")