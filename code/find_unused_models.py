import os
import re

def get_used_models(gen_summary_path):
    """
    Extracts model timestamps from the model_name_map in gen_summary_table.py.
    """
    used_models = set()
    try:
        with open(gen_summary_path, 'r') as f:
            content = f.read()
            # Simple regex to find the model_name_map entries
            # Matches strings like "12281023": "CCSR Run 1"
            matches = re.findall(r'"(\d{8})":\s*"[^"]*"', content)
            used_models.update(matches)
    except Exception as e:
        print(f"Error reading {gen_summary_path}: {e}")
    return used_models

def get_all_models(exp_base_path):
    """
    Finds all 8-digit timestamp directories in the exp directory.
    """
    all_models = set()
    if not os.path.exists(exp_base_path):
        print(f"Error: Path {exp_base_path} does not exist.")
        return all_models

    # Walk through the exp directory
    for root, dirs, files in os.walk(exp_base_path):
        for d in dirs:
            # Check if directory name is exactly 8 digits
            if re.fullmatch(r'\d{8}', d):
                all_models.add(d)
    return all_models

def main():
    base_dir = "/storage/yotam/SAR-SSL/SAR-SSL"
    exp_path = os.path.join(base_dir, "exp")
    gen_summary_path = os.path.join(base_dir, "code/gen_summary_table.py")

    used_models = get_used_models(gen_summary_path)
    all_models = get_all_models(exp_path)

    unused_models = sorted(list(all_models - used_models))

    print(f"Found {len(all_models)} total models in {exp_path}")
    print(f"Found {len(used_models)} used models in {gen_summary_path}")
    print(f"Found {len(unused_models)} unused models.")
    print("-" * 30)
    
    if unused_models:
        print("Unused models (comma separated):")
        print(",".join(unused_models))
    else:
        print("No unused models found.")

if __name__ == "__main__":
    main()
