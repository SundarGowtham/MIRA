import json
import re
from collections import defaultdict
from pathlib import Path


DATA_RAW      = Path("data/raw")


def analyze_corpus_for_nuance(json_filepath):
    # Define the chemical nuances we suspect might be missing
    nuance_categories = {
        "Separation":         r"\b(centrifug\w*|filter\w*|wash\w*|decant\w*|siev\w*|chromatography)\b",
        "Aging_or_Passive":   r"\b(ag(e|ing)|cur(e|ing)|incubat\w*|standing|left to sit)\b",
        "Atmosphere_Control": r"\b(vacuum|argon|Ar|nitrogen|N2|glovebox|schlenk|purged|flow)\b",
        "Special_Energy":     r"\b(sonicat\w*|ultrasound|microwave\w*|irradiat\w*|UV|reflux\w*)\b",
    }
    
    # Initialize counters
    total_recipes = 0
    matches = defaultdict(int)
    flagged_examples = defaultdict(list)
    
    # Load your data (assuming a JSON array of synthesis entries)
    with open(json_filepath, 'r') as f:
        data = json.load(f)
        
    for entry in data:
        paragraph = entry.get("paragraph_string", "")  # don't lowercase
        if not paragraph:
            continue
        total_recipes += 1
        for category, pattern in nuance_categories.items():
            if re.search(pattern, paragraph, re.IGNORECASE):  # flag handles case
                matches[category] += 1
                if len(flagged_examples[category]) < 3:
                    flagged_examples[category].append(entry.get("doi", "Unknown DOI"))

    # Print out the verdict
    print(f"--- Corpus Analysis Complete ---")
    print(f"Total recipes analyzed: {total_recipes}")
    print("-" * 30)
    
    for category in nuance_categories.keys():
        count = matches[category]
        percentage = (count / total_recipes) * 100 if total_recipes > 0 else 0
        print(f"{category}: Found in {count} recipes ({percentage:.2f}%)")
        
        if flagged_examples[category]:
            print(f"  Example DOIs: {', '.join(flagged_examples[category])}")
    print("-" * 30)

# To run it on your data:
analyze_corpus_for_nuance(DATA_RAW / 'synthesis.json')