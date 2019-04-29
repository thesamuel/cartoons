import json
import os
from pathlib import Path
from collections import Counter

_DATA_DIR = Path('./raw-data')
_TOP_K = 10


def split_keywords(keywords: str) -> set:
    return set(t.strip().lower() for t in keywords.split(','))


# Keep statistics for keywords
unsplittable = 0
empty = 0
total = 0

# Get all of the available tags
all_keywords = Counter()
for filename in os.listdir(_DATA_DIR):
    cid, ext = os.path.splitext(filename)

    if ext != '.json':
        continue

    total += 1

    with open(_DATA_DIR / filename, 'r') as f:
        metadata = json.load(f)
        keywords = metadata['keywords']
        if not keywords:
            empty += 1
        elif "," not in keywords:
            unsplittable += 1
        else:
            all_keywords.update(split_keywords(keywords))

# Remove entry for empty string
del all_keywords[""]

# Print results
print(f"Keywords found for {total - unsplittable - empty} cartoons:")
print(f"{unsplittable} unsplittable, {empty} empty")
print()

print(f"{_TOP_K} most common tags:")
print(*all_keywords.most_common(_TOP_K), sep='\n')
