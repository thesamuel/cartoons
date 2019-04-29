import json
import os
from pathlib import Path
from collections import Counter

_DATA_DIR = Path('./raw-data')
_TOP_K = 10


def split_keywords(keywords: str) -> list:
    return [t.strip().lower() for t in keywords.split(',')]


# Keep statistics for keywords
unsplittable = 0
empty = 0
total = 0

# Get all of the available tags
tags = {}
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
            tags[cid] = split_keywords(keywords)

print(f"Keywords found for {total - unsplittable - empty} cartoons:")
print(f"{unsplittable} unsplittable, {empty} empty")
print()

# See what the most common tags are
all_tags = [t for tags_for_cartoon in tags.values() for t in tags_for_cartoon]
tag_counts = Counter(all_tags)

most_common = [x[0].strip().lower() for x in tag_counts.most_common(_TOP_K)]
most_common = list(dict.fromkeys(most_common))
print(*most_common, sep='\n')
