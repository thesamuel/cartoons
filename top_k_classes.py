import json
import os
from pathlib import Path
from collections import Counter

_DATA_DIR = Path('data')

def split_tags(tags: str) -> list:
    return [t.strip().lower() for t in tags.split(',')]

# Get all of the available tags
tags = {}
for filename in os.listdir(_DATA_DIR):
    parts = os.path.splitext(filename)
    cid = parts[0]
    ext = parts[1]

    if ext != '.json':
        continue

    with open(_DATA_DIR / filename, 'r') as f:
        metadata = json.load(f)
        keywords = metadata['keywords']
        if keywords:
            if not isinstance(keywords, str):
                # Fix old keword splitting scheme
                keywords = ", ".join(keywords)

            tags[cid] = split_tags(keywords)

# See what the most common tags are
all_tags = [t for tags_for_cartoon in tags.values() for t in tags_for_cartoon]
tag_counts = Counter(all_tags)

most_common = [x[0].strip().lower() for x in tag_counts.most_common(1000)]
most_common = list(dict.fromkeys(most_common))
print(*most_common, sep='\n')
