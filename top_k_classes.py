import json
import os
from pathlib import Path
from collections import Counter

_DATA_DIR = Path('./scraped-data')
_TOP_K = 1000


def split_keywords(keywords: str) -> list:
    return [t.strip().lower() for t in keywords.split(',')]


# Get all of the available tags
tags = {}
for filename in os.listdir(_DATA_DIR):
    cid, ext = os.path.splitext(filename)

    if ext != '.json':
        continue

    with open(_DATA_DIR / filename, 'r') as f:
        metadata = json.load(f)
        keywords = metadata['keywords']
        if keywords:
            tags[cid] = split_keywords(keywords)

            if "," not in keywords:
                print(keywords)

# See what the most common tags are
all_tags = [t for tags_for_cartoon in tags.values() for t in tags_for_cartoon]
tag_counts = Counter(all_tags)

most_common = [x[0].strip().lower() for x in tag_counts.most_common(_TOP_K)]
most_common = list(dict.fromkeys(most_common))
print(*most_common, sep='\n')
