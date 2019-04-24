import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse
from shutil import copyfile

_DATA_DIR = Path('data')
_SORTED_DATA_DIR = Path('data-sorted')

# Get all of the available tags
tags = {}
for filename in os.listdir(_DATA_DIR):
    parts = filename.split(os.extsep)
    cid = parts[0]
    ext = parts[-1]

    local_image_filename = cid + ".jpg"

    if ext != 'json':
        continue

    with open(_DATA_DIR / filename, 'r') as f:
        metadata = json.load(f)
        url = urlparse(metadata["image_url"])
        author, year, image_filename = url.path.split(os.sep)[2:]
        year = int(year)

        # Parse date from image filename
        date = re.search(r'[0-9]{8}', image_filename)[0]
        year_2, month, day = int(date[:4]), int(date[4:6]), int(date[6:])
        assert year == year_2

        os.makedirs(_SORTED_DATA_DIR / author, exist_ok=True)
        copyfile(_DATA_DIR / local_image_filename, _SORTED_DATA_DIR / author / local_image_filename)



