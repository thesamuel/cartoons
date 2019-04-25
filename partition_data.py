import random
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse
from shutil import copyfile
from tqdm import tqdm

_DATA_DIR = Path('./data')
_SORTED_DATA_DIR = Path('./data-sorted')

filenames = set(os.listdir(_DATA_DIR))
downloaded_ids = list(set(int(filename.split(os.extsep)[0])
                          for filename in os.listdir(_DATA_DIR)))

# Split the data
random.seed(1996)
random.shuffle(downloaded_ids)

split = int(0.8 * len(downloaded_ids))
train_ids = set(downloaded_ids[:split])
val_ids = set(downloaded_ids[split:])

for cid in tqdm(downloaded_ids):
    try:
        image_filename = f"{cid}.jpg"

        json_filename = f"{cid}.json"
        if json_filename not in filenames:
            json_filename = f"{cid}.v2.json"

        with open(_DATA_DIR / json_filename, 'r') as f:
            metadata = json.load(f)
            url = urlparse(metadata["image_url"])
            author, year, web_image_filename = url.path.split(os.sep)[2:]
            year = int(year)

            # Parse date from image filename
            # FIXME: some usernames end in a "1", so this regex doesn't work
            date = re.search(r'[0-9]{8}', web_image_filename)[0]
            year_2, month, day = int(date[:4]), int(date[4:6]), int(date[6:])

            # Add train or val to path
            sorted_path = _SORTED_DATA_DIR
            if cid in train_ids:
                sorted_path = sorted_path / "train"
            elif cid in val_ids:
                sorted_path = sorted_path / "val"
            else:
                raise Exception()

            sorted_path = sorted_path / author
            os.makedirs(sorted_path, exist_ok=True)
            copyfile(_DATA_DIR / image_filename, sorted_path / image_filename)
    except Exception as e:
        print(e)
        print("error parsing", json_filename)
        continue
