"""
Our data was collected in a non-ideal way. To compensate for this, we have created this
cleaning script to generate better JSON files for the data.

Rather than overwriting the existing JSON files, this script copies them to a new folder.

Also, this script will attempt to open each associated JPEG file, and skip those which
were corrupted during the scraping process.
"""

import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse
from shutil import copyfile
from tqdm import tqdm
from PIL import Image

_IN_DATA_DIR = Path('./scraped-data')
_OUT_DATA_DIR = Path('./raw-data')
_RANDOM_SEED = 2019

_DATE_REGEX = re.compile(r'(?=([0-9]{8}))')


# Reference: https://opensource.com/article/17/2/python-tricks-artists
def is_readable(image_path: str) -> bool:
    try:
        img = Image.open(image_path)  # open the image file
        img.verify()  # verify that it is, in fact an image
        return True
    except (IOError, SyntaxError):
        return False


def update_metadata(metadata: dict) -> dict:
    metadata = metadata.copy()

    url = urlparse(metadata["image_url"])
    author, year, web_image_filename = url.path.split(os.sep)[2:]
    year = int(year)

    # Parse date from image filename
    date = _DATE_REGEX.findall(web_image_filename)[-1]
    year_2, month, day = int(date[:4]), int(date[4:6]), int(date[6:])

    if year != year_2:
        raise RuntimeError(f"path year ({year}) did not equal filename year ({year_2})")

    # Update dictionary with author and date
    metadata["author"] = author
    metadata["date"] = f"{year}-{month}-{day}"

    # Reverse old keyword splitting scheme
    if isinstance(metadata["keywords"], list):
        metadata["keywords"] = ", ".join(metadata["keywords"])

    # Replace all empty strings with None
    return {key: (value if value != "" else None) for (key, value) in metadata.items()}


# Read filenames from input directory and parse ids
filenames = set(os.listdir(_IN_DATA_DIR))
ids = list(set(int(filename.split(os.extsep)[0]) for filename in filenames))

# Make output directory
os.makedirs(_OUT_DATA_DIR)

# Copy data to output path in PyTorch's ImageFolder structure
for cid in tqdm(ids):
    # Get filenames and paths
    image_filename = f"{cid}.jpg"
    input_image_path = _IN_DATA_DIR / image_filename

    json_filename = f"{cid}.json"
    input_json_path = _IN_DATA_DIR / (json_filename if json_filename in filenames else f"{cid}.v2.json")

    # Ensure that paths exist
    if not os.path.exists(input_json_path):
        tqdm.write(f"{cid} json missing")
        continue

    if not os.path.exists(input_image_path):
        tqdm.write(f"{cid} json missing")
        continue

    # Ensure that image is readable
    if not is_readable(input_image_path):
        tqdm.write(f"{cid} image was not readable")
        continue

    # Read and update metadata
    metadata = json.load(open(input_json_path, 'r'))
    try:
        metadata = update_metadata(metadata)
    except RuntimeError as e:
        tqdm.write(f"{cid} metadata malformed: {e}")
        continue

    # Copy image file
    copyfile(input_image_path, _OUT_DATA_DIR / image_filename)

    # Write new metadata file
    with open(_OUT_DATA_DIR / json_filename, 'w') as f:
        json.dump(metadata, f)
