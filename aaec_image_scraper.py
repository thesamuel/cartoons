import json
import os
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

_NUM_THREADS = 100

_ID_FILE = "cartoon_ids.txt"
_DATA_PATH = Path("./data")

_BASE_URL = "http://editorialcartoonists.com/cartoon/display.cfm/"
_METADATA_HEADERS = ("Cartoon Title", "Keywords", "Caption")


def parse_description(soup: BeautifulSoup, header: str) -> Optional[str]:
    header_soup = soup.find(text=header + ":")
    if not header_soup:
        return None
    description = header_soup.find_parent().next_sibling
    if not description:
        return None
    return description.strip()


def parse_metadata(soup: BeautifulSoup) -> dict:
    img = soup.find("img", {"name": "Toon"})
    if not img:
        raise Exception("No image url found.")

    title, keywords, caption = (parse_description(soup, h) for h in _METADATA_HEADERS)
    keywords = keywords.split(", ") if keywords else None
    return {"title": title, "keywords": keywords, "caption": caption, "image_url": img["src"]}


def download_cartoon(cid: int) -> Optional[str]:
    cartoon_url = _BASE_URL + str(cid)
    try:
        # Request cartoon page
        r = requests.get(cartoon_url)
        soup = BeautifulSoup(r.content, 'html.parser')

        # Parse metadata from cartoon page
        metadata = parse_metadata(soup)

        # Download cartoon image
        image = requests.get(metadata["image_url"]).content
    except Exception as e:
        return f"Error occurred while downloading cartoon {cid}: {e}"

    image_path = _DATA_PATH / f"{cid}.jpg"
    metadata_path = _DATA_PATH / f"{cid}.json"
    try:
        # Save files
        with open(image_path, 'wb') as f:
            f.write(image)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    except:  # Handle any interruption while writing
        # Remove partially-saved files
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return f"Error occurred while saving cartoon {cid}"

    return None


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def download_batch(cartoon_ids: set):
    pool = ThreadPool(_NUM_THREADS)
    with tqdm(total=len(cartoon_ids), desc="Downloading cartoons") as t:
        num_errors = 0
        for err in pool.imap_unordered(download_cartoon, cartoon_ids):
            if err:
                t.write(err)
                num_errors += 1
                t.set_description(f"Downloading cartoons ({num_errors} errors)")
            t.update(1)


def download_cartoons_from_file(filename: str):
    # Get all cartoon ids from file
    lines = tuple(open(filename, 'r'))
    cartoon_ids = set(int(line.strip()) for line in lines
                      if line.strip() and not line.startswith('#'))

    # Remove all previously downloaded ids
    downloaded_ids = set(int(os.path.splitext(filename)[0]) for filename in os.listdir(_DATA_PATH))
    cartoon_ids = list(cartoon_ids - downloaded_ids)

    for batch in batches(cartoon_ids, 500):
        download_batch(batch)
        time.sleep(15)


if __name__ == "__main__":
    download_cartoons_from_file(_ID_FILE)
