import json
import os
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


def parse_metadata(soup: BeautifulSoup) -> Optional[dict]:
    image_url = soup.find("img", {"name": "Toon"})["src"]
    title, keywords, caption = (parse_description(soup, h) for h in _METADATA_HEADERS)
    keywords = keywords.split(", ") if keywords else None
    return {"title": title, "keywords": keywords, "caption": caption, "image_url": image_url}


def download_cartoon(cid: int):
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
        print(f"Error occurred while downloading cartoon {cid}:")
        print(e)
        return

    image_path = _DATA_PATH / f"{cid}.jpg"
    metadata_path = _DATA_PATH / f"{cid}.json"
    try:
        # Save files
        with open(image_path, 'wb') as f:
            f.write(image)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    except:
        # Remove partially-saved files
        print("Removing any partially-saved files")
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)


def download_cartoons_from_file(filename: str):
    # Get all cartoon ids from file
    lines = tuple(open(filename, 'r'))
    cartoon_ids = set(int(line.strip()) for line in lines
                      if line.strip() and not line.startswith('#'))

    # Remove all previously downloaded ids
    downloaded_ids = set(int(os.path.splitext(filename)[0]) for filename in os.listdir(_DATA_PATH))
    cartoon_ids -= downloaded_ids

    # Download cartoons concurrently
    pool = ThreadPool(_NUM_THREADS)
    with tqdm(total=len(cartoon_ids)) as t:
        for _ in pool.imap_unordered(download_cartoon, cartoon_ids):
            t.update(1)


if __name__ == "__main__":
    download_cartoons_from_file(_ID_FILE)
