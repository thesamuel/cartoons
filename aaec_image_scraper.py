# Example URL:
# http://editorialcartoonists.com/cartoon/browse.cfm/?membertype=Regular
# &cartoonist=Regular&datefrom=2018-01-01&orderby=publicationdate&sortorder=desc
# &totalcount=0&submit=Search

import json
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

_NUM_THREADS = 16

_ID_FILE = "cartoon_ids.txt"
_DATA_PATH = Path("./data")

_BASE_URL = "http://editorialcartoonists.com/cartoon/display.cfm/"
_METADATA_HEADERS = ("Cartoon Title", "Keywords", "Caption")


def cartoon_request(cid: int) -> BeautifulSoup:
    cartoon_url = _BASE_URL + str(cid)
    r = requests.get(cartoon_url)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup


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


def save(id: int, metadata: dict):
    # Save image
    image_path = _DATA_PATH / f"{id}.jpg"
    with open(image_path, 'wb') as f:
        f.write(requests.get(metadata["image_url"]).content)

    # Save metadata
    metadata_path = _DATA_PATH / f"{id}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def download_cartoon(cid: int):
    # Parse and save cartoon
    soup = cartoon_request(cid)
    metadata = parse_metadata(soup)
    save(cid, metadata)


def parse():
    # Get all cartoon ids from file
    lines = tuple(open(_ID_FILE, 'r'))
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
    parse()
