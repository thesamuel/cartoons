import json
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

_NUM_THREADS = 20

_ID_FILE = "cartoon_ids.txt"
_DATA_PATH = Path("./scraped-data")

_CARTOON_URL = "http://editorialcartoonists.com/cartoon/display.cfm/"
_METADATA_HEADERS = ("Cartoon Title", "Keywords", "Caption")
_TIMEOUT = 20


def parse_image_url(soup: BeautifulSoup) -> str:
    img = soup.find("img", {"name": "Toon"})
    if not img:
        raise Exception("No image url found.")

    img_url = urlparse(img["src"])
    if not img_url.netloc:
        # If no domain was specified, its a relative url for the new ".com" domain.
        # Otherwise, its an absolute url for the old ".org" domain.
        img_url = img_url._replace(scheme='http', netloc="editorialcartoonists.com")

    return img_url.geturl()


def parse_description(soup: BeautifulSoup, header: str) -> Optional[str]:
    # Find the description header (ie. "Title:")
    header_soup = soup.find(text=header + ":")
    if not header_soup:
        return None

    # Description will be adjacent to the header
    description = header_soup.find_parent().next_sibling
    if not description:
        return None

    # Return whitespace-stripped description
    return description.strip()


def parse_metadata(soup: BeautifulSoup) -> dict:
    image_url = parse_image_url(soup)
    title, keywords, caption = (parse_description(soup, h) for h in _METADATA_HEADERS)
    return {"title": title, "keywords": keywords, "caption": caption, "image_url": image_url}


def download_cartoon(cid: int) -> Optional[str]:
    cartoon_url = _CARTOON_URL + str(cid)
    try:
        # Request cartoon page
        r = requests.get(cartoon_url, timeout=_TIMEOUT)
        soup = BeautifulSoup(r.content, 'html.parser')

        # Parse metadata from cartoon page
        metadata = parse_metadata(soup)

        # Download cartoon image
        image = requests.get(metadata["image_url"], timeout=_TIMEOUT).content
    except Exception as e:
        return f"Error occurred while downloading cartoon {cid}: {e}"

    image_path = _DATA_PATH / f"{cid}.jpg"
    metadata_path = _DATA_PATH / f"{cid}.v2.json"
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


def download_cartoons_from_file(filename: str):
    # Get all cartoon ids from file
    lines = tuple(open(filename, 'r'))
    cartoon_ids = set(int(line.strip()) for line in lines
                      if line.strip() and not line.startswith('#'))

    # Remove all previously downloaded ids
    # Uses os.extsep rather than os.path.splitext because some files
    # have multiple extensions (ie. '1003.v2.json')
    downloaded_ids = set(int(filename.split(os.extsep)[0])
                         for filename in os.listdir(_DATA_PATH))
    cartoon_ids = list(cartoon_ids - downloaded_ids)

    # Download all cartoons
    pool = ThreadPool(_NUM_THREADS)
    with tqdm(total=len(cartoon_ids), desc="Downloading cartoons") as t:
        num_errors = 0
        for err in pool.imap_unordered(download_cartoon, cartoon_ids):
            if err:
                t.write(err)
                num_errors += 1
                t.set_description(f"Downloading cartoons ({num_errors} errors)")
            t.update(1)


if __name__ == "__main__":
    download_cartoons_from_file(_ID_FILE)
