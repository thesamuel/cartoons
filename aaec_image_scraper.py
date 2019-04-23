# Example URL:
# http://editorialcartoonists.com/cartoon/browse.cfm/?membertype=Regular
# &cartoonist=Regular&datefrom=2018-01-01&orderby=publicationdate&sortorder=desc
# &totalcount=0&submit=Search

import requests
from typing import Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from pathlib import Path

_ID_FILE = "cartoon_ids.txt"
_DATA_PATH = Path("./data")

_BASE_URL = "http://editorialcartoonists.com/cartoon/display.cfm/"
_METADATA_HEADERS = ("Cartoon Title", "Keywords", "Caption")


def cartoon_request(cid: int) -> BeautifulSoup:
    cartoon_url = _BASE_URL + str(cid)
    r = requests.get(cartoon_url)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup


def parse_metadata(soup: BeautifulSoup) -> Optional[dict]:
    image_url = soup.find("img", {"name": "Toon"})["src"]
    title, keywords, caption = (soup.find(text=h + ":")
                                    .find_parent()
                                    .next_sibling
                                    .strip()
                                for h in _METADATA_HEADERS)
    keywords = keywords.split(", ")
    return {"title": title, "keywords": keywords,
            "caption": caption, "image_url": image_url}


def save(id: int, metadata: dict):
    # Save image
    image_path = _DATA_PATH / f"{id}.jpg"
    with open(image_path, 'wb') as f:
        f.write(requests.get(metadata["image_url"]).content)

    # Save metadata
    metadata_path = _DATA_PATH / f"{id}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def parse():
    downloaded_ids = set()

    with open(_ID_FILE, 'r') as f:
        lines = f.readlines()
        try:
            for line in tqdm(lines):
                # Ignore comments
                if line.startswith('#'):
                    continue

                # Ignore blank lines
                cid = line.strip()
                if not cid:
                    continue

                # Ignore duplicated ids
                if cid in downloaded_ids:
                    print(cid, "already downloaded; skipping")
                    continue

                # Download cartoon webpage
                cid = int(cid)
                soup = cartoon_request(cid)

                # Parse and save cartoon
                metadata = parse_metadata(soup)
                save(cid, metadata)
        except:
            print("AAEC image scraper crashed; printing already downloaded ids")
            print(downloaded_ids)


if __name__ == "__main__":
    parse()
