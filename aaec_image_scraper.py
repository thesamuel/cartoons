# Example URL:
# http://editorialcartoonists.com/cartoon/browse.cfm/?membertype=Regular
# &cartoonist=Regular&datefrom=2018-01-01&orderby=publicationdate&sortorder=desc
# &totalcount=0&submit=Search

import requests
from typing import Optional
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

_IN_FILE = "cartoon_ids.txt"
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
    return {"title": title, "keywords": keywords, "caption": caption, "image_url": image_url}


def parse():
    with open(_IN_FILE, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            cid = line.strip()
            if not cid:
                continue

            cid = int(cid)
            soup = cartoon_request(cid)

            metadata = parse_metadata(soup)


if __name__ == "__main__":
    parse()
