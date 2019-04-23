# Example URL:
# http://editorialcartoonists.com/cartoon/browse.cfm/?membertype=Regular
# &cartoonist=Regular&datefrom=2018-01-01&orderby=publicationdate&sortorder=desc
# &totalcount=0&submit=Search

import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

_IN_FILE = "cartoon_ids.txt"
_BASE_URL = "http://editorialcartoonists.com/cartoon/display.cfm/"


def cartoon_request(cid: int) -> BeautifulSoup:
    cartoon_url = _BASE_URL + str(cid)
    r = requests.get(cartoon_url)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup


def parse_metadata(soup: BeautifulSoup) -> dict:

    return {}


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
