# Example URL:
# http://editorialcartoonists.com/cartoon/browse.cfm/?membertype=Regular
# &cartoonist=Regular&datefrom=2018-01-01&orderby=publicationdate&sortorder=desc
# &totalcount=0&submit=Search

import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

# 5 cartoon urls per search page
_CARTOONS_PER_PAGE = 5

_SEARCH_URL = "http://editorialcartoonists.com/cartoon/browse.cfm"
_SEARCH_PAYLOAD = {
    "membertype": "Regular",
    "cartoonist": "Regular",
    "orderby": "publicationdate",
    "sortorder": "desc",
    "submit": "Search"
}

_CARTOON_URL_REGEX = re.compile(r'/cartoon/display.cfm/([0-9]*)/')
_TOTAL_COUNT_URL_REGEX = re.compile(r'totalcount')
_TOTAL_COUNT_PARAM_REGEX = re.compile(r'totalcount=([0-9]*)')


def search_request(date_from: str, offset: int = 1) -> BeautifulSoup:
    assert offset >= 1
    payload = {**_SEARCH_PAYLOAD, **{"datefrom": date_from, "count": offset}}
    r = requests.get(_SEARCH_URL, payload)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup


def parse_cartoon_ids(soup: BeautifulSoup) -> int:
    soup.find_all('a')
    cartoon_urls = soup.find_all('a', href=_CARTOON_URL_REGEX)
    return [a['href'] for a in cartoon_urls]


def parse_total_count(soup: BeautifulSoup) -> int:
    total_count_url = soup.find('a', href=_TOTAL_COUNT_URL_REGEX)
    groups = _TOTAL_COUNT_PARAM_REGEX.search(total_count_url['href']).groups()
    return int(groups[0]) if len(groups) > 0 else 0


def parse():
    cartoon_ids = set()

    for year in range(2019, 2020):
        date_from = str(year) + "-01-01"

        soup = search_request(date_from)
        total_count = parse_total_count(soup)

        if total_count == 0:
            return

        for offset in tqdm(range(1, total_count, _CARTOONS_PER_PAGE)):
            soup = search_request(date_from, offset)
            cartoon_ids.add(parse_cartoon_ids(soup))

    print(cartoon_ids)


if __name__ == "__main__":
    parse()
