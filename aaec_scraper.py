import requests
from bs4 import BeautifulSoup
import re

# http://editorialcartoonists.com/cartoon/browse.cfm/?membertype=Regular&cartoonist=Regular&datefrom=2018-01-01&orderby=publicationdate&sortorder=desc&totalcount=0&submit=Search

_SEARCH_URL = "http://editorialcartoonists.com/cartoon/browse.cfm"
_SEARCH_PAYLOAD = {"membertype": "Regular",
                   "cartoonist": "Regular",
                   "orderby": "publicationdate",
                   "sortorder": "desc",
                   "submit": "Search"}

_CARTOON_URL_REGEX = re.compile(r'/cartoon/display.cfm/[0-9]*/')
_TOTAL_COUNT_URL_REGEX = re.compile(r'totalcount')
_TOTAL_COUNT_PARAM_REGEX = re.compile(r'totalcount=([0-9]*)')


def parse_cartoon_urls(soup: BeautifulSoup):
    soup.find_all('a')
    cartoon_urls = soup.find_all('a', href=_CARTOON_URL_REGEX)
    return [a['href'] for a in cartoon_urls]


def parse_total_count(soup: BeautifulSoup):
    total_count_url = soup.find('a', href=_TOTAL_COUNT_URL_REGEX)
    return int(_TOTAL_COUNT_PARAM_REGEX.search(total_count_url['href']).groups()[0])


def parse():
    cartoon_urls = []

    for year in range(2019, 2020):
        date_from = str(year) + "-01-01"

        i = 1
        total_count = 1
        while i <= total_count:
            payload = {**_SEARCH_PAYLOAD, **{"datefrom": date_from, "count": i}}
            r = requests.get(_SEARCH_URL, payload)

            soup = BeautifulSoup(r.content, 'html.parser')
            cartoon_urls.append(parse_cartoon_urls(soup))
            total_count = parse_total_count(soup)

            # 5 cartoon urls per page
            i += 5

    print(cartoon_urls)


parse()
