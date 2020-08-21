#!/usr/bin/python3

"""
    get_allpages.py

    MediaWiki API Demos
    Demo of `Allpages` module: Get all pages whose title contains the text
    "Jungle," in whole or part.

    MIT License
"""

import requests
import yaml

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

p0 = "Þórfi"

flag = True
start = True
titles = []

PARAMS = {"action": "query",
          "format": "json",
          "list": "allpages",
          "aplimit": 500}

while flag:

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    PAGES = DATA["query"]["allpages"]
    titles += [x['title'] for x in PAGES]
    PARAMS['apfrom'] = titles[-1]

    print(f"-- Downloading ... {titles[-1]}                        \r", end='')
    if len(PAGES) < 500:
        flag = False

breakpoint()

# Save categories to file
with open('wikipedia_titles.yml', 'w', encoding="utf-8") as f:
    yaml.dump(PAGES, f, default_flow_style=False, sort_keys=True)

# for page in PAGES:
#     print(page["title"])

