import json, httpx, urllib.parse as up

GETITEMS = "https://api.steampowered.com/IStoreBrowseService/GetItems/v1"  # no trailing slash

payload = {
    "ids": [{"appid": 570}, {"appid": 730}],
    "context": {"country_code": "US", "language": "english"},
    "data_request": {"include_assets": True}
}

q = up.urlencode({"input_json": json.dumps(payload)})
url = f"{GETITEMS}?{q}"

with httpx.Client(timeout=15.0, follow_redirects=True, headers={"User-Agent": "assets-sanity/1.0"}) as c:
    r = c.get(url)
    print("STATUS:", r.status_code)
    r.raise_for_status()
    print(r.json())
