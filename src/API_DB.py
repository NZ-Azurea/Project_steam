from fastapi import FastAPI
from pydantic import BaseModel,constr
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError,DuplicateKeyError
from bson import json_util
from typing import Any,List, Dict
import json
import requests
import time
import httpx
from urllib.parse import urlencode, quote
import logging
from logging.config import dictConfig
from rich.logging import RichHandler
from rich.traceback import install as rich_traceback_install
from rich import pretty as rich_pretty
from datetime import datetime
# Pretty print and tracebacks in the whole app
rich_pretty.install()                       # nicer pprint for dicts/lists in logs
rich_traceback_install(show_locals=False)   # colored, clickable tracebacks
dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # Let RichHandler render; keep format minimal
        "rich": {
            "format": "%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "rich_console": {
            "class": "rich.logging.RichHandler",
            "formatter": "rich",
            "level": "INFO",
            # RichHandler kwargs:
            "rich_tracebacks": True,
            "markup": True,          # allow [bold], [green], etc. in messages
            "show_path": True,       # show module:line
            "enable_link_path": True # clickable paths in some terminals
        },
    },
    "loggers": {
        # Root logger (your app)
        "": {"handlers": ["rich_console"], "level": "INFO"},
        # Make uvicorn logs go through Rich too
        "uvicorn":         {"handlers": ["rich_console"], "level": "INFO", "propagate": False},
        "uvicorn.error":   {"handlers": ["rich_console"], "level": "INFO", "propagate": False},
        "uvicorn.access":  {"handlers": ["rich_console"], "level": "INFO", "propagate": False},
    },
})
logger = logging.getLogger(__name__)



# ---------- MongoDB CONFIG ----------

username = "NebuZard"
password = "7YAMTHHD"
uri = f"mongodb://{username}:{password}@127.0.0.1:27017/?authSource=admin"

# Short timeout so we don't block forever if Mongo is down
client = MongoClient(uri, serverSelectionTimeoutMS=2000)
db = client["Steam_Project"]

# ---------- FASTAPI APP ----------

app = FastAPI()


class CommandRequest(BaseModel):
    command: dict  # raw MongoDB command as a JSON object


class ApiResponse(BaseModel):
    message: Any
    Status: bool

NameStr = constr(strip_whitespace=True, min_length=1, max_length=64)

class UserCreate(BaseModel):
    name: NameStr

class UserDelete(BaseModel):
    name: NameStr

class UserRename(BaseModel):
    old_name: NameStr
    new_name: NameStr

def is_mongo_up() -> tuple[bool, str | None]:
    """
    Check if MongoDB is reachable.
    Returns (ok, error_message).
    """
    try:
        client.admin.command("ping")
        return True, None
    except ServerSelectionTimeoutError as e:
        return False, f"Cannot reach MongoDB (timeout): {e}"
    except PyMongoError as e:
        return False, f"MongoDB error: {e}"
    except Exception as e:
        return False, f"Unexpected error while pinging MongoDB: {e}"

@app.post("/command", response_model=ApiResponse)
def run_command(req: CommandRequest):
    """
    Run a MongoDB command sent by the client and return our wrapped JSON:
    {
      "message": <mongo_result or error>,
      "Status": true/false
    }
    """
    ok, err = is_mongo_up()
    if not ok:
        return ApiResponse(message=err, Status=False)

    try:
        # Example: { "ping": 1 } or { "serverStatus": 1 }
        result = db.command(req.command)

        # Convert BSON (ObjectId, dates, etc.) to JSON-safe types
        json_str = json_util.dumps(result)
        json_result = json.loads(json_str)

        return ApiResponse(message=json_result, Status=True)

    except Exception as e:
        # No HTTPException: we always return our own JSON envelope
        return ApiResponse(message=f"Error executing command: {e}", Status=False)

GETITEMS_URL = "https://api.steampowered.com/IStoreBrowseService/GetItems/v1"  # no trailing slash
CDN_BASE = "https://shared.steamstatic.com/store_item_assets/"

def _chunked(seq: List[int], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]
def _build_url(fmt: str, v: str) -> str:
    """Turn an asset filename into a full CDN URL using asset_url_format."""
    if not isinstance(v, str) or not v:
        return ""
    if v.startswith("http://") or v.startswith("https://"):
        return v
    # if it's clearly a filename/path, resolve via the format; else leave as-is
    if "${FILENAME}" in fmt and ("/" in v or "." in v):
        return CDN_BASE + fmt.replace("${FILENAME}", v)
    if "/" in v or "." in v:
        return CDN_BASE + v
    # e.g., bare hashes like community_icon -> leave as the raw value
    return v
def _extract_items(obj: Dict[str, Any]) -> list[dict]:
    resp = obj.get("response", {}) if isinstance(obj, dict) else {}
    items = resp.get("store_items")
    if isinstance(items, list):
        return items
    # legacy fallback
    items = resp.get("items")
    return items if isinstance(items, list) else []
def _extract_appid(item: Dict[str, Any]) -> int | None:
    if isinstance(item.get("appid"), int):
        return item["appid"]
    if isinstance(item.get("id"), dict) and isinstance(item["id"].get("appid"), int):
        return item["id"]["appid"]
    if isinstance(item.get("id"), int):
        return item["id"]
    return None
def _extract_assets(item: Dict[str, Any]) -> Dict[str, Any] | None:
    a = item.get("assets")
    if isinstance(a, dict):
        return a
    inner = item.get("item")
    if isinstance(inner, dict) and isinstance(inner.get("assets"), dict):
        return inner["assets"]
    return None
def _call_getitems(client: httpx.Client, payload: dict) -> dict:
    # Try POST (form-encoded) first
    r = client.post(
        GETITEMS_URL,
        data={"input_json": json.dumps(payload)},
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "assets-fetcher/1.0",
        },
    )
    if r.status_code == 200:
        return r.json()
    # Fallback to GET with URL-encoded input_json
    q = urlencode({"input_json": json.dumps(payload)})
    r = client.get(f"{GETITEMS_URL}?{q}", headers={"User-Agent": "assets-fetcher/1.0"})
    r.raise_for_status()
    return r.json()
def fetch_assets_for_appids(
    appids: List[int],
    country: str = "US",
    language: str = "english",
    batch_size: int = 100,
    timeout: float = 15.0,
    sleep_between_batches: float = 0.1,
) -> Dict[int, Dict[str, str]]:
    by_app: Dict[int, Dict[str, str]] = {}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for group in _chunked(appids, batch_size):
            payload = {
                "ids": [{"appid": a} for a in group],
                "context": {"country_code": country, "language": language},
                "data_request": {"include_assets": True},
            }
            raw = _call_getitems(client, payload)
            items = _extract_items(raw)

            for it in items:
                appid = _extract_appid(it)
                assets = _extract_assets(it)
                if appid is None or not isinstance(assets, dict):
                    continue

                fmt = assets.get("asset_url_format", "")
                urls: Dict[str, str] = {}
                for k, v in assets.items():
                    if k == "asset_url_format":
                        continue
                    if isinstance(v, str) and v:
                        u = _build_url(fmt, v)
                        if u:
                            urls[k] = u
                by_app[appid] = urls

            if sleep_between_batches:
                time.sleep(sleep_between_batches)

    # Return in the exact same order as input; missing apps -> {}
    return {appid: by_app.get(appid, {}) for appid in appids}

@app.get("/recommandation", response_model=ApiResponse)
def recommandation():
    """
    Run a MongoDB command sent by the client and return our wrapped JSON:
    {
      "message": <mongo_result or error>,
      "Status": true/false
    }
    """
    ok, err = is_mongo_up()
    if not ok:
        return ApiResponse(message=err, Status=False)

    try:
        # Example: { "ping": 1 } or { "serverStatus": 1 }
        result = db.games.find({},{}).sort({ "positive": -1, "_id": 1 }).limit(12)
        missing_asset = []
        result = [game for game in result]
        for game in result:
            if "assets" not in game or not game["assets"]:
                missing_asset.append(game["_id"])
        if missing_asset:
            logger.info(f"Fetching assets for missing appids: {missing_asset}")
            assets = fetch_assets_for_appids(missing_asset)
            for game in result:
                appid = game["_id"]
                if appid in assets:
                    game["assets"] = assets[appid]
                    db.games.update_one(
                        {"_id": appid},
                        {"$set": {"assets": assets[appid]}}
                    )
        
        return ApiResponse(message=result, Status=True)

    except Exception as e:
        # No HTTPException: we always return our own JSON envelope
        return ApiResponse(message=f"Error executing command: {e}", Status=False)

@app.post("/Game", response_model=ApiResponse)
def get_Game(game_id: int):
    """
    Run a MongoDB command sent by the client and return our wrapped JSON:
    {
      "message": <mongo_result or error>,
      "Status": true/false
    }
    """
    ok, err = is_mongo_up()
    if not ok:
        return ApiResponse(message=err, Status=False)

    try:
        # Example: { "ping": 1 } or { "serverStatus": 1 }
        result = db.games.find({},{}).sort({ "positive": -1, "_id": 1 }).limit(1)
        missing_asset = []
        result = [game for game in result]
        for game in result:
            if "assets" not in game or not game["assets"]:
                missing_asset.append(game["_id"])
        if missing_asset:
            logger.info(f"Fetching assets for missing appids: {missing_asset}")
            assets = fetch_assets_for_appids(missing_asset)
            for game in result:
                appid = game["_id"]
                if appid in assets:
                    game["assets"] = assets[appid]
                    db.games.update_one(
                        {"_id": appid},
                        {"$set": {"assets": assets[appid]}}
                    )
        
        return ApiResponse(message=result[0], Status=True)

    except Exception as e:
        # No HTTPException: we always return our own JSON envelope
        return ApiResponse(message=f"Error executing command: {e}", Status=False)

@app.post("/user", response_model=ApiResponse)
def create_user(payload: UserCreate):
    ok, err = is_mongo_up()
    if not ok:
        return ApiResponse(message=err, Status=False)

    name = payload.name
    doc = {"name": name, "created_at": datetime.utcnow()}

    try:
        res = db.users.insert_one(doc)
        out = {"_id": str(res.inserted_id), "name": name}
        return ApiResponse(message=out, Status=True)
    except DuplicateKeyError:
        return ApiResponse(message=f"User '{name}' already exists.", Status=False)
    except Exception as e:
        return ApiResponse(message=f"Error creating user: {e}", Status=False)

@app.delete("/user", response_model=ApiResponse)
def delete_user(payload: UserDelete):
    ok, err = is_mongo_up()
    if not ok:
        return ApiResponse(message=err, Status=False)

    name = payload.name
    try:
        res = db.users.delete_one({"name": name})
        if res.deleted_count == 0:
            return ApiResponse(message=f"User '{name}' not found.", Status=False)
        return ApiResponse(message=f"User '{name}' deleted.", Status=True)
    except Exception as e:
        return ApiResponse(message=f"Error deleting user: {e}", Status=False)
    
@app.patch("/user", response_model=ApiResponse)
def rename_user(payload: UserRename):
    ok, err = is_mongo_up()
    if not ok:
        return ApiResponse(message=err, Status=False)

    if payload.old_name == payload.new_name:
        return ApiResponse(message="Old and new names are identical.", Status=False)

    try:
        res = db.users.update_one(
            {"name": payload.old_name},
            {"$set": {"name": payload.new_name, "updated_at": datetime.utcnow()}}
        )
        if res.matched_count == 0:
            return ApiResponse(message=f"User '{payload.old_name}' not found.", Status=False)
        # If unique index exists, DuplicateKeyError would have been raised before matched_count>0,
        # but in some race cases it can be thrown—handle it explicitly above if needed.
        return ApiResponse(
            message=f"User '{payload.old_name}' renamed to '{payload.new_name}'.",
            Status=True
        )
    except DuplicateKeyError:
        return ApiResponse(
            message=f"User '{payload.new_name}' already exists.",
            Status=False
        )
    except Exception as e:
        return ApiResponse(message=f"Error renaming user: {e}", Status=False)
@app.get("/user", response_model=ApiResponse)
def get_user(name: NameStr):
    """
    Fetch a user document by name.
    Returns { "message": <user_doc>, "Status": true } when found,
    else { "message": "...not found", "Status": false }.
    """
    ok, err = is_mongo_up()
    if not ok:
        return ApiResponse(message=err, Status=False)

    try:
        doc = db.users.find_one({"name": name})
        if not doc:
            return ApiResponse(message=f"User '{name}' not found.", Status=False)

        # Convert ObjectId to str for JSON-compat
        doc["_id"] = str(doc["_id"])
        return ApiResponse(message=doc, Status=True)

    except Exception as e:
        return ApiResponse(message=f"Error fetching user: {e}", Status=False)
@app.get("/health", response_model=ApiResponse)
def health():
    """
    Health check endpoint using the same JSON envelope.
    """
    ok, err = is_mongo_up()
    if not ok:
        return ApiResponse(message=err, Status=False)

    return ApiResponse(
        message={"fastapi": "ok", "mongodb": "ok"},
        Status=True
    )
