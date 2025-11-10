#!/usr/bin/env python3
"""
Import games (JSON) and reviews (CSVs) into MongoDB with PyMongo.
- Games -> collection 'games'
- Reviews -> collection 'reviews'
Field parsing:
  - post_date: "October 22, 2024" -> datetime
  - recommend: "Recommended"/"Not Recommended" -> True/False
  - early_access_review: Null or "Early Access Review" -> False/True
  - playtime: float
  - helpfulness: int
"""

from pathlib import Path
from datetime import datetime
import json
import csv
from typing import Optional, List

from pymongo import MongoClient, UpdateOne, InsertOne, ASCENDING, DESCENDING
from pymongo.errors import BulkWriteError


# =========================
# ====== CONFIG HERE ======
# =========================
# Option A: Full connection string (recommended if you have one).
MONGO_URI = ""  # e.g. "mongodb://USERNAME:PASSWORD@localhost:27017/?authSource=admin"

# Option B: Or fill host/port/user/pass if you leave MONGO_URI empty.
HOST = "localhost"
PORT = 27017
USERNAME = "NebuZard"
PASSWORD = "7YAMTHHD"
AUTH_SOURCE = "admin"

# Target database and collections
DB_NAME = "Steam_Project"
GAMES_COLLECTION = "games"
REVIEWS_COLLECTION = "reviews"

# Paths (Windows-friendly; space in folder name is fine)
GAMES_JSON_PATH = Path("data") / "games.json"
REVIEWS_DIR = Path("data") / "Game Reviews"

# Batch size for bulk inserts
BATCH_SIZE = 1000
# =========================


def parse_date_mdy_long(s: Optional[str]) -> Optional[datetime]:
    """Parse 'October 22, 2024' (and a couple fallbacks) -> datetime."""
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.strptime(s, "%B %d, %Y")
    except ValueError:
        for fmt in ("%b %d, %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
    return None


def coerce_bool_recommend(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    t = s.strip().lower()
    if t == "recommended":
        return True
    if t == "not recommended":
        return False
    return None  # unexpected value


def coerce_bool_early_access(s: Optional[str]) -> bool:
    # Null/empty -> False; "Early Access Review" -> True
    if not s:
        return False
    return s.strip().lower() == "early access review"


def load_games_array(games_json_path: Path) -> List[dict]:
    """
    Load games from JSON. Accepts either:
      - an array of documents, or
      - a {id: {...}, id2: {...}} map and converts to array with _id set.
    Tries to parse release_date if it's a human string.
    """
    with games_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [
            {**v, "_id": int(k) if str(k).isdigit() else k}
            for k, v in data.items()
        ]

    for doc in data:
        rd = doc.get("release_date")
        if isinstance(rd, str):
            parsed = parse_date_mdy_long(rd)
            if parsed:
                doc["release_date"] = parsed
    return data


def import_games(db):
    col = db[GAMES_COLLECTION]
    print(f"[games] Loading: {GAMES_JSON_PATH}")
    games = load_games_array(GAMES_JSON_PATH)
    if not games:
        print("[games] No documents found.")
        return

    ops = []
    for g in games:
        if "_id" in g:
            ops.append(UpdateOne({"_id": g["_id"]}, {"$set": g}, upsert=True))
        else:
            ops.append(InsertOne(g))

    print(f"[games] Bulk writing {len(ops)} ops…")
    res = col.bulk_write(ops, ordered=False)
    upserted = getattr(res, "upserted_count", 0) or 0
    modified = getattr(res, "modified_count", 0) or 0
    inserted = getattr(res, "inserted_count", 0) or 0
    print(f"[games] upserted: {upserted}, modified: {modified}, inserted: {inserted}")

    # Indexes
    print("[games] Ensuring indexes…")
    col.create_index([("name", "text")])
    col.create_index([("price", ASCENDING)])
    print("[games] Indexes ready.")


def import_reviews(db):
    col = db[REVIEWS_COLLECTION]
    reviews_dir = REVIEWS_DIR
    if not reviews_dir.exists():
        print(f"[reviews] Directory not found: {reviews_dir}")
        return

    files = sorted(reviews_dir.glob("*.csv"))
    if not files:
        print("[reviews] No CSV files found.")
        return

    total_inserted = 0
    for csv_path in files:
        stem = csv_path.stem  # "<app_id>_<review_Count>"
        try:
            app_id = int(stem.split("_", 1)[0])
        except Exception:
            print(f"[reviews] Skip (cannot parse app_id): {csv_path.name}")
            continue

        print(f"[reviews] Importing {csv_path.name} (app_id={app_id})…")
        batch = []
        inserted_for_file = 0

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # user: str
                user = (row.get("user") or "").strip()
                # playtime: float
                playtime = None
                raw_playtime = row.get("playtime")
                if raw_playtime not in (None, ""):
                    try:
                        playtime = float(raw_playtime)
                    except ValueError:
                        playtime = None
                # post_date: parse "October 22, 2024"
                post_date = parse_date_mdy_long(row.get("post_date"))
                # helpfulness: int
                helpfulness = None
                raw_help = row.get("helpfulness")
                if raw_help not in (None, ""):
                    try:
                        helpfulness = int(raw_help)
                    except ValueError:
                        helpfulness = None
                # review: str
                review_text = (row.get("review") or "").strip()
                # recommend: "Recommended"/"Not Recommended" -> bool
                recommend = coerce_bool_recommend(row.get("recommend"))
                # early_access_review: Null or "Early Access Review" -> bool
                early_access = coerce_bool_early_access(row.get("early_access_review"))

                doc = {
                    "app_id": app_id,
                    "user": user or None,
                    "playtime": playtime,                    # float
                    "post_date": post_date,                  # datetime
                    "helpfulness": helpfulness,              # int
                    "review_text": review_text or None,      # str
                    "recommend": recommend,                  # bool
                    "early_access": early_access,            # bool
                    "source_file": csv_path.name,
                }
                # strip None values for cleanliness
                doc = {k: v for k, v in doc.items() if v is not None}

                batch.append(InsertOne(doc))
                if len(batch) >= BATCH_SIZE:
                    try:
                        res = col.bulk_write(batch, ordered=False)
                        inserted_now = getattr(res, "inserted_count", 0) or 0
                        inserted_for_file += inserted_now
                        total_inserted += inserted_now
                    except BulkWriteError as bwe:
                        print(f"[reviews] Bulk write error: {bwe.details}")
                    batch = []

        if batch:
            try:
                res = col.bulk_write(batch, ordered=False)
                inserted_now = getattr(res, "inserted_count", 0) or 0
                inserted_for_file += inserted_now
                total_inserted += inserted_now
            except BulkWriteError as bwe:
                print(f"[reviews] Bulk write error (final batch): {bwe.details}")

        print(f"[reviews] {csv_path.name}: inserted ~{inserted_for_file} docs")

    # Indexes for reviews
    print("[reviews] Ensuring indexes…]")
    col.create_index([("app_id", ASCENDING), ("post_date", DESCENDING)])
    print("[reviews] Created index on (app_id, post_date DESC)")
    col.create_index([("app_id", ASCENDING), ("recommend", ASCENDING)])
    print("[reviews] Created index on (app_id, recommend)")
    col.create_index([("user", ASCENDING), ("app_id", ASCENDING)])
    # Text index on review_text (optional; comment out if you don't want it)
    print("[reviews] Creating text index on review_text (may take time)…")
    col.create_index([("review_text", "text")])
    print(f"[reviews] Done. Total inserted ≈ {total_inserted}")


def get_db():
    if MONGO_URI:
        client = MongoClient(MONGO_URI)
    else:
        client = MongoClient(
            host=HOST,
            port=PORT,
            username=USERNAME or None,
            password=PASSWORD or None,
            authSource=AUTH_SOURCE,
        )
    return client[DB_NAME]


def main():
    db = get_db()
    import_games(db)
    import_reviews(db)
    print("All done.")


if __name__ == "__main__":
    main()
