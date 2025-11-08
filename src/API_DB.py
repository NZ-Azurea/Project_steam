from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError
from bson import json_util
from typing import Any
import json

# ---------- MongoDB CONFIG ----------

username = "NebuZard"
password = "7YAMTHHD"
uri = f"mongodb://{username}:{password}@127.0.0.1:27017/?authSource=admin"

# Short timeout so we don't block forever if Mongo is down
client = MongoClient(uri, serverSelectionTimeoutMS=2000)
db = client["Steam_data"]
db_users = client["Steam_Users"]

# ---------- FASTAPI APP ----------

app = FastAPI()


class CommandRequest(BaseModel):
    command: dict  # raw MongoDB command as a JSON object


class ApiResponse(BaseModel):
    message: Any
    Status: bool


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
