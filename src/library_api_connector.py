import requests
BASE_URL = "http://10.242.216.203:27099"
def get_default_game_reco(verbose=False):
    """
    get top recommended games from /recommandation endpoint
    using the {message, Status} format.
    """
    try:
        resp = requests.get(f"{BASE_URL}/recommandation", timeout=10)
        # Same note: you now always return HTTP 200,
        # but keep it anyway:
        resp.raise_for_status()

        data = resp.json()
        status = data.get("Status")
        message = data.get("message")
        
        if status is True:
            if verbose:
                print("✅ /health OK:", message)
            return message
        else:
            print("❌ /health reported a problem:", message)

    except Exception as e:
        print("❌ /health failed:", e)
def get_game_info(game_id,verbose=False):
    """
    get top recommended games from /recommandation endpoint
    using the {message, Status} format.
    """
    try:
        resp = requests.post(f"{BASE_URL}/Game",params={f"game_id": {game_id}}, timeout=10)
        # Same note: you now always return HTTP 200,
        # but keep it anyway:
        resp.raise_for_status()

        data = resp.json()
        status = data.get("Status")
        message = data.get("message")
        
        if status is True:
            if verbose:
                print("✅ /health OK:", message)
            return message
        else:
            print("❌ /health reported a problem:", message)
    except Exception as e:
        print("❌ /health failed:", e)
        
def add_user(name: str, verbose: bool = False):
    """
    Create a user via POST /user with payload {"name": <name>}.
    Returns `message` on success; prints an error otherwise.
    """
    try:
        resp = requests.post(f"{BASE_URL}/user", json={"name": name}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("Status")
        message = data.get("message")

        if status is True:
            if verbose:
                print(f"✅ /user create OK:", message)
            return message
        else:
            print(f"❌ /user create reported a problem:", message)
    except Exception as e:
        print("❌ /user create failed:", e)
def delete_user(name: str, verbose: bool = False):
    """
    Delete a user via DELETE /user with payload {"name": <name>}.
    Returns `message` on success; prints an error otherwise.
    """
    try:
        resp = requests.delete(f"{BASE_URL}/user", json={"name": name}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("Status")
        message = data.get("message")

        if status is True:
            if verbose:
                print(f"✅ /user delete OK:", message)
            return message
        else:
            print(f"❌ /user delete reported a problem:", message)
    except Exception as e:
        print("❌ /user delete failed:", e)
def rename_user(old_name: str, new_name: str, verbose: bool = False):
    """
    Rename a user via PATCH /user with payload {"old_name": <old>, "new_name": <new>}.
    Returns `message` on success; prints an error otherwise.
    """
    try:
        payload = {"old_name": old_name, "new_name": new_name}
        resp = requests.patch(f"{BASE_URL}/user", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("Status")
        message = data.get("message")

        if status is True:
            if verbose:
                print(f"✅ /user rename OK:", message)
            return message
        else:
            print(f"❌ /user rename reported a problem:", message)
    except Exception as e:
        print("❌ /user rename failed:", e) 
def get_user_by_name(name: str, verbose: bool = False):
    """
    Fetch a user from GET /user?name=<name>.
    Returns `message` on success (the user document), else prints the error.
    """
    try:
        resp = requests.get(f"{BASE_URL}/user", params={"name": name}, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        status = data.get("Status")
        message = data.get("message")

        if status is True:
            if verbose:
                print("✅ /user fetch OK:", message)
            return message
        else:
            print("❌ /user fetch reported a problem:", message)

    except Exception as e:
        print("❌ /user fetch failed:", e)