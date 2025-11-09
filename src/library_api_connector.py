import requests
BASE_URL = "http://10.242.254.198:27099"
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