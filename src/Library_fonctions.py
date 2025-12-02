import streamlit as st
import json
from streamlit_js_eval import streamlit_js_eval
import time
import base64
import urllib.parse

# def _eval_js(js: str, key_prefix: str):
#     # unique key so Streamlit re-runs the JS each call
#     return streamlit_js_eval(js_expressions=js, key=f"{key_prefix}_{uuid.uuid4().hex}")
# def set_cookie(name: str, data, days: int = 365):
#     """
#     Save a cookie `name` with JSON-serialized `data`. Default expiry: 365 days.
#     After setting, st.session_state[name] = data (as Python object), then st.stop().
#     """
#     payload = json.dumps(data)  # serialize in Python
#     js = f"""
#     (function() {{
#         const name = {json.dumps(name)};
#         const value = encodeURIComponent({json.dumps(payload)});
#         const days = {days};
#         const d = new Date();
#         d.setTime(d.getTime() + days*24*60*60*1000);
#         document.cookie = name + "=" + value + "; expires=" + d.toUTCString() + "; path=/; SameSite=Lax";
#         return true;
#     }})()
#     """
#     _eval_js(js, "save_cookie")
#     st.session_state[name] = data
# def get_cookie(name: str):
#     """
#     Get cookie `name`. Stores decoded JSON value into st.session_state[name],
#     or None if it doesn't exist. Then st.stop().
#     """
#     js = f"""
#     (function() {{
#         const target = {json.dumps(name)} + "=";
#         const parts = document.cookie.split(';');
#         for (let c of parts) {{
#             c = c.trim();
#             if (c.indexOf(target) === 0) {{
#                 const raw = c.substring(target.length);
#                 try {{
#                     return decodeURIComponent(raw);
#                 }} catch (e) {{
#                     return raw;
#                 }}
#             }}
#         }}
#         return null;
#     }})()
#     """
#     if name in st.session_state or st.session_state.get(name, None) is not None:
#         return st.session_state[name]
#     result = _eval_js(js, "get_cookie")
#     st.session_state[name] = json.loads(result) if result else None
#     st.stop()
# def delete_cookie(name: str):
#     """
#     Delete cookie `name`. After deletion, st.session_state[name] = None, then st.stop().
#     """
#     js = f"""
#     (function() {{
#         const n = {json.dumps(name)};
#         // Expire immediately (also include past date for broad browser support)
#         document.cookie = n + "=; Max-Age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax";
#         return true;
#     }})()
#     """
#     _eval_js(js, "delete_cookie")
#     st.session_state[name] = None
#     st.stop()
# def check_cookie(name: str):
#     """
#     Check if cookie `name` exists. Sets st.session_state[name] to its JSON-decoded
#     value (or None), returns True/False to Python, then st.stop().
#     """
#     js = f"""
#     (function() {{
#         const target = {json.dumps(name)} + "=";
#         const parts = document.cookie.split(';');
#         for (let c of parts) {{
#             c = c.trim();
#             if (c.indexOf(target) === 0) {{
#                 const raw = c.substring(target.length);
#                 try {{
#                     return ["1", decodeURIComponent(raw)]; // ["exists","value"]
#                 }} catch (e) {{
#                     return ["1", raw];
#                 }}
#             }}
#         }}
#         return ["0", null];
#     }})()
#     """
#     exists_and_val = _eval_js(js, "check_cookie") or ["0", None]
#     exists, raw_val = exists_and_val[0] == "1", exists_and_val[1]
#     st.session_state[name] = json.loads(raw_val) if raw_val else None
#     # (optional) also return exists if you call this from Python code
#     st.stop()
#     return exists

def _unwrap_local_storage(x):
    if isinstance(x, list) and x:
        return x[0]
    return x
def _storage_api(api: str) -> str:
    # 'local' -> window.localStorage ; 'session' -> window.sessionStorage
    return "window.sessionStorage" if api == "session" else "window.localStorage"
def load_state_from_web_storage(
    storage_key_name: str,
    allowed_keys: list[str] | None = None,
    *,
    storage_api: str = "local",     # 'local' or 'session'
    max_age_seconds: int | None = None,  # TTL; if None, no expiry
    version: int | None = None,     # bump when your schema changes
    spinner_msg: str = "Loading saved state…",
) -> None:
    """
    Hydrate st.session_state from Web Storage exactly once per Streamlit session.
    - Blocks only until JS is ready (not when the key is missing).
    - Honors TTL (max_age_seconds) and schema version.
    - storage_api: 'local' (default) or 'session' for per-tab storage.
    Stored format: {"v": version|None, "savedAt": epoch_s, "data": {...}}
    """
    flag = f"__hydrated__{storage_api}__{storage_key_name}"
    if st.session_state.get(flag, False):
        return

    js = f"""
    (function(){{
      const store = {_storage_api(storage_api)};
      const raw = store.getItem({json.dumps(storage_key_name)});
      return {{ ready: true, value: raw }};
    }})()
    """
    resp = _unwrap_local_storage(streamlit_js_eval(js_expressions=js, key=f"hydrate_{storage_api}_{storage_key_name}", want_output=True))

    if resp is None or not isinstance(resp, dict) or not resp.get("ready", False):
        with st.spinner(spinner_msg):
            st.stop()

    raw = resp.get("value", None)  # may be null/None if key missing

    # Decide whether to accept, ignore, or purge the stored value
    accept = False
    data_dict = {}
    if raw:
        try:
            blob = json.loads(raw)
            # accept old plain dicts (no envelope) for backward compat
            if isinstance(blob, dict) and ("data" in blob or "savedAt" in blob or "v" in blob):
                blob_v = blob.get("v", None)
                blob_ts = blob.get("savedAt", None)
                blob_data = blob.get("data", None)
                # version check
                if version is not None and blob_v is not None and blob_v != version:
                    accept = False
                else:
                    # TTL check
                    if max_age_seconds is not None and isinstance(blob_ts, (int, float)):
                        now_s = time.time()
                        if now_s - float(blob_ts) <= max_age_seconds:
                            accept = True
                        else:
                            accept = False
                    else:
                        accept = True
                if accept and isinstance(blob_data, dict):
                    data_dict = blob_data
                elif accept and isinstance(blob, dict) and "data" not in blob:
                    # Fall back if someone saved plain dict into the envelope
                    data_dict = {k: v for k, v in blob.items() if k not in ("v", "savedAt")}
            elif isinstance(blob, dict):
                # legacy: direct dict state
                accept = True
                data_dict = blob
        except Exception:
            accept = False

    if accept:
        for k, v in data_dict.items():
            if (allowed_keys is None or k in allowed_keys) and k not in st.session_state:
                st.session_state[k] = v

    st.session_state[flag] = True  # mark hydrated even if we didn’t accept (missing/expired/version-mismatch)
def save_state_to_web_storage(
    storage_key_name: str,
    allowed_keys: list[str] | None = None,
    *,
    storage_api: str = "local",  # 'local' or 'session'
    version: int | None = None,
) -> None:
    """
    Save selected st.session_state entries into Web Storage as:
      {"v": version|None, "savedAt": epoch_s, "data": {...}}
    """
    payload = {
        k: st.session_state[k]
        for k in (allowed_keys or list(st.session_state.keys()))
        if k in st.session_state
    }
    envelope = {"v": version, "savedAt": time.time(), "data": payload}
    streamlit_js_eval(
        js_expressions=(
            f"{_storage_api(storage_api)}.setItem("
            f"{json.dumps(storage_key_name)}, {json.dumps(json.dumps(envelope))})"
        ),
        key=f"save_{storage_api}_{storage_key_name}",
        want_output=False,
    )

def load_state_from_query() -> None:
    """
    Load all saved data from the `state` query param into st.session_state.

    For each key in the saved state:
      - if the key is missing in st.session_state, it will be added
      - if the key already exists in st.session_state, it is left as-is
    """
    qp = st.query_params
    raw = qp.get("state")

    if not raw:
        return  # nothing saved yet

    # st.query_params can return str or list[str]
    if isinstance(raw, list):
        raw = raw[0]

    try:
        data = json.loads(raw)
    except Exception:
        # bad JSON, just ignore
        return

    if not isinstance(data, dict):
        return

    for k, v in data.items():
        if k not in st.session_state:
            st.session_state[k] = v
def save_key_to_query(key: str) -> None:
    """
    Save a single key from st.session_state into the `state` query param.

    - Raises KeyError if the key is not in st.session_state.
    - Keeps any other existing query params untouched.
    """
    if key not in st.session_state:
        raise KeyError(f"Key '{key}' not found in st.session_state")

    qp = st.query_params
    raw = qp.get("state")

    current_state = {}
    if raw:
        if isinstance(raw, list):
            raw = raw[0]
        try:
            current_state = json.loads(raw)
            if not isinstance(current_state, dict):
                current_state = {}
        except Exception:
            current_state = {}

    current_state[key] = st.session_state[key]

    # Rebuild full query params dict (so we keep other query params)
    new_qp = {k: v for k, v in qp.items()}
    new_qp["state"] = json.dumps(current_state)

    st.query_params.from_dict(new_qp)
def ensure_key_in_query(key: str) -> None:
    """
    Ensure that a key is present in the `state` query param.

    - If the key is already present in the query `state`, do nothing.
    - Else, if the key exists in st.session_state, add/update it in `state`.
    - Else, do nothing.
    """
    qp = st.query_params
    raw = qp.get("state")

    current_state = {}
    if raw:
        if isinstance(raw, list):
            raw = raw[0]
        try:
            current_state = json.loads(raw)
            if not isinstance(current_state, dict):
                current_state = {}
        except Exception:
            current_state = {}

    # If key already in query state, nothing to do
    if key in current_state:
        return

    # If not in session_state, nothing to save
    if key not in st.session_state:
        return

    # Add the value from session_state
    current_state[key] = st.session_state[key]

    # Rebuild full query params dict (keeping other params)
    new_qp = {k: v for k, v in qp.items()}
    new_qp["state"] = json.dumps(current_state)

    st.query_params.from_dict(new_qp)
