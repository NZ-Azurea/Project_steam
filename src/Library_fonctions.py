import streamlit as st
import json
from streamlit_js_eval import streamlit_js_eval
from typing import Any, Optional
import extra_streamlit_components as stx
import time
import datetime as dt


@st.cache_resource
def get_cookie_mgr():
    return stx.CookieManager(key="cookie_manager")
cookie_manager = get_cookie_mgr()


if "cookies_ready" not in st.session_state:
    _ = cookie_manager.get_all()
    time.sleep(0.05)  # tiny delay helps the component settle
    st.session_state["cookies_ready"] = True
    # On Streamlit Cloud or slow networks, you can uncomment:
    # st.rerun()

def set_cookie_value(
    name: str,
    data: Any,
    days: int = 30,
    path: str = "/",
    secure: bool = True,
    samesite: str = "Lax",  # "Lax" | "Strict" | "None"
) -> None:
    """
    Save ANY JSON-serializable value (str/int/float/bool/None/list/dict) in a cookie.
    Stored as JSON text so types round-trip correctly.
    """
    json_text = json.dumps(data, separators=(",", ":"))  # compact JSON
    expires_at = dt.datetime.utcnow() + dt.timedelta(days=days)

    # CookieManager accepts either expires_at *or* max_age (seconds).
    # We'll prefer expires_at for clarity.
    try:
        cookie_manager.set(
            name,
            json_text,
            expires_at=expires_at,   # or max_age=days*24*60*60
            path=path,
            secure=secure,
            same_site=samesite,      # if your version doesn't support this, remove it
        )
    except TypeError:
        # Fallback for older versions without same_site kwarg
        cookie_manager.set(
            name,
            json_text,
            expires_at=expires_at,
            path=path,
            secure=secure,
        )
def get_cookie_value(name: str) -> Optional[Any]:
    """
    Read a cookie and return the original Python value via JSON decoding.
    Returns None if the cookie doesn't exist or JSON is malformed.
    """
    raw = cookie_manager.get(name)
    if raw is None:
        st.session_state[name] = None
    try:
        st.session_state[name] = json.loads(raw)
    except Exception:
        st.session_state[name] = None
def cookie_exists(name: str) -> bool:
    """
    True if the cookie exists in the browser (non-HttpOnly).
    """
    return cookie_manager.get(name) is not None
def delete_cookie(name: str, path: str = "/") -> None:
    st.session_state.pop(name, None)
    cookie_manager.delete(name, path=path)

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