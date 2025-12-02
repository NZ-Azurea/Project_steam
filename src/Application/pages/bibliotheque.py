from pathlib import Path
import sys
SRC = Path(__file__).resolve().parents[1]  # ...\src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import streamlit as st
from st_clickable_images import clickable_images
from Library_fonctions import load_state_from_query,save_key_to_query,ensure_key_in_query
from library_api_connector import get_user_by_name, get_games_info  # adapte selon ton fichier
from Library_fonctions import load_state_from_web_storage, save_state_to_web_storage
STORAGE_KEY = "biblioteque_Page"
PERSIST_KEYS = ["user_data"]
# load_state_from_web_storage(STORAGE_KEY,PERSIST_KEYS,max_age_seconds=15)
load_state_from_query()
ensure_key_in_query("User")
ensure_key_in_query("game_name_magasin")
st.set_page_config(page_title="Bibliothèque", page_icon="📚", layout="wide")

# Cacher la sidebar
hide_sidebar_style = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>📚 Votre Bibliothèque</h1>", unsafe_allow_html=True)

if st.button("🏠 Accueil"):
    st.switch_page("app.py")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------
# 1) Récupérer l'utilisateur depuis le cookie
# -----------------------------------------------------
username = st.session_state["User"]

if not username or username == None:
    st.warning("⚠️ Vous devez être connecté pour voir votre bibliothèque.")
    st.stop()

user_data = get_user_by_name(username)

# if "user_data" in st.session_state and st.session_state["user_data"] != None:
#     user_data = st.session_state["user_data"]
# else:
#     st.session_state["user_data"] = get_user_by_name(username)
#     user_data = st.session_state["user_data"]
#     save_state_to_web_storage(STORAGE_KEY,PERSIST_KEYS)

if not user_data:
    st.error("❌ Impossible de charger les données utilisateur.")
    st.stop()

user_status, user_info = user_data  # décompose le tuple
if not user_status:
    st.error("❌ Erreur : utilisateur introuvable.")
    st.stop()

owned_ids = user_info.get("owned_app_ids", [])

if not owned_ids:
    st.info("📭 Votre bibliothèque est vide pour l'instant.")
    st.stop()

games = get_games_info(owned_ids)  # cette fonction renvoie les documents complets des jeux

if not games:
    st.warning("Impossible de charger les informations des jeux.")
    st.stop()

st.subheader("🎮 Vos jeux")

cols = st.columns(1)

imgs = [game.get("assets", {}).get("main_capsule", "") for game in games]
titles = [game.get("name", "Nom inconnu") for game in games]

idx = clickable_images(
    imgs,
    titles=titles,
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap", "gap": "14px"},
    img_style={"border-radius": "12px", "height": "170px"},
)

if idx > -1:
    selected_game = games[idx]
    game_id = selected_game.get("_id")
    st.session_state["game_name_magasin"] = game_id
    st.switch_page("pages/magasin.py")