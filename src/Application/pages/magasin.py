import streamlit as st
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1]  # ...\src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from library_api_connector import get_default_game_reco, get_game_info
from Library_fonctions import get_cookie_value, delete_cookie
from Library_fonctions import load_state_from_web_storage, save_state_to_web_storage
from st_clickable_images import clickable_images

STORAGE_KEY = "magasin_Page"
PERSIST_KEYS = ["game_reco","game_data"]
load_state_from_web_storage(STORAGE_KEY,PERSIST_KEYS,max_age_seconds=180)

# mis dans le session_state
get_cookie_value("game_name_magasin")


st.set_page_config(page_title="Magasin", page_icon="🏬", layout="wide")
# CSS pour cacher le menu latéral
hide_sidebar_style = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>Page du magasin</h1>", unsafe_allow_html=True)

if st.button("🏠 Accueil"):
    st.switch_page("app.py")

cookie_game_id = st.session_state.get("game_name_magasin", "None")

# --- CAS : Page du jeu dans le magasin ---
if cookie_game_id !="None":
    game_id = cookie_game_id
    st.subheader("📚 Jeu ajouté à votre bibliothèque")
    st.write(f"game_id: {game_id}")
    try:
        #  rempli : récupération du jeu via son _id (API ou fonction locale)
        if "game_data" not in st.session_state:
            st.session_state["game_data"] = get_game_info(game_id)
            save_state_to_web_storage(STORAGE_KEY,PERSIST_KEYS)
        game_data = st.session_state["game_data"]
    except Exception as e:
        st.error(f"Erreur lors de la récupération du jeu : {e}")
        game_data = None

    if game_data:
        st.markdown(f"### 🎮 {game_data.get('name', 'Nom inconnu')}")
        st.image(
            
            game_data.get("assets", {}).get("main_capsule", ""),
            caption=game_data.get("name", "Jeu inconnu"),
            use_container_width=True
        )
        st.write(game_data.get("about_the_game", "Aucune description disponible."))
    else:
        st.warning("Le jeu n’a pas pu être trouvé dans la base de données.")
else:
    st.write(f"Pas de cookie ")