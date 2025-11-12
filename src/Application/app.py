#### = sessionstate used... might need to reset it later
import streamlit as st
from st_clickable_images import clickable_images
from pathlib import Path
import sys
SRC = Path(__file__).resolve().parents[1]  # ...\src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from library_api_connector import get_default_game_reco
from Library_fonctions import load_state_from_web_storage, save_state_to_web_storage
from Library_fonctions import set_cookie, get_cookie, delete_cookie

## --- Local Storage for session persistence ---
STORAGE_KEY = "App_Page"
PERSIST_KEYS = ["game_reco"]

load_state_from_web_storage(STORAGE_KEY,PERSIST_KEYS,max_age_seconds=180)

st.set_page_config(page_title="Accueil", page_icon="🏠", layout="wide")
# Masquer la sidebar automatique
hide_sidebar_style = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

st.title("Bienvenue sur l'application !")
st.write("Ceci est la page principale `/app`.")

# Boutons centrés en grille
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔐 Login"):
        st.switch_page("pages/login.py")
    if st.button("🏬 Magasin"):
        st.switch_page("pages/magasin.py")

with col2:
    if st.button("🔍 Recherche"):
        st.switch_page("pages/recherche.py")
    if st.button("📚 Bibliothèque"):
        st.switch_page("pages/bibliotheque.py")

with col3:
    if st.button("💬 Chatbot"):
        st.switch_page("pages/chatbot.py")
    if st.button("❓ Questionnaire"):
        st.switch_page("pages/question.py")
    
# --- Récupération des données via ton API Mongo ---

####
try:
    if "game_reco" not in st.session_state:
        st.session_state.game_reco = get_default_game_reco()
        save_state_to_web_storage(STORAGE_KEY,PERSIST_KEYS)
    message = st.session_state.game_reco  # appel réel de ta fonction
except Exception as e:
    st.error(f"Erreur lors du chargement des jeux : {e}")
    message = []

# ---- Gallery with overlapping targets (image open + heart favorite) ----
if message:
    st.subheader("🎮 Jeux recommandés")

    # Récupère les 8 premiers éléments
    items = message[:12] if len(message) >= 12 else message

    imgs = [game.get("assets", {}).get("main_capsule", "") for game in items]
    
    idx = clickable_images(
    imgs,
    titles=[f"Image {i}" for i in range(len(imgs))],
    div_style={"display": "flex","justify-content": "center", "flex-wrap": "wrap", "gap": "12px"},
    img_style={"border-radius": "12px", "height": "160px"},
    )

    if idx > -1:
        selected_game = items[idx]
        game_id = selected_game.get("_id", None)
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.image(imgs[idx], caption=f"Selected: #{idx}")
        with col2:
            st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
            if st.button("➕ Aller au magasin", key=f"add_lib_{idx}"):
                set_cookie("game_name_magasin", game_id)
                st.write(f"Jeu '{get_cookie("game_name_magasin")}' aller au magasin")
                st.switch_page("pages/magasin.py")

    
    # # Affiche les images 2 par ligne
    # for i in range(0, len(items), 2):
    #     _,colA, colB,_ = st.columns([1,1, 1,1])

    #     # Première image
    #     with colA:
    #         if "main_capsule" in items[i]["assets"]:
    #             st.image(items[i]["assets"]["main_capsule"], use_container_width=True)

    #     # Deuxième image (si elle existe)
    #     if i + 1 < len(items):
    #         with colB:
    #             if "main_capsule" in items[i + 1]["assets"]:
    #                 st.image(items[i + 1]["assets"]["main_capsule"], use_container_width=True)
else:
    st.warning("Aucun jeu trouvé pour le moment.")