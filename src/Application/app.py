#### = sessionstate used... might need to reset it later
import streamlit as st
from st_clickable_images import clickable_images
from pathlib import Path
import sys
SRC = Path(__file__).resolve().parents[1]  # ...\src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from library_api_connector import get_default_game_reco,get_user_recommendation
from Library_fonctions import load_state_from_web_storage, save_state_to_web_storage
from Library_fonctions import load_state_from_query,save_key_to_query,ensure_key_in_query
from library_api_connector import gensar_recommendation_connector,recomandation_genral_connector
import traceback
load_state_from_query()
ensure_key_in_query("User")
ensure_key_in_query("game_name_magasin")

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


# Boutons centrés en grille
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔐 Login"):
        st.switch_page("pages/login.py")

with col2:
    if st.button("🔍 Recherche"):
        st.switch_page("pages/recherche.py")
    

with col3:
    if st.button("📚 Bibliothèque"):
        st.switch_page("pages/bibliotheque.py")
    # if st.button("💬 Chatbot"):
    #     st.switch_page("pages/chatbot.py")
    # if st.button("❓ Questionnaire"):
    #     st.switch_page("pages/question.py")

####
try:
    # if "game_reco" not in st.session_state:
    #     if "User" in st.session_state:
    #         st.session_state.game_reco = get_user_recommendation(st.session_state["User"])
    #     else:
    #         st.session_state.game_reco = get_default_game_reco()
    #     save_state_to_web_storage(STORAGE_KEY,PERSIST_KEYS)
    # message = st.session_state.game_reco  # appel réel de ta fonction

    if "User" in st.session_state:
        st.session_state.game_reco = get_user_recommendation(st.session_state["User"])[1]
        # st.write(st.session_state.game_reco)
        # st.write(gensar_recommendation_connector(st.session_state["User"],top_k=20))
        # st.session_state.game_reco = gensar_recommendation_connector(st.session_state["User"],top_k=20)
        # st.session_state.game_reco = recomandation_genral_connector(st.session_state["User"],top_k_Gensar=20)
    else:
        st.session_state.game_reco = get_default_game_reco()
    st.session_state.default_reco = get_default_game_reco()
    message =  st.session_state.game_reco
except Exception as e:
    st.error(f"Erreur lors du chargement des jeux : {e}")
    st.error(traceback.format_exc())
    message = []

st.markdown("<br><hr><h2 style='text-align:center;'> À la une</h2><br>", unsafe_allow_html=True)

try:
    carousel_items = st.session_state.default_reco[:5]
except:
    carousel_items = []

if len(carousel_items) >= 3:

    # Index actuel dans la session
    if "carousel_index" not in st.session_state:
        st.session_state.carousel_index = 2  # centre

    center = st.session_state.carousel_index
    left = (center - 1) % len(carousel_items)
    right = (center + 1) % len(carousel_items)

    # IMAGES DU CARROUSEL
    imgs = [
        carousel_items[left].get("assets", {}).get("main_capsule", ""),
        carousel_items[center].get("assets", {}).get("main_capsule", ""),
        carousel_items[right].get("assets", {}).get("main_capsule", "")
    ]

    # Cliques possibles : 0 = gauche, 1 = centre, 2 = droite
    clicked = clickable_images(
        imgs,
        titles=["left", "center", "right"],
        div_style={"display": "flex", "justify-content": "center", "gap": "25px"},
        img_style={"border-radius": "14px", "height": "260px"}
    )

    # LOGIQUE DES CLICS
    if clicked == 0:   # image gauche → carrousel vers la gauche
        st.session_state.carousel_index = left

    elif clicked == 2: # image droite → carrousel vers la droite
        st.session_state.carousel_index = right

    elif clicked == 1:  # image centrale → ouvrir magasin
        game_id = carousel_items[center].get("_id", "")
        st.session_state["game_name_magasin"] = game_id
        save_key_to_query("game_name_magasin")
        st.switch_page("pages/magasin.py")

else:
    st.info("Pas assez de jeux pour afficher le carrousel.")



# ---- Gallery with overlapping targets (image open + heart favorite) ----
if message:
    st.subheader(" Jeux recommandés")

    # Récupère les 8 premiers éléments
    items = [game if "main_capsule" in game.get("assets", {}) else None for game in message]
    items = [x for x in items if x is not None]
    imgs = [game.get("assets", {}).get("main_capsule", "") for game in items]
    
    imgs = imgs[:12] if len(imgs) >= 12 else imgs


    idx = clickable_images(
        imgs,
        titles=[f"Image {i}" for i in range(len(imgs))],
        div_style={
            "display": "flex",
            "justify-content": "center",
            "flex-wrap": "wrap",
            "gap": "12px"
        },
        img_style={"border-radius": "12px", "height": "160px"},
    )

    # 👉 Si une image est cliquée → on va DIRECTEMENT au magasin
    if idx > -1:
        selected_game = items[idx]
        game_id = selected_game.get("_id", None)

        # set cookie / query
        st.session_state["game_name_magasin"] = game_id
        save_key_to_query("game_name_magasin")

        # redirection instantanée
        st.switch_page("pages/magasin.py")


else:
    st.warning("Aucun jeu trouvé pour le moment.")

