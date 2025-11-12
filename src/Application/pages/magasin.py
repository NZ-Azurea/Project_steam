import streamlit as st
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1]  # ...\src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from library_api_connector import get_default_game_reco, get_game_info
from Library_fonctions import get_cookie, delete_cookie
from Library_fonctions import load_state_from_web_storage, save_state_to_web_storage
from st_clickable_images import clickable_images

STORAGE_KEY = "magasin_Page"
PERSIST_KEYS = ["game_data","game_name_magasin"]
load_state_from_web_storage(STORAGE_KEY,PERSIST_KEYS,max_age_seconds=180)

# mis dans le session_state
get_cookie("game_name_magasin")


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

if "game_name_magasin" not in st.session_state or st.session_state["game_name_magasin"]==None:
    st.session_state["game_name_magasin"] = get_cookie("game_name_magasin")
    save_state_to_web_storage(STORAGE_KEY,PERSIST_KEYS)
    cookie_game_id = st.session_state["game_name_magasin"]
else:
    cookie_game_id = st.session_state["game_name_magasin"]
# --- CAS : Page du jeu dans le magasin ---
if cookie_game_id !=None:
    game_id = cookie_game_id
    st.subheader("📚 Jeu du magasin")
    try:
        st.session_state["game_data"] = get_game_info(game_id)
        save_state_to_web_storage(STORAGE_KEY,PERSIST_KEYS)
        game_data = st.session_state["game_data"]
    except Exception as e:
        st.error(f"Erreur lors de la récupération du jeu : {e}")
        game_data = None

    if game_data:
        # En-tête avec le nom du jeu
        st.markdown(
            f"<h2 style='text-align:center;'>🎮 {game_data.get('name', 'Nom inconnu')}</h2>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Deux colonnes : image principale / infos + petit visuel
        col1, col2 = st.columns([2, 1], gap="large")

        with col1:
            st.image(
                game_data.get("assets", {}).get("main_capsule", ""),
                caption="Aperçu du jeu",
                use_container_width=True,
            )

        with col2:
            

            # Espace vertical avant le prix
            # st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

            # Prix
            price = game_data.get("price", "Non spécifié")
            st.markdown(
                f"<h3 style='text-align:center; color:#00BFFF;'>💰 Prix : {price}</h3>",
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Bouton d’ajout à la bibliothèque
            if st.button("➕ Ajouter à la bibliothèque", use_container_width=True):
                # from Library_fonctions import set_cookie_value
                # game_id = game_data.get("_id", None)
                # if game_id:
                #     set_cookie_value("game_name_biblioteque", game_id)
                #     st.success(f"✅ {game_data.get('name', 'Jeu')} ajouté à la bibliothèque !")
                # else:
                #     st.error("❌ Impossible d'ajouter ce jeu : ID manquant.")
                st.write("Ajouter")

        # Ligne de séparation + description du jeu
        st.divider()
        st.markdown("### 📝 À propos du jeu")
        st.write(game_data.get("about_the_game", "Aucune description disponible."))
    else:
        st.warning("Le jeu n’a pas pu être trouvé dans la base de données.")
else:
    st.write(f"Pas de cookie ")