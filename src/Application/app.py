import streamlit as st
from st_clickable_images import clickable_images
from pathlib import Path
import sys
import html

SRC = Path(__file__).resolve().parents[1]  # ...\src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from library_api_connector import get_default_game_reco


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
try:
    message = get_default_game_reco()  # appel réel de ta fonction
except Exception as e:
    st.error(f"Erreur lors du chargement des jeux : {e}")
    message = []

# ---- Gallery with overlapping targets (image open + heart favorite) ----
if message:
    st.subheader("🎮 Jeux recommandés")

    # Récupère les 8 premiers éléments
    items = message[:8]

    # Build a small normalized list we can use safely in HTML
    norm = []
    for i, it in enumerate(items):
        # try to get a stable id; fallback to index
        gid = str(it.get("appid", i))
        name = str(it.get("name", f"Jeu {i+1}"))
        cover = str(it.get("assets", {}).get("main_capsule", ""))  # your field
        about = str(it.get("about_the_game", ""))

        norm.append({
            "id": gid,
            "name": name,
            "cover": cover,
            "about": about
        })

    # ---- state: favorites + selected (opened) ----
    if "favorites" not in st.session_state:
        st.session_state.favorites = set()
    selected_id = None

    # ---- handle URL actions ----
    qp = st.query_params

    if "fav" in qp:
        fid = str(qp.get("fav"))
        if fid in st.session_state.favorites:
            st.session_state.favorites.remove(fid)
            st.toast(f"Retiré des favoris : {fid}")
        else:
            st.session_state.favorites.add(fid)
            st.toast(f"Ajouté aux favoris : {fid}")
        st.query_params.clear()

    if "open" in qp:
        selected_id = str(qp.get("open"))
        st.query_params.clear()

    # ---- render cards (image opens; heart toggles favorite) ----
    cards = []
    for g in norm:
        fav = g["id"] in st.session_state.favorites
        heart = "♥" if fav else "♡"
        heart_color = "#e0245e" if fav else "#333"

        # Escape text for HTML safety
        title_esc = html.escape(g["name"])
        cover_esc = html.escape(g["cover"])
        gid_esc = html.escape(g["id"])

        cards.append(f"""
        <div class="card">
          <!-- Big target: open the game -->
          <a class="cover" href="?open={gid_esc}" title="Ouvrir {title_esc}">
            <img src="{cover_esc}" alt="{title_esc}" />
          </a>

          <!-- Small overlapping target: toggle favorite -->
          <a class="fav" href="?fav={gid_esc}" title="Mettre en favori">{heart}</a>

          <div class="meta">{title_esc}</div>
        </div>
        """)

    st.markdown(
        f"""
        <style>
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            justify-items: center;
            align-items: start;
            width: 100%;
          }}
          .card {{
            position: relative;
            width: 320px;
            max-width: 90vw;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 4px 18px rgba(0,0,0,.12);
            transition: transform .08s ease;
            background: #fff;
          }}
          .card:hover {{ transform: translateY(-2px); }}
          .card .cover img {{
            display: block;
            width: 100%;
            height: auto;
            aspect-ratio: 16/9;
            object-fit: cover;
          }}
          .card .fav {{
            position: absolute;
            top: 10px; right: 10px;
            z-index: 2; /* sits above the big link */
            text-decoration: none;
            font-size: 18px; line-height: 18px;
            padding: 8px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,.92);
            color: #333;
            box-shadow: 0 2px 8px rgba(0,0,0,.18);
            color: inherit;
          }}
          /* color comes from inline style we set below if favorited */
          .card .meta {{
            padding: 10px 12px;
            font: 600 14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            text-align: center;
          }}
        </style>
        <div class="grid">
          {''.join(cards).replace('class="fav"', 'class="fav" style="color:#e0245e;"' ) if st.session_state.favorites else ''.join(cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If a game was opened, show its details below
    if selected_id:
        game = next((g for g in norm if g["id"] == selected_id), None)
        if game:
            st.markdown(f"### {game['name']}")
            if game["cover"]:
                st.image(game["cover"], use_column_width=True)
            if game["about"]:
                st.write(game["about"])
        else:
            st.info("Jeu introuvable.")

    # Summary of favorites (names)
    if st.session_state.favorites:
        fav_names = [g["name"] for g in norm if g["id"] in st.session_state.favorites]
        st.write("Favoris :", fav_names)
    
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