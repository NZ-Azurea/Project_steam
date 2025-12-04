from pathlib import Path
import sys
import streamlit as st
import requests
import os
from agent import ask_model  # Assure-toi que agent.py est correct
from st_click_detector import click_detector

# --- CONFIGURATION DU PATH ---
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Chatbot & Recherche", page_icon="🎮", layout="wide")

# ---- STYLES CSS (Masquer Sidebar + Style Cartes) ----
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
        .stButton button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
col_header_1, col_header_2 = st.columns([8, 1])
with col_header_1:
    st.markdown("## 🤖 Assistant Jeux Vidéo")
with col_header_2:
    if st.button("🏠 Accueil"):
        st.switch_page("app.py") # Vérifie que app.py existe à la racine

# ---- INITIALISATION STATE ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = None # Liste vide ou None au départ

# ---- LAYOUT PRINCIPAL (2 COLONNES) ----
# On divise l'écran : Chat à gauche, Résultats à droite
col_chat, col_results = st.columns([0.55, 0.45], gap="large")

# =========================================================
# COLONNE 1 : LE CHATBOT
# =========================================================
with col_chat:
    st.subheader("Discussion")
    
    # Conteneur scrollable pour l'historique (hauteur fixe)
    # Cela permet de garder l'historique propre sans pousser l'input hors de l'écran
    chat_container = st.container(height=600)
    
    with chat_container:
        # Affichage de l'historique
        for sender, msg in st.session_state.chat_history:
            with st.chat_message(sender):
                st.markdown(msg)

    # Zone de saisie (input)
    # Callback pour envoyer le message
    def handle_submit():
        user_msg = st.session_state.user_input.strip()
        if user_msg:
            st.session_state.chat_history.append(("user", user_msg))
            st.session_state.user_input = "" # Clear input

    st.text_input(
        "Posez votre question (ex: 'Prix de Elden Ring ?')",
        key="user_input",
        on_change=handle_submit
    )

    # LOGIQUE DE RÉPONSE DU BOT
    # On vérifie si le dernier message est de l'utilisateur pour déclencher le bot
    if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
        last_user_msg = st.session_state.chat_history[-1][1]
        previous_history = st.session_state.chat_history[:-1]

        with chat_container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                bot_text = ""
                
                try:
                    # Appel à ton agent (qui remplit last_search_results si besoin)
                    for chunk in ask_model(last_user_msg, history=previous_history):
                        bot_text += chunk
                        placeholder.markdown(bot_text)
                except Exception as e:
                    bot_text = f"❌ Erreur : {e}"
                    placeholder.error(bot_text)
                
                # Sauvegarde historique
                st.session_state.chat_history.append(("assistant", bot_text))
                
                # IMPORTANT : On force le rerun pour mettre à jour la colonne de droite
                # dès que la réponse est finie et que les données sont là
                st.rerun()

# =========================================================
# COLONNE 2 : RÉSULTATS DE RECHERCHE (Ton Code)
# =========================================================
with col_results:
    st.subheader("Résultats de la recherche")

    games = st.session_state.get("last_search_results")

    if games:
        st.success(f"{len(games)} jeux trouvés")
        
        with st.container(height=650):
            html_content = ""
            
            for game in games:
                html_content += f"""
                <a href='#' id='{game["_id"]}' style='text-decoration: none; color: inherit; display: block; margin-bottom: 10px;'>
                    
                    <div style="
                        display: flex; 
                        align-items: center; 
                        background-color: #262730; 
                        border: 1px solid #464b5f; 
                        border-radius: 10px; 
                        padding: 10px; 
                        transition: all 0.2s ease-in-out;
                        "
                        onmouseover="this.style.backgroundColor='#3b3d4a'; this.style.borderColor='#ff4b4b'; this.style.transform='scale(1.01)';"
                        onmouseout="this.style.backgroundColor='#262730'; this.style.borderColor='#464b5f'; this.style.transform='scale(1.0)';"
                        >
                        
                        <div style="flex: 0 0 170px; margin-right: 15px;">
                            <img src="{game['assets']['main_capsule']}" style="width: 100%; border-radius: 5px;">
                        </div>
                        
                        <div style="flex: 1; color: white;">
                            <div style="font-size: 1.2rem; font-weight: 600;">{game['name']}</div>
                            <div style="font-size: 0.85rem; color: #b0b0b0;">
                                genre: {', '.join(game['genres'])}
                            </div>
                        </div>
                        
                        <div style="flex: 0 0 auto; font-size: 1.2rem; font-weight: bold; color: #eee; padding-left: 10px;">
                            {game['price']}€
                        </div>
                    </div>
                </a>
                """
            # 2. DETECTOR
            clicked_id = click_detector(html_content, key="game_list_detector")
            # 3. LOGIC
            if clicked_id:
                st.session_state["game_name_magasin"] = clicked_id
                st.switch_page("pages/magasin.py")
    
    else:
        # Message si aucun jeu n'est chargé
        st.info("Posez une question sur un jeu pour voir les résultats apparaître ici !")
        st.markdown("""
        *Exemples :*
        - *"Combien coûte Cyberpunk 2077 ?"*
        - *"Trouve-moi des jeux de course pas chers"*
        - *"Qui a développé Hades ?"*
        """)